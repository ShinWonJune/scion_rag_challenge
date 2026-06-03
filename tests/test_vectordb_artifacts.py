from __future__ import annotations

import json
from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path

from shrag.data_handler.for_embedding import save_results
from shrag.pipeline import run as pipeline_run
from shrag.pipeline.steps.step3_build_vectordb import run_step3


@dataclass
class _Doc:
    doc_id: str
    title: str
    embedding: list[float]


def test_save_results_uses_run_output_dir_override(tmp_path: Path) -> None:
    config_path = tmp_path / "encoder.json"
    legacy_dir = tmp_path / "legacy-vectordb"
    run_dir = tmp_path / "run" / "vectordb"
    config = {
        "nickname": "test",
        "output_dir": str(legacy_dir),
        "output_file": "old.csv",
    }
    config_path.write_text(json.dumps(config), encoding="utf-8")

    save_results(
        config=config,
        documents_to_save=[_Doc("D1", "Title", [0.1, 0.2])],
        embedding_shape=(1, 2),
        document_class=_Doc,
        config_path=str(config_path),
        model_name="model/name",
        output_dir_override=str(run_dir),
    )

    updated = json.loads(config_path.read_text(encoding="utf-8"))
    output_file = Path(updated["output_file"])

    assert output_file.exists()
    assert output_file.parent.parent == run_dir
    assert output_file.name == "vector_db_test_model_name.csv"
    assert not legacy_dir.exists()


def test_run_step3_forwards_output_dir(monkeypatch) -> None:
    calls = []

    def fake_run(cmd, check, env):
        calls.append((cmd, check, env))

    monkeypatch.setattr("shrag.pipeline.steps.step3_build_vectordb.subprocess.run", fake_run)

    run_step3(
        encoder="encoder.json",
        docs="docs.jsonl",
        schema="schema.json",
        output_dir="outputs/run_x/vectordb",
    )

    cmd, check, env = calls[0]
    assert check is True
    assert env["PYTHONIOENCODING"] == "utf-8"
    assert "--output_dir" in cmd
    assert cmd[cmd.index("--output_dir") + 1] == "outputs/run_x/vectordb"


def test_pipeline_manifest_tracks_run_local_vectordb(monkeypatch, tmp_path: Path) -> None:
    output_root = tmp_path / "run"
    encoder_config = tmp_path / "encoder.json"
    questions_csv = tmp_path / "questions.csv"
    encoder_config.write_text(json.dumps({"output_file": None}), encoding="utf-8")
    questions_csv.write_text("id,question\nQ1,What is alpha?\n", encoding="utf-8")

    monkeypatch.setattr(
        pipeline_run,
        "parse_args",
        lambda: Namespace(
            questions=str(questions_csv),
            encoder=str(encoder_config),
            llm="gemini",
            sources="scienceon",
            extractor="gemini",
            chatgpt_model=None,
            extractor_model="gpt-4o-mini",
            llm_model="gpt-5.4",
            extractor_temperature="0",
            keyword_lang="all",
            vllm_url="http://localhost:8000/v1",
            vllm_model="openai/gpt-oss-20b",
            openai_base_url=None,
            openai_reasoning_effort=None,
            max_answer_tokens=4000,
            scienceon_credentials="scienceon.json",
            scienceon_max_pages=5,
            scienceon_max_concurrency=1,
            scienceon_min_interval_sec=0.5,
            scienceon_fixed_concurrency=False,
            scienceon_max_retries=5,
            scienceon_retry_base_sleep_sec=2.0,
            scienceon_retry_max_sleep_sec=60.0,
            cache_root="outputs/_shared_cache",
            no_cache=False,
            frozen_queries=None,
            pubmed_credentials="pubmed.json",
            schema="schema.json",
            output=str(output_root),
            decompose=False,
            decompose_model="gemini-2.5-flash",
            target_documents=50,
            top_k=50,
            max_rank=5,
        ),
    )
    monkeypatch.setattr(pipeline_run, "build_components", lambda args: (object(), {"scienceon": object()}))
    monkeypatch.setattr(pipeline_run, "_git_sha_and_dirty", lambda: ("sha", False))
    monkeypatch.setattr(pipeline_run, "_file_sha1", lambda path: "sha1")
    monkeypatch.setattr(pipeline_run, "_credential_fingerprint", lambda path: "fingerprint")

    def fake_step1(**kwargs):
        out_dir = Path(kwargs["output_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)
        meta = out_dir / "search_meta_results.json"
        docs = out_dir / "search_documents.jsonl"
        meta.write_text(json.dumps({"request_stats": {"cache": {}}}), encoding="utf-8")
        docs.write_text(json.dumps({"doc_id": "D1"}) + "\n", encoding="utf-8")
        return str(meta), str(docs)

    def fake_step3(encoder, docs, schema, gpu_id=None, output_dir=None):
        vectordb_csv = Path(output_dir) / "260101_000000" / "vector_db_test.csv"
        vectordb_csv.parent.mkdir(parents=True, exist_ok=True)
        vectordb_csv.write_text("doc_id,embedding\nD1,\"[0.1]\"\n", encoding="utf-8")
        encoder_config.write_text(
            json.dumps({"output_file": str(vectordb_csv)}),
            encoding="utf-8",
        )

    def fake_step4(**kwargs):
        assert kwargs["vectordb"] == str(output_root / "vectordb" / "260101_000000" / "vector_db_test.csv")
        return kwargs["output_dir"]

    monkeypatch.setattr(pipeline_run, "run_step1", fake_step1)
    monkeypatch.setattr(pipeline_run, "run_step3", fake_step3)
    monkeypatch.setattr(pipeline_run, "run_step4", fake_step4)
    monkeypatch.setattr(pipeline_run, "run_step5", lambda **kwargs: None)

    pipeline_run.main()

    manifest = json.loads((output_root / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["artifacts"]["vectordb_output_dir"] == str(output_root / "vectordb")
    assert manifest["artifacts"]["vectordb_csv"] == str(
        output_root / "vectordb" / "260101_000000" / "vector_db_test.csv"
    )
