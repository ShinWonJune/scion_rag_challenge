"""exp7d driver — subprocess per cell with VRAM/RAM monitor + abort."""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
import yaml

PROJECT_ROOT = Path("/mnt/c/Users/wonjune/workspace/RAG/scion_rag_challenge")
PYTHON = "/home/wonjune/miniconda3/envs/shrag/bin/python"


def _vram() -> int:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True, timeout=5,
        )
        return int(out.strip().splitlines()[0])
    except Exception:
        return -1


def _rss(pid: int) -> int:
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) // 1024
    except Exception:
        return -1
    return -1


def run_case(case: dict, output: Path, corpus: str, queries: str, gold: str,
             vram_limit: int, ram_limit: int, max_runtime: int = 1500) -> dict:
    case_id = case["case_id"]
    case_dir = output / case_id
    case_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        PYTHON, str(PROJECT_ROOT / "experiments/exp7d_retriever_reranker_stack/run_one_cell.py"),
        "--retriever", case["retriever"],
        "--reranker", case["reranker"],
        "--candidates", str(case["candidates"]),
        "--corpus", corpus,
        "--queries", queries,
        "--gold", gold,
        "--output", str(case_dir),
        "--retriever-batch", str(case.get("retriever_batch", 32)),
        "--reranker-batch", str(case.get("reranker_batch", 16)),
        "--max-seq-length", str(case.get("max_seq_length", 512)),
        "--torch-dtype", str(case.get("torch_dtype", "float16")),
    ]
    if case.get("prefix_passage"):
        cmd += ["--prefix-passage", case["prefix_passage"]]
    if case.get("prefix_query"):
        cmd += ["--prefix-query", case["prefix_query"]]

    print(f"\n=== {case_id} START at {time.strftime('%H:%M:%S')} ===", flush=True)

    proc = subprocess.Popen(cmd, cwd=str(PROJECT_ROOT))
    start = time.time()
    peak_rss, peak_vram, aborted = 0, 0, None
    while True:
        rc = proc.poll()
        if rc is not None:
            break
        rss, vram = _rss(proc.pid), _vram()
        peak_rss = max(peak_rss, rss) if rss > 0 else peak_rss
        peak_vram = max(peak_vram, vram) if vram > 0 else peak_vram
        if vram > vram_limit:
            aborted = f"vram_exceeded ({vram}>{vram_limit})"; break
        if rss > ram_limit:
            aborted = f"ram_exceeded ({rss}>{ram_limit})"; break
        if time.time() - start > max_runtime:
            aborted = f"runtime_exceeded ({max_runtime}s)"; break
        time.sleep(1.0)
    if aborted:
        print(f"  ABORT: {aborted}", flush=True)
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()

    elapsed = round(time.time() - start, 2)
    rj = case_dir / "result.json"
    result = json.loads(rj.read_text()) if rj.exists() else {"case_id": case_id, "status": "no_result"}
    result["case_id"] = case_id
    result["wall_sec"] = elapsed
    result["peak_rss_mib"] = peak_rss
    result["peak_vram_mib"] = peak_vram
    if aborted:
        result["status"] = "aborted"
        result["aborted_reason"] = aborted
    rj.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"  status={result.get('status')} wall={elapsed}s vram_peak={peak_vram}MiB", flush=True)
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--queries", required=True)
    ap.add_argument("--gold", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--vram-limit-mib", type=int, default=11500)
    ap.add_argument("--ram-limit-mib", type=int, default=28000)
    ap.add_argument("--cooldown-sec", type=int, default=15)
    ap.add_argument("--max-runtime-per-case", type=int, default=1500)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.cases).read_text(encoding="utf-8"))
    cases = cfg.get("cases", [])
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    summary = {"experiment": "exp7d_retriever_reranker_stack", "started_at": time.time(), "cases": []}
    for i, case in enumerate(cases):
        r = run_case(case, output, args.corpus, args.queries, args.gold,
                     vram_limit=args.vram_limit_mib, ram_limit=args.ram_limit_mib,
                     max_runtime=args.max_runtime_per_case)
        summary["cases"].append(r)
        Path(output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
        if i < len(cases) - 1:
            print(f"  cooldown {args.cooldown_sec}s ...", flush=True)
            time.sleep(args.cooldown_sec)

    summary["finished_at"] = time.time()
    Path(output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nSWEEP DONE → {output}/summary.json", flush=True)


if __name__ == "__main__":
    main()
