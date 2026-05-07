"""exp18 sweep driver — runs each case in an isolated subprocess.

Per-case:
  1. Spawn `run_one_case.py` subprocess
  2. Poll process RSS + nvidia-smi VRAM at 1 Hz
  3. If VRAM > vram_limit_mib OR RSS > ram_limit_mib → SIGTERM (then SIGKILL after 5s)
  4. Wait for exit, read result.json
  5. Append to summary, cooldown, next case

Outputs:
  experiments/outputs/exp18_isolated_speed/cases/<case_id>/result.json + embeddings.npy
  experiments/outputs/exp18_isolated_speed/summary.json
  experiments/outputs/exp18_isolated_speed/report.md
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
import yaml

PROJECT_ROOT = Path("/mnt/c/Users/wonjune/workspace/RAG/scion_rag_challenge")
PYTHON = "/home/wonjune/miniconda3/envs/shrag/bin/python"


def get_vram_used_mib() -> int:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True, timeout=5,
        )
        return int(out.strip().splitlines()[0])
    except Exception:
        return -1


def get_proc_rss_mib(pid: int) -> int:
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    kb = int(line.split()[1])
                    return kb // 1024
    except Exception:
        return -1
    return -1


def run_case(case: dict, output_root: Path, corpus: str, vram_limit_mib: int, ram_limit_mib: int, poll_interval: float = 1.0, max_runtime: int = 1800) -> dict:
    case_id = case["case_id"]
    case_dir = output_root / "cases" / case_id
    case_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        PYTHON, str(PROJECT_ROOT / "experiments/exp18_isolated_speed/run_one_case.py"),
        "--model-name", case["model_name"],
        "--embedding-dim", str(case["embedding_dim"]),
        "--embedding-mode", case.get("embedding_mode", "3T+A"),
        "--batch-size", str(case["batch_size"]),
        "--corpus", corpus,
        "--case-id", case_id,
        "--output-dir", str(case_dir),
    ]
    if case.get("torch_dtype"):
        cmd += ["--torch-dtype", str(case["torch_dtype"])]
    if case.get("max_seq_length"):
        cmd += ["--max-seq-length", str(case["max_seq_length"])]
    if case.get("attn_implementation"):
        cmd += ["--attn-impl", str(case["attn_implementation"])]
    if case.get("length_bucket"):
        cmd += ["--length-bucket"]

    print(f"\n=== {case_id} START at {time.strftime('%H:%M:%S')} ===", flush=True)
    print(" ".join(cmd[:6]) + " ... [truncated]", flush=True)

    start = time.time()
    proc = subprocess.Popen(cmd, cwd=str(PROJECT_ROOT))
    peak_rss = 0
    peak_vram = 0
    aborted_reason = None

    while True:
        rc = proc.poll()
        if rc is not None:
            break
        rss = get_proc_rss_mib(proc.pid)
        vram = get_vram_used_mib()
        if rss > 0:
            peak_rss = max(peak_rss, rss)
        if vram > 0:
            peak_vram = max(peak_vram, vram)
        if vram > vram_limit_mib:
            aborted_reason = f"vram_exceeded ({vram} > {vram_limit_mib} MiB)"
            break
        if rss > ram_limit_mib:
            aborted_reason = f"ram_exceeded ({rss} > {ram_limit_mib} MiB)"
            break
        if time.time() - start > max_runtime:
            aborted_reason = f"runtime_exceeded ({max_runtime}s)"
            break
        time.sleep(poll_interval)

    if aborted_reason:
        print(f"  ABORTING ({aborted_reason}) — sending SIGTERM", flush=True)
        try:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                print("  SIGTERM ignored — sending SIGKILL", flush=True)
                proc.kill()
                proc.wait(timeout=5)
        except Exception as e:
            print(f"  kill error: {e}", flush=True)

    elapsed = round(time.time() - start, 2)
    result_path = case_dir / "result.json"
    if result_path.exists():
        result = json.loads(result_path.read_text())
    else:
        result = {"case_id": case_id, "status": "no_result"}
    result["wall_sec"] = elapsed
    result["peak_rss_mib"] = peak_rss
    result["peak_vram_mib"] = peak_vram
    if aborted_reason:
        result["status"] = "aborted"
        result["aborted_reason"] = aborted_reason
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"  status={result.get('status')} wall={elapsed}s vram_peak={peak_vram}MiB rss_peak={peak_rss}MiB", flush=True)
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", required=True, help="YAML with cases list")
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--vram-limit-mib", type=int, default=11500)
    ap.add_argument("--ram-limit-mib", type=int, default=28000)
    ap.add_argument("--cooldown-sec", type=int, default=15)
    ap.add_argument("--max-runtime-per-case", type=int, default=1800)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.cases).read_text(encoding="utf-8"))
    cases = cfg.get("cases", [])
    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)

    summary = {
        "experiment": "exp18_isolated_speed",
        "corpus": args.corpus,
        "vram_limit_mib": args.vram_limit_mib,
        "ram_limit_mib": args.ram_limit_mib,
        "cooldown_sec": args.cooldown_sec,
        "started_at": time.time(),
        "cases": [],
    }

    for i, case in enumerate(cases):
        result = run_case(
            case, output_root, args.corpus,
            vram_limit_mib=args.vram_limit_mib,
            ram_limit_mib=args.ram_limit_mib,
            max_runtime=args.max_runtime_per_case,
        )
        summary["cases"].append(result)
        # write summary incrementally
        Path(output_root / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
        # cooldown
        if i < len(cases) - 1:
            print(f"  cooldown {args.cooldown_sec}s ...", flush=True)
            time.sleep(args.cooldown_sec)

    summary["finished_at"] = time.time()
    Path(output_root / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nSWEEP COMPLETE → {output_root}/summary.json", flush=True)


if __name__ == "__main__":
    main()
