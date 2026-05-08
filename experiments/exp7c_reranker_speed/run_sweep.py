"""Driver: subprocess per reranker model, RAM/VRAM monitor + abort."""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

PROJECT_ROOT = Path("/mnt/c/Users/wonjune/workspace/RAG/scion_rag_challenge")
PYTHON = "/home/wonjune/miniconda3/envs/shrag/bin/python"


def _vram_used_mib() -> int:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True, timeout=5,
        )
        return int(out.strip().splitlines()[0])
    except Exception:
        return -1


def _rss_mib(pid: int) -> int:
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) // 1024
    except Exception:
        return -1
    return -1


def run_model(model: str, candidates: str, queries: str, output_dir: Path,
              vram_limit: int = 11500, ram_limit: int = 28000,
              max_runtime: int = 1200) -> dict:
    case_id = "M_" + model.replace("/", "__").replace("-", "_")
    case_dir = output_dir / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    result_path = case_dir / "result.json"

    cmd = [
        PYTHON, str(PROJECT_ROOT / "experiments/exp7c_reranker_speed/run_one_model.py"),
        "--model", model,
        "--candidates", candidates,
        "--queries", queries,
        "--output", str(result_path),
    ]
    print(f"\n=== {model} START at {time.strftime('%H:%M:%S')} ===", flush=True)

    proc = subprocess.Popen(cmd, cwd=str(PROJECT_ROOT))
    start = time.time()
    peak_rss, peak_vram, aborted = 0, 0, None
    while True:
        rc = proc.poll()
        if rc is not None:
            break
        rss, vram = _rss_mib(proc.pid), _vram_used_mib()
        peak_rss = max(peak_rss, rss) if rss > 0 else peak_rss
        peak_vram = max(peak_vram, vram) if vram > 0 else peak_vram
        if vram > vram_limit:
            aborted = f"vram_exceeded ({vram} > {vram_limit})"
            break
        if rss > ram_limit:
            aborted = f"ram_exceeded ({rss} > {ram_limit})"
            break
        if time.time() - start > max_runtime:
            aborted = f"runtime_exceeded ({max_runtime}s)"
            break
        time.sleep(1.0)
    if aborted:
        print(f"  ABORT: {aborted}", flush=True)
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()

    elapsed = round(time.time() - start, 2)
    if result_path.exists():
        result = json.loads(result_path.read_text())
    else:
        result = {"model": model, "status": "no_result"}
    result["wall_sec"] = elapsed
    result["peak_rss_mib"] = peak_rss
    result["peak_vram_mib"] = peak_vram
    if aborted:
        result["status"] = "aborted"
        result["aborted_reason"] = aborted
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"  status={result.get('status')} wall={elapsed}s vram_peak={peak_vram}MiB", flush=True)
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--queries", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--vram-limit-mib", type=int, default=11500)
    ap.add_argument("--ram-limit-mib", type=int, default=28000)
    ap.add_argument("--cooldown-sec", type=int, default=10)
    args = ap.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    summary = {
        "experiment": "exp7c_reranker_speed",
        "started_at": time.time(),
        "models": [],
    }
    for i, m in enumerate(args.models):
        r = run_model(m, args.candidates, args.queries, output,
                      vram_limit=args.vram_limit_mib, ram_limit=args.ram_limit_mib)
        summary["models"].append(r)
        Path(output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
        if i < len(args.models) - 1:
            print(f"  cooldown {args.cooldown_sec}s ...", flush=True)
            time.sleep(args.cooldown_sec)

    summary["finished_at"] = time.time()
    Path(output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nSWEEP DONE → {output}/summary.json", flush=True)


if __name__ == "__main__":
    main()
