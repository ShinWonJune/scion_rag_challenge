import os
import re
import json
import pandas as pd


# =========================
# 설정 로드: ../datasets.json 또는 ../datastes.json (오타 대비)
# =========================
def load_datasets_config():
    for p in "/workspace/configs/datasets/datasets.json":
        try:
            with open(p, "r", encoding="utf-8") as f:
                print(f"[INFO] Loaded config: {p}")
                return json.load(f)
        except Exception:
            continue
    print("[WARN] Failed to load datasets config. Using empty list.")
    return []


# =========================
# 경로 생성 규칙
# /workspace/src/data/{dataset}/{config|default}/{split}/sample_{N}.csv
# =========================
def build_candidate_paths(entry):
    raw_dataset = entry.get("dataset", "unknown_dataset")
    dataset_name = raw_dataset.split("/")[-1]
    config = entry.get("config", "default")
    splits = entry.get("splits", ["train"])
    sample_rows = entry.get("sample_rows", 100)

    out = []
    for split in splits:
        base_dir = f"/workspace/src/data/{dataset_name}/{config}/{split}"
        csv_path = os.path.join(base_dir, f"sample_{sample_rows}.csv")
        out.append(
            {
                "dataset_name": dataset_name,
                "config": config,
                "split": split,
                "path": csv_path,
            }
        )
    return out


# =========================
# JSON-like 문자열 정규화
# - 단일따옴표 → 이스케이프된 쌍따옴표
# - numpy 스타일 array([...], dtype=object) → [...]
# - 공백/줄바꿈 정리
# =========================
_array_pat = re.compile(
    r"array\(\s*(\[[\s\S]*?\])\s*(?:,\s*dtype\s*=\s*[^)]*)?\)", re.MULTILINE
)


def normalize_jsonish(s: str) -> str:
    if not isinstance(s, str):
        return s
    txt = s.strip()

    # numpy array(...) -> [...]
    txt = _array_pat.sub(r"\1", txt)

    # dict/str 단일따옴표 -> 쌍따옴표 (간단 치환)
    # 단, 내부 따옴표 충돌을 줄이기 위해 먼저 이스케이프 처리
    txt = txt.replace("\\", "\\\\")  # 백슬래시 이스케이프
    txt = txt.replace('"', '\\"')  # 기존 쌍따옴표 이스케이프
    txt = txt.replace("'", '"')  # 단일따옴표 -> 쌍따옴표

    # python True/False/None -> JSON true/false/null
    txt = re.sub(r"\bTrue\b", "true", txt)
    txt = re.sub(r"\bFalse\b", "false", txt)
    txt = re.sub(r"\bNone\b", "null", txt)

    return txt


def try_parse_jsonish(s):
    if not isinstance(s, str):
        return None
    t = s.strip()
    if not (
        (t.startswith("{") and t.endswith("}"))
        or (t.startswith("[") and t.endswith("]"))
    ):
        return None
    try:
        return json.loads(normalize_jsonish(t))
    except Exception:
        return None


# =========================
# 트리 출력 (길이 제한)
# =========================
def print_json_tree(obj, indent=0, max_list_items=2, max_str=160):
    prefix = "  " * indent
    if isinstance(obj, dict):
        for k, v in obj.items():
            print(f"{prefix}- {k}:")
            print_json_tree(v, indent + 1, max_list_items, max_str)
    elif isinstance(obj, list):
        print(f"{prefix}list[{len(obj)}]")
        for i, item in enumerate(obj[:max_list_items]):
            print(f"{prefix}  [{i}]")
            print_json_tree(item, indent + 2, max_list_items, max_str)
        if len(obj) > max_list_items:
            print(f"{prefix}  ... ({len(obj) - max_list_items} more items)")
    else:
        if isinstance(obj, str):
            s = obj.replace("\n", " ")
            if len(s) > max_str:
                s = s[:max_str] + f"... (+{len(obj) - max_str} chars)"
            print(f"{prefix}str: {s}")
        else:
            print(f"{prefix}{type(obj).__name__}: {obj}")


# =========================
# CSV 스캔
# - 파일이 없으면 건너뜀(에러 무시)
# - JSON/JSON-like 셀을 찾으면
#   · 어떤 데이터셋의 것인지 (dataset/config/split) 출력
#   · 트리 구조로 1~N개만 풀어 보여줌
# =========================
def process_csv_for_json_trees(
    csv_path,
    dataset_name,
    config,
    split,
    scan_rows=50,
    max_hits=2,
    candidate_cols=("context", "contexts"),
):
    if not os.path.exists(csv_path):
        print(f"[MISS] {csv_path}")
        return

    try:
        df = pd.read_csv(csv_path, low_memory=False)
    except Exception as e:
        print(f"[WARN] skip {csv_path}: read error ({e})")
        return

    # 스캔 대상 컬럼 (우선순위: candidate_cols -> 나머지 object형 컬럼)
    cols = [c for c in candidate_cols if c in df.columns]
    cols += [c for c in df.columns if c not in cols and df[c].dtype == "object"]

    hits = 0
    for col in cols:
        for idx in range(min(scan_rows, len(df))):
            cell = df.iloc[idx][col]
            parsed = try_parse_jsonish(cell)
            if parsed is not None:
                if hits == 0:
                    print(f"\n[DATASET] {dataset_name} / {config} / {split}")
                print(f"  - file: {csv_path}")
                print(f"  - column: {col}, row: {idx}")
                print("  - JSON tree:")
                print_json_tree(parsed, indent=2)
                hits += 1
                if hits >= max_hits:
                    return
    if hits == 0:
        print(f"[NO-JSON] {csv_path} (scanned up to {scan_rows} rows)")


# =========================
# 메인
# =========================
if __name__ == "__main__":
    entries = load_datasets_config()
    for entry in entries:
        paths = build_candidate_paths(entry)
        for p in paths:
            process_csv_for_json_trees(
                csv_path=p["path"],
                dataset_name=p["dataset_name"],
                config=p["config"],
                split=p["split"],
                scan_rows=50,  # 처음 50행만 훑어 JSON-like 셀 탐지
                max_hits=2,  # 파일당 최대 2개 결과만 출력
                candidate_cols=("context", "contexts", "metadata", "json", "data"),
            )
