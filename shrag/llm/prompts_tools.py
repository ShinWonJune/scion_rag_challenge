from typing import Dict, Any, List, Optional
import json
import re


def build_prompts(
    mode: str, question: str, context: Optional[str], DEFAULT_PROMPT
) -> List[Dict[str, str]]:
    system = DEFAULT_PROMPT[mode]
    user_parts = []
    user_parts.append(f"Question:\n{question}".strip())
    if context:
        user_parts.append(f"\nContext:\n{context}".strip())
    # For Gemini, we pass as a single string with system-style instruction included in the prompt
    prompt = system + "\n\n" + "\n\n".join(user_parts)
    return [{"role": "user", "parts": [prompt]}]


def _extract_json(text: str) -> Any:
    """Extract a JSON object/array from a model response.

    Handles fenced blocks and stray prose. Raises ValueError on failure.
    """
    if text is None:
        raise ValueError("Empty response from model.")
    # Try code fences first
    fence = re.search(r"```(?:json)?\n(.*?)\n```", text, re.DOTALL | re.IGNORECASE)
    candidate = fence.group(1) if fence else text
    # Trim leading/trailing non-json
    start = candidate.find("{")
    start_arr = candidate.find("[")
    if start == -1 or (start_arr != -1 and start_arr < start):
        start = start_arr
    if start == -1:
        raise ValueError("No JSON object/array found in response.")
    # Find last closing brace/bracket
    end = max(candidate.rfind("}"), candidate.rfind("]")) + 1
    payload = candidate[start:end]
    return json.loads(payload)


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    """
    Robust JSONL reader.
    - utf-8-sig 로 열어 BOM 제거.
    - 한 줄에 여분 문자열이 있어도 JSON 부분만 추출 시도.
    - 실패 시 몇 번째 줄인지 함께 에러 표시.
    """
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8-sig", newline="\n") as f:
        for i, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                # 모델 응답 등에서 코드펜스/문구가 섞인 경우를 대비해 JSON 부분만 추출
                try:
                    rows.append(_extract_json(line))
                except Exception as e:
                    raise ValueError(
                        f"Invalid JSON on line {i}: {e}\n"
                        f"Line content (truncated): {line[:200]}..."
                    )
    return rows
