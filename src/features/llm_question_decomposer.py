# llm_processor.py
"""
LLM을 사용하여 multi-hop 질문을 single-hop으로 변환하는 핵심 로직.
이 모듈은 단일 레코드 처리에만 집중하며, 파일 입출력이나 배치 처리는 다루지 않습니다.
"""

from typing import Dict, Any, Optional

# 필요한 유틸리티 및 클라이언트 함수들을 임포트합니다.
# 실제 프로젝트 구조에 맞게 경로를 조정해야 할 수 있습니다.
from llm_client.call_gemini import call_gemini
from llm_client.prompts_tools import build_prompts
from prompts.general.breakdown_multi_hop_question_as_single_hop import DEFAULT_PROMPT


def process_single_record(
    record: Dict[str, Any],
    mode: str,
    model_obj: Any,
    qfield: str,
    cfield: Optional[str],
) -> Dict[str, Any]:
    """
    하나의 데이터 레코드를 받아 multi-hop 질문을 single-hop으로 변환합니다.

    이 함수는 프롬프트를 만들고, LLM을 호출하며, 반환된 결과를 구조화합니다.

    Args:
        record (Dict[str, Any]): 질문과 컨텍스트를 포함한 입력 데이터 딕셔너리.
        mode (str): 처리 모드 ('decompose', 'chain', 'rewrite').
        model_obj (Any): 초기화된 Gemini 모델 객체.
        qfield (str): 레코드에서 질문에 해당하는 필드 이름.
        cfield (Optional[str]): 레코드에서 컨텍스트에 해당하는 필드 이름.

    Returns:
        Dict[str, Any]: LLM이 처리한 결과가 담긴 딕셔너리.

    Raises:
        ValueError: 필수 필드가 누락되었거나 모델의 출력 형식이 예상과 다를 경우 발생합니다.
    """
    question = record.get(qfield)
    if not question:
        raise ValueError(f"레코드에 '{qfield}' 필드가 없습니다: {record}")

    context = record.get(cfield) if cfield else None

    # 1. 모드에 따라 프롬프트 메시지를 생성합니다.
    messages = build_prompts(mode, question, context, DEFAULT_PROMPT)

    # 2. LLM을 호출하고 파싱된 JSON 응답을 받습니다.
    parsed = call_gemini(model_obj, messages)

    # 3. 기본 출력 구조를 설정합니다.
    out: Dict[str, Any] = {
        "id": record.get("id"),
        "mode": mode,
        **(
            {"question": question} if "question" not in record else {}
        ),
    }

    # 4. 모드에 따라 파싱된 결과를 out 딕셔너리에 추가합니다.
    if mode == "decompose":
        if not isinstance(parsed, dict) or "single_hop_questions" not in parsed:
            raise ValueError(
                f"'decompose' 모드의 모델 출력이 예상과 다릅니다: {parsed}"
            )
        out["single_hop_questions"] = parsed["single_hop_questions"]
    elif mode == "chain":
        if not isinstance(parsed, dict) or "steps" not in parsed:
            raise ValueError(f"'chain' 모드의 모델 출력이 예상과 다릅니다: {parsed}")
        out["steps"] = parsed["steps"]
    elif mode == "rewrite":
        key = "single_hop_question"
        if not isinstance(parsed, dict) or key not in parsed:
            raise ValueError(f"'rewrite' 모드의 모델 출력이 예상과 다릅니다: {parsed}")
        out[key] = parsed[key]
    else:
        raise ValueError(f"알 수 없는 모드입니다: {mode}")

    # 5. 원본 레코드의 다른 필드(예: context, answer)를 결과에 포함시킵니다.
    for k in ("context", "answer"):
        if k in record:
            out[k] = record[k]

    return out
