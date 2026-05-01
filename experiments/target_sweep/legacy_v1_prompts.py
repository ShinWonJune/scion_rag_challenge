"""v1 (legacy) 키워드 추출 프롬프트.

이 프롬프트는 본 프로젝트의 Exp1 (target=50, R@10=0.7561 달성) 시점에 사용된
`src/search_pipeline/core/chatgpt_keyword_extractor.py`의 `language="all"` 프롬프트다.
당시 ChatGPT(gpt-4.1-mini, temperature=0)에 system+user 메시지로 단 1회 호출하여
한국어·영어 키워드를 한꺼번에 추출했고, 단순 OR(`|`) 검색어 2개를 생성했다.

본 모듈은 그 v1 동작을 **그대로 재현**할 수 있도록 프롬프트와 카테고라이즈 로직을
별도로 보존한다. `LLMKeywordExtractor`(unified base)와는 호환되지 않는 별개 흐름이다.

사용 예 (codex CLI로 v1 프롬프트를 다시 실험):
    python -m experiments.target_sweep.extract_keywords_codex_v1 \
      --questions data/test.csv --gold data/gold/scienceon_gold.json \
      --output experiments/outputs/frozen/queries_codex_gold41_v1.jsonl
"""
from __future__ import annotations

import re
from typing import Dict, List

# v1 chatgpt extractor가 OpenAI Chat Completions에 보낸 system/user 메시지.
# 그 시점엔 Exp1이 `language="all"` 모드로 돌아갔으므로 "all" 프롬프트만 보존.

V1_SYSTEM_PROMPT = (
    "당신은 과학 논문과 의학 문헌의 키워드 추출 전문가입니다. "
    "주어진 질문이나 텍스트에서 가장 중요하고 검색에 유용한 키워드를 "
    "한국어와 영어 모두 추출해주세요."
)

V1_USER_TEMPLATE = """
다음 질문에서 중요한 키워드를 한국어와 영어로 추출해주세요:

질문: "{query}"

요구사항:
1. 한국어와 영어 키워드 모두 추출
2. 의학, 과학 용어 우선
3. 검색에 유용한 핵심 키워드만
4. 총 5-10개 단어
5. 쉼표로 구분

키워드:"""


def v1_combined_prompt(query: str) -> str:
    """Codex CLI처럼 system/user 분리가 없는 백엔드용 — 두 메시지를 단일 프롬프트로 결합."""
    return f"{V1_SYSTEM_PROMPT}\n\n{V1_USER_TEMPLATE.format(query=query)}"


# ---------------------------------------------------------------------------
# v1 의 _parse_keywords / _categorize_keywords 동작 재현
# ---------------------------------------------------------------------------

def v1_parse_keywords(text: str) -> List[str]:
    """원본 chatgpt_keyword_extractor._parse_keywords 와 동일한 동작.

    - "키워드:" 접두사 제거
    - 콤마 split → strip → len>1 필터
    - 공백·하이픈 split 같은 후처리는 **하지 않음** (v1 원본 그대로)
    """
    text = (text or "").strip()
    text = text.replace("키워드:", "").replace("Keywords:", "").strip()
    parts = [kw.strip() for kw in text.split(",")]
    return [kw for kw in parts if kw and len(kw) > 1]


def v1_categorize(keywords: List[str]) -> Dict[str, List[str]]:
    """원본 _categorize_keywords와 동일: 한글 포함이면 'korean', a-zA-Z\\s\\- 만이면 'english'."""
    korean: List[str] = []
    english: List[str] = []
    for kw in keywords:
        if re.search(r"[가-힣]", kw):
            korean.append(kw)
        elif re.search(r"^[a-zA-Z\s\-]+$", kw):
            english.append(kw)
        # 그 외 (숫자/특수문자 단독 등)는 v1과 동일하게 drop
    return {"korean": korean, "english": english}


def v1_generate_search_terms(keywords: Dict[str, List[str]]) -> List[str]:
    """원본 generate_search_terms 와 동일: 한국어 OR 1개 + 영어 OR 1개."""
    out: List[str] = []
    if keywords.get("korean"):
        out.append("|".join(keywords["korean"]))
    if keywords.get("english"):
        out.append("|".join(keywords["english"]))
    return out
