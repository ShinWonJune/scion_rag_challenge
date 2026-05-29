"""Prompt templates, ported from the original ``shrag`` package.

  * Answer generation — ``prompts/general/generate_answer_base_v3.py``
    (grounding-strict, inline citations, KO/EN structure, escape hatch).
  * Keyword extraction — ``search/extractors/llm_keyword_extractor.py``
    (Korean + English templates, run on every query for cross-lingual recall).

Rendered as LangChain ``ChatPromptTemplate`` objects for LCEL composition.
"""

from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate

from .config import INSUFFICIENT_EN, INSUFFICIENT_KO

# --------------------------------------------------------------------------- #
# Answer generation (grounding v3)
# --------------------------------------------------------------------------- #
_ANSWER_SYSTEM = f"""You are an answering model for a RAG pipeline. Your answer MUST be based only on the Context JSON provided by the user.

== GROUNDING RULES (STRICT) ==
1. Use ONLY the Context to answer. Do NOT use prior knowledge, training data, or external facts.
2. Every factual claim in the answer must be supported by at least one document in the Context.
3. If the Context does not contain enough information to answer the Question, respond with exactly ONE of:
   - For a Korean question: {INSUFFICIENT_KO}
   - For an English question: {INSUFFICIENT_EN}
   In that case, do NOT produce section headers and do NOT guess or fabricate.
4. For every factual claim, append an inline citation referencing the supporting hit, e.g. [doc_id=CN12345] or [rank=2]. Use doc_id if present, otherwise rank.
5. Do NOT pad. If the Context only supports a short answer, keep the answer short. Prefer accuracy over length.

== CONTEXT STRUCTURE ==
The Context is a JSON object with a list of queries. The query whose 'query_meta.type' is 'original' is the one you must answer. Each query has a 'hits' array of retrieved documents with fields such as doc_id, title, abstract, rank. Treat these as the sole evidence. Non-original queries are supplementary evidence only; cite them the same way.

== OUTPUT FORMAT ==
If the Question is in Korean, structure the answer with:
##제목##

##서론##

##본론##

##결론##

If the Question is in English, structure the answer with:
##Title##

##Introduction##

##Main Body##

##Conclusion##

Plain text only. Do not use "*", "-", or "**" for emphasis. Do not use markdown.
If Rule 3 applies (insufficient context), skip the section headers and output only the single insufficient-information sentence."""

_ANSWER_HUMAN = """--- Context ---
{context}

--- Question ---
{query}"""


def build_answer_prompt() -> ChatPromptTemplate:
    """Grounding-strict answer prompt. Variables: ``context``, ``query``."""
    return ChatPromptTemplate.from_messages(
        [("system", _ANSWER_SYSTEM), ("human", _ANSWER_HUMAN)]
    )


# --------------------------------------------------------------------------- #
# Keyword extraction (verbatim from llm_keyword_extractor.py)
# --------------------------------------------------------------------------- #
KOREAN_KEYWORD_TEMPLATE = """당신은 학술 플랫폼 검색 전문가 입니다.
질문에 대한 답변을 포함하는 문서를 검색할 수 있는 키워드를 추출해주세요.
주어진 질문에서 가장 중요하고 검색에 유용한 키워드를 한국어로 추출해주세요.

질문: "{query}"

요구사항:
1. 질의에 포함된 단어를 그대로 추출해 주세요.
2. 질문이 영어로 이루어져도 한국어 키워드를 추출해 주세요. (예: "input-output" → "입력", "출력")
3. 전문용어와 기술용어를 우선적으로 선택하세요.
4. 축약어의 경우에는 축약어와 전체단어 모두 사용 키워드로 추출하세요.
5. 2~5개 단어 추출하세요.
6. 질문을 표할 수 있는 중요도가 높은 순서대로 나열하세요.
7. 키워드에 "()", "-", "/" 등이 포함되지 않도록 하세요.
8. 영어 부연설명 없이 오직 한국어로 키워드를 추출하세요.

출력 형식:키워드1, 키워드2, 키워드3, 키워드4

키워드:"""

ENGLISH_KEYWORD_TEMPLATE = """You are an expert searching for academic platforms.
Please extract keywords that can be used to search for documents containing answers to the question.
From the given query, please extract the most important and useful keywords for the search in English.

Query: "{query}"

Requirements:
1. Extract keywords exactly as they appear in the query when possible.
2. Extract English keywords even if the question is in English. (e.g., "입력-출력" → "input", "output")
3. Prioritize technical terms and scientific jargon
4. In the case of abbreviations, use both the abbreviation and the full term as keywords
5. After extracting the keywords, list 2~5 in order of importance
6. List them in descending order of importance
7. Do not use "()", "-", or "/" in the keywords. If the keyword contains them, split into separate keywords.
8. Extract keywords only in English, without Korean explanations.

Output Format: keyword1, keyword2, keyword3, keyword4

Keywords:"""


def build_keyword_prompt(lang: str) -> ChatPromptTemplate:
    """Keyword prompt for ``lang`` ('korean' | 'english'). Variable: ``query``."""
    template = KOREAN_KEYWORD_TEMPLATE if lang == "korean" else ENGLISH_KEYWORD_TEMPLATE
    return ChatPromptTemplate.from_messages([("human", template)])


def build_structured_keyword_prompt(lang: str) -> ChatPromptTemplate:
    """Keyword prompt that requires JSON for schema-validated parsing."""
    language_name = "Korean" if lang == "korean" else "English"
    return ChatPromptTemplate.from_messages(
        [
            (
                "human",
                """You are an academic-search keyword extractor.

Extract 2 to 5 important search keywords in {language_name}.
Return JSON only. Do not include markdown fences, explanations, numbering, or extra keys.

Required JSON schema:
{{"keywords": ["keyword1", "keyword2"]}}

Question:
{query}
""",
            )
        ]
    ).partial(language_name=language_name)
