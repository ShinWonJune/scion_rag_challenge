def build_scifact_prompt(original_query: str, context_json_string: str) -> str:
    """
    SciFact 데이터셋을 위한 과학적 사실 검증 프롬프트를 생성합니다.
    
    Args:
        original_query: 검증할 과학적 주장
        context_json_string: 검색된 문서들의 JSON 컨텍스트
        
    Returns:
        SciFact 검증용 프롬프트 문자열
    """
    prompt_template = """You are a scientific fact verification system. Your task is to determine whether a scientific claim is SUPPORTED or REFUTED by the provided scientific literature.

Given a scientific claim and relevant research documents, you must:
1. Carefully analyze the provided research Documents
2. Determine if the claim is supported or refuted.
3. Provide a clear, evidence-based response

Do not use external knowledge. Base your analysis solely on the provided context.

--- Context (Research Documents) ---
{context}

--- Scientific Claim to Verify ---
{query}

--- Instructions ---
Analyze the scientific claim above using only the provided research documents.
Determine whether the claim is SUPPORT or REFUTE by the evidence.
Provide a clear explanation based on the scientific evidence in the documents.

Your response should only include:
1. Begin your determination (SUPPORT/REFUTE)
2. Key evidence from the provided documents

"""
    return prompt_template.format(context=context_json_string, query=original_query)