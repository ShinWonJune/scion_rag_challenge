# ### Instruction ###
# You will receive a user's question that requires retrieving relevant content through internet search to provide an answer.
# There are now the following four rewriting methods, General Search, Rewriting, Keyword Rewriting, Pseudo-Answer Rewriting, Core Content Extraction.
# Please apply four rewriting methods to rewrite the question.
# 
# ### General Search Rewriting ###
# Rewrite the question into a general query for internet search.
# 
# ### Keyword Rewriting ###
# Extract all keywords from the question and separate them with commas, preserving the amount
# of information as in the original question.
# 
# ### Pseudo-Answer Rewriting ###
# Generate an answer for the question, and use the answer to match the real answers from the
# search engine.
# 
# ### Core Content Extraction ###
# Reduce the amount of information in the original question, only extracting the most core content.
# The rewritten query should be more brief than Keyword Rewriting.
# 
# ### Example ###
# Question: Which city was the site where the armistice agreement officially ending World War I
# was signed?
# 
# Output:
#  + General Search Rewriting: City where World War I armistice agreement was signed
#  + Keyword Rewriting: World War I, Armistice, Signing Location
#  + Pseudo-Answer Rewriting: The armistice that ended World War I was signed in the city of Compiegne.
#  + Core Content Extraction: World War I armistice signing city
# 
# Begin! Only output the final result without any additional content. Do not generate any other
# unrelated content.
# 
# Question: {query}
# Output:

def create_query_rewrite_prompt(query: str) -> str:
    """
    사용자 질문을 4가지 방식(일반 검색, 키워드, 유사 답변, 핵심 내용)으로
    재작성하도록 지시하는 프롬프트를 생성합니다.

    Args:
        query (str): 재작성할 사용자의 원본 질문입니다.

    Returns:
        str: AI 모델에 전달할 수 있도록 완성된 프롬프트 문자열입니다.
    """
    # f-string을 사용하여 함수의 인자로 받은 query를 프롬프트에 삽입합니다.
   prompt = f"""### Instruction ###
You will receive a user\'s question that requires retrieving relevant content through internet search to provide an answer. 
There are now the following four rewriting methods, General Search Rewriting, Keyword Rewriting, Pseudo-Answer Rewriting, Core Content Extraction.
Please apply four rewriting methods to rewrite the question.

### General Search Rewriting ###
Rewrite the question into a general query for internet search.

### Keyword Rewriting ###
Extract all keywords from the question and separate them with commas, preserving the amount of information as in the original question.

### Pseudo-Answer Rewriting ###
Generate an answer for the question, and use the answer to match the real answers from the search engine.

### Core Content Extraction ###
Reduce the amount of information in the original question, only extracting the most core content.
The rewritten query should be more brief than Keyword Rewriting.

### Example ###
Question: Which city was the site where the armistice agreement officially ending World War I was signed?

Output:
  + General Search Rewriting: City where World War I armistice agreement was signed
  + Keyword Rewriting: World War I, Armistice, Signing Location
  + Pseudo-Answer Rewriting: The armistice that ended World War I was signed in the city of Compiegne.
  + Core Content Extraction: World War I armistice signing city

Begin! Only output the final result without any additional content. Do not generate any other
unrelated content.

Question: {query}
Output:"""
    return prompt
         
if __name__ == "__main__":
    # --- 함수 사용 예시 ---
    user_question = "How to format a prompt into a Python function?"
    formatted_prompt = create_query_rewrite_prompt(user_question)
    print(formatted_prompt)
