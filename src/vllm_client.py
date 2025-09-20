"""
vLLM OpenAI 호환 클라이언트 예제
기존 Gemini API 대신 사용할 클라이언트
"""
from openai import OpenAI
import os

class VLLMClient:
    def __init__(self, base_url="http://localhost:8000/v1"):
        self.client = OpenAI(
            base_url=base_url,
            api_key="token-abc123",  # vLLM은 dummy key 사용
        )
    
    def extract_keywords(self, text, max_keywords=10):
        """키워드 추출"""
        prompt = f"""Extract {max_keywords} key terms from the following text:
        
        Text: {text}
        
        Return only the keywords, separated by commas:"""
        
        response = self.client.chat.completions.create(
            model="openai/gpt-oss-120B",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=8192,
            temperature=0.3
        )
        
        # 전체 응답 확인
        print("="*50)
        print("Full response:", response)
        print("="*50)
        print("Content:", response.choices[0].message.content)
        print("="*50)
        
        if response.choices[0].message.content is None:
            return "No content generated"

        return response.choices[0].message.content.strip()

    def generate_answer(self, question, context, max_tokens=8192):
        """답변 생성"""
        prompt = f"""Answer the question based on the provided context.
        
        Context: {context}
        
        Question: {question}
        
        Answer:"""
        
        response = self.client.chat.completions.create(
            model="openai/gpt-oss-120B",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.1
        )

        print("="*50)
        print("Full response:", response)
        print("="*50)
        print("Content:", response.choices[0].message.content)
        print("="*50)

        return response.choices[0].message.content.strip()

# 사용 예시
if __name__ == "__main__":
    client = VLLMClient()  # 같은 컨테이너 내에서 localhost 사용
    
    # 키워드 추출 테스트
    text = "Machine learning is transforming various industries."
    keywords = client.extract_keywords(text)
    print(f"Keywords: {keywords}")
    
    # 답변 생성 테스트
    question = "What is machine learning?"
    context = "Machine learning is a subset of artificial intelligence that enables computers to learn from data."
    answer = client.generate_answer(question, context)
    print(f"Answer: {answer}")
