from openai import OpenAI


class VLLMClient:
    def __init__(self, base_url: str = "http://localhost:8000/v1", model: str = "openai/gpt-oss-20b"):
        self.client = OpenAI(
            base_url=base_url,
            api_key="token-abc123",
        )
        self.model = model

    def extract_keywords(self, text: str, max_keywords: int = 10) -> str:
        prompt = f"""Extract {max_keywords} key terms from the following text:\n\nText: {text}\n\nReturn only the keywords, separated by commas:"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=8192,
            temperature=0.0,
        )
        content = response.choices[0].message.content
        return (content or "").strip()

    def generate_answer(self, question: str, context: str, max_tokens: int = 8192) -> str:
        prompt = f"""Answer the question based on the provided context.
Answer should be in 2-3 sentences. Final decision should be 'yes', 'no', or 'maybe'.

Context: {context}

Question: {question}

Answer:
Final decision:"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.0,
        )
        content = response.choices[0].message.content
        return (content or "").strip()

    def generate_answer_with_prompt(self, prompt: str, max_tokens: int = 2000) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.0,
        )
        content = response.choices[0].message.content
        return (content or "").strip()
