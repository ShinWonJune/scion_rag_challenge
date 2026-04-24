from __future__ import annotations

import json
from typing import Any

import google.generativeai as genai
from openai import OpenAI


class HyDETransform:
    def __init__(
        self,
        backend: str = "vllm",
        model: str = "openai/gpt-oss-20b",
        base_url: str = "http://localhost:8000/v1",
        api_key: str | None = None,
        n_samples: int = 1,
        language: str = "auto",
        seed: int = 42,
    ) -> None:
        self.backend = backend
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.n_samples = n_samples
        self.language = language
        self.seed = seed
        self._client: Any = None
        if backend == "vllm":
            self._client = OpenAI(base_url=base_url, api_key="dummy-key")
        elif backend == "gemini":
            genai.configure(api_key=api_key)
            self._client = genai.GenerativeModel(model)
        else:
            raise ValueError(f"Unsupported HyDE backend: {backend}")

    def transform(self, question: str) -> list[str]:
        prompt = (
            "Write a short hypothetical academic abstract that would answer the question. "
            "Keep it factual in tone, concise, and focused on retrieval cues.\n\n"
            f"Question: {question}"
        )
        outputs: list[str] = []
        for _ in range(self.n_samples):
            if self.backend == "vllm":
                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=512,
                    temperature=0.0,
                    extra_body={"seed": self.seed},
                )
                outputs.append((response.choices[0].message.content or "").strip())
            else:
                response = self._client.generate_content(
                    prompt,
                    generation_config=genai.GenerationConfig(
                        temperature=0.0,
                        candidate_count=1,
                    ),
                )
                outputs.append((response.text or "").strip())
        return [text for text in outputs if text]
