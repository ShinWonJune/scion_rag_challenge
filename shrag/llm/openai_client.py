import os
import random
import time
from typing import Any

from openai import OpenAI, RateLimitError, APIError


class OpenAIClient:
    def __init__(
        self,
        model: str = "gpt-5.4",
        api_key: str | None = None,
        base_url: str | None = None,
        reasoning_effort: str | None = None,
    ):
        resolved_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not resolved_key:
            raise ValueError("OPENAI_API_KEY or --openai-api-key is required for OpenAI generation.")
        kwargs: dict[str, Any] = {"api_key": resolved_key}
        if base_url:
            kwargs["base_url"] = base_url
        self.client = OpenAI(**kwargs)
        self.model = model
        self.reasoning_effort = reasoning_effort

    def generate_answer_with_prompt(self, prompt: str, max_tokens: int = 4000) -> str:
        request: dict[str, Any] = {
            "model": self.model,
            "input": prompt,
            "max_output_tokens": max_tokens,
        }
        if self.reasoning_effort:
            request["reasoning"] = {"effort": self.reasoning_effort}

        max_attempts = 6
        last_error: Exception | None = None
        for attempt in range(max_attempts):
            try:
                response = self.client.responses.create(**request)
                break
            except RateLimitError as exc:
                last_error = exc
                msg = str(exc)
                if attempt + 1 == max_attempts:
                    raise
                wait = min(120.0, 10.0 * (2 ** attempt)) + random.uniform(0, 3)
                print(f"[openai_client] 429 retry {attempt + 1}/{max_attempts} after {wait:.1f}s: {msg[:120]}")
                time.sleep(wait)
            except APIError as exc:
                last_error = exc
                if attempt + 1 == max_attempts:
                    raise
                wait = min(60.0, 5.0 * (2 ** attempt)) + random.uniform(0, 2)
                print(f"[openai_client] APIError retry {attempt + 1}/{max_attempts} after {wait:.1f}s: {str(exc)[:120]}")
                time.sleep(wait)
        else:
            assert last_error is not None
            raise last_error
        text = getattr(response, "output_text", None)
        if text:
            return str(text).strip()

        chunks: list[str] = []
        for item in getattr(response, "output", []) or []:
            for content in getattr(item, "content", []) or []:
                value = getattr(content, "text", None)
                if value:
                    chunks.append(str(value))
        return "".join(chunks).strip()
