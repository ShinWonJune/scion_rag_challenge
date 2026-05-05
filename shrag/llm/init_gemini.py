from typing import List, Dict, Optional, Any
import google.generativeai as genai
import os
from google.generativeai.types import HarmCategory, HarmBlockThreshold
# def init_gemini(
#     model_name: str, api_key: str, temperature: float, seed: Optional[int]
# ) -> genai.GenerativeModel:
#     """
#     Gemini 모델을 초기화합니다.
#     사용자 제공 코드 기반.
#     """
#     genai.configure(api_key=api_key)
#     generation_config = genai.GenerationConfig(
#         temperature=temperature,
#         candidate_count=1,
#     )
#     # 결정론적 출력을 위해 시드 설정
#     if seed is not None:
#         generation_config.seed = seed

#     return genai.GenerativeModel(
#         model_name=model_name,
#         generation_config=generation_config,
#     )


def init_gemini(
    model: str, api_key: Optional[str], temperature: float, seed: Optional[int]
) -> Any:
    if genai is None:
        raise RuntimeError(
            "google-generativeai is not installed.\nInstall: pip install google-generativeai"
        )
    key = api_key or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError("Missing API key. Set --api_key or GOOGLE_API_KEY env var.")
    genai.configure(api_key=key)
    generation_config = {
        "temperature": temperature,
        "candidate_count": 1,
        "max_output_tokens": 2048,
        "response_mime_type": "application/json",
    }
    safety_settings = None  # use defaults
    # Safty release
    safety_settings = {
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    }

    model_obj = genai.GenerativeModel(
        model_name=model,
        generation_config=generation_config,
        safety_settings=safety_settings,
    )
    # Set seed if supported
    if seed is not None:
        try:
            model_obj._generation_config["seed"] = seed  # type: ignore[attr-defined]
        except Exception:
            pass
    return model_obj
