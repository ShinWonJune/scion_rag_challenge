import os
import json
import glob
import argparse
import copy
from typing import List, Dict, Optional
import google.generativeai as genai
import concurrent.futures
from functools import partial
import time

# --- 사용자 요청 기능 구현 ---
from llm_client.init_gemini import init_gemini
from llm_client.call_gemini import call_gemini
from prompts.general.generate_answer_base_v2 import build_prompt
from prompts.scifact.generate_scifact_prompt import build_scifact_prompt
from vllm_client import VLLMClient

# --- Gemini API 호출을 위한 기본 설정 및 함수 ---
"""

python preprocess_and_generate_answer.py     \
  --input_dir /workspace/results/retrieval_docs/250908_235532 --max_rank 1 --parallel True
"""


def process_file(
    filepath: str, max_rank: int, model_obj: Optional[genai.GenerativeModel] = None, 
    vllm_client: Optional[VLLMClient] = None, use_scifact: bool = False
) -> List[Dict]:
    """
    단일 JSON 파일을 처리하는 주 함수입니다.
    1. 파일을 읽고 'original' 타입의 질문을 찾습니다.
    2. 전체 데이터의 'hits'를 max_rank 기준으로 필터링합니다.
    3. 필터링된 전체 JSON 객체를 문자열로 변환하여 컨텍스트로 사용합니다.
    4. 'original' 질문과 컨텍스트로 Gemini API 또는 vLLM을 호출합니다.
    5. 최종 결과를 지정된 형식으로 반환합니다.
    """
    print(f"--- Processing file: {os.path.basename(filepath)} ---")

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"Error reading or parsing {filepath}: {e}")
        return []

    # 1. 'original' 타입의 쿼리를 찾습니다.
    original_query_obj = next(
        (
            q
            for q in data.get("retrieval_results", [])
            if q.get("query_meta", {}).get("type") == "original"
        ),
        None,
    )

    if not original_query_obj:
        print(f"Warning: No 'original' query found in {filepath}. Skipping file.")
        return []

    original_query_text = original_query_obj.get("query")
    if not original_query_text:
        print(f"Warning: 'original' query in {filepath} has no text. Skipping file.")
        return []

    # 2. 전체 데이터를 복사하고 모든 query의 'hits'를 필터링합니다.
    filtered_data = copy.deepcopy(data)
    for query_item in filtered_data.get("retrieval_results", []):
        hits = query_item.get("hits", [])
        # rank가 max_rank 이하인 것들만 남깁니다.
        query_item["hits"] = [
            hit for hit in hits if hit.get("rank", float("inf")) <= max_rank
        ]

    # 3. 필터링된 전체 데이터를 컨텍스트용 JSON 문자열로 변환합니다.
    context_str = json.dumps(filtered_data, ensure_ascii=False, indent=2)

    # 4. 프롬프트를 생성하고 API를 호출합니다.
    # SciFact 사용 여부에 따라 다른 프롬프트 사용
    if use_scifact:
        prompt = build_scifact_prompt(original_query_text, context_str)
    else:
        prompt = build_prompt(original_query_text, context_str)
    
    if vllm_client:
        # vLLM 사용
        api_result = vllm_client.generate_answer_with_prompt(prompt)
        model_name = "vllm"
    elif model_obj:
        # Gemini 사용
        messages = [{"role": "user", "parts": [prompt]}]
        api_result = call_gemini(model_obj, messages)
        model_name = model_obj.model_name
    else:
        raise ValueError("Either model_obj or vllm_client must be provided")

    # 5. 최종 결과 데이터를 구성합니다.
    file_id = data.get("id", os.path.basename(filepath).split("_")[1])
    result_data = {
        "id": file_id,
        "result": api_result,
        "prompt": prompt,
        "model": model_name,
        "prompt_type": "scifact" if use_scifact else "general",
        "retrival": filtered_data,
    }

    print(f"Successfully processed file {os.path.basename(filepath)}.")
    return [result_data]  # 파일 당 하나의 결과가 있으므로 리스트에 담아 반환


# --- ✨ 새로운 병렬 파일 처리 함수 ✨ ---


def process_files_parallel(
    filepaths: List[str],
    max_rank: int,
    model_obj: Optional[genai.GenerativeModel] = None,
    vllm_client: Optional[VLLMClient] = None,
    max_workers: int = 10,
    use_scifact: bool = False,
) -> List[Dict]:
    """
    여러 파일을 병렬로 처리합니다.

    Args:
        filepaths (List[str]): 처리할 JSON 파일 경로의 리스트.
        max_rank (int): 'hits'를 필터링할 최대 순위.
        model_obj (genai.GenerativeModel): 초기화된 Gemini 모델 객체.
        vllm_client (VLLMClient): 초기화된 vLLM 클라이언트 객체.
        max_workers (int): 동시에 실행할 최대 스레드(작업자) 수.

    Returns:
        List[Dict]: 모든 파일에서 집계된 결과 데이터의 리스트.
    """
    if not filepaths:
        print("Warning: No filepaths provided for parallel processing.")
        return []

    # functools.partial을 사용하여 process_file 함수에 고정 인자들을 미리 전달합니다.
    worker_func = partial(
        process_file, 
        max_rank=max_rank, 
        model_obj=model_obj,
        vllm_client=vllm_client,
        use_scifact=use_scifact
    )

    all_results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # executor.map은 각 파일 경로에 대해 worker_func를 실행하고,
        # 결과(각 파일 처리 결과가 담긴 리스트)를 순서대로 반환합니다.
        future_to_path = {
            executor.submit(worker_func, path): path for path in filepaths
        }

        for future in concurrent.futures.as_completed(future_to_path):
            path = future_to_path[future]
            try:
                # process_file은 결과를 리스트([result_data])로 반환하므로,
                # extend를 사용하여 전체 결과 리스트에 추가합니다.
                result = future.result()
                if result:
                    all_results.extend(result)
            except Exception as exc:
                print(f"File '{os.path.basename(path)}' generated an exception: {exc}")

    return all_results


def main():
    """
    스크립트의 메인 실행 함수.
    """
    parser = argparse.ArgumentParser(
        description="JSON 파일을 전처리하고 Gemini API를 사용하여 답변을 생성합니다."
    )
    # 필수 인자
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="처리할 JSON 파일이 있는 입력 디렉토리 경로.",
    )
    parser.add_argument(
        "--max_rank",
        type=int,
        required=True,
        help="포함할 최대 'hit' 순위. 이 순위보다 높은 'hit'은 제외됩니다.",
    )
    # API 관련 인자 (사용자 제공 코드 기반)
    parser.add_argument(
        "--model",
        default="gemini-2.5-flash",
        help="사용할 Gemini 모델 이름 (예: gemini-2.5-pro, gemini-2.5-flash).",
    )
    parser.add_argument(
        "--api_key",
        default=os.environ.get("GOOGLE_API_KEY"),
        help="Google API 키. 설정하지 않으면 GOOGLE_API_KEY 환경 변수를 사용합니다.",
    )
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=None)
    # vLLM 관련 인자 추가
    parser.add_argument(
        "--use-vllm",
        action="store_true",
        help="vLLM을 사용하여 답변을 생성합니다.",
    )
    parser.add_argument(
        "--vllm-url",
        default="http://localhost:8000/v1",
        help="vLLM 서버 URL (기본값: http://localhost:8000/v1).",
    )
    # 출력 디렉토리 인자
    parser.add_argument(
        "--output_dir",
        default="/workspace/data/expr/final_result",
        help="각 결과를 개별 JSON 파일로 저장할 디렉토리 경로.",
    )
    parser.add_argument(
        "--scifact",
        action="store_true",
        help="SciFact 데이터셋용 프롬프트를 사용합니다.",
    )
    parser.add_argument("--parallel", required=False)
    MAX_PARALLEL_FILES = 10  # 동시에 처리할 최대 파일 수
    args = parser.parse_args()

    # vLLM 또는 Gemini 초기화
    model_obj = None
    vllm_client = None
    
    if args.use_vllm:
        print(f"Initializing vLLM client with URL: {args.vllm_url}")
        vllm_client = VLLMClient(base_url=args.vllm_url)
    else:
        if not args.api_key:
            raise ValueError(
                "API 키가 필요합니다. --api_key 인자를 사용하거나 GOOGLE_API_KEY 환경 변수를 설정해주세요."
            )
        print(f"Initializing Gemini model: {args.model}")
        model_obj = init_gemini(args.model, args.api_key, args.temperature, args.seed)

    # 출력 디렉토리 생성
    os.makedirs(args.output_dir, exist_ok=True)

    # 입력 디렉토리에서 모든 관련 JSON 파일 찾기
    search_pattern = os.path.join(args.input_dir, "row_*.json")
    file_paths = glob.glob(search_pattern)

    if not file_paths:
        print(
            f"경고: '{args.input_dir}'에서 'row_*.json' 패턴과 일치하는 파일을 찾을 수 없습니다."
        )
        return

    processed_count = 0
    if not args.parallel:
        for path in file_paths:
            processed_data_list = process_file(
                path, args.max_rank, model_obj=model_obj, vllm_client=vllm_client, use_scifact=args.scifact
            )

            if processed_data_list:
                result_data = processed_data_list[0]
                result_id = result_data["id"]
                output_filename = os.path.join(args.output_dir, f"{result_id}.json")

                with open(output_filename, "w", encoding="utf-8") as f:
                    json.dump(result_data, f, ensure_ascii=False, indent=4)

                processed_count += 1
                print(
                    "Done:"
                    + "["
                    + str(processed_count)
                    + "/"
                    + str(len(processed_data_list))
                    + "]"
                )

        print(
            f"\n✅ 처리가 완료되었습니다. 총 {processed_count}개의 결과 파일이 '{args.output_dir}'에 저장되었습니다."
        )
    if args.parallel:
        # --- 병렬 처리 실행 ---

        print(
            f"Starting parallel processing for {len(file_paths)} files with max {MAX_PARALLEL_FILES} workers..."
        )
        start_time = time.time()

        final_results = process_files_parallel(
            filepaths=file_paths,
            max_rank=args.max_rank,
            model_obj=model_obj,
            vllm_client=vllm_client,
            max_workers=MAX_PARALLEL_FILES,
            use_scifact=args.scifact,
        )

        end_time = time.time()

        print("\n" + "=" * 50)
        print("           PARALLEL PROCESSING COMPLETE")
        print("=" * 50 + "\n")
        # --- 결과 출력 ---
        print(
            f"\n✅ Successfully processed {len(final_results)} out of {len(file_paths)} files."
        )
        print(f"Total execution time: {end_time - start_time:.2f} seconds.")

        for dataaa in final_results:
            result_id = dataaa["id"]
            output_filename = os.path.join(args.output_dir, f"{result_id}.json")

            with open(output_filename, "w", encoding="utf-8") as f:
                json.dump(dataaa, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    main()
