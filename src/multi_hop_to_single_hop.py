#!/usr/bin/env python3
"""
변경사항:
output "original_question" -> "question" 

이유:
멀티홉을 거치지 않은 원본 question.jsonl 도 retrieve_system.main.py에서 처리할 수 있도록 하기 위함.
retrieval_system.main 의 output key "query"와 "question" 이 매핑되도록 수정


multi_hopt_to_single_hop.py

Convert multi-hop questions into single-hop questions using Google's Gemini models.

Features
- Reads JSONL input with fields for question/context.
- Modes: 
    * decompose: produce a minimal set of single-hop Qs that collectively solve the original.
    * chain: ordered decomposition where each Q depends on prior answers.
    * rewrite: rewrite into a single simpler (single-hop) question that preserves answerability.
- Strict JSON outputs, robust parsing, retries with exponential backoff, optional rate limiting.
- Works with environment variable GOOGLE_API_KEY or --api_key.

Example
-------
python multi_hop_to_single_hop.py \
  --input /workspace/data/rag_test_data/questions.jsonl \
  --output /workspace/data/expr/singlehop_decompose.jsonl \
  --question_field question --context_field context \
  --mode decompose --model gemini-2.5-flash 

Input JSONL example (one JSON per line):
{"id":"hotpot_0001", "question":"Which author wrote the novel adapted into the film directed by...", "context":"<optional extra context or passages>", "answer":"<optional>"}

Output JSONL example (decompose):
{"id":"hotpot_0001", "mode":"decompose", "single_hop_questions":["Who directed ...?", "Which novel was ...?", "Who wrote that novel?"], "meta":{"model":"gemini-1.5-pro","temperature":0.2}}

Notes
-----
- If your dataset lacks context, the model will rely on world knowledge. You can pass retrieved passages in `context` to improve faithfulness.
- To throttle requests: use --qps 1.5 (max queries per second).
- To ensure deterministic output: set --temperature 0.0 and --seed.
"""

#!/usr/bin/env python3
# main.py
"""
Multi-hop 질문을 Single-hop으로 변환하는 프로세스 실행 스크립트.

이 스크립트의 역할:
- 커맨드 라인 인자 파싱
- 입력 데이터 로딩 (JSONL 파일 또는 직접 입력)
- LLM 클라이언트 초기화
- 데이터 순회, 속도 제한(rate limiting), 오류 처리 관리
- 각 데이터에 대한 핵심 로직(llm_processor) 호출
- 최종 결과를 파일에 저장하거나 콘솔에 출력
"""
import argparse
import json
import time
from typing import Dict, Any, List

# --- 핵심 로직 임포트 ---
# llm_processor.py 파일이 같은 디렉토리나 PYTHONPATH에 있다고 가정합니다.
from features.llm_question_decomposer import process_single_record

# --- 유틸리티 및 클라이언트 임포트 ---
from llm_client.init_gemini import init_gemini
from llm_client.prompts_tools import read_jsonl
from utils.write_jsonl import write_jsonl


def run_batch_processing(args: argparse.Namespace, model_obj: Any):
    """입력 파일에서 레코드를 읽어 처리하고, 출력 파일에 저장합니다."""
    if not args.output:
        raise ValueError("--input 사용 시에는 반드시 --output을 지정해야 합니다.")

    records = read_jsonl(args.input)
    results: List[Dict[str, Any]] = []

    last_request_time = 0.0
    min_interval = 1.0 / args.qps if args.qps and args.qps > 0 else 0.0

    print(
        f"'{args.input}' 파일의 총 {len(records)}개 레코드에 대한 배치 처리를 시작합니다..."
    )

    for idx, rec in enumerate(records, start=1):
        # 1. API 요청 속도 제어
        now = time.time()
        if min_interval > 0 and now - last_request_time < min_interval:
            time.sleep(min_interval - (now - last_request_time))

        # 2. 핵심 로직 호출 및 예외 처리
        try:
            print(f"({idx}/{len(records)}) 처리 중 (ID: {rec.get('id', 'N/A')})...")
            processed_result = process_single_record(
                rec, args.mode, model_obj, args.question_field, args.context_field
            )
            # 실행 관련 메타데이터 추가
            processed_result.setdefault("meta", {})
            processed_result["meta"].update(
                {"model": args.model, "temperature": args.temperature}
            )
            results.append(processed_result)

        except Exception as e:
            print(f"오류 발생 (레코드 {idx}, ID: {rec.get('id', 'N/A')}): {e}")
            # 실패한 경우, 에러 정보를 결과에 기록하고 계속 진행
            results.append(
                {
                    "id": rec.get("id"),
                    "mode": args.mode,
                    "error": str(e),
                    "original_record": rec,
                }
            )

        last_request_time = time.time()

    # 3. 최종 결과 저장
    write_jsonl(args.output, results)
    print(f"\n✅ 총 {len(results)}개의 결과를 '{args.output}' 파일에 저장했습니다.")


def run_single_processing(args: argparse.Namespace, model_obj: Any):
    """커맨드 라인으로 직접 입력받은 단일 질문을 처리합니다."""
    record = {"id": None, "question": args.question}
    if args.context:
        record["context"] = args.context

    try:
        result = process_single_record(
            record, args.mode, model_obj, "question", "context"
        )
        result.setdefault("meta", {})
        result["meta"].update({"model": args.model, "temperature": args.temperature})

        print(json.dumps(result, ensure_ascii=False, indent=2))

    except Exception as e:
        print(f"처리 중 오류가 발생했습니다: {e}")
        error_output = {"error": str(e), "original_record": record}
        print(json.dumps(error_output, ensure_ascii=False, indent=2))


def main():
    """메인 함수: 인자를 파싱하고 전체 프로세스를 조율합니다."""
    parser = argparse.ArgumentParser(
        description="Gemini 모델을 사용하여 Multi-hop 질문을 Single-hop으로 변환합니다.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    # I/O 관련 인자
    parser.add_argument(
        "--input",
        help="입력으로 사용할 JSONL 파일 경로. 생략 시 --question으로 단일 처리.",
    )
    parser.add_argument("--output", help="결과를 저장할 JSONL 파일 경로.")
    parser.add_argument("--question", help="--input 대신 처리할 단일 질문 문자열.")
    parser.add_argument("--context", help="단일 질문에 사용할 선택적 컨텍스트 문자열.")

    # 데이터 필드 관련 인자
    parser.add_argument(
        "--question_field",
        default="question",
        help="입력 JSONL에서 질문이 담긴 필드 이름.",
    )
    parser.add_argument(
        "--context_field",
        default="context",
        help="입력 JSONL에서 컨텍스트가 담긴 필드 이름.",
    )

    # 모델 및 처리 관련 인자
    parser.add_argument(
        "--mode",
        choices=["decompose", "chain", "rewrite"],
        default="decompose",
        help="처리 모드:\n  - decompose: 질문을 독립적인 여러 개의 단일 hop 질문으로 분해\n  - chain: 이전 답변에 의존하는 순차적인 질문들로 분해\n  - rewrite: 질문 자체를 더 단순한 단일 hop 질문으로 재작성",
    )
    parser.add_argument(
        "--model",
        default="gemini-2.5-flash",
        help="사용할 Gemini 모델 이름 (예: gemini-1.5-pro).",
    )
    parser.add_argument(
        "--api_key", help="Google API 키. 미제공 시 GOOGLE_API_KEY 환경 변수 사용."
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="모델의 샘플링 온도를 조절합니다.",
    )
    parser.add_argument("--seed", type=int, help="결과의 일관성을 위한 시드 값.")

    # 실행 제어 관련 인자
    parser.add_argument(
        "--qps",
        type=float,
        help="초당 최대 요청(Queries Per Second) 수를 제한합니다 (예: 1.5).",
    )

    args = parser.parse_args()

    if not args.input and not args.question:
        parser.error(
            "--input 파일이나 --question 문자열 중 하나는 반드시 제공해야 합니다."
        )

    # 모델 초기화
    print(f"모델을 초기화합니다: {args.model}...")
    model_obj = init_gemini(args.model, args.api_key, args.temperature, args.seed)
    print("모델 초기화 완료.")

    # 실행 흐름 분기
    if args.input:
        run_batch_processing(args, model_obj)
    else:
        run_single_processing(args, model_obj)


if __name__ == "__main__":
    main()
