#!/usr/bin/env python3
"""
검색 메타데이터 시스템 - 메인 실행 파일
- 사용자 친화적 명령행 인터페이스
- 단일/배치/변환 모드 지원
- 상세한 사용법 안내
"""

import sys
import os
import logging
from pathlib import Path
from typing import Optional
import argparse
import json

# 상위 디렉토리의 모듈 import를 위한 경로 추가
sys.path.append(str(Path(__file__).parent.parent))

from search_meta_system import SearchMetaSystem
from utils.file_manager import FileManager

def print_usage():
    """사용법 출력"""
    print("""
🔍 검색 메타데이터 시스템 v2.0
=====================================

사용법: python main.py [명령] [옵션]

📋 명령어:
  single [질문]           - 단일 질문 처리
  batch [CSV파일]         - CSV 파일에서 배치 처리
  convert                 - 최신 JSON 결과를 CSV/JSONL로 변환
  info                    - 시스템 정보 출력
  help                    - 이 도움말 출력

📝 예시:
  python main.py single "인공지능의 미래는 어떻게 될까요?"
  python main.py batch test.csv
  python main.py --input-dir /path/to/data batch questions.csv
  python main.py convert
  python main.py info

⚙️  주요 옵션:
  --input-dir DIR        - 입력 파일이 위치한 디렉토리 (기본: 현재 디렉토리)
  --use-vllm             - vLLM 사용 (기본: Gemini)
  --use-chatgpt          - ChatGPT API 사용
  --chatgpt-model MODEL  - ChatGPT 모델 (기본: gpt-4o-mini)
  --keyword-lang LANG    - 키워드 언어 (all/korean/english, 기본: all)
  --use-pubmed           - PubMed 검색 활성화 (기본: ScienceON 사용)
  --use-scienceon        - ScienceON 검색 명시적 활성화
  --use-wiki             - Wikipedia 검색 활성화
  --skip-keyword-extraction - 키워드 추출 건너뛰기

⚙️  환경 변수:
  GEMINI_API_KEY         - Gemini API 키 (필수)
  TARGET_DOCUMENTS       - 목표 문서 수 (기본: 50)
  OUTPUT_DIRECTORY       - 출력 디렉토리 (기본: ./outputs)
  LOG_LEVEL              - 로그 레벨 (기본: INFO)

📁 출력 파일:
  - search_meta_results_YYYYMMDD_HHMMSS.json  (원본 결과)
  - search_results_YYYYMMDD_HHMMSS.csv        (CSV 형식)
  - search_documents_YYYYMMDD_HHMMSS.jsonl    (JSONL 형식)
  - elapsed_times.json                         (실행 시간)

🔧 설정 파일:
  - ./configs/scienceon_api_credentials.json  (ScienceON API 자격증명)
""")

def parse_arguments():
    """명령행 인수 파싱"""
    parser = argparse.ArgumentParser(description='검색 메타데이터 시스템', add_help=False)
    
    # AI 모델 옵션
    parser.add_argument('--use-vllm', action='store_true', help='vLLM 사용')
    parser.add_argument('--vllm-url', default='http://localhost:8000/v1', help='vLLM 서버 URL')
    parser.add_argument('--vllm-model', default='openai/gpt-oss-120B', help='vLLM 모델명')
    
    # ChatGPT 옵션
    parser.add_argument('--use-chatgpt', action='store_true', help='ChatGPT API 사용')
    parser.add_argument('--chatgpt-model', default='gpt-4o-mini', help='ChatGPT 모델명')
    parser.add_argument('--keyword-lang', choices=['all', 'korean', 'english'], default='all', 
                       help='키워드 추출 언어 선택')
    
    # 검색 플랫폼 옵션
    parser.add_argument('--use-scienceon', action='store_true', help='ScienceON 사용')
    parser.add_argument('--use-pubmed', action='store_true', help='PubMed 사용')
    parser.add_argument('--use-wiki', action='store_true', help='Wikipedia 사용')
    parser.add_argument('--simple-test', action='store_true', help='간단 테스트 모드')
    
    # 검색 방식 옵션
    parser.add_argument('--skip-keyword-extraction', action='store_true', 
                       help='키워드 추출을 건너뛰고 질문을 직접 검색에 사용')
    
    # 파일 경로 옵션
    parser.add_argument('--input-dir', default='.', help='입력 파일이 위치한 디렉토리 경로')

    # 기존 위치 인수들
    parser.add_argument('command', nargs='?', help='실행할 명령어')
    parser.add_argument('args', nargs='*', help='명령어 인수')
    
    return parser.parse_args()

def get_gemini_api_key() -> str:
    """Gemini API 키 가져오기"""
    # 환경 변수에서 먼저 확인
    api_key = os.getenv("GEMINI_API_KEY")
    if api_key:
        return api_key
    
    # 설정 파일에서 확인
    script_dir = Path(__file__).parent
    config_path = script_dir / "configs" / "gemini_api_credentials.json"
    if config_path.exists():
        try:
            import json
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                api_key = config.get("api_key", "")
                if api_key:  # 빈 문자열이 아닌 경우에만 반환
                    return api_key
        except Exception as e:
            logging.warning(f"설정 파일 읽기 실패: {e}")
    
    # 사용자 입력 요청
    print("⚠️  GEMINI_API_KEY 환경 변수가 설정되지 않았습니다.")
    api_key = input("Gemini API 키를 입력하세요: ").strip()
    
    if not api_key:
        print("❌ API 키가 필요합니다.")
        sys.exit(1)
    
    return api_key

def get_openai_api_key() -> str:
    """OpenAI API 키 가져오기"""
    # 환경 변수에서 먼저 확인
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        return api_key
    
    # 설정 파일에서 확인
    script_dir = Path(__file__).parent
    config_path = script_dir / "configs" / "chatgpt_api_credentials.json"
    if config_path.exists():
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                api_key = config.get("api_key", "")
                if api_key:  # 빈 문자열이 아닌 경우에만 반환
                    return api_key
        except Exception as e:
            logging.warning(f"ChatGPT 설정 파일 읽기 실패: {e}")
    
    # 사용자 입력 요청
    print("⚠️  OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")
    api_key = input("OpenAI API 키를 입력하세요: ").strip()
    
    if not api_key:
        print("❌ API 키가 필요합니다.")
        sys.exit(1)
    
    return api_key

def get_pubmed_credentials() -> tuple[str, str]:
    # 환경변수에서 먼저 확인
    api_key = os.getenv("PUBMED_API_KEY")
    email = os.getenv("PUBMED_EMAIL")
    
    if api_key:
        return api_key, email or ""
    
    # 설정 파일에서 확인
    config_path = Path("./configs/pubmed_credentials.json")
    if config_path.exists():
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                return config.get("api_key", ""), config.get("email", "")
        except Exception as e:
            logging.warning(f"PubMed 설정 파일 읽기 실패: {e}")
    
    return "", ""


def run_single_mode(system: SearchMetaSystem, query: str, use_pubmed: bool = False, use_wiki: bool = False, simple_test: bool = False):
    """단일 모드 실행"""
    print(f"🔍 단일 질문 처리 시작")
    print("=" * 50)
    print(f"질문: {query}")
    if simple_test:
        print("🧪 간단 테스트 모드")
    print("=" * 50)
    
    try:
        # 간단 테스트 모드
        if simple_test and use_pubmed:
            result = system.process_single_query_with_pubmed_simple(query)
        # Wikipedia 사용
        elif use_wiki:
            if hasattr(system, 'process_single_query_with_wikipedia'):
                result = system.process_single_query_with_wikipedia(query)
            else:
                print("⚠️  Wikipedia 검색 기능이 아직 구현되지 않았습니다. 기본 검색을 사용합니다.")
                result = system.process_single_query(query)
        # PubMed 사용 여부에 따라 다른 메서드 호출
        elif use_pubmed:
            result = system.process_single_query_with_pubmed(query, True)
        else:
            result = system.process_single_query(query)
        
        if result["status"] == "success":
            print(f"✅ 처리 완료!")
            print(f"   찾은 문서: {result['document_count']}개")
            print(f"   처리 시간: {result['processing_time']:.2f}초")
            print(f"   검색어: {len(result.get('search_terms', []))}개")
            
            # 간단한 결과 미리보기
            if result['document_count'] > 0:
                print(f"\n📄 첫 번째 문서:")
                first_doc = result['documents'][0]
                print(f"   제목: {first_doc.get('title', 'N/A')[:100]}...")
                print(f"   저자: {first_doc.get('authors', 'N/A')[:50]}...")
        else:
            print(f"❌ 처리 실패: {result.get('error_message', '알 수 없는 오류')}")
            
    except Exception as e:
        print(f"❌ 실행 중 오류 발생: {e}")
        logging.error(f"단일 모드 실행 실패: {e}")

def process_batch_with_platform(system: SearchMetaSystem, csv_path: str, max_queries: int = None, platform: str = 'scienceon'):
    """플랫폼별 배치 처리"""
    import pandas as pd
    import time
    from pathlib import Path
    
    try:
        # CSV 파일 읽기
        df = pd.read_csv(csv_path)
        
        # 컬럼 이름을 유연하게 찾기 (대소문자 무관)
        question_col = None
        for col in df.columns:
            if col.lower() in ['question', 'query', 'text']:
                question_col = col
                break
        
        if question_col is None:
            available_cols = ', '.join(df.columns.tolist())
            return {"error": f"CSV 파일에 질문 컬럼을 찾을 수 없습니다. 사용 가능한 컬럼: {available_cols}"}
        
        questions = df[question_col].tolist()
        if max_queries:
            questions = questions[:max_queries]
        
        results = []
        successful_queries = 0
        failed_queries = 0
        total_documents_found = 0
        
        print(f"📊 {len(questions)}개의 질문을 {platform.upper()}로 처리합니다...")
        
        for i, question in enumerate(questions):
            print(f"진행률: {i+1}/{len(questions)} - {question[:30]}...")
            
            try:
                if platform == 'wikipedia':
                    result = system.process_single_query_with_wikipedia(question)
                elif platform == 'pubmed':
                    result = system.process_single_query_with_pubmed(question, True)
                else:
                    result = system.process_single_query(question)
                
                if result.get("status") == "success":
                    successful_queries += 1
                    total_documents_found += result.get('document_count', 0)
                else:
                    failed_queries += 1
                
                results.append(result)
                
            except Exception as e:
                print(f"❌ 질문 {i+1} 처리 실패: {e}")
                failed_queries += 1
                results.append({"status": "error", "query": question, "error_message": str(e)})
        
        # 통계 계산
        success_rate = (successful_queries / len(questions) * 100) if questions else 0
        avg_documents_per_query = (total_documents_found / successful_queries) if successful_queries > 0 else 0
        
        # 결과를 기존 format과 맞추기
        batch_result = {
            "batch_statistics": {
                "total_queries": len(questions),
                "successful_queries": successful_queries,
                "failed_queries": failed_queries,
                "success_rate": success_rate,
                "total_documents_found": total_documents_found,
                "avg_documents_per_query": avg_documents_per_query
            },
            "results": results
        }
        
        # 기존 로직과 동일하게 결과 저장
        try:
            output_file = system.file_manager.save_json_results(batch_result)
            logging.info(f"결과가 {output_file}에 저장되었습니다.")
            print(f"📁 결과 파일: {output_file}")
        except Exception as e:
            logging.error(f"결과 저장 실패: {e}")
            print(f"⚠️  결과 저장 실패: {e}")
        
        return batch_result
        
    except FileNotFoundError:
        return {"error": f"CSV 파일을 찾을 수 없습니다: {csv_path}"}
    except pd.errors.EmptyDataError:
        return {"error": f"CSV 파일이 비어있습니다: {csv_path}"}
    except pd.errors.ParserError as e:
        return {"error": f"CSV 파일 파싱 오류: {str(e)}"}
    except Exception as e:
        import traceback
        return {"error": f"배치 처리 중 오류 발생: {str(e)}\n상세 오류:\n{traceback.format_exc()}"}

def run_batch_mode(system: SearchMetaSystem, csv_path: str, input_dir: str = '.', max_queries: int = None, use_wiki: bool = False, use_pubmed: bool = False):
    """배치 모드 실행"""
    print(f"📊 배치 처리 시작")
    print("=" * 50)
    
    # 상대 경로인 경우 input_dir과 결합
    if not os.path.isabs(csv_path):
        full_csv_path = os.path.join(input_dir, csv_path)
    else:
        full_csv_path = csv_path
    
    print(f"CSV 파일: {full_csv_path}")
    if max_queries:
        print(f"최대 처리 질문 수: {max_queries}개")
    print("=" * 50)
    
    try:
        # CSV 파일 존재 확인
        if not Path(full_csv_path).exists():
            print(f"❌ CSV 파일을 찾을 수 없습니다: {full_csv_path}")
            return
        
        # 플랫폼에 따라 다른 배치 처리 방식 사용
        if use_wiki:
            print("🔍 Wikipedia 배치 처리 모드")
            result = process_batch_with_platform(system, full_csv_path, max_queries, 'wikipedia')
        elif use_pubmed:
            print("🔍 PubMed 배치 처리 모드")
            result = process_batch_with_platform(system, full_csv_path, max_queries, 'pubmed')
        else:
            print("🔍 ScienceON 배치 처리 모드")
            result = system.process_batch_from_csv(full_csv_path, max_queries=max_queries)
        
        # 결과에 오류가 있는지 먼저 확인
        if "error" in result:
            print(f"❌ 배치 처리 실패: {result['error']}")
            return
        
        batch_info = result.get("batch_statistics", {})
        if batch_info.get("total_queries", 0) > 0:
            print(f"✅ 배치 처리 완료!")
            print(f"   총 질문: {batch_info['total_queries']}개")
            print(f"   성공: {batch_info['successful_queries']}개")
            print(f"   실패: {batch_info['failed_queries']}개")
            print(f"   성공률: {batch_info['success_rate']:.1f}%")
            print(f"   총 문서: {batch_info['total_documents_found']}개")
            print(f"   평균 문서/질문: {batch_info['avg_documents_per_query']:.1f}개")
        else:
            print(f"❌ 배치 처리 실패: {batch_info.get('error', '알 수 없는 오류')}")
            
    except Exception as e:
        print(f"❌ 실행 중 오류 발생: {e}")
        logging.error(f"배치 모드 실행 실패: {e}")

def run_convert_mode(system: SearchMetaSystem):
    """변환 모드 실행"""
    print(f"🔄 결과 변환 시작")
    print("=" * 50)
    
    try:
        files = system.convert_latest_results()
        
        print(f"✅ 변환 완료!")
        print(f"   CSV 파일: {files['csv']}")
        print(f"   JSONL 파일: {files['jsonl']}")
        
    except Exception as e:
        print(f"❌ 변환 실패: {e}")
        logging.error(f"변환 모드 실행 실패: {e}")

def run_info_mode(system: SearchMetaSystem):
    """정보 모드 실행"""
    system.print_system_info()

def main():
    """메인 함수"""
    if len(sys.argv) < 2:
        print_usage()
        return
    args = parse_arguments()

    if not args.command or args.command == "help":
        print_usage()
        return

    command = args.command.lower()

    # 플랫폼 선택 로직 개선
    use_pubmed = args.use_pubmed
    use_wiki = args.use_wiki
    use_scienceon = args.use_scienceon or (not use_pubmed and not use_wiki)  # 다른 플랫폼이 선택되지 않으면 기본적으로 ScienceON 사용
    
    # 사용할 플랫폼 출력
    platforms = []
    if use_scienceon:
        platforms.append("ScienceON")
    if use_pubmed:
        platforms.append("PubMed")
    if use_wiki:
        platforms.append("Wikipedia")
    
    print(f"🔍 검색 플랫폼: {', '.join(platforms)}")
    
    if args.use_vllm:
        print(f"🤖 AI 모델: vLLM ({args.vllm_model})")
    elif args.use_chatgpt:
        print(f"🤖 AI 모델: ChatGPT ({args.chatgpt_model})")
    else:
        print(f"🤖 AI 모델: Gemini")
    
    # API 키 가져오기
    try:
        if args.use_vllm:
            # vLLM 사용
            system = SearchMetaSystem(
                gemini_api_key=None,
                use_vllm=True,
                vllm_base_url=args.vllm_url,
                vllm_model=args.vllm_model,
                skip_keyword_extraction=args.skip_keyword_extraction
            )
        elif args.use_chatgpt:
            # ChatGPT 사용
            try:
                openai_api_key = get_openai_api_key()
            except KeyboardInterrupt:
                print("\n❌ 사용자가 취소했습니다.")
                return
                
            system = SearchMetaSystem(
                gemini_api_key=None,
                use_chatgpt=True,
                chatgpt_model=args.chatgpt_model,
                openai_api_key=openai_api_key,
                keyword_lang=args.keyword_lang,
                skip_keyword_extraction=args.skip_keyword_extraction
            )
        else:
            # Gemini 사용 (기본)
            try:
                api_key = get_gemini_api_key()
                if not api_key:
                    print("❌ Gemini API 키가 필요합니다.")
                    return
            except KeyboardInterrupt:
                print("\n❌ 사용자가 취소했습니다.")
                return
            except Exception as e:
                print(f"❌ API 키 가져오기 실패: {e}")
                return
            
            system = SearchMetaSystem(
                gemini_api_key=api_key,
                keyword_lang=args.keyword_lang,
                skip_keyword_extraction=args.skip_keyword_extraction
            )
            
    except Exception as e:
        print(f"❌ 시스템 초기화 실패: {e}")
        logging.error(f"시스템 초기화 실패: {e}")
        return

    try:
        if command == "single":
            if len(args.args) < 1:
                print("❌ 질문을 입력해주세요.")
                print("사용법: python main.py single \"질문내용\"")
                return

            query = args.args[0]
            run_single_mode(system, query, use_pubmed, use_wiki, args.simple_test)
            
        elif command == "batch":
            if len(args.args) < 1:
                print("❌ CSV 파일 경로를 입력해주세요.")
                print("사용법: python main.py batch test.csv [최대질문수]")
                return

            csv_path = args.args[0]
            max_queries = None
            if len(args.args) > 1:
                try:
                    max_queries = int(args.args[1])
                except ValueError:
                    print("❌ 최대 질문 수는 숫자여야 합니다.")
                    return
            
            run_batch_mode(system, csv_path, args.input_dir, max_queries, use_wiki, use_pubmed)
            
        elif command == "convert":
            run_convert_mode(system)
            
        elif command == "info":
            run_info_mode(system)
            
        else:
            print(f"❌ 알 수 없는 명령어: {command}")
            print_usage()
            return
            
    except KeyboardInterrupt:
        print("\n❌ 사용자가 취소했습니다.")
    except Exception as e:
        print(f"❌ 예상치 못한 오류 발생: {e}")
        logging.error(f"메인 실행 실패: {e}")
    finally:
        system.cleanup()

if __name__ == "__main__":
    main()
