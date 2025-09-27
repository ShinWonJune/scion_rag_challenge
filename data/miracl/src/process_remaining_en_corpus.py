#!/usr/bin/env python3
"""
영어 코퍼스 나머지 파일들을 처리하여 누락된 문서 찾기
"""

import json
import gzip
import os
from collections import defaultdict
from huggingface_hub import hf_hub_download, list_repo_files

def analyze_missing_documents():
    """현재 JSON과 QRELs를 비교하여 누락된 문서 확인"""
    print("🔍 누락된 영어 문서 분석...")
    
    # 1. 현재 JSON에서 찾은 문서들
    with open('miracl_en_query_documents.json', 'r', encoding='utf-8') as f:
        current_data = json.load(f)
    
    found_docs = set()
    for query in current_data:
        for doc in query['documents']:
            found_docs.add(doc['doc_id'])
    
    print(f"   현재 찾은 문서: {len(found_docs)}개")
    
    # 2. QRELs에서 모든 target 문서들 확인
    target_query_ids = {query['query_id'] for query in current_data}
    target_docs = set()
    
    with open('miracl-v1.0-en/qrels/qrels.miracl-v1.0-en-train.tsv', 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 4:
                q_id = parts[0]
                doc_id_full = parts[2]
                relevance = int(parts[3])
                
                if q_id in target_query_ids and relevance == 1:
                    if '#' in doc_id_full:
                        doc_id_q = doc_id_full.split('#')[0]
                        target_docs.add(doc_id_q)
    
    print(f"   찾아야 할 문서: {len(target_docs)}개")
    
    # 3. 누락된 문서들
    missing_docs = target_docs - found_docs
    print(f"   누락된 문서: {len(missing_docs)}개")
    
    if missing_docs:
        print("   누락된 문서 예시:")
        for i, doc_id in enumerate(sorted(missing_docs)[:10]):
            print(f"      - {doc_id}")
        if len(missing_docs) > 10:
            print(f"      ... 외 {len(missing_docs)-10}개")
    
    return missing_docs, target_query_ids

def download_remaining_corpus_files():
    """나머지 영어 코퍼스 파일들을 다운로드"""
    print("\n📥 나머지 영어 코퍼스 파일 다운로드...")
    
    # 전체 파일 목록
    files = list_repo_files('miracl/miracl-corpus', repo_type='dataset')
    en_files = [f for f in files if f.startswith('miracl-corpus-v1.0-en/') and f.endswith('.jsonl.gz')]
    en_files.sort()
    
    # 현재 있는 파일들
    existing_files = set(os.listdir('miracl-corpus-v1.0-en/'))
    
    # 다운로드할 파일들
    files_to_download = []
    for f in en_files:
        filename = f.split('/')[-1]  # docs-X.jsonl.gz
        if filename not in existing_files:
            files_to_download.append(f)
    
    print(f"   다운로드할 파일: {len(files_to_download)}개")
    
    # 배치 단위로 다운로드 (메모리 고려)
    batch_size = 10
    downloaded_files = []
    
    for i in range(0, len(files_to_download), batch_size):
        batch = files_to_download[i:i+batch_size]
        print(f"\n   배치 {i//batch_size + 1}: {len(batch)}개 파일 다운로드 중...")
        
        for j, file_path in enumerate(batch):
            try:
                print(f"      [{j+1}/{len(batch)}] {file_path}")
                local_path = hf_hub_download(
                    repo_id='miracl/miracl-corpus',
                    filename=file_path,
                    local_dir='.',
                    repo_type='dataset'
                )
                downloaded_files.append(local_path)
                
            except Exception as e:
                print(f"         ❌ 다운로드 실패: {e}")
                continue
    
    print(f"\n   총 {len(downloaded_files)}개 파일 다운로드 완료")
    return downloaded_files

def process_remaining_files_for_missing_docs(missing_docs, target_query_ids):
    """나머지 파일들에서 누락된 문서들만 검색"""
    print(f"\n🔍 나머지 파일에서 누락된 {len(missing_docs)}개 문서 검색...")
    
    # 현재 JSON 데이터 로드
    with open('miracl_en_query_documents.json', 'r', encoding='utf-8') as f:
        current_data = json.load(f)
    
    # query_id로 빠른 접근을 위한 인덱스
    query_index = {q['query_id']: i for i, q in enumerate(current_data)}
    
    # QRELs에서 missing_docs에 해당하는 쿼리 매핑
    doc_to_queries = defaultdict(list)
    
    with open('miracl-v1.0-en/qrels/qrels.miracl-v1.0-en-train.tsv', 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 4:
                q_id = parts[0]
                doc_id_full = parts[2]
                relevance = int(parts[3])
                
                if q_id in target_query_ids and relevance == 1:
                    if '#' in doc_id_full:
                        doc_id_q = doc_id_full.split('#')[0]
                        if doc_id_q in missing_docs:
                            doc_to_queries[doc_id_q].append(q_id)
    
    # 모든 코퍼스 파일 처리
    pattern = "miracl-corpus-v1.0-en/docs-*.jsonl.gz"
    import glob
    corpus_files = sorted(glob.glob(pattern))
    
    found_missing = 0
    total_missing = len(missing_docs)
    
    for file_idx, corpus_file in enumerate(corpus_files):
        if not missing_docs:  # 모든 문서를 찾았으면 종료
            break
            
        print(f"\n   📄 [{file_idx+1}/{len(corpus_files)}] 처리: {corpus_file}")
        
        file_found = 0
        try:
            with gzip.open(corpus_file, 'rt', encoding='utf-8') as f:
                for line_num, line in enumerate(f):
                    doc = json.loads(line.strip())
                    docid = doc['docid']  # "x#y" 형태
                    
                    if '#' in docid:
                        doc_id_c = docid.split('#')[0]
                        
                        # 찾던 누락 문서인가?
                        if doc_id_c in missing_docs:
                            doc_info = {
                                'doc_id': doc_id_c,
                                'title': doc['title']
                            }
                            
                            # 해당 문서가 속하는 모든 query에 추가
                            for q_id in doc_to_queries[doc_id_c]:
                                if q_id in query_index:
                                    query_idx = query_index[q_id]
                                    current_data[query_idx]['documents'].append(doc_info)
                            
                            missing_docs.discard(doc_id_c)
                            file_found += 1
                            found_missing += 1
                    
                    if line_num % 100000 == 0 and line_num > 0:
                        print(f"      진행: {line_num:,}개 문서 처리 (남은 누락 문서: {len(missing_docs)}개)")
            
            print(f"      완료: {file_found}개 누락 문서 발견")
            
            # 메모리 절약을 위해 처리된 파일 삭제
            if file_found > 0:
                os.remove(corpus_file)
                print(f"      💾 {corpus_file} 삭제")
                
        except Exception as e:
            print(f"      ❌ 파일 처리 오류: {e}")
            continue
    
    print(f"\n   📊 누락 문서 검색 완료:")
    print(f"      찾은 문서: {found_missing}개")
    print(f"      여전히 누락: {len(missing_docs)}개")
    
    # 업데이트된 JSON 저장
    with open('miracl_en_query_documents_updated.json', 'w', encoding='utf-8') as f:
        json.dump(current_data, f, indent=2, ensure_ascii=False)
    
    return current_data

def main():
    """메인 실행 함수"""
    print("🚀 영어 코퍼스 완전 처리 시작...\n")
    
    # 1. 누락된 문서 분석
    missing_docs, target_query_ids = analyze_missing_documents()
    
    if not missing_docs:
        print("✅ 누락된 문서가 없습니다!")
        return
    
    # 2. 나머지 코퍼스 파일 다운로드
    downloaded_files = download_remaining_corpus_files()
    
    # 3. 누락된 문서들 검색
    updated_data = process_remaining_files_for_missing_docs(missing_docs, target_query_ids)
    
    # 4. 최종 통계
    total_docs = sum(len(q['documents']) for q in updated_data)
    print(f"\n🎉 영어 코퍼스 완전 처리 완료!")
    print(f"   총 쿼리: {len(updated_data)}개")
    print(f"   총 문서: {total_docs}개")
    print(f"   평균 문서/쿼리: {total_docs/len(updated_data):.1f}개")
    print(f"   업데이트된 파일: miracl_en_query_documents_updated.json")

if __name__ == "__main__":
    main()