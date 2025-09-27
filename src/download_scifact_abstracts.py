#!/usr/bin/env python3
"""
SciFact 데이터셋에서 doc_id에 해당하는 abstract들을 다운로드하는 스크립트

사용법:
    python download_scifact_abstracts.py --input_csv /app/scifact/data/questions/subquestion.csv --output_dir /app/scifact/data/abstracts
"""

import os
import csv
import json
import requests
import argparse
from typing import Dict, List, Set
from pathlib import Path
import time


def load_doc_ids_from_csv(csv_path: str) -> Set[str]:
    """
    CSV 파일에서 doc_id들을 추출
    
    Args:
        csv_path: subquestion.csv 파일 경로
        
    Returns:
        Set[str]: 고유한 doc_id들
    """
    doc_ids = set()
    
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                doc_id = row.get('doc_id', '').strip()
                if doc_id:
                    doc_ids.add(doc_id)
                    
        print(f"✅ CSV에서 {len(doc_ids)}개의 고유한 doc_id 발견")
        return doc_ids
        
    except Exception as e:
        print(f"❌ CSV 파일 로드 오류: {e}")
        return set()


def download_scifact_corpus():
    """
    SciFact 데이터셋의 corpus.jsonl 파일을 다운로드
    
    Returns:
        str: 다운로드된 파일 경로 또는 None
    """
    urls = [
        "https://scifact.s3-us-west-2.amazonaws.com/release/2020-05-01/corpus.jsonl",
        "https://github.com/allenai/scifact/raw/master/data/corpus.jsonl"
    ]
    
    output_path = "/tmp/scifact_corpus.jsonl"
    
    for url in urls:
        try:
            print(f"📥 SciFact corpus 다운로드 시도: {url}")
            response = requests.get(url, timeout=30)
            
            if response.status_code == 200:
                with open(output_path, 'wb') as f:
                    f.write(response.content)
                print(f"✅ SciFact corpus 다운로드 완료: {output_path}")
                return output_path
                
        except Exception as e:
            print(f"⚠️ {url} 다운로드 실패: {e}")
            continue
    
    print("❌ 모든 URL에서 다운로드 실패")
    return None


def load_scifact_corpus(corpus_path: str) -> Dict[str, Dict]:
    """
    SciFact corpus.jsonl 파일을 로드하여 딕셔너리로 변환
    
    Args:
        corpus_path: corpus.jsonl 파일 경로
        
    Returns:
        Dict[str, Dict]: {doc_id: document_data}
    """
    corpus = {}
    
    try:
        with open(corpus_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    doc = json.loads(line.strip())
                    doc_id = str(doc.get('doc_id', ''))
                    if doc_id:
                        corpus[doc_id] = doc
                except json.JSONDecodeError as e:
                    print(f"⚠️ 라인 {line_num} JSON 파싱 오류: {e}")
                    continue
                    
        print(f"✅ SciFact corpus 로드 완료: {len(corpus)}개 문서")
        return corpus
        
    except Exception as e:
        print(f"❌ Corpus 파일 로드 오류: {e}")
        return {}


def extract_abstracts(doc_ids: Set[str], corpus: Dict[str, Dict]) -> Dict[str, Dict]:
    """
    필요한 doc_id들의 abstract를 추출
    
    Args:
        doc_ids: 추출할 doc_id들
        corpus: SciFact corpus 딕셔너리
        
    Returns:
        Dict[str, Dict]: {doc_id: abstract_data}
    """
    abstracts = {}
    found_count = 0
    
    for doc_id in doc_ids:
        if doc_id in corpus:
            doc = corpus[doc_id]
            abstract_data = {
                'doc_id': doc_id,
                'title': doc.get('title', ''),
                'abstract': doc.get('abstract', ''),
                'structured': doc.get('structured', False)
            }
            abstracts[doc_id] = abstract_data
            found_count += 1
        else:
            print(f"⚠️ doc_id {doc_id}를 corpus에서 찾을 수 없음")
    
    print(f"✅ {found_count}/{len(doc_ids)}개 abstract 추출 완료")
    return abstracts


def save_abstracts(abstracts: Dict[str, Dict], output_dir: str):
    """
    추출된 abstract들을 파일로 저장
    
    Args:
        abstracts: abstract 데이터 딕셔너리
        output_dir: 출력 디렉토리
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 전체 abstracts를 하나의 JSON 파일로 저장
    all_abstracts_path = os.path.join(output_dir, "scifact_abstracts.json")
    with open(all_abstracts_path, 'w', encoding='utf-8') as f:
        json.dump(abstracts, f, indent=2, ensure_ascii=False)
    print(f"💾 전체 abstracts 저장: {all_abstracts_path}")
    
    # 2. 각 abstract를 개별 파일로 저장 (옵션)
    individual_dir = os.path.join(output_dir, "individual")
    os.makedirs(individual_dir, exist_ok=True)
    
    for doc_id, data in abstracts.items():
        individual_path = os.path.join(individual_dir, f"abstract_{doc_id}.json")
        with open(individual_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"💾 개별 abstracts 저장: {individual_dir} ({len(abstracts)}개 파일)")
    
    # 3. CSV 형태로도 저장
    csv_path = os.path.join(output_dir, "scifact_abstracts.csv")
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['doc_id', 'title', 'abstract', 'structured'])
        
        for doc_id, data in abstracts.items():
            writer.writerow([
                data['doc_id'],
                data['title'],
                data['abstract'],
                data['structured']
            ])
    
    print(f"💾 CSV 형태 저장: {csv_path}")


def main():
    parser = argparse.ArgumentParser(
        description="SciFact 데이터셋에서 doc_id에 해당하는 abstract들을 다운로드"
    )
    parser.add_argument(
        "--input_csv",
        required=True,
        help="doc_id가 포함된 CSV 파일 경로 (예: /app/scifact/data/questions/subquestion.csv)"
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="abstract들을 저장할 디렉토리"
    )
    parser.add_argument(
        "--corpus_path",
        help="기존 corpus.jsonl 파일 경로 (제공하지 않으면 자동 다운로드)"
    )
    
    args = parser.parse_args()
    
    # 입력 파일 검증
    if not os.path.exists(args.input_csv):
        print(f"❌ 입력 CSV 파일이 존재하지 않습니다: {args.input_csv}")
        return 1
    
    try:
        # 1. CSV에서 doc_id 추출
        doc_ids = load_doc_ids_from_csv(args.input_csv)
        if not doc_ids:
            print("❌ 유효한 doc_id를 찾을 수 없습니다.")
            return 1
        
        # 2. SciFact corpus 획득
        corpus_path = args.corpus_path
        if not corpus_path or not os.path.exists(corpus_path):
            corpus_path = download_scifact_corpus()
            if not corpus_path:
                print("❌ SciFact corpus를 다운로드할 수 없습니다.")
                return 1
        
        # 3. Corpus 로드
        corpus = load_scifact_corpus(corpus_path)
        if not corpus:
            print("❌ Corpus를 로드할 수 없습니다.")
            return 1
        
        # 4. Abstract 추출
        abstracts = extract_abstracts(doc_ids, corpus)
        if not abstracts:
            print("❌ 매칭되는 abstract를 찾을 수 없습니다.")
            return 1
        
        # 5. 결과 저장
        save_abstracts(abstracts, args.output_dir)
        
        print(f"\n🎉 Abstract 다운로드 완료!")
        print(f"   - 요청된 doc_id: {len(doc_ids)}개")
        print(f"   - 성공적으로 추출: {len(abstracts)}개")
        print(f"   - 저장 위치: {args.output_dir}")
        
        return 0
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        return 1


if __name__ == "__main__":
    exit(main())