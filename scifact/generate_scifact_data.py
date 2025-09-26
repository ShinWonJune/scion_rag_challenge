#!/usr/bin/env python3
"""
SciFact 데이터셋 생성기
CSV와 JSONL 형식 모두 생성
"""

import csv
import json
from pathlib import Path
from typing import List, Dict, Tuple

def generate_scifact_data() -> List[Tuple[str, str, str]]:
    """SciFact 과학적 주장 데이터 생성"""
    
    scifact_data = [
        # 의학/건강
        ("Coronavirus disease 2019 (COVID-19) spreads primarily through respiratory droplets.", "SUPPORT", "101"),
        ("Vitamin C supplements prevent the common cold.", "REFUTE", "201"),
        ("Regular exercise improves cardiovascular health.", "SUPPORT", "301"),
        ("Smoking increases the risk of lung cancer.", "SUPPORT", "401"),
        ("Antioxidant supplements reduce cancer risk in healthy individuals.", "REFUTE", "501"),
        ("Moderate alcohol consumption protects against heart disease.", "SUPPORT", "601"),
        ("Mobile phone radiation causes brain tumors.", "REFUTE", "701"),
        ("Meditation reduces symptoms of anxiety and depression.", "SUPPORT", "801"),
        ("Probiotics improve digestive health in healthy adults.", "SUPPORT", "901"),
        ("Artificial sweeteners are safer than sugar for diabetics.", "SUPPORT", "1001"),
        ("Sleep deprivation impairs immune function.", "SUPPORT", "1101"),
        ("Homeopathic remedies are effective for treating flu symptoms.", "REFUTE", "1201"),
        ("Omega-3 fatty acids reduce inflammation in rheumatoid arthritis.", "SUPPORT", "1301"),
        ("Excessive screen time causes myopia in children.", "SUPPORT", "1401"),
        ("Acupuncture is effective for chronic pain management.", "SUPPORT", "1501"),
        ("Low-carbohydrate diets are more effective for weight loss than low-fat diets.", "SUPPORT", "1601"),
        ("Fluoride in drinking water prevents tooth decay.", "SUPPORT", "1701"),
        ("Cranberry juice prevents urinary tract infections.", "REFUTE", "1801"),
        ("Wearing masks reduces transmission of respiratory infections.", "SUPPORT", "1901"),
        ("Multivitamins improve cognitive function in elderly adults.", "REFUTE", "2001"),
        
        # 환경/기후
        ("Climate change is caused primarily by human activities.", "SUPPORT", "2101"),
        ("Renewable energy sources can meet global energy demands.", "SUPPORT", "2201"),
        ("Deforestation contributes significantly to global warming.", "SUPPORT", "2301"),
        ("Ocean acidification threatens marine ecosystems.", "SUPPORT", "2401"),
        ("Solar panels reduce greenhouse gas emissions over their lifetime.", "SUPPORT", "2501"),
        ("Plastic pollution has no significant impact on marine life.", "REFUTE", "2601"),
        ("Carbon capture technology can reverse climate change.", "REFUTE", "2701"),
        ("Urban green spaces improve air quality.", "SUPPORT", "2801"),
        ("Electric vehicles produce zero emissions.", "REFUTE", "2901"),
        ("Biodegradable plastics completely decompose in marine environments.", "REFUTE", "3001"),
        
        # 기술/AI
        ("Artificial intelligence can diagnose diseases more accurately than human doctors.", "SUPPORT", "3101"),
        ("5G networks pose health risks to humans.", "REFUTE", "3201"),
        ("Quantum computers will break current encryption methods.", "SUPPORT", "3301"),
        ("Autonomous vehicles are safer than human-driven cars.", "SUPPORT", "3401"),
        ("Social media algorithms contribute to political polarization.", "SUPPORT", "3501"),
        ("Blockchain technology is more secure than traditional databases.", "SUPPORT", "3601"),
        ("Virtual reality therapy is effective for treating phobias.", "SUPPORT", "3701"),
        ("Machine learning models are immune to bias.", "REFUTE", "3801"),
        ("Internet of Things devices improve home energy efficiency.", "SUPPORT", "3901"),
        ("Facial recognition technology is 100% accurate.", "REFUTE", "4001"),
        
        # 생물학/과학
        ("Evolution by natural selection explains the diversity of life on Earth.", "SUPPORT", "4101"),
        ("CRISPR gene editing can cure genetic diseases.", "SUPPORT", "4201"),
        ("GMO foods are harmful to human health.", "REFUTE", "4301"),
        ("Stem cell therapy can regenerate damaged organs.", "SUPPORT", "4401"),
        ("Antibiotics are effective against viral infections.", "REFUTE", "4501"),
        ("Biodiversity loss threatens ecosystem stability.", "SUPPORT", "4601"),
        ("Telomeres determine human lifespan.", "SUPPORT", "4701"),
        ("Organic foods have higher nutritional value than conventional foods.", "REFUTE", "4801"),
        ("Microbiome diversity is linked to overall health.", "SUPPORT", "4901"),
        ("Epigenetic changes can be inherited across generations.", "SUPPORT", "5001"),
    ]
    
    return scifact_data

def create_csv_format(data: List[Tuple[str, str, str]], output_path: str):
    """CSV 형식으로 데이터 저장"""
    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['question', 'label', 'doc_id'])  # 헤더
        
        for question, label, doc_id in data:
            writer.writerow([question, label, doc_id])
    
    print(f"✅ CSV 파일 생성: {output_path}")

def create_jsonl_format(data: List[Tuple[str, str, str]], output_path: str):
    """JSONL 형식으로 데이터 저장"""
    with open(output_path, 'w', encoding='utf-8') as jsonlfile:
        for idx, (question, label, doc_id) in enumerate(data, 1):
            json_obj = {
                "id": f"row_{idx:06d}",  # row_000001 형식
                "question": question
            }
            jsonlfile.write(json.dumps(json_obj, ensure_ascii=False) + '\n')
    
    print(f"✅ JSONL 파일 생성: {output_path}")

def create_metadata_file(data: List[Tuple[str, str, str]], output_path: str):
    """메타데이터 파일 생성 (레이블 정보 포함)"""
    metadata = {
        "dataset": "SciFact",
        "description": "Scientific fact verification dataset with SUPPORT/REFUTE labels",
        "total_questions": len(data),
        "created_at": "2025-09-26",
        "categories": {
            "medical_health": "의학 및 건강 관련 주장",
            "climate_environment": "기후 및 환경 관련 주장", 
            "technology_ai": "기술 및 AI 관련 주장",
            "biology_science": "생물학 및 과학 관련 주장"
        },
        "labels": {
            "SUPPORT": "과학적 증거로 뒷받침되는 주장",
            "REFUTE": "과학적 증거로 반박되는 주장"
        },
        "data_samples": []
    }
    
    for question, label, doc_id in data:
        metadata["data_samples"].append({
            "doc_id": doc_id,
            "question": question,
            "label": label
        })
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    print(f"✅ 메타데이터 파일 생성: {output_path}")

def main():
    """메인 함수"""
    # 출력 디렉토리 설정
    base_dir = Path("/app/scifact/data/questions")
    base_dir.mkdir(parents=True, exist_ok=True)
    
    # 데이터 생성
    print("🔬 SciFact 데이터셋 생성 시작...")
    data = generate_scifact_data()
    
    # CSV 형식 생성
    csv_path = base_dir / "subquestion.csv"
    create_csv_format(data, str(csv_path))
    
    # JSONL 형식 생성
    jsonl_path = base_dir / "questions.jsonl"
    create_jsonl_format(data, str(jsonl_path))
    
    # 메타데이터 생성
    metadata_path = base_dir / "dataset_info.json"
    create_metadata_file(data, str(metadata_path))
    
    print(f"\n🎉 SciFact 데이터셋 생성 완료!")
    print(f"📁 출력 디렉토리: {base_dir}")
    print(f"📊 총 {len(data)}개 과학적 주장")
    print(f"📄 파일:")
    print(f"   - subquestion.csv (CSV 형식)")
    print(f"   - questions.jsonl (JSONL 형식)")
    print(f"   - dataset_info.json (메타데이터)")

if __name__ == "__main__":
    main()