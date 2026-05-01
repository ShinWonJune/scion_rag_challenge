from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path


# Manual LLM-as-a-judge review.
# Criteria:
# - FULL: the selected abstract directly supports an answer to the question's
#   main target and most requested sub-points.
# - PARTIAL: the selected document is relevant or likely correct, but the
#   abstract alone omits requested details such as concrete factors, procedures,
#   comparison results, or structure.
# - INSUFFICIENT: the abstract does not support answering the question.
LLM_JUDGMENTS: dict[str, tuple[str, str]] = {
    "0": ("PARTIAL", "초록은 AI 수학교육 교재의 필요성과 한두 학기 분량, 무료 전자교과서 개발은 설명하지만 교재의 구체적 구조나 수학 주제 구성은 충분히 제시하지 않는다."),
    "1": ("FULL", "입력/은닉/출력 벡터, weight matrix 연결, 행렬곱을 통한 입력-출력 매핑이 직접 제시된다."),
    "2": ("FULL", "TurKontrol의 POMDP 파라미터를 Mechanical Turk 데이터로 학습하고 실제 작업 최적화에 적용한 방법과 결과가 직접 제시된다."),
    "3": ("PARTIAL", "지속가능 AI와 기업문화의 관계 및 여섯 명제를 다룬다는 점은 맞지만, 질문이 요구하는 구체적 문화 특징 목록은 초록에 드러나지 않는다."),
    "4": ("FULL", "산과 영역에서 조기진단 대상, 기계학습 활용, 윤리 고려 필요성이 모두 직접 제시된다."),
    "5": ("FULL", "설명 가능한 AI 교육의 의미, 문제해결, 알고리즘 이해, 실제 삶의 예, 교수학습 도구 활용이 직접 제시된다."),
    "6": ("FULL", "생성형 AI 콘텐츠 가치 유형, 매체 형식, 소비자 인식, AI 사용 공개 인식 결과가 직접 제시된다."),
    "7": ("FULL", "AI 기본 개념, 항공 포함 산업적 의미, 윤리와 전문직 역할 등 적용 과제가 직접 제시된다."),
    "8": ("FULL", "구성주의 관점, 지능 이해 확장, 방법론 검토, 관련 AI 기술 동향이라는 질문 핵심이 직접 제시된다."),
    "9": ("FULL", "AI 정의에 함축된 지능 이해, 지능과 뇌의 관계, 컴퓨터 모의/복제의 논리 구조 쟁점이 직접 제시된다."),
    "10": ("FULL", "SDG2 enabling framework, stakeholder 연결, 데이터 자유 교환, near real-time analytics, IoT/Big Data 역할이 직접 제시된다."),
    "11": ("FULL", "신경학 의료영상에서 AI/Big Data의 발전, RWD, 대용량 분석, 데이터 품질과 윤리 과제가 직접 제시된다."),
    "12": ("FULL", "Big Data의 물질성, 환경영향, 윤리적 함의, 거버넌스 용어/정책 긴장/공정 분배 논점이 직접 제시된다."),
    "13": ("FULL", "현상론적 big data 분석과 VPH/기계론적 모델 결합, 맞춤의료 요구사항이 직접 제시된다."),
    "14": ("FULL", "사회기술 평가, Big Data 발전 과정, 사용자 채택 요인, 사용자 중심 설계 결론이 직접 제시된다."),
    "15": ("PARTIAL", "Big Data 기반 WMS 방향성은 제시되지만, 질문이 요구하는 핵심 개념과 모델 세부 구조는 초록만으로 부족하다."),
    "16": ("FULL", "DTG/GPS, 공간 빅데이터 맵매칭, 운행 패턴, 도로정보 입력변수, MapReduce 처리, 검증 성능이 직접 제시된다."),
    "17": ("PARTIAL", "처리 단계별 분류와 전문가 우선순위 조사 방식은 제시되지만, 실제 위험요인별 우선순위 결과는 초록에 없다."),
    "18": ("FULL", "패키징 분야 빅데이터 적용, 데이터 수집/저장/분석, 텍스트마이닝/오피니언마이닝/SNA를 통한 소비자 인식 분석 방안이 직접 제시된다."),
    "19": ("PARTIAL", "식스시그마 실행 방법을 기준으로 Big Data 활용 방법을 제안한다는 방향은 있으나 주요 절차 자체는 초록에 충분히 설명되지 않는다."),
    "20": ("FULL", "레이더 영상에서 딥러닝 역할, SAR 모델 기반 네트워크, 비선형 forward model, phase error/autofocus 문제가 직접 제시된다."),
    "21": ("FULL", "복잡 영상분석에서 DL/DRL 비교, CNN 장점과 한계, MIL의 맥락 손실, DRL의 제한 데이터 학습 장점이 직접 제시된다."),
    "22": ("FULL", "HRV 기반 비침습 당뇨 탐지, LSTM/CNN/CNN-LSTM+SVM 방법, 성능 향상 및 95.7% 정확도가 직접 제시된다."),
    "23": ("FULL", "eMLP, ARIMA, CNN-LSTM 비교와 MSE/MAPE 평가, CNN-LSTM 우수 결과가 직접 제시된다."),
    "24": ("FULL", "10년간 딥러닝 문헌 수집, LDA 토픽모델링, 국가별 차이 분석, 연구 방향 시사점이 직접 제시된다."),
    "25": ("PARTIAL", "클라우드/OpenStack 환경에서 프레임워크 성능을 비교한다는 목적은 명확하지만 실제 비교 결과와 의의는 초록에 없다."),
    "26": ("PARTIAL", "DBN 기반 PID 모방 제어기 설계와 시뮬레이션 비교는 제시되지만, 구체적인 성능 비교 결과는 초록에 없다."),
    "27": ("FULL", "DBN과 SVM 비교, 국내 기업 데이터, DBN 우수성, 부도기업 민감도 5% 이상 향상이 직접 제시된다."),
    "28": ("FULL", "리뷰 텍스트에서 특징 추출, ReLU/Dropout 모델, 좋음/보통/나쁨 3분류, 90% 정확도가 직접 제시된다."),
    "29": ("FULL", "Go-game 적용, 3개 CNN 레이어/5개 hidden layer, HuuDucGo/Orego/Fuego 비교와 결과 수용 가능성이 직접 제시된다."),
    "30": ("FULL", "한국 IT융합 전략 환경, 정부 노력, LG/Samsung/POSCO 사례, 주요 개념과 동향 요약이 직접 제시된다."),
    "31": ("FULL", "u-IT convergence 정의, 센서/기기 평가 기준, 기술성/경제성/경영성, 표준화 기준이 직접 제시된다."),
    "32": ("FULL", "ADMS 개념, IT/OT convergence, 시스템 구성과 기능 설계가 직접 제시된다."),
    "33": ("FULL", "국방 IT융합에서 IT기업 역할, control tower, SW 비용 산정 이슈가 직접 제시된다."),
    "34": ("PARTIAL", "구매태도와 효용성, 경쟁력 시사점은 제시되지만 소비자 구매태도를 구성하는 구체 요인들은 초록에 충분히 없다."),
    "35": ("PARTIAL", "델파이 결과로 역량강화요인을 도출했다는 연구 설계는 있으나 실제 요인 목록은 초록에 없다."),
    "36": ("FULL", "IT-BT 융합, 특허 인용/공분류 분석, 융합 강도와 범위, 포트폴리오 매트릭스가 직접 제시된다."),
    "37": ("FULL", "스마트팜, 저탄소 녹색산업 정책, IT융합기술 적용과 효과/개선방안 검토가 직접 제시된다."),
    "38": ("FULL", "국방 IT융합 비즈니스 모델 4유형과 프레임워크/프로세스 분석이 직접 제시된다."),
    "39": ("FULL", "AHP로 EIT 융합기술을 도출하고 green building 및 건물 내 에너지소비기기 우선 결과가 직접 제시된다."),
    "40": ("FULL", "CYP450 ligand binding 예측, ligand-based/hybrid ML 모델 비교, molecular docking 및 벤치마크 성능이 직접 제시된다."),
    "41": ("FULL", "large-scale ML의 상업/과학적 전환, database와 ML 커뮤니티 교차, 시스템 지원과 open research question이 직접 제시된다."),
    "42": ("FULL", "공간/시간 적응 개념, 다구치 방법과 공진화 유전 알고리듬, 적용 시스템과 수행도 결과가 직접 제시된다."),
    "43": ("FULL", "시청각 feature fusion, alignment neural network/RNN attention, GMM-HMM 및 DNN-HMM backend 성능 향상이 직접 제시된다."),
    "44": ("FULL", "항우울제 pharmacogenomics에 ML/DL, neuroimaging, multi-omics, 치료반응 예측과 biomarker 탐색이 직접 제시된다."),
    "45": ("FULL", "A3 AR 조절자 발굴, Laplacian-modified naive Bayes/recursive partitioning/SVM, docking, 주요 성과가 직접 제시된다."),
    "46": ("FULL", "공간/시간 적응 개념과 두 지식획득 방법, 반도체/연료전송장치 적용, 성능 개선과 한계가 직접 제시된다."),
    "47": ("FULL", "잡음 환경 AV-ASR, acoustic/visual fusion, AliNN/RNN attention, GMM-HMM/DNN-HMM backend 성능 향상이 직접 제시된다."),
    "48": ("FULL", "PSO로 fuzzy ELM 활성화 함수 파라미터를 최적화하고 FCM activation level function을 쓰는 원리가 직접 제시된다."),
    "49": ("FULL", "스팸 패턴 변화 문제, LL/LLML 개념, Naive Bayes와 ELLA 앙상블 적용 아이디어가 직접 제시된다."),
}

MULTI_GOLD_REVIEW: dict[str, tuple[str, str]] = {
    "21": ("DUPLICATE_RECORD", "동일 제목의 중복 ScienceON 레코드가 있어 같은 논문을 가리키는 대체 record가 존재한다."),
    "24": ("DUPLICATE_RECORD", "동일 제목의 중복 ScienceON 레코드가 있어 같은 논문을 가리키는 대체 record가 존재한다."),
    "32": ("DUPLICATE_RECORD", "동일 제목의 중복 ScienceON 레코드가 있어 같은 논문을 가리키는 대체 record가 존재한다."),
    "36": ("DUPLICATE_RECORD", "동일/근접 제목의 특허분석 논문 record가 있어 같은 연구의 대체 record 가능성이 있다."),
    "37": ("POSSIBLE_SECONDARY", "스마트팜 시스템 논문도 일부 답변 가능하지만, 현재 gold가 질문의 농업 IT융합기술 적용과 효과를 더 직접 포괄한다."),
    "39": ("DUPLICATE_RECORD", "동일 제목의 ScienceON record가 추가로 있어 같은 논문을 가리키는 대체 record가 존재한다."),
}


def main() -> None:
    base = Path("experiments/gold_scienceon/artifacts")
    gold_rows = [
        json.loads(line)
        for line in (base / "scienceon_gold_docs.jsonl").open(encoding="utf-8")
        if line.strip()
    ]

    review_rows = []
    for row in gold_rows:
        qid = str(row["qid"])
        label, rationale = LLM_JUDGMENTS[qid]
        multi_label, multi_note = MULTI_GOLD_REVIEW.get(
            qid,
            ("NO", "gold 외 후보는 주제적으로 관련될 수 있으나, 질문의 특정 문서 기반 답변을 충분히 대체하는 독립 gold로 보기는 어렵다."),
        )
        review_rows.append(
            {
                "qid": qid,
                "llm_judge_label": label,
                "llm_judge_rationale": rationale,
                "gold_doc_id": row["gold_doc_id"],
                "gold_title": row["gold_title"],
                "multi_gold_label": multi_label,
                "multi_gold_note": multi_note,
                "question": row["question"],
                "gold_abstract": row["gold_abstract"],
            }
        )

    with (base / "scienceon_gold_llm_judge.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        for row in review_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    with (base / "scienceon_gold_llm_judge.csv").open("w", encoding="utf-8-sig", newline="") as f:
        fieldnames = [
            "qid",
            "llm_judge_label",
            "gold_doc_id",
            "gold_title",
            "multi_gold_label",
            "llm_judge_rationale",
            "multi_gold_note",
            "question",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in review_rows:
            writer.writerow({key: row[key] for key in fieldnames})

    label_counts = Counter(row["llm_judge_label"] for row in review_rows)
    multi_counts = Counter(row["multi_gold_label"] for row in review_rows)
    summary = {
        "total_questions": len(review_rows),
        "llm_judge_label_counts": dict(label_counts),
        "multi_gold_label_counts": dict(multi_counts),
        "partial_qids": [
            row["qid"] for row in review_rows if row["llm_judge_label"] == "PARTIAL"
        ],
        "insufficient_qids": [
            row["qid"] for row in review_rows if row["llm_judge_label"] == "INSUFFICIENT"
        ],
        "criteria": {
            "FULL": "초록만으로 질문의 핵심 대상과 대부분의 하위 요구에 답변 가능",
            "PARTIAL": "문서는 관련되거나 맞지만 초록만으로 결과/절차/요인/구조 일부가 부족",
            "INSUFFICIENT": "초록이 질문 답변 근거로 부족",
        },
    }
    (base / "scienceon_gold_llm_judge_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
