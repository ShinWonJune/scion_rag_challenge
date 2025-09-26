#!/usr/bin/env python3
"""
SciFact 예측 결과 평가 도구

사용법:
    python scifact_evaluation.py --ground_truth /app/scifact/data/questions/subquestion.csv --predictions /app/results/scifact_final_answers/scifact_predictions_20250926_171411.csv
"""

import os
import csv
import argparse
import pandas as pd
from typing import Dict, List, Tuple
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, classification_report
import json


def load_ground_truth(file_path: str) -> Dict[str, str]:
    """
    정답 파일(subquestion.csv)을 로드하여 딕셔너리로 반환
    
    Args:
        file_path: subquestion.csv 파일 경로
        
    Returns:
        Dict[str, str]: {question: label} 매핑
    """
    ground_truth = {}
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                question = row['question'].strip()
                label = row['label'].strip()
                ground_truth[question] = label
                
        print(f"✅ 정답 데이터 로드 완료: {len(ground_truth)}개 항목")
        return ground_truth
        
    except Exception as e:
        print(f"❌ 정답 파일 로드 오류: {e}")
        return {}


def load_predictions(file_path: str) -> Dict[str, str]:
    """
    예측 파일(scifact_predictions_*.csv)을 로드하여 딕셔너리로 반환
    
    Args:
        file_path: scifact_predictions_*.csv 파일 경로
        
    Returns:
        Dict[str, str]: {question: label} 매핑
    """
    predictions = {}
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                question = row['question'].strip()
                label = row['label'].strip()
                predictions[question] = label
                
        print(f"✅ 예측 데이터 로드 완료: {len(predictions)}개 항목")
        return predictions
        
    except Exception as e:
        print(f"❌ 예측 파일 로드 오류: {e}")
        return {}


def normalize_labels(label: str) -> str:
    """
    라벨을 정규화 (대소문자 통일, 공백 제거 등)
    
    Args:
        label: 원본 라벨
        
    Returns:
        str: 정규화된 라벨
    """
    label = label.strip().upper()
    
    # 다양한 형태의 라벨을 표준화
    if label in ['SUPPORT', 'SUPPORTED', 'TRUE', 'YES']:
        return 'SUPPORT'
    elif label in ['REFUTE', 'REFUTED', 'FALSE', 'NO']:
        return 'REFUTE'
    else:
        return label


def align_data(ground_truth: Dict[str, str], predictions: Dict[str, str]) -> Tuple[List[str], List[str], List[str]]:
    """
    정답과 예측 데이터를 정렬하여 매칭되는 항목들의 리스트를 반환
    
    Args:
        ground_truth: 정답 데이터
        predictions: 예측 데이터
        
    Returns:
        Tuple[List[str], List[str], List[str]]: (questions, true_labels, pred_labels)
    """
    questions = []
    true_labels = []
    pred_labels = []
    
    matched_count = 0
    missing_predictions = []
    
    for question, true_label in ground_truth.items():
        if question in predictions:
            questions.append(question)
            true_labels.append(normalize_labels(true_label))
            pred_labels.append(normalize_labels(predictions[question]))
            matched_count += 1
        else:
            missing_predictions.append(question)
    
    print(f"📊 데이터 매칭 결과:")
    print(f"   - 매칭된 항목: {matched_count}개")
    print(f"   - 누락된 예측: {len(missing_predictions)}개")
    
    if missing_predictions:
        print(f"⚠️ 다음 질문들에 대한 예측이 누락되었습니다:")
        for question in missing_predictions[:5]:  # 처음 5개만 표시
            print(f"     - {question[:60]}...")
        if len(missing_predictions) > 5:
            print(f"     - ... 및 {len(missing_predictions) - 5}개 더")
    
    return questions, true_labels, pred_labels


def calculate_metrics(true_labels: List[str], pred_labels: List[str]) -> Dict[str, float]:
    """
    분류 성능 지표를 계산
    
    Args:
        true_labels: 정답 라벨 리스트
        pred_labels: 예측 라벨 리스트
        
    Returns:
        Dict[str, float]: 성능 지표들
    """
    # 전체 정확도
    accuracy = sum(1 for true, pred in zip(true_labels, pred_labels) if true == pred) / len(true_labels)
    
    # 클래스별 precision, recall, f1-score 계산
    labels = ['SUPPORT', 'REFUTE']
    precision, recall, f1, support = precision_recall_fscore_support(
        true_labels, pred_labels, labels=labels, average=None, zero_division=0
    )
    
    # 매크로 평균
    macro_precision = precision.mean()
    macro_recall = recall.mean()
    macro_f1 = f1.mean()
    
    # 가중 평균
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        true_labels, pred_labels, labels=labels, average='weighted', zero_division=0
    )
    
    return {
        'accuracy': accuracy,
        'support_precision': precision[0] if len(precision) > 0 else 0.0,
        'support_recall': recall[0] if len(recall) > 0 else 0.0,
        'support_f1': f1[0] if len(f1) > 0 else 0.0,
        'support_count': support[0] if len(support) > 0 else 0,
        'refute_precision': precision[1] if len(precision) > 1 else 0.0,
        'refute_recall': recall[1] if len(recall) > 1 else 0.0,
        'refute_f1': f1[1] if len(f1) > 1 else 0.0,
        'refute_count': support[1] if len(support) > 1 else 0,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        'weighted_precision': weighted_precision,
        'weighted_recall': weighted_recall,
        'weighted_f1': weighted_f1
    }


def print_confusion_matrix(true_labels: List[str], pred_labels: List[str]):
    """
    혼동 행렬을 출력
    
    Args:
        true_labels: 정답 라벨 리스트
        pred_labels: 예측 라벨 리스트
    """
    labels = ['SUPPORT', 'REFUTE']
    cm = confusion_matrix(true_labels, pred_labels, labels=labels)
    
    print(f"\n📊 혼동 행렬 (Confusion Matrix):")
    print(f"{'':>12} {'SUPPORT':>10} {'REFUTE':>10}")
    print(f"{'SUPPORT':>12} {cm[0][0]:>10} {cm[0][1]:>10}")
    print(f"{'REFUTE':>12} {cm[1][0]:>10} {cm[1][1]:>10}")


def print_detailed_report(true_labels: List[str], pred_labels: List[str]):
    """
    상세한 분류 리포트를 출력
    
    Args:
        true_labels: 정답 라벨 리스트
        pred_labels: 예측 라벨 리스트
    """
    labels = ['SUPPORT', 'REFUTE']
    report = classification_report(true_labels, pred_labels, labels=labels, target_names=labels, digits=4)
    
    print(f"\n📈 상세 분류 리포트:")
    print(report)


def save_evaluation_results(metrics: Dict[str, float], output_path: str):
    """
    평가 결과를 JSON 파일로 저장
    
    Args:
        metrics: 평가 지표 딕셔너리
        output_path: 출력 파일 경로
    """
    try:
        # numpy 타입을 Python 기본 타입으로 변환
        serializable_metrics = {}
        for key, value in metrics.items():
            if hasattr(value, 'item'):  # numpy 스칼라
                serializable_metrics[key] = value.item()
            else:
                serializable_metrics[key] = value
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_metrics, f, indent=2, ensure_ascii=False)
        print(f"💾 평가 결과 저장 완료: {output_path}")
    except Exception as e:
        print(f"❌ 결과 저장 오류: {e}")


def analyze_errors(questions: List[str], true_labels: List[str], pred_labels: List[str], max_examples: int = 10):
    """
    오분류 사례들을 분석하고 출력
    
    Args:
        questions: 질문 리스트
        true_labels: 정답 라벨 리스트
        pred_labels: 예측 라벨 리스트
        max_examples: 출력할 최대 예시 수
    """
    errors = []
    
    for question, true_label, pred_label in zip(questions, true_labels, pred_labels):
        if true_label != pred_label:
            errors.append({
                'question': question,
                'true_label': true_label,
                'pred_label': pred_label
            })
    
    if errors:
        print(f"\n❌ 오분류 사례 분석 ({len(errors)}개 중 {min(max_examples, len(errors))}개 표시):")
        
        for i, error in enumerate(errors[:max_examples]):
            print(f"\n{i+1}. {error['question']}")
            print(f"   정답: {error['true_label']} → 예측: {error['pred_label']}")
    else:
        print(f"\n🎉 모든 예측이 정확합니다!")


def evaluate_scifact_predictions(ground_truth_path: str, predictions_path: str, output_dir: str = None) -> Dict[str, float]:
    """
    SciFact 예측 결과를 평가하는 메인 함수
    
    Args:
        ground_truth_path: 정답 CSV 파일 경로
        predictions_path: 예측 CSV 파일 경로
        output_dir: 결과 저장 디렉토리 (옵션)
        
    Returns:
        Dict[str, float]: 평가 지표들
    """
    # 데이터 로드
    ground_truth = load_ground_truth(ground_truth_path)
    predictions = load_predictions(predictions_path)
    
    if not ground_truth or not predictions:
        return {}
    
    # 데이터 정렬
    questions, true_labels, pred_labels = align_data(ground_truth, predictions)
    
    if not questions:
        print("❌ 매칭되는 데이터가 없습니다.")
        return {}
    
    # 성능 지표 계산
    metrics = calculate_metrics(true_labels, pred_labels)
    
    # 결과 출력
    print(f"\n🎯 SciFact 평가 결과:")
    print(f"{'='*50}")
    print(f"전체 정확도 (Accuracy): {metrics['accuracy']:.4f}")
    print(f"\n클래스별 성능:")
    print(f"  SUPPORT - Precision: {metrics['support_precision']:.4f}, Recall: {metrics['support_recall']:.4f}, F1: {metrics['support_f1']:.4f} (n={int(metrics['support_count'])})")
    print(f"  REFUTE  - Precision: {metrics['refute_precision']:.4f}, Recall: {metrics['refute_recall']:.4f}, F1: {metrics['refute_f1']:.4f} (n={int(metrics['refute_count'])})")
    print(f"\n평균 성능:")
    print(f"  Macro   - Precision: {metrics['macro_precision']:.4f}, Recall: {metrics['macro_recall']:.4f}, F1: {metrics['macro_f1']:.4f}")
    print(f"  Weighted- Precision: {metrics['weighted_precision']:.4f}, Recall: {metrics['weighted_recall']:.4f}, F1: {metrics['weighted_f1']:.4f}")
    
    # 혼동 행렬 및 상세 리포트
    print_confusion_matrix(true_labels, pred_labels)
    print_detailed_report(true_labels, pred_labels)
    
    # 오분류 분석
    analyze_errors(questions, true_labels, pred_labels)
    
    # 결과 저장
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(output_dir, f"scifact_evaluation_{timestamp}.json")
        save_evaluation_results(metrics, output_path)
    
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="SciFact 예측 결과를 평가합니다."
    )
    parser.add_argument(
        "--ground_truth",
        required=True,
        help="정답이 포함된 CSV 파일 경로 (예: /app/scifact/data/questions/subquestion.csv)"
    )
    parser.add_argument(
        "--predictions", 
        required=True,
        help="예측 결과가 포함된 CSV 파일 경로 (예: /app/results/scifact_final_answers/scifact_predictions_*.csv)"
    )
    parser.add_argument(
        "--output_dir",
        help="평가 결과를 저장할 디렉토리 (옵션)"
    )
    
    args = parser.parse_args()
    
    # 입력 파일 검증
    if not os.path.exists(args.ground_truth):
        print(f"❌ 정답 파일이 존재하지 않습니다: {args.ground_truth}")
        return 1
    
    if not os.path.exists(args.predictions):
        print(f"❌ 예측 파일이 존재하지 않습니다: {args.predictions}")
        return 1
    
    try:
        metrics = evaluate_scifact_predictions(
            args.ground_truth, 
            args.predictions, 
            args.output_dir
        )
        
        if metrics:
            print(f"\n🎉 평가 완료!")
            return 0
        else:
            print(f"\n❌ 평가 실패!")
            return 1
            
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        return 1


if __name__ == "__main__":
    exit(main())