#!/usr/bin/env python3
"""
PubMedQA Evaluation Script

이 스크립트는 PubMedQA 데이터셋에 대한 모델 성능을 평가합니다.
- BLEU score (corpus-level) for long_answer
- METEOR score (sentence-level averaged) for long_answer  
- Context inclusion ratio (retrieval effectiveness)
- Final decision accuracy rate
"""

import pandas as pd
import numpy as np
import argparse
import datetime
import os
from typing import Dict, List, Tuple
import re
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.meteor_score import meteor_score
import nltk

# NLTK 데이터 다운로드 (필요시)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

def load_data(ground_truth_path: str, prediction_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Ground truth와 prediction 데이터를 로드합니다.
    
    Args:
        ground_truth_path: 정답 데이터 CSV 파일 경로
        prediction_path: 예측 데이터 CSV 파일 경로
        
    Returns:
        (ground_truth_df, prediction_df) 튜플
    """
    print("Loading datasets...")
    gt_df = pd.read_csv(ground_truth_path)
    pred_df = pd.read_csv(prediction_path)
    
    print(f"Ground truth: {len(gt_df)} samples")
    print(f"Predictions: {len(pred_df)} samples")
    
    # pubid로 매칭하여 정렬
    gt_df = gt_df.sort_values('pubid').reset_index(drop=True)
    pred_df = pred_df.sort_values('pubid').reset_index(drop=True)
    
    # pubid가 일치하는지 확인
    common_pubids = set(gt_df['pubid']) & set(pred_df['pubid'])
    print(f"Common pubids: {len(common_pubids)}")
    
    # 공통 pubid만 필터링
    gt_df = gt_df[gt_df['pubid'].isin(common_pubids)].sort_values('pubid').reset_index(drop=True)
    pred_df = pred_df[pred_df['pubid'].isin(common_pubids)].sort_values('pubid').reset_index(drop=True)
    
    return gt_df, pred_df

def preprocess_text(text: str) -> List[str]:
    """
    텍스트를 전처리하여 토큰 리스트로 변환합니다.
    
    Args:
        text: 입력 텍스트
        
    Returns:
        토큰 리스트
    """
    if pd.isna(text):
        return []
    
    # 소문자 변환 및 구두점 정리
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # 토큰화
    tokens = nltk.word_tokenize(text)
    return tokens

def calculate_bleu_score(references: List[str], predictions: List[str]) -> float:
    """
    BLEU 점수를 corpus-level에서 계산합니다.
    
    Args:
        references: 정답 텍스트 리스트
        predictions: 예측 텍스트 리스트
        
    Returns:
        BLEU 점수 (0-1)
    """
    print("Calculating BLEU score (corpus-level)...")
    
    # 전처리된 토큰 리스트로 변환
    ref_tokens = [[preprocess_text(ref)] for ref in references]  # 각 reference를 리스트로 감싸기
    pred_tokens = [preprocess_text(pred) for pred in predictions]
    
    # corpus-level BLEU 계산
    bleu_score = corpus_bleu(ref_tokens, pred_tokens)
    
    return bleu_score

def calculate_meteor_scores(references: List[str], predictions: List[str]) -> float:
    """
    METEOR 점수를 sentence-level에서 계산하고 평균을 구합니다.
    
    Args:
        references: 정답 텍스트 리스트
        predictions: 예측 텍스트 리스트
        
    Returns:
        평균 METEOR 점수 (0-1)
    """
    print("Calculating METEOR scores (sentence-level averaged)...")
    
    meteor_scores = []
    
    for ref, pred in zip(references, predictions):
        ref_tokens = preprocess_text(ref)
        pred_tokens = preprocess_text(pred)
        
        if len(ref_tokens) == 0 or len(pred_tokens) == 0:
            meteor_scores.append(0.0)
        else:
            try:
                score = meteor_score([ref_tokens], pred_tokens)
                meteor_scores.append(score)
            except:
                meteor_scores.append(0.0)
    
    avg_meteor = np.mean(meteor_scores)
    return avg_meteor

def calculate_context_inclusion_ratio(gt_contexts: List[str], pred_contexts: List[str]) -> float:
    """
    Ground truth context가 prediction context에 포함된 비율을 계산합니다.
    
    Args:
        gt_contexts: 정답 context 리스트  
        pred_contexts: 예측 context 리스트
        
    Returns:
        포함 비율 (0-1)
    """
    print("Calculating context inclusion ratio...")
    
    inclusion_count = 0
    total_count = 0
    
    for gt_ctx, pred_ctx in zip(gt_contexts, pred_contexts):
        total_count += 1
        
        if pd.isna(gt_ctx) or pd.isna(pred_ctx):
            continue
            
        gt_ctx = str(gt_ctx).lower()
        pred_ctx = str(pred_ctx).lower()
        
        # Ground truth context를 문장으로 분할
        gt_sentences = [s.strip() for s in gt_ctx.split('.') if s.strip() and len(s.strip()) > 20]
        
        if not gt_sentences:
            continue
            
        # 각 문장에 대해 키워드 기반 매칭 확인
        sentence_matches = 0
        for sentence in gt_sentences:
            # 문장에서 의미있는 단어들 추출 (3글자 이상, 불용어 제외)
            words = [w.strip() for w in sentence.split() 
                    if len(w.strip()) > 3 and w.strip() not in ['this', 'that', 'with', 'from', 'were', 'have', 'been', 'they', 'their', 'there']]
            
            if len(words) < 3:
                continue
                
            # 핵심 단어들이 prediction context에 포함되어 있는지 확인
            word_matches = sum(1 for word in words if word in pred_ctx)
            match_ratio = word_matches / len(words)
            
            # 70% 이상의 단어가 매칭되면 해당 문장이 포함된 것으로 간주
            if match_ratio >= 0.7:
                sentence_matches += 1
        
        # 전체 문장의 50% 이상이 포함되어 있으면 context가 포함된 것으로 간주
        if len(gt_sentences) > 0 and sentence_matches / len(gt_sentences) >= 0.5:
            inclusion_count += 1
    
    ratio = inclusion_count / total_count if total_count > 0 else 0.0
    print(f"Context inclusion: {inclusion_count}/{total_count} = {ratio:.3f}")
    
    return ratio

def calculate_final_decision_accuracy(gt_decisions: List[str], pred_decisions: List[str]) -> Tuple[float, Dict]:
    """
    Final decision의 정확도를 계산합니다.
    
    Args:
        gt_decisions: 정답 final_decision 리스트
        pred_decisions: 예측 final_decision 리스트
        
    Returns:
        (정확도, confusion_matrix_dict) 튜플
    """
    print("Calculating final decision accuracy...")
    
    correct_count = 0
    total_count = 0
    
    # Confusion matrix를 위한 딕셔너리
    confusion_matrix = {}
    labels = ['yes', 'no', 'maybe']
    
    for label1 in labels:
        confusion_matrix[label1] = {}
        for label2 in labels:
            confusion_matrix[label1][label2] = 0
    
    for gt_dec, pred_dec in zip(gt_decisions, pred_decisions):
        if pd.isna(gt_dec) or pd.isna(pred_dec):
            continue
            
        total_count += 1
        
        # 정규화 (소문자, 공백 제거)
        gt_dec = str(gt_dec).lower().strip()
        pred_dec = str(pred_dec).lower().strip()
        
        # Confusion matrix 업데이트
        if gt_dec in labels and pred_dec in labels:
            confusion_matrix[gt_dec][pred_dec] += 1
        
        if gt_dec == pred_dec:
            correct_count += 1
    
    accuracy = correct_count / total_count if total_count > 0 else 0.0
    print(f"Final decision accuracy: {correct_count}/{total_count} = {accuracy:.3f}")
    
    return accuracy, confusion_matrix

def main():
    """메인 평가 함수"""
    
    parser = argparse.ArgumentParser(
        description="Evaluate PubMedQA model predictions against ground truth"
    )
    
    parser.add_argument(
        "--ground_truth_path",
        default="/app/pubmedqa/data/evaluation/pubmedqa_answer.csv",
        help="Path to ground truth CSV file (default: /app/pubmedqa/data/evaluation/pubmedqa_answer.csv)"
    )
    
    parser.add_argument(
        "--input_path",
        default="/app/results/pubmedqa_final_answers/pubmedqa_final_predictions.csv",
        help="Path to prediction CSV file (default: /app/results/pubmedqa_final_answers/pubmedqa_final_predictions.csv)"
    )
    
    parser.add_argument(
        "--output_dir",
        default="../results/pubmedqa_evaluation",
        help="Output directory for evaluation results (default: ../results/pubmedqa_evaluation)"
    )
    
    parser.add_argument(
        "--output_file",
        default=None,
        help="Output CSV filename. If not specified, uses timestamp-based filename."
    )
    
    args = parser.parse_args()
    
    # 현재 날짜와 시간을 가져옵니다.
    now = datetime.datetime.now()
    # 원하는 형식(yymmdd_hhmmss)으로 문자열을 만듭니다.
    timestamp_str = now.strftime("%y%m%d_%H%M%S")
    
    # 출력 파일명 설정
    if args.output_file:
        output_file = args.output_file
    else:
        output_file = f"pubmedqa_evaluation_results_{timestamp_str}.csv"
    
    # 출력 디렉토리 생성
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    output_path = os.path.join(args.output_dir, output_file)
    
    print("="*60)
    print("PubMedQA Evaluation Report")
    print("="*60)
    print(f"Ground truth file: {args.ground_truth_path}")
    print(f"Prediction file: {args.input_path}")
    print(f"Output file: {output_path}")
    print(f"Timestamp: {timestamp_str}")
    
    # 데이터 로드
    gt_df, pred_df = load_data(args.ground_truth_path, args.input_path)
    
    print(f"\nEvaluating {len(gt_df)} samples...")
    print("-"*40)
    
    # 1. BLEU 점수 계산 (corpus-level)
    bleu_score = calculate_bleu_score(
        gt_df['long_answer'].tolist(),
        pred_df['long_answer'].tolist()
    )
    
    # 2. METEOR 점수 계산 (sentence-level averaged)
    meteor_score_avg = calculate_meteor_scores(
        gt_df['long_answer'].tolist(),
        pred_df['long_answer'].tolist()
    )
    
    # 3. Context inclusion ratio 계산
    context_inclusion_ratio = calculate_context_inclusion_ratio(
        gt_df['context'].tolist(),
        pred_df['context'].tolist()
    )
    
    # 4. Final decision accuracy 계산
    final_decision_accuracy, confusion_matrix = calculate_final_decision_accuracy(
        gt_df['final_decision'].tolist(),
        pred_df['final_decision'].tolist()
    )
    
    # 결과 출력
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"📊 Long Answer Quality:")
    print(f"   - BLEU Score (corpus-level):     {bleu_score:.4f}")
    print(f"   - METEOR Score (averaged):       {meteor_score_avg:.4f}")
    print()
    print(f"🔍 Retrieval Effectiveness:")
    print(f"   - Context Inclusion Ratio:       {context_inclusion_ratio:.4f}")
    print()
    print(f"🎯 Final Decision Performance:")
    print(f"   - Accuracy Rate:                 {final_decision_accuracy:.4f}")
    print()
    
    # 분포 정보 추가
    print(f"📈 Data Distribution:")
    print(f"   - Total samples evaluated:       {len(gt_df)}")
    
    gt_dist = gt_df['final_decision'].value_counts()
    pred_dist = pred_df['final_decision'].value_counts()
    
    print(f"   - Ground Truth distribution:")
    for decision, count in gt_dist.items():
        print(f"     * {decision}: {count}")
    
    print(f"   - Prediction distribution:")
    for decision, count in pred_dist.items():
        print(f"     * {decision}: {count}")
    
    print()
    print(f"🔍 Confusion Matrix (GT → Predicted):")
    print(f"     {'':>8} {'yes':>6} {'no':>6} {'maybe':>6}")
    for gt_label in ['yes', 'no', 'maybe']:
        row = f"     {gt_label:>8}"
        for pred_label in ['yes', 'no', 'maybe']:
            count = confusion_matrix.get(gt_label, {}).get(pred_label, 0)
            row += f" {count:>6}"
        print(row)
    
    print("="*60)
    
    # 결과를 파일로 저장
    results = {
        'timestamp': timestamp_str,
        'bleu_score': bleu_score,
        'meteor_score': meteor_score_avg,
        'context_inclusion_ratio': context_inclusion_ratio,
        'final_decision_accuracy': final_decision_accuracy,
        'total_samples': len(gt_df),
        'ground_truth_file': args.ground_truth_path,
        'prediction_file': args.input_path
    }
    
    results_df = pd.DataFrame([results])
    results_df.to_csv(output_path, index=False)
    print(f"Results saved to: {output_path}")

if __name__ == "__main__":
    main()