# 🚀 Quick Start Guide

## 📋 필수 준비사항
1. **API 자격 증명 파일**:
   - `configs/gemini_api_credentials.json`
   - `configs/scienceon_api_credentials.json`

2. **질문 파일**: `test.csv` (첫 번째 컬럼에 질문)

## 🎯 기본 명령어

### 단일 질문 처리
```bash
python main.py single "질문 내용"
```

### 배치 처리 (전체)
```bash
python main.py batch test.csv
```

### 배치 처리 (제한)
```bash
python main.py batch test.csv 5  # 처음 5개만
```

### 목표 문서 수 설정
```bash
TARGET_DOCUMENTS=50 python main.py batch test.csv
```

## 📊 결과 파일
- **JSON**: `outputs/search_meta_results_*.json` (상세 결과)
- **CSV**: `outputs/search_results_*.csv` (표 형식)
- **JSONL**: `outputs/search_documents_*.jsonl` (문서별)

## ⚙️ 주요 설정
- **목표 문서 수**: 50개 (기본값)
- **검색어 수**: 15개
- **최대 페이지**: 5페이지
- **품질 필터링**: 자동 적용

## 🔧 문제 해결
- **API 오류**: 자격 증명 파일 확인
- **파일 없음**: CSV 파일 경로 확인
- **문서 부족**: 검색어/페이지 수 증가

---
**자세한 내용은 `USAGE_GUIDE.md` 참조**
