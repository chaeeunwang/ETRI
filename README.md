# ETRI 2026 EDA Template

## 1. 데이터 배치
아래 구조로 파일을 넣어주세요.

```text
etri_2026_eda_template/
├── data/
│   ├── ch2025_data_items/
│   ├── ch2026_metrics_train.csv
│   └── ch2026_submission_sample.csv
├── eda/
└── run_all_eda.py
```

## 2. 패키지 설치

```bash
pip install -r requirements.txt
```

## 3. 전체 실행

```bash
python run_all_eda.py
```

## 4. 생성되는 주요 시각화

### 01_label_distribution
- target별 count distribution
- target별 ratio distribution
- majority ratio by target
- class1 ratio by target
- target-class ratio heatmap

### 02_subject_prior
- subject별 labeled day 수
- subject-target prior heatmap
- target별 subject prior barplot
- target별 subject-class ratio heatmap
- subject prior range by target

### 03_naive_baseline
- global prior probability heatmap
- naive baseline logloss by target
- naive baseline accuracy by target
- naive baseline macro-F1 by target

### 04_target_correlation
- Pearson correlation heatmap
- Spearman correlation heatmap
- target pair joint-ratio heatmaps
- top target pair correlations

## 5. 경로 수정
기본은 `PROJECT_ROOT = Path(__file__).resolve().parents[2]`로 자동 인식됩니다.
수동으로 바꿔야 한다면 `eda/common/config.py`에서 `PROJECT_ROOT`만 프로젝트 최상위 폴더로 지정하세요.
