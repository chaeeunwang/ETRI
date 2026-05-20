# Subject Distribution EDA Summary
## 1. 생성 목적
- subject별 데이터 수, 날짜 범위, 날짜 연속성, target prior를 확인했습니다.
- 현재 sequence 모델이 lookback=14를 사용한다고 가정하고, subject별 최대 연속 일수가 14일 이상인지 점검했습니다.

## 2. Subject별 row 수
- train row 최소값: 33
- train row 최대값: 57
- train row 표준편차: 8.300
- train row가 적은 subject TOP3:
  - id03: train_rows=33, test_rows=21
  - id10: train_rows=33, test_rows=22
  - id09: train_rows=41, test_rows=27

## 3. 날짜 연속성
- train에서 날짜 gap이 있는 subject 수: 10
- train에서 max_consecutive_days < 14인 subject 수: 5
- lookback 위험 subject:
  - id03: max_consecutive_days=6, missing_days_count=25, max_gap_days=14
  - id05: max_consecutive_days=10, missing_days_count=35, max_gap_days=12
  - id06: max_consecutive_days=13, missing_days_count=29, max_gap_days=9
  - id09: max_consecutive_days=11, missing_days_count=24, max_gap_days=17
  - id10: max_consecutive_days=5, missing_days_count=38, max_gap_days=25

## 4. Target prior 차이
- subject별 class ratio 변동성이 큰 target/class TOP10:
  - target=S3, class=0: mean=0.348, std=0.247, min=0.042, max=0.864
  - target=S3, class=1: mean=0.652, std=0.247, min=0.136, max=0.958
  - target=S2, class=0: mean=0.365, std=0.219, min=0.083, max=0.750
  - target=S2, class=1: mean=0.635, std=0.219, min=0.250, max=0.917
  - target=S4, class=0: mean=0.451, std=0.195, min=0.146, max=0.848
  - target=S4, class=1: mean=0.549, std=0.195, min=0.152, max=0.854
  - target=S1, class=1: mean=0.678, std=0.180, min=0.446, max=0.938
  - target=S1, class=0: mean=0.322, std=0.180, min=0.062, max=0.554
  - target=Q1, class=1: mean=0.507, std=0.175, min=0.146, max=0.848
  - target=Q1, class=0: mean=0.493, std=0.175, min=0.152, max=0.854

## 5. 1차 판단
- subject별 row 수 차이가 크면 subject별 sampling 또는 subject-aware validation을 검토해야 합니다.
- max_consecutive_days가 14보다 짧은 subject가 있으면 lookback 축소, masking, sequence padding 정책을 재검토해야 합니다.
- subject별 target prior 차이가 크면 subject prior feature, subject embedding, personalized normalization을 검토해야 합니다.
- 날짜 gap이 크면 random split보다 time-aware split을 유지하는 것이 안전합니다.

## 6. 생성 파일
- subject_row_count.csv
- subject_date_range.csv
- subject_missing_dates.csv
- subject_timeline_plot.png
- subject_target_prior.csv
- subject_target_prior_wide.csv
- subject_summary.md
