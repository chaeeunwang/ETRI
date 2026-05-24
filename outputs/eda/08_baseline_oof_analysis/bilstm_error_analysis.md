# 08 Baseline OOF Analysis

## 1. 분석 대상

- 사용 target: Q1, Q2, Q3, S1, S2, S3, S4
- OOF target 수: 7

## 2. Target별 LogLoss

- Q2: 0.686806 (n=450)
- Q1: 0.668629 (n=450)
- Q3: 0.667613 (n=450)
- S4: 0.640768 (n=450)
- S1: 0.583339 (n=450)
- S2: 0.572020 (n=450)
- S3: 0.548021 (n=450)

## 3. 주요 취약 구간

- 가장 약한 target: Q2 / logloss=0.686806
- 가장 약한 subject: id06 / target=Q3 / logloss=0.779359

## 4. Confidence 분석

- high confidence wrong cases: 16
- low confidence cases: 1592

## 5. 다음 액션

1. logloss가 큰 target부터 feature와 loss를 분리 검토
2. 특정 subject에서만 성능이 낮으면 subject-aware feature 추가
3. high-confidence wrong case는 calibration 또는 blending 우선 검토
4. low-confidence case가 많으면 feature 부족 또는 모델 underfitting 가능성 점검
