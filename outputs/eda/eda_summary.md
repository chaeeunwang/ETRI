# EDA Summary

## 0. Executive Summary

본 EDA는 ETRI 휴먼이해 인공지능 논문경진대회 데이터의 기본 구조, target 분포, subject 편차, timestamp/leakage 위험, 센서 처리 우선순위, missing 전략, train/test shift, baseline OOF 취약점을 통합 점검하기 위해 수행하였다.

핵심 결론은 다음과 같다.

1. train/test 기본 구조는 정상이다.
   - train: 450 rows, 10 columns
   - test: 250 rows, 10 columns
   - subject 수: train/test 모두 10명
   - target: `Q1, Q2, Q3, S1, S2, S3, S4`
   - `subject_id + lifelog_date` 중복 없음
   - `sleep_date - lifelog_date = 1일`로 일관됨

2. 전체 target class imbalance는 심하지 않다.
   - 모든 target은 binary classification target이다.
   - 모든 target의 valid row 수는 450이며 missing label은 없다.
   - majority ratio는 0.504~0.682 범위다.
   - 80:20 이상의 심한 불균형은 없다.
   - 따라서 현재 단계에서는 `pos_weight`, `focal loss`, aggressive resampling을 우선 적용하지 않는다.

3. 가장 큰 병목은 class imbalance가 아니라 subject별 label prior 차이다.
   - 모든 target에서 subject별 positive rate range가 크다.
   - 특히 `S3`, `S2`, `S4`, `Q1`에서 subject별 기준 차이가 크다.
   - subject-aware modeling, subject-wise normalization, personalized deviation feature가 핵심이다.

4. subject prior만 사용해도 naive baseline 대비 log-loss가 크게 개선된다.
   - 이는 센서 feature 이전에 개인별 기준선 자체가 강한 signal임을 의미한다.
   - 단, 현재 subject prior baseline은 in-sample reference이므로 실제 성능 추정용으로 사용하면 안 된다.
   - subject prior feature는 반드시 chronological validation fold 내부에서만 계산해야 한다.

5. target 간 상관 구조가 존재한다.
   - `S2 ↔ S4`, `S2 ↔ S3`, `S1 ↔ S2`, `Q1 ↔ S1`, `Q2 ↔ Q3` 상관이 상대적으로 높다.
   - multi-task learning, target-wise blending, target prediction stacking 후보가 될 수 있다.
   - 단, target label 자체를 feature로 직접 사용하는 것은 leakage이므로 금지한다.

6. `LOOKBACK=14` 기반 BiLSTM은 데이터 연속성 측면에서 위험하다.
   - 전체 10명 중 5명이 14일 이상 연속 관측 구간을 충분히 확보하지 못한다.
   - `id03`, `id10`은 최대 연속 관측일이 각각 6일, 5일이다.
   - sequence 모델은 실제 생활 패턴보다 padding/결측/불연속성을 학습할 위험이 있다.

7. timestamp 기반 leakage 위험은 낮지만 feature 생성 단계의 leakage는 계속 검증해야 한다.
   - 센서 timestamp 파싱 정상
   - invalid timestamp 없음
   - boundary risk 없음
   - subject별 fold 시간 순서 보존
   - 단, 수면 이후 timestamp, raw sleep-derived feature, train+test 전체 통계 사용은 금지한다.

8. 초기 feature engineering은 핵심 센서 중심으로 진행한다.
   - 1차 핵심 센서: `mScreenStatus`, `mACStatus`, `mActivity`, `mLight`, `wPedo`, `wLight`, `mUsageStats`
   - 주의해서 사용할 센서: `wHr`
   - 후순위 센서: `mBle`, `mWifi`, `mGps`, `mAmbience`

9. missing은 단순 보간보다 coverage feature로 활용하는 전략이 적절하다.
   - missing ratio 0.5 이상 센서 없음
   - train/test missing shift 큼 없음
   - target-dependent missingness 큼 없음
   - 다만 `wHr`, `mUsageStats`, `wPedo`, `wLight`는 야간/새벽 coverage feature를 추가한다.

10. 현재 baseline OOF 기준 취약 target은 `Q2`, 취약 subject-target은 `id06-Q3`이다.
    - Q2 logloss: 0.686806
    - Q1 logloss: 0.668629
    - Q3 logloss: 0.667613
    - 가장 약한 subject-target: `id06 / Q3 / logloss=0.779359`
    - high-confidence wrong case 16건 → calibration/blending 필요
    - low-confidence case 1592건 → feature 부족 또는 underfitting 가능성 점검 필요

---

## 1. Data Overview

| 항목                     |                       결과 |
| ------------------------ | -------------------------: |
| train rows               |                        450 |
| train columns            |                         10 |
| test rows                |                        250 |
| test columns             |                         10 |
| train subject count      |                         10 |
| test subject count       |                         10 |
| train duplicate key rows |                          0 |
| test duplicate key rows  |                          0 |
| target columns           | Q1, Q2, Q3, S1, S2, S3, S4 |
| S4 in train              |                       True |
| S4 in submission         |                       True |

### Date Range

| split | lifelog_date min | lifelog_date max | sleep_date min | sleep_date max |
| ----- | ---------------- | ---------------- | -------------- | -------------- |
| train | 2024-06-03       | 2024-11-14       | 2024-06-04     | 2024-11-15     |
| test  | 2024-07-06       | 2024-11-19       | 2024-07-07     | 2024-11-20     |

### 판단

- `lifelog_date` 당일의 라이프로그로 다음날 `sleep_date`의 target을 예측하는 구조로 해석된다.
- `subject_id + lifelog_date`는 train/test 모두 unique하다.
- S4가 submission에도 존재하므로 현재 모델의 7-target 구조는 유지한다.

---

## 2. Target Distribution

### 2.1 Label Distribution

| Target | Valid | Classes | Majority Class | Majority Ratio | Missing |
| ------ | ----: | ------: | -------------: | -------------: | ------: |
| Q1     |   450 |       2 |              0 |          0.504 |       0 |
| Q2     |   450 |       2 |              1 |          0.562 |       0 |
| Q3     |   450 |       2 |              1 |          0.600 |       0 |
| S1     |   450 |       2 |              1 |          0.682 |       0 |
| S2     |   450 |       2 |              1 |          0.651 |       0 |
| S3     |   450 |       2 |              1 |          0.662 |       0 |
| S4     |   450 |       2 |              1 |          0.560 |       0 |

### 판단

- 모든 target은 binary classification target이다.
- label missing은 없다.
- 전체 majority ratio는 0.504~0.682 범위로 심각한 불균형은 아니다.
- 현재 단계에서는 `pos_weight`, `focal loss`, aggressive resampling은 보류한다.
- 본 대회는 log-loss 기반이므로 불필요한 loss weighting은 probability calibration을 악화시킬 수 있다.

---

## 3. Subject Prior Analysis

### 3.1 Subject-wise Prior Difference

| Target | Subject Prior Range | Subject Prior Std | 판단        |
| ------ | ------------------: | ----------------: | ----------- |
| Q1     |               0.703 |             0.175 | High        |
| Q2     |               0.422 |             0.147 | High        |
| Q3     |               0.333 |             0.124 | Medium-High |
| S1     |               0.491 |             0.180 | High        |
| S2     |               0.667 |             0.219 | High        |
| S3     |               0.822 |             0.247 | Very High   |
| S4     |               0.703 |             0.195 | High        |

### 판단

- 전체 class imbalance보다 subject별 label 기준 차이가 훨씬 중요하다.
- `S3`는 subject prior range 0.822, std 0.247로 개인차가 가장 크다.
- `S2`, `S4`, `Q1`도 subject별 기준 차이가 매우 크다.
- subject 정보를 반영하지 않으면 전체 평균 패턴에 치우친 예측이 발생할 가능성이 높다.

### 적용 결정

| 항목                               | 결정                   |
| ---------------------------------- | ---------------------- |
| `subject_id` one-hot               | 우선 적용              |
| subject embedding                  | BiLSTM 실험 시 적용    |
| subject prior feature              | smoothing 적용 후 검토 |
| subject-wise normalization         | 우선 적용              |
| personalized deviation feature     | 우선 적용              |
| subject별 단독 calibration         | 표본 수 부족으로 보류  |
| 전체 calibration + subject feature | 우선 적용              |

### 주의사항

- subject prior feature는 강한 baseline이지만 leakage 위험이 크다.
- 반드시 fold train 구간에서만 계산해야 한다.
- validation/test 대상 row의 label 정보를 prior 계산에 포함하면 안 된다.
- smoothing 없이 subject prior를 직접 넣으면 소표본 subject에서 과적합될 수 있다.

---

## 4. Naive Baseline Analysis

### 4.1 Baseline 정의

| Baseline                                     | 설명                                              | 사용 목적                |
| -------------------------------------------- | ------------------------------------------------- | ------------------------ |
| `global_prior`                               | 전체 train label 분포만 사용한 가장 기본 baseline | target별 최소 기준선     |
| `subject_prior_smoothed_in_sample_reference` | subject별 prior 효과 확인용 reference             | subject effect 강도 확인 |

> `subject_prior_smoothed_in_sample_reference`는 in-sample reference이다.  
> 실제 성능 추정용으로 해석하면 안 되며, chronological validation에서 fold별로 다시 계산해야 한다.

### 4.2 Target별 Naive Baseline 성능

| Target | Baseline                                   | LogLoss | Accuracy | Macro F1 |
| ------ | ------------------------------------------ | ------: | -------: | -------: |
| Q1     | global_prior                               |  0.6931 |    0.504 |    0.335 |
| Q1     | subject_prior_smoothed_in_sample_reference |  0.6404 |    0.607 |    0.603 |
| Q2     | global_prior                               |  0.6854 |    0.562 |    0.360 |
| Q2     | subject_prior_smoothed_in_sample_reference |  0.6464 |    0.609 |    0.579 |
| Q3     | global_prior                               |  0.6730 |    0.600 |    0.375 |
| Q3     | subject_prior_smoothed_in_sample_reference |  0.6431 |    0.624 |    0.559 |
| S1     | global_prior                               |  0.6252 |    0.682 |    0.406 |
| S1     | subject_prior_smoothed_in_sample_reference |  0.5558 |    0.696 |    0.558 |
| S2     | global_prior                               |  0.6468 |    0.651 |    0.394 |
| S2     | subject_prior_smoothed_in_sample_reference |  0.5558 |    0.720 |    0.636 |
| S3     | global_prior                               |  0.6396 |    0.662 |    0.398 |
| S3     | subject_prior_smoothed_in_sample_reference |  0.5180 |    0.733 |    0.609 |
| S4     | global_prior                               |  0.6859 |    0.560 |    0.359 |
| S4     | subject_prior_smoothed_in_sample_reference |  0.6232 |    0.636 |    0.605 |

### 4.3 개선 폭 요약

| Target | Global Prior LogLoss | Subject Prior LogLoss | 개선폭 |
| ------ | -------------------: | --------------------: | -----: |
| Q1     |               0.6931 |                0.6404 | 0.0527 |
| Q2     |               0.6854 |                0.6464 | 0.0390 |
| Q3     |               0.6730 |                0.6431 | 0.0299 |
| S1     |               0.6252 |                0.5558 | 0.0694 |
| S2     |               0.6468 |                0.5558 | 0.0910 |
| S3     |               0.6396 |                0.5180 | 0.1216 |
| S4     |               0.6859 |                0.6232 | 0.0627 |

### 판단

- subject prior만으로 모든 target에서 global prior 대비 log-loss가 개선된다.
- 개선폭은 `S3 > S2 > S1 > S4 > Q1 > Q2 > Q3` 순으로 크다.
- 특히 `S3`, `S2`는 센서 feature보다 개인별 기준선 반영이 먼저 필요하다.
- subject-aware feature 없이 복잡한 모델을 확장하면 개인차를 충분히 반영하지 못할 수 있다.

### 적용 결정

- LightGBM/XGBoost/CatBoost baseline에 smoothed subject prior feature를 추가한다.
- smoothing은 global prior와 subject prior를 혼합한다.
- subject prior는 fold별 train split에서만 계산한다.
- subject prior 단독 성능을 별도 baseline으로 저장해 이후 모델 개선폭을 비교한다.

---

## 5. Target Correlation Analysis

### 5.1 Target Pair Correlation

| Target Pair | Spearman | Pearson |   n | 해석                                           |
| ----------- | -------: | ------: | --: | ---------------------------------------------- |
| S2 ↔ S4     |    0.478 |   0.478 | 450 | 수면 효율과 WASO 계열 지표 간 관련성 큼        |
| S2 ↔ S3     |    0.394 |   0.394 | 450 | 수면 효율과 수면 시작 지연 간 관련성 있음      |
| S1 ↔ S2     |    0.382 |   0.382 | 450 | 총 수면 시간 적절성과 수면 효율 간 관련성 있음 |
| Q1 ↔ S1     |    0.361 |   0.361 | 450 | 주관적 수면 만족도와 수면 시간 적절성 관련     |
| Q2 ↔ Q3     |    0.340 |   0.340 | 450 | 피로도와 스트레스 간 관련                      |
| Q1 ↔ Q2     |    0.122 |   0.122 | 450 | 약한 양의 상관                                 |
| Q1 ↔ S3     |   -0.119 |  -0.119 | 450 | 약한 음의 상관                                 |
| S1 ↔ S3     |    0.118 |   0.118 | 450 | 약한 양의 상관                                 |
| S1 ↔ S4     |    0.107 |   0.107 | 450 | 약한 양의 상관                                 |
| Q1 ↔ Q3     |    0.102 |   0.102 | 450 | 약한 양의 상관                                 |

### 판단

- 수면 관련 target 간 상관이 상대적으로 높다.
  - `S2 ↔ S4`
  - `S2 ↔ S3`
  - `S1 ↔ S2`
- 주관식 target 중에서는 `Q2 ↔ Q3` 상관이 가장 높다.
- `Q1`은 `S1`과 가장 관련이 크며, 수면 만족도가 수면 시간 적절성과 연결될 가능성이 있다.
- target 간 상관은 multi-task learning 또는 target-level blending의 근거가 된다.

### 적용 가능 전략

| 전략                                  | 적용 여부 | 설명                             |
| ------------------------------------- | --------- | -------------------------------- |
| multi-task learning                   | 후보      | target 간 공통 signal 활용 가능  |
| target-wise ensemble                  | 우선 적용 | target별 강한 모델을 다르게 선택 |
| prediction stacking                   | 후보      | OOF prediction 기반으로만 가능   |
| label 직접 feature 사용               | 금지      | target leakage                   |
| correlated target group별 calibration | 후보      | 수면 계열과 설문 계열 분리 가능  |

### target group 후보

| Group                        | Targets        | 설명                       |
| ---------------------------- | -------------- | -------------------------- |
| Subjective state group       | Q1, Q2, Q3     | 설문 기반 주관 상태        |
| Sleep metric group           | S1, S2, S3, S4 | 수면 지표 기반 target      |
| Sleep efficiency/onset group | S2, S3, S4     | 수면 효율·입면·각성 관련   |
| Fatigue/stress group         | Q2, Q3         | 취침 전 피로·스트레스 관련 |

---

## 6. Subject Distribution

### 6.1 Subject별 row 수

| 항목               |    값 |
| ------------------ | ----: |
| train row 최소값   |    33 |
| train row 최대값   |    57 |
| train row 표준편차 | 8.300 |

train row가 적은 subject는 다음과 같다.

| subject_id | train_rows | test_rows |
| ---------- | ---------: | --------: |
| id03       |         33 |        21 |
| id10       |         33 |        22 |
| id09       |         41 |        27 |

### 판단

- `id03`, `id10`은 학습 가능한 일자 수가 가장 적다.
- sequence 모델에서는 생성 가능한 window 수가 제한될 가능성이 높다.
- subject별 성능을 반드시 분리 평가해야 한다.
- subject별 calibration을 독립적으로 적용하기에는 표본 수가 부족하다.

---

## 7. Sequence Continuity & LOOKBACK Risk

### 7.1 날짜 연속성

| 항목                                   | 결과 |
| -------------------------------------- | ---: |
| 날짜 gap이 있는 subject 수             |   10 |
| `max_consecutive_days < 14` subject 수 |    5 |

`LOOKBACK=14` 위험 subject는 다음과 같다.

| subject_id | max_consecutive_days | missing_days_count | max_gap_days |
| ---------- | -------------------: | -----------------: | -----------: |
| id03       |                    6 |                 25 |           14 |
| id05       |                   10 |                 35 |           12 |
| id06       |                   13 |                 29 |            9 |
| id09       |                   11 |                 24 |           17 |
| id10       |                    5 |                 38 |           25 |

### 판단

- 현재 BiLSTM의 `LOOKBACK=14`는 과한 설정일 가능성이 높다.
- 14일 window를 강제로 만들면 실제 14일 연속 생활 패턴이 아니라 불연속 날짜, padding, 결측 패턴을 함께 학습할 위험이 있다.
- 특히 `id03`, `id10`은 sequence 모델에서 학습 안정성이 낮을 수 있다.

### 후속 실험

| 실험 | lookback | 목적                             |
| ---- | -------: | -------------------------------- |
| A    |       14 | 기존 baseline 유지               |
| B    |        7 | 1주일 단위 생활 패턴 검증        |
| C    |        5 | id03/id10까지 고려한 현실적 후보 |
| D    |        3 | 짧은 시계열 기준 안전 비교군     |

### 결정

- `lookback=14` 단독 사용은 보류한다.
- `lookback=5`, `lookback=7`을 우선 비교한다.
- `sequence_window_quality.csv`를 생성해 lookback별 usable window 수와 valid day ratio를 검증한다.

---

## 8. Timestamp & Leakage Risk

### 8.1 Date Relationship

| split | diff_days | count | ratio |
| ----- | --------: | ----: | ----: |
| train |         1 |   450 |   1.0 |
| test  |         1 |   250 |   1.0 |

### 8.2 Sensor Timestamp Check

| 항목                      | 결과 |
| ------------------------- | ---: |
| 정상 파싱된 센서 파일 수  |   12 |
| invalid timestamp row     |    0 |
| boundary risk sensor file |    0 |

### 판단

- timestamp 형식 문제는 발견되지 않았다.
- 전체 lifelog date 범위를 벗어나는 sensor timestamp도 발견되지 않았다.
- subject별 fold는 시간 순서를 보존한다.
- 명확한 timestamp 기반 leakage 위험은 낮다.

### 지속 관리해야 할 leakage checklist

- [ ] feature 생성 시 `lifelog_date` 이후 데이터가 포함되지 않는지 확인
- [ ] `sleep_date` 기준 미래 timestamp를 사용하지 않는지 확인
- [ ] S1/S2/S3/S4 계산 원천이 되는 raw sleep sensor를 직접 feature로 쓰지 않는지 확인
- [ ] train+test 전체 통계로 scaling/normalization하지 않는지 확인
- [ ] subject prior feature는 fold train 구간에서만 계산
- [ ] calibration도 validation fold 기준으로만 fitting
- [ ] `LOOKBACK` window가 실제 관측일 기준으로 유효한지 확인
- [ ] target correlation을 이용한 stacking은 OOF prediction으로만 수행

---

## 9. Sensor Schema & Processing Priority

### 9.1 센서 처리 결과

| 항목                        | 결과 |
| --------------------------- | ---: |
| 총 센서 파일 수             |   12 |
| 정상 로딩 센서 수           |   12 |
| manual inspection 필요 센서 |    0 |
| object/list 센서            |    6 |
| discrete 센서               |    3 |
| continuous 센서             |    2 |
| continuous/multi-value 센서 |    1 |

### 9.2 1차 처리 대상 센서

| Sensor        | Type                   |  Freq | Value Column                              | Feature 방향                                       |
| ------------- | ---------------------- | ----: | ----------------------------------------- | -------------------------------------------------- |
| mACStatus     | discrete               |  1min | m_charging                                | 충전 시간, 야간 충전 여부                          |
| mActivity     | discrete               |  1min | m_activity                                | activity code별 count/duration/ratio               |
| mScreenStatus | discrete               |  1min | m_screen_use                              | 화면 사용 시간, 야간/취침 전 screen-on             |
| mLight        | continuous             | 10min | m_light                                   | 조도 평균/최대, 야간 밝기                          |
| wLight        | continuous             |  1min | w_light                                   | 조도 평균/최대, 야간 밝기                          |
| wPedo         | continuous/multi-value |  1min | step, distance, speed, burned_calories 등 | 걸음수/거리/칼로리/속도 집계                       |
| wHr           | object/list            |  1min | heart_rate                                | 심박 mean/std/min/max, 야간 심박                   |
| mUsageStats   | object/list            | 10min | m_usage_stats                             | 앱 사용량, 야간/취침 전 사용량, top-k app/category |

### 9.3 후순위 센서

| Sensor    | Type        | 처리 방향                                        |
| --------- | ----------- | ------------------------------------------------ |
| mAmbience | object/list | 초기 제외, 필요 시 sound label count/probability |
| mBle      | object/list | 초기 제외 또는 count/presence/RSSI               |
| mGps      | object/list | 초기 제외, 필요 시 movement coverage/speed       |
| mWifi     | object/list | 초기 제외 또는 AP count/RSSI                     |

### 결정

- 1차 feature engineering은 `mScreenStatus`, `mACStatus`, `mActivity`, `mLight`, `wPedo`, `wLight`, `mUsageStats` 중심으로 진행한다.
- `wHr`는 수면 예측에 중요하지만 야간 결측이 크므로 coverage feature와 함께 사용한다.
- `mGps`, `mBle`, `mWifi`, `mAmbience`는 noise/sparsity 가능성이 높아 1차 baseline 이후 성능 개선 실험에서 추가한다.

---

## 10. Missing Analysis

### 10.1 Overall Result

| 항목                                     | 결과 |
| ---------------------------------------- | ---- |
| missing ratio >= 0.5 센서                | 없음 |
| train/test missing-ratio gap >= 0.2 센서 | 없음 |
| target-dependent missingness gap >= 0.15 | 없음 |

### 판단

- 하루 단위 센서 결측은 심각하지 않다.
- 특정 센서를 결측률 때문에 즉시 제거할 필요는 없다.
- 결측 자체가 강한 target leakage 신호로 보이지는 않는다.
- 복잡한 보간보다 coverage feature를 추가하는 전략이 적절하다.

### 10.2 Stable Sensors

대부분 시간대에서 결측률이 낮은 센서:

- `mACStatus`
- `mActivity`
- `mAmbience`
- `mLight`
- `mScreenStatus`

### 10.3 주의 센서별 전략

#### wHr

- 야간/새벽 시간대 결측률이 높다.
- 특히 00~07시, 21~23시 결측이 크다.
- `night_hr_mean` 단독 사용은 위험하다.

적용 feature:

- `wHr_00_06_mean`
- `wHr_00_06_std`
- `wHr_00_06_present_ratio`
- `wHr_00_06_missing_ratio`
- `wHr_21_24_mean`
- `wHr_21_24_present_ratio`
- `wHr_21_24_missing_ratio`
- `wHr_daytime_mean`
- `wHr_daytime_std`

#### mUsageStats

- 새벽 02~06시 결측률이 높다.
- 이는 센서 오류보다 “앱 사용 없음”이라는 수면 관련 신호일 가능성이 있다.
- 0-fill과 missing indicator를 함께 사용한다.

적용 feature:

- `usage_00_06_total`
- `usage_00_06_present_ratio`
- `usage_00_06_missing_ratio`
- `usage_18_24_total`
- `usage_21_24_total`
- `usage_21_24_count`
- `usage_late_night_flag`

#### wPedo / wLight

- 사용 가능하지만 새벽 시간대 결측이 일부 존재한다.
- 시간대별 missing ratio를 함께 추가한다.

적용 feature:

- `wPedo_day_sum`
- `wPedo_00_06_sum`
- `wPedo_00_06_missing_ratio`
- `wPedo_evening_sum`
- `wLight_00_06_mean`
- `wLight_00_06_missing_ratio`
- `wLight_21_24_mean`
- `wLight_night_bright_count`

### 10.4 모델별 결측 처리

| 모델                      | 결측 처리                                                               |
| ------------------------- | ----------------------------------------------------------------------- |
| LightGBM/XGBoost/CatBoost | NaN 유지 가능, count/duration은 0-fill 후보, present/missing ratio 추가 |
| BiLSTM                    | NaN 입력 불가, 0-fill + mask feature 조합 권장                          |

---

## 11. Sensor Statistics

### 11.1 Feature 생성 결과

| 항목                      |  값 | 처리                            |
| ------------------------- | --: | ------------------------------- |
| Sensors processed         |  12 | 정상                            |
| Skipped sensors           |   0 | 정상                            |
| Daily feature count       | 248 | 전체 후보                       |
| Actual behavior features  |  75 | 메인 feature                    |
| Coverage proxy features   |  61 | coverage/missing feature로 분리 |
| Invalid/constant features | 112 | 제거 대상                       |

### 판단

- 센서 처리 자체는 정상이다.
- invalid/constant feature가 많으므로 학습 전 제거해야 한다.
- `actual_behavior`와 `coverage_proxy`는 의미가 다르므로 feature group을 분리한다.
- raw correlation만 보고 feature를 선택하면 subject prior에 과적합될 수 있다.
- subject-adjusted correlation에서 유지되는 feature를 우선 사용한다.

### 11.2 Log Transform 후보

| feature                   |  skew | zero_ratio | q99/median | group         |
| ------------------------- | ----: | ---------: | ---------: | ------------- |
| mLight_m_light_sum        | 5.970 |      0.000 |      6.467 | light         |
| wPedo_burned_calories_std | 2.556 |      0.067 |     13.836 | activity_pedo |
| wPedo_burned_calories_max | 2.539 |      0.067 |     50.351 | activity_pedo |
| wLight_w_light_sum        | 2.395 |      0.011 |      6.281 | light         |

### 결정

- 위 feature는 `log1p` 변환 후보로 둔다.
- 원본 feature와 `log1p` feature를 모두 생성한 뒤 CV log-loss 기준으로 선택한다.
- ratio, z-score, heart rate mean 계열에는 기본적으로 log transform을 적용하지 않는다.

### 11.3 Target별 주요 signal

| Target | 주요 signal                                   | 해석                                                     |
| ------ | --------------------------------------------- | -------------------------------------------------------- |
| Q1     | light, wPedo q75, 04~08 light, 20~24 activity | 조도와 활동량이 반복적으로 등장                          |
| Q2     | charging, activity code, screen, wPedo        | 피로도는 충전/활동/화면/활동량 후보가 중요               |
| Q3     | mLight/wLight, wPedo q75                      | 스트레스는 조도 계열 signal이 강함                       |
| S1     | screen, wPedo, 00~04 activity                 | 수면 시간 적절성은 screen 상태와 야간 활동량 후보가 중요 |
| S2     | screen, wPedo, BLE/WiFi/GPS coverage proxy    | 수면 효율은 screen 및 coverage proxy까지 점검 필요       |
| S3     | light, charging, activity, wPedo coverage     | 수면 시작 지연은 전체 상관이 약해 개인화 필요            |
| S4     | activity code, pedo, screen, 04~08 charging   | activity/charging/screen 조합 후보                       |

### 추가 결정

- `S2`, `S3`는 일반 calendar day 기준만으로 부족할 수 있다.
- `16:00 ~ 다음날 16:00` analysis-day 기준 feature를 별도로 생성해 비교한다.
- 시간대 feature는 4시간 block과 수면 전후 도메인 block을 모두 실험한다.

---

## 12. Train/Test Distribution Shift

### 12.1 Subject 구성

| 구분        | train | test |
| ----------- | ----: | ---: |
| subject 수  |    10 |   10 |
| 전체 row 수 |   450 |  250 |

### 판단

- train/test 모두 `id01~id10`이 포함되어 있다.
- 특정 subject가 test에만 존재하거나 train에서 누락된 구조는 아니다.
- subject coverage 자체는 안정적이다.

### 12.2 Subject별 데이터 규모

| subject_id | train rows | test rows | test/train ratio |
| ---------- | ---------: | --------: | ---------------: |
| id01       |         41 |        27 |             0.66 |
| id02       |         48 |        32 |             0.67 |
| id03       |         33 |        21 |             0.64 |
| id04       |         57 |        27 |             0.47 |
| id05       |         44 |        21 |             0.48 |
| id06       |         48 |        24 |             0.50 |
| id07       |         49 |        30 |             0.61 |
| id08       |         56 |        19 |             0.34 |
| id09       |         41 |        27 |             0.66 |
| id10       |         33 |        22 |             0.67 |

### 판단

- `id08`은 train 56일 대비 test 19일로 test 표본이 가장 적다.
- `id04`, `id05`, `id08`은 train 대비 test 비율이 낮다.
- subject별 calibration을 독립적으로 적용하기에는 표본 수가 부족하다.

### 결정

- subject prior feature는 사용하되 smoothing을 적용한다.
- subject별 평균/표준편차는 train fold 기준으로만 계산한다.
- calibration은 subject별 단독 calibration보다 전체 calibration + subject feature 방식이 안전하다.

### 12.3 Date Range

- train/test 날짜는 subject별로 일부 overlap된다.
- test가 모든 subject에서 train 이후 기간으로만 구성된 것은 아니다.
- 단순 chronological holdout만으로 실제 test 구조를 완전히 대변하기 어렵다.

### 결정

- 기본 validation은 subject별 time-aware split을 유지한다.
- 추가로 Stratified/Group 기반 CV 결과를 함께 비교한다.
- 날짜 feature는 `month`, `dayofweek`, `is_weekend` 수준으로 제한한다.
- 절대 날짜를 강하게 쓰는 feature는 과적합 위험이 있으므로 주의한다.

### 12.4 Missing Shift

- day-level presence 기준 train/test missing 차이는 전반적으로 작다.
- test에서 missing이 크게 증가한 센서는 없다.
- `wLight`, `mUsageStats`, `mBle`, `mWifi`는 오히려 test missing이 train보다 낮다.

### 결정

- missing shift 위험은 현재 기준 낮다.
- 다만 hour-level missing은 별도로 feature화한다.
- `sensor_present`, `hourly_missing_ratio`, `night_missing_ratio`를 추가한다.

---

## 13. Baseline OOF Analysis

### 13.1 Target별 LogLoss

| Target |  LogLoss |   n |
| ------ | -------: | --: |
| Q2     | 0.686806 | 450 |
| Q1     | 0.668629 | 450 |
| Q3     | 0.667613 | 450 |
| S4     | 0.640768 | 450 |
| S1     | 0.583339 | 450 |
| S2     | 0.572020 | 450 |
| S3     | 0.548021 | 450 |

### 13.2 취약 구간

| 항목                        | 결과      |
| --------------------------- | --------- |
| 가장 약한 target            | Q2        |
| Q2 logloss                  | 0.686806  |
| 가장 약한 subject-target    | id06 / Q3 |
| id06-Q3 logloss             | 0.779359  |
| high confidence wrong cases | 16        |
| low confidence cases        | 1592      |

### 판단

- Q2, Q1, Q3가 우선 개선 target이다.
- `id06-Q3`는 subject-specific failure case로 별도 분석이 필요하다.
- high-confidence wrong case는 calibration 또는 blending으로 개선 가능성이 있다.
- low-confidence case가 많다는 것은 feature 부족 또는 모델 underfitting 가능성을 의미한다.
- naive subject prior reference와 비교했을 때, 현재 OOF 모델이 일부 target에서 subject prior 수준의 개인화 효과를 충분히 활용하지 못했을 가능성이 있다.

---

## 14. Final Modeling Decision

### 14.1 1차 모델 전략

| 항목                    | 결정                                  |
| ----------------------- | ------------------------------------- |
| 메인 모델               | LightGBM / XGBoost / CatBoost         |
| BiLSTM                  | 보조 실험으로 유지                    |
| Transformer             | 현재 데이터 규모상 후순위             |
| 주요 metric             | Average Log-Loss                      |
| 검증 방식               | subject-aware + time-aware split 우선 |
| 추가 검증               | Stratified/Group CV 비교              |
| calibration             | 필수 검토                             |
| ensemble                | target별 blending 검토                |
| target correlation 활용 | OOF prediction 기반 stacking 후보     |

### 14.2 Feature Engineering 우선순위

1. subject-aware feature
   - `subject_id`
   - subject별 sensor mean/std
   - subject별 z-score
   - 개인 평균 대비 deviation
   - smoothed subject target prior
   - rolling subject baseline

2. 시간대 feature
   - 00~04
   - 04~08
   - 08~12
   - 12~16
   - 16~20
   - 20~24
   - 16:00~다음날 16:00 analysis-day

3. 핵심 센서 feature
   - screen-on duration/ratio
   - charging duration/ratio
   - activity code count/ratio
   - light mean/median/max/q25/q75
   - wPedo step/distance/calories/speed
   - mUsageStats total/count/late-night usage
   - wHr mean/std/present_ratio/missing_ratio

4. missing/coverage feature
   - sensor_present
   - hourly_missing_ratio
   - night_missing_ratio
   - present_ratio
   - row_count
   - nonnull_count

5. 안정화 feature
   - log1p 변환 후보
   - clipping/winsorizing 후보
   - invalid/constant feature 제거
   - high-cardinality object feature는 후순위

6. target relation 기반 feature
   - target label 직접 사용 금지
   - OOF prediction stacking만 허용
   - `S2/S3/S4`, `Q2/Q3` group별 ensemble 검토

---

## 15. Target별 개선 전략

### Q2

현재 가장 취약한 target이다.

우선 적용:

- charging feature
- activity code feature
- screen feature
- wPedo activity feature
- subject-adjusted activity feature
- Q3와의 correlation을 고려한 multi-task/blending 후보
- calibration/blending

검증 포인트:

- fold별 Q2 ratio gap 확인
- subject별 Q2 prior 확인
- Q2 high-confidence wrong case 분석

### Q1

우선 적용:

- light median/q25/q75
- wPedo q75 계열
- 04~08 light
- 20~24 activity
- subject deviation feature
- S1과의 correlation을 고려한 group 분석

### Q3

우선 적용:

- mLight/wLight feature
- wPedo q75
- 12~16 light
- 00~04 light
- id06 전용 error case 분석
- Q2와의 correlation을 고려한 multi-task/blending 후보

### S1

우선 적용:

- screen status
- screen-off/screen-on ratio
- 00~04 wPedo activity
- 야간 활동량 feature
- S2와의 correlation을 고려한 sleep metric group 분석

### S2

우선 적용:

- 16:00~다음날 16:00 analysis-day feature
- screen 20~24
- wHr coverage
- wPedo 야간 feature
- missing/coverage proxy
- S3/S4와의 correlation을 고려한 sleep efficiency group 분석

### S3

우선 적용:

- 16:00~다음날 16:00 analysis-day feature
- 00~04 light
- 04~08 light
- charging/activity feature
- subject prior/deviation feature
- subject prior smoothing 강화

### S4

우선 적용:

- activity code 8
- pedo
- screen
- 04~08 charging
- 야간 light feature
- S2와의 correlation을 고려한 group-level blending

---

## 16. Experiment Backlog

### Priority 1. Tabular Feature Baseline 개선

- [ ] invalid/constant feature 제거
- [ ] 핵심 센서 기반 daily feature 생성
- [ ] 4시간 block feature 생성
- [ ] 16:00~다음날 16:00 analysis-day feature 생성
- [ ] subject z-score/deviation feature 생성
- [ ] smoothed subject prior feature 생성
- [ ] present/missing ratio feature 추가
- [ ] log1p 후보 feature 추가
- [ ] LightGBM/XGBoost/CatBoost target별 학습
- [ ] OOF log-loss 저장
- [ ] target별 feature importance/SHAP 확인

### Priority 2. Validation 안정화

- [ ] subject별 time-aware CV 유지
- [ ] Stratified/Group CV 추가 비교
- [ ] Q2 fold distribution 재점검
- [ ] subject별 OOF log-loss 리포트 생성
- [ ] id06-Q3 error case 분석
- [ ] subject prior feature가 fold 내부에서만 계산되는지 검증
- [ ] calibration fitting이 validation leakage 없이 수행되는지 검증

### Priority 3. Calibration & Blending

- [ ] target별 Platt Scaling
- [ ] target별 Isotonic Regression
- [ ] target별 Temperature Scaling 후보 검토
- [ ] high-confidence wrong case 감소 여부 확인
- [ ] LGBM/XGB/CatBoost probability blending
- [ ] target별 ensemble weight 탐색
- [ ] `S2/S3/S4`, `Q2/Q3` group별 probability consistency 점검

### Priority 4. Target Correlation 활용 실험

- [ ] OOF prediction 기반 target stacking 실험
- [ ] sleep metric group model과 subjective group model 분리 비교
- [ ] multi-task BiLSTM에서 correlated target group loss 분석
- [ ] target별 단독 모델 vs multi-output 모델 비교
- [ ] correlation이 높은 target 간 error overlap 분석

### Priority 5. BiLSTM 재실험

- [ ] `lookback=14/7/5/3` 비교
- [ ] sequence_window_quality.csv 생성
- [ ] 0-fill + mask feature 적용
- [ ] subject embedding 추가
- [ ] fold 수 유지/축소/확대 비교
- [ ] tabular model 대비 개선 여부 확인
- [ ] BiLSTM output을 tabular model과 blending

### Priority 6. Analysis-day 실험

- [ ] calendar day 기준 feature 생성
- [ ] 16:00~다음날 16:00 기준 feature 생성
- [ ] S2/S3 target별 성능 비교
- [ ] 수면 전 3시간/6시간 feature 추가
- [ ] 새벽 00~06 feature 추가

---

## 17. Final Conclusion

현재 EDA 결과 기준으로, 이 대회는 복잡한 딥러닝 구조를 바로 확장하는 것보다 subject-aware tabular feature engineering과 probability calibration을 우선하는 전략이 가장 합리적이다.

전체 class imbalance는 심하지 않다. 모든 target은 valid row 450개, class 2개, missing 0개이며 majority ratio도 0.504~0.682 범위에 머문다. 따라서 현재 단계에서 `pos_weight`, `focal loss`, aggressive resampling을 우선 적용할 근거는 약하다.

반면 subject별 label prior 차이는 매우 크다. `S3`, `S2`, `S4`, `Q1`에서 개인별 positive rate range가 특히 크며, subject prior만 사용하는 naive reference도 global prior 대비 모든 target에서 log-loss를 개선한다. 이는 모델이 센서 signal 이전에 개인별 기준선을 반드시 반영해야 함을 의미한다.

target 간 상관도 모델링 전략에 반영할 가치가 있다. `S2 ↔ S4`, `S2 ↔ S3`, `S1 ↔ S2`, `Q1 ↔ S1`, `Q2 ↔ Q3`는 상대적으로 높은 상관을 보인다. 다만 target label을 직접 feature로 사용하는 것은 leakage이므로 금지하며, OOF prediction 기반 stacking 또는 target group별 blending으로만 활용한다.

또한 `LOOKBACK=14` 기반 BiLSTM은 데이터 연속성 측면에서 위험하다. sequence 모델을 유지하려면 lookback을 5 또는 7로 줄이고, 실제 window 품질을 먼저 검증해야 한다.

1차 개선 방향은 다음과 같다.

1. LightGBM/XGBoost/CatBoost 기반 tabular baseline 강화
2. smoothed subject prior, subject z-score, personalized deviation feature 추가
3. 4시간 block 및 16:00~다음날 16:00 analysis-day feature 실험
4. missing/coverage feature를 명시적으로 추가
5. target별 calibration과 probability blending 수행
6. target correlation을 OOF stacking 또는 group-wise blending으로 제한적으로 활용
7. Q2, Q1, Q3 및 id06-Q3를 우선 개선 대상으로 관리

최종적으로 본 프로젝트의 핵심은 모델 구조 자체보다 다음 다섯 가지다.

- Feature Engineering
- Leakage Control
- Subject Personalization
- Target-wise Calibration
- Probability Blending
