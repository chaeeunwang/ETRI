# Subject Distribution EDA Summary

## 1. 생성 목적

본 분석의 목적은 subject별 데이터 규모, 날짜 범위, 날짜 연속성, target prior를 확인하여 현재 sequence 기반 BiLSTM 구조가 적절한지 판단하는 것이다.

특히 현재 모델이 `lookback=14`를 사용하고 있으므로, 각 subject가 최소 14일 이상의 연속 관측 구간을 충분히 보유하고 있는지 확인하였다. 또한 subject별 target prior 차이를 분석하여 subject embedding, personalized feature, subject-aware validation의 필요성을 점검하였다.

---

## 2. Subject별 row 수

### 2.1 전체 요약

- train row 최소값: 33
- train row 최대값: 57
- train row 표준편차: 8.300

subject별 train row 수에는 일정 수준의 차이가 존재한다. 특히 일부 subject는 학습 데이터가 30일대에 불과하므로, 전체 모델이 데이터가 많은 subject 중심으로 학습될 가능성이 있다.

### 2.2 train row가 적은 subject TOP3

| subject_id | train_rows | test_rows |
| ---------- | ---------: | --------: |
| id03       |         33 |        21 |
| id10       |         33 |        22 |
| id09       |         41 |        27 |

### 2.3 판단

id03, id10은 train row가 33개로 가장 적다. 이들은 sequence 모델에서 학습 가능한 window 수 자체가 적을 가능성이 높다. 따라서 단순 K-Fold 또는 무작위 sampling만으로는 subject별 학습 안정성을 보장하기 어렵다.

후속 실험에서는 subject별 성능을 반드시 분리해서 확인해야 하며, 특정 subject에서만 성능이 급락하는지 확인할 필요가 있다.

---

## 3. 날짜 연속성

### 3.1 전체 요약

- train에서 날짜 gap이 있는 subject 수: 10
- train에서 `max_consecutive_days < 14`인 subject 수: 5

모든 subject에서 날짜 gap이 존재한다. 즉, lifelog_date가 완전히 연속적인 시계열로 제공되지 않는다.

### 3.2 lookback=14 위험 subject

| subject_id | max_consecutive_days | missing_days_count | max_gap_days |
| ---------- | -------------------: | -----------------: | -----------: |
| id03       |                    6 |                 25 |           14 |
| id05       |                   10 |                 35 |           12 |
| id06       |                   13 |                 29 |            9 |
| id09       |                   11 |                 24 |           17 |
| id10       |                    5 |                 38 |           25 |

### 3.3 판단

현재 `lookback=14`는 그대로 사용하기 어렵다.

전체 subject 10명 중 5명이 14일 연속 구간을 확보하지 못한다. 특히 id03과 id10은 최대 연속 구간이 각각 6일, 5일에 불과하다. 이 상태에서 14일 window를 강제로 구성하면 모델은 실제 14일 생활 패턴을 학습하는 것이 아니라, 불연속 날짜, padding, 결측 패턴을 함께 학습할 가능성이 높다.

따라서 현재 BiLSTM의 `lookback=14`는 과한 설정일 가능성이 크다.

### 3.4 후속 조치

lookback 길이를 고정하지 말고 아래 후보를 비교해야 한다.

| 실험 | lookback | 목적                              |
| ---- | -------: | --------------------------------- |
| A    |       14 | 기존 baseline 유지                |
| B    |        7 | 1주일 단위 생활 패턴 확인         |
| C    |        5 | id03, id10까지 고려한 현실적 후보 |
| D    |        3 | 짧은 시계열 기준 안전한 비교군    |

현재 EDA 결과 기준으로는 `lookback=5` 또는 `lookback=7`이 가장 현실적인 후보이다.

---

## 4. Sequence window 품질 검증 필요성

현재 분석은 subject별 최대 연속 일수와 missing date를 확인한 단계이다. 그러나 실제 BiLSTM 입력 window가 얼마나 유효한지는 아직 별도 검증이 필요하다.

따라서 다음 파일을 추가 생성해야 한다.

| 생성 파일                            | 목적                             |
| ------------------------------------ | -------------------------------- |
| `sequence_window_quality.csv`        | lookback별 window 품질 수치 저장 |
| `sequence_window_quality_summary.md` | lookback별 판단 요약             |

### 4.1 확인할 지표

| 지표                             | 설명                                           |
| -------------------------------- | ---------------------------------------------- |
| lookback별 생성 가능한 window 수 | subject별 학습 가능한 sequence 개수            |
| valid day ratio                  | window 안에서 실제 관측된 날짜 비율            |
| missing day count                | window 안에서 비어 있는 날짜 수                |
| subject별 usable window 수       | 특정 subject가 학습에서 사실상 배제되는지 확인 |
| lookback=14/7/5/3 비교           | 적정 lookback 길이 결정                        |

### 4.2 판단 기준

| 관찰                                   | 후속 조치                                  |
| -------------------------------------- | ------------------------------------------ |
| lookback=14에서 valid_ratio가 낮음     | lookback=14 폐기                           |
| lookback=7에서 대부분 subject가 안정적 | lookback=7 후보                            |
| lookback=5에서 id03/id10까지 커버      | lookback=5 강력 후보                       |
| lookback=3만 안정적                    | LSTM보다 tabular/rolling feature 우선 검토 |

---

## 5. Target prior 차이

### 5.1 subject별 class ratio 변동성이 큰 target/class TOP10

| target | class |  mean |   std |   min |   max |
| ------ | ----: | ----: | ----: | ----: | ----: |
| S3     |     0 | 0.348 | 0.247 | 0.042 | 0.864 |
| S3     |     1 | 0.652 | 0.247 | 0.136 | 0.958 |
| S2     |     0 | 0.365 | 0.219 | 0.083 | 0.750 |
| S2     |     1 | 0.635 | 0.219 | 0.250 | 0.917 |
| S4     |     0 | 0.451 | 0.195 | 0.146 | 0.848 |
| S4     |     1 | 0.549 | 0.195 | 0.152 | 0.854 |
| S1     |     1 | 0.678 | 0.180 | 0.446 | 0.938 |
| S1     |     0 | 0.322 | 0.180 | 0.062 | 0.554 |
| Q1     |     1 | 0.507 | 0.175 | 0.146 | 0.848 |
| Q1     |     0 | 0.493 | 0.175 | 0.152 | 0.854 |

### 5.2 판단

subject별 target prior 차이가 매우 크다.

특히 S3, S2의 class ratio 차이가 가장 크다. S3 class 0 ratio는 subject에 따라 0.042에서 0.864까지 차이가 난다. S2 class 0 ratio도 0.083에서 0.750까지 차이가 난다.

이는 같은 센서 패턴이라도 subject마다 label 기준이나 생활 패턴이 다를 가능성이 높다는 의미이다. 따라서 subject 정보를 모델에 넣지 않으면 전체 평균 패턴에 치우친 예측이 나올 수 있다.

### 5.3 후속 조치

| 방법                     | 설명                                        |
| ------------------------ | ------------------------------------------- |
| subject_id embedding     | BiLSTM 내부에서 개인별 패턴 학습            |
| subject prior feature    | subject별 target 평균/분포를 feature로 사용 |
| subject feature baseline | subject별 센서 평균, 표준편차               |
| personalized deviation   | 현재 값 - 개인 평균                         |
| subject z-score          | 개인 평균 대비 표준화 값                    |
| subject-aware validation | subject별 날짜 순서를 보존한 검증           |

단, subject target prior를 사용할 경우 validation leakage에 주의해야 한다. validation/test 예측 시에는 해당 fold의 train 구간에서 계산한 prior만 사용해야 한다.

---

## 6. Target별 우선 관리 대상

### 6.1 S3

S3는 subject별 prior 차이가 가장 크다.

- class 0 ratio 평균: 0.348
- class 0 ratio 표준편차: 0.247
- subject별 범위: 0.042 ~ 0.864

S3는 수면 시작 지연과 관련된 target이므로, 전체 공통 패턴보다 개인별 수면 습관과 취침 전 행동 차이가 크게 작용할 가능성이 있다.

### 6.2 S2

S2도 subject별 prior 차이가 크다.

- class 0 ratio 평균: 0.365
- class 0 ratio 표준편차: 0.219
- subject별 범위: 0.083 ~ 0.750

S2는 수면 효율과 관련된 target이므로, 야간 심박, 야간 조도, charging, screen-on, 수면 중 움직임 등 야간 특화 feature가 필요하다.

### 6.3 Q1

Q1 역시 subject별 class ratio 차이가 크다.

- class 1 ratio 평균: 0.507
- class 1 ratio 표준편차: 0.175
- subject별 범위: 0.146 ~ 0.848

Q1은 개인이 느끼는 주관적 수면 만족도이므로 subject별 기준선 차이가 클 가능성이 높다.

---

## 7. 현재 BiLSTM 구조에 대한 판단

### 7.1 유지하기 어려운 부분

| 항목                    | 판단           |
| ----------------------- | -------------- |
| lookback=14 고정        | 위험           |
| mask 없는 sequence 구성 | 비추천         |
| subject 정보 없는 모델  | 비추천         |
| 모든 target 동일 처리   | S2/S3에서 위험 |
| 단순 random split       | 비추천         |

### 7.2 개선 방향

현재 결과 기준으로 BiLSTM은 구조 자체보다 입력 window 구성부터 재검토해야 한다.

1. lookback별 window 품질 분석
2. lookback=14/7/5/3 비교 실험
3. valid mask 추가
4. day gap feature 추가
5. subject embedding 또는 subject prior feature 추가
6. target별 logloss 및 subject별 logloss 분석
7. S2/S3 전용 feature 또는 calibration 검토

---

## 8. 추가해야 할 입력 feature 후보

### 8.1 Sequence validity feature

| feature                     | 설명                                  |
| --------------------------- | ------------------------------------- |
| is_observed                 | 해당 날짜가 실제 관측된 날짜인지 여부 |
| days_from_prev_observed     | 이전 관측일과의 날짜 차이             |
| valid_day_count_in_window   | window 내 실제 관측일 수              |
| missing_day_count_in_window | window 내 결측 날짜 수                |
| valid_ratio_in_window       | window 내 실제 관측일 비율            |

### 8.2 Subject-aware feature

| feature                | 설명                       |
| ---------------------- | -------------------------- |
| subject_id             | categorical 또는 embedding |
| subject_Q1_prior       | subject별 Q1 평균          |
| subject_Q2_prior       | subject별 Q2 평균          |
| subject_Q3_prior       | subject별 Q3 평균          |
| subject_S1_prior       | subject별 S1 평균          |
| subject_S2_prior       | subject별 S2 평균          |
| subject_S3_prior       | subject별 S3 평균          |
| subject_feature_mean   | subject별 센서 평균        |
| subject_feature_zscore | 개인 기준 표준화 feature   |

---

## 9. 다음 실험 액션

### 9.1 EDA 추가

다음 산출물을 추가 생성한다.

| 생성 파일                            | 목적                                           |
| ------------------------------------ | ---------------------------------------------- |
| `sequence_window_quality.csv`        | lookback별 usable window 수와 valid ratio 저장 |
| `sequence_window_quality_summary.md` | lookback 후보 판단 요약                        |

확인할 내용은 다음과 같다.

- lookback별 usable window 수
- subject별 usable window 수
- window별 valid_ratio
- lookback=14/7/5/3 비교
- id03/id10에서 window가 실제로 얼마나 생성되는지

### 9.2 모델 실험

| 우선순위 | 실험                                     |
| -------: | ---------------------------------------- |
|        1 | BiLSTM lookback=14 baseline 재확인       |
|        2 | BiLSTM lookback=7                        |
|        3 | BiLSTM lookback=5                        |
|        4 | BiLSTM lookback=3                        |
|        5 | lookback + valid_mask                    |
|        6 | lookback + valid_mask + day_gap          |
|        7 | subject embedding 추가                   |
|        8 | target별 pos_weight 또는 focal loss 적용 |
|        9 | S2/S3 전용 feature 또는 별도 head 검토   |

---

## 10. 최종 결론

이번 subject distribution EDA 결과, 현재 데이터는 sequence 모델을 그대로 적용하기에 날짜 연속성이 충분하지 않다.

핵심 결론은 다음과 같다.

1. 모든 subject에서 날짜 gap이 존재한다.
2. 전체 subject의 절반이 14일 연속 관측 구간을 확보하지 못한다.
3. id03, id10은 최대 연속 구간이 5~6일에 불과하다.
4. 따라서 `lookback=14`는 현재 데이터 구조에 비해 과한 설정일 가능성이 크다.
5. subject별 target prior 차이가 매우 크므로 subject-aware modeling이 필요하다.
6. 특히 S3, S2는 subject별 label prior 차이가 커서 별도 관리가 필요하다.
7. 다음 단계에서는 sequence window quality를 계산한 뒤 lookback=14/7/5/3을 비교해야 한다.

현재 기준으로는 `lookback=5` 또는 `lookback=7`이 가장 현실적인 후보이며, mask 없이 `lookback=14`를 유지하는 것은 비추천한다.
