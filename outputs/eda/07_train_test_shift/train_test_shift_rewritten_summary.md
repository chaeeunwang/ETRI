# 07. Train/Test Distribution Shift 분석

## 1. 목적

train/test 간 분포 차이를 확인하여 public/private 성능 변동 위험을 점검한다.

본 EDA에서는 다음 항목을 확인했다.

- subject별 train/test row 수 차이
- subject별 train/test 날짜 범위 차이
- 센서 단위 day-level missing ratio 차이
- numeric feature distribution shift
- adversarial validation AUC

---

## 2. 핵심 결론

### 2.1 Train/Test subject 구성

train/test 모두 `id01`~`id10` 전체 subject가 포함되어 있다.

| 구분        | train | test |
| ----------- | ----: | ---: |
| subject 수  |    10 |   10 |
| 전체 row 수 |   450 |  250 |

subject별 row 수는 train 33~57일, test 19~32일 범위다. 특정 subject가 완전히 누락된 구조는 아니므로 subject coverage 자체는 안정적이다.

다만 subject별 test row 수가 균등하지 않다. 특히 `id08`, `id03`, `id05`, `id10`은 test row 수가 상대적으로 적어 subject별 예측 확률 안정성이 낮을 수 있다.

---

## 3. Subject별 데이터 규모

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

### 해석

- `id04`, `id05`, `id08`은 train 대비 test 비율이 낮다.
- `id08`은 train 56일 대비 test 19일로 test 표본이 가장 적다.
- subject별 prior, subject별 calibration, subject-aware feature를 적용할 때 일부 subject는 표본 수 부족으로 과적합될 수 있다.

### 후속 조치

- subject별 target prior feature는 사용하되, smoothing을 적용한다.
- subject별 평균/표준편차 기반 feature는 train 기준으로만 계산한다.
- subject별 calibration은 단독 적용보다 전체 calibration + subject feature 방식이 안전하다.

---

## 4. Date Range 분석

train/test 날짜는 subject별로 일부 겹친다. 즉, test가 모든 subject에서 train 이후 기간으로만 구성된 것은 아니다.

| subject_id | train range             | test range              | 해석           |
| ---------- | ----------------------- | ----------------------- | -------------- |
| id01       | 2024-06-26 ~ 2024-08-31 | 2024-07-30 ~ 2024-09-14 | 일부 overlap   |
| id02       | 2024-07-17 ~ 2024-09-27 | 2024-08-25 ~ 2024-10-15 | 일부 overlap   |
| id03       | 2024-07-17 ~ 2024-09-12 | 2024-08-16 ~ 2024-10-09 | 일부 overlap   |
| id04       | 2024-07-31 ~ 2024-10-26 | 2024-09-09 ~ 2024-10-29 | 대부분 overlap |
| id05       | 2024-08-28 ~ 2024-11-14 | 2024-09-28 ~ 2024-11-19 | 대부분 overlap |
| id06       | 2024-06-03 ~ 2024-08-18 | 2024-07-06 ~ 2024-08-24 | 대부분 overlap |
| id07       | 2024-06-09 ~ 2024-08-13 | 2024-07-13 ~ 2024-09-01 | 일부 overlap   |
| id08       | 2024-06-25 ~ 2024-09-16 | 2024-07-31 ~ 2024-09-19 | 대부분 overlap |
| id09       | 2024-07-01 ~ 2024-09-03 | 2024-08-05 ~ 2024-09-21 | 일부 overlap   |
| id10       | 2024-07-06 ~ 2024-09-14 | 2024-08-04 ~ 2024-09-26 | 일부 overlap   |

### 해석

- test가 완전히 미래 구간만은 아니므로 단순 chronological holdout과 실제 test 구조가 다를 수 있다.
- train/test 날짜 overlap이 있으므로 계절성·요일성·월별 패턴은 어느 정도 공유될 가능성이 있다.
- 반대로, 날짜가 겹친다고 해서 같은 날짜의 target 또는 센서 정보를 누설하면 안 된다.

### 후속 조치

- validation은 subject별 time-aware split을 유지한다.
- 단, 실제 test 구조가 완전 미래 예측은 아니므로 `time-aware CV`와 `Stratified/Group CV` 결과를 함께 비교할 필요가 있다.
- 날짜 기반 feature는 `month`, `dayofweek`, `is_weekend` 정도만 사용하고, 절대 날짜를 강하게 쓰는 feature는 주의한다.

---

## 5. Sensor Missing Shift

센서 day-level presence 기준 train/test missing 차이는 전반적으로 작다.

| sensor        | train missing | test missing |    diff |
| ------------- | ------------: | -----------: | ------: |
| mGps          |        0.0556 |       0.0600 | +0.0044 |
| mACStatus     |        0.0000 |       0.0000 |  0.0000 |
| mAmbience     |        0.0000 |       0.0000 |  0.0000 |
| mActivity     |        0.0000 |       0.0000 |  0.0000 |
| mScreenStatus |        0.0000 |       0.0000 |  0.0000 |
| mLight        |        0.0000 |       0.0000 |  0.0000 |
| mWifi         |        0.0244 |       0.0160 | -0.0084 |
| mBle          |        0.0756 |       0.0600 | -0.0156 |
| mUsageStats   |        0.0222 |       0.0000 | -0.0222 |
| wLight        |        0.0800 |       0.0000 | -0.0800 |

### 해석

- test에서 missing이 크게 증가한 센서는 없다.
- `mGps`만 test missing이 소폭 증가했지만 차이는 매우 작다.
- `wLight`, `mUsageStats`, `mBle`, `mWifi`는 오히려 test missing이 train보다 낮다.
- day-level 존재 여부 기준으로는 train/test 센서 결측 shift가 크지 않다.

### 주의

현재 missing 분석은 “해당 날짜에 센서 row가 하나라도 존재하는가” 기준이다.  
따라서 하루 내 시간대별 결측, 야간 결측, 착용 중단, 특정 시간대 데이터 공백은 반영되지 않는다.

### 후속 조치

- 07 결과만 보면 missing shift 위험은 낮다.
- 하지만 모델 feature에는 `sensor_present`, `hourly_missing_ratio`, `night_missing_ratio`를 추가하는 것이 안전하다.
- 특히 `wHr`, `wLight`, `mUsageStats`, `mScreenStatus`, `mACStatus`는 야간/수면 전 결측률을 따로 봐야 한다.

---

## 6. Numeric Feature Shift

현재 `Top PSI Features`가 비어 있다.

이는 train/test 공통 numeric feature가 없거나, 현재 코드가 원본 metrics CSV의 numeric feature만 대상으로 PSI를 계산했기 때문으로 보인다. 센서 aggregation feature가 아직 생성되지 않은 상태라면 정상적인 결과다.

### 해석

- 현재 07 EDA는 numeric feature distribution shift를 충분히 평가하지 못했다.
- 실질적인 PSI 분석은 feature engineering 이후 다시 수행해야 한다.
- 특히 시간대별 통계 feature, subject-normalized feature, rolling feature 생성 후 train/test shift를 재계산해야 한다.

### 후속 조치

Feature engineering 이후 아래 feature를 대상으로 PSI를 다시 계산한다.

- `wPedo_*`
- `wHr_*`
- `mUsageStats_*`
- `mScreenStatus_*`
- `mACStatus_*`
- `mLight_*`
- `wLight_*`
- subject z-score feature
- rolling mean/deviation feature
- timeblock feature

---

## 7. Adversarial Validation

| metric      |                                      value |
| ----------- | -----------------------------------------: |
| AUC mean    |                                     0.5678 |
| AUC std     |                                     0.0355 |
| fold scores | 0.5317 / 0.5731 / 0.5393 / 0.6320 / 0.5627 |

### 해석

Adversarial validation AUC가 0.568로 낮다.

일반적으로 AUC가 0.5에 가까우면 train/test 구분이 어렵고, 0.7 이상이면 train/test shift가 크다고 본다. 현재 결과는 0.5에 가깝기 때문에 train/test 분포 차이는 크지 않은 편이다.

다만 이 결과는 대부분 센서 presence feature 기반으로 계산된 것으로 보인다. 따라서 최종 feature set 기준의 train/test shift를 의미한다고 보기는 어렵다.

### 후속 조치

- 현재 기준으로는 train/test shift 위험이 크지 않다.
- feature engineering 후 adversarial validation을 재실행한다.
- 재실행 후 AUC가 0.70 이상이면 PSI 높은 feature 제거 또는 robust scaling을 검토한다.

---

## 8. 종합 판단

| 항목                           | 판단                    |
| ------------------------------ | ----------------------- |
| subject coverage               | 양호                    |
| subject별 row 수 균형          | 보통                    |
| date range                     | train/test overlap 존재 |
| sensor day-level missing shift | 낮음                    |
| numeric PSI 분석               | 미완료                  |
| adversarial validation AUC     | 낮음, shift 크지 않음   |
| 즉시 위험도                    | 낮음~중간               |
| 재분석 필요성                  | 높음                    |

현재 07 EDA 기준으로는 train/test distribution shift가 심하다는 근거는 없다.  
다만 numeric feature shift 분석이 비어 있으므로, 이 결론은 “원본 key 및 sensor presence 기준”에 한정된다.

최종 모델링 전에는 feature engineering 결과물을 기준으로 07 EDA를 다시 수행해야 한다.

---

## 9. 모델링 반영 사항

### 적용 권장

- `subject_id` categorical feature 유지
- sensor presence feature 추가
- subject-aware normalization 적용
- train 기준 statistics만 사용
- LightGBM/XGBoost/CatBoost에서 missing value를 그대로 활용
- calibration은 전체 OOF 기준으로 우선 적용

### 주의 필요

- train+test 전체 평균으로 scaling 금지
- test 날짜 정보를 활용한 rolling/statistics 계산 금지
- subject별 test row가 적은 경우 subject 단독 calibration 지양
- PSI 재분석 전 feature 제거 판단 금지

---

## 10. 다음 액션

1. 06에서 생성한 sensor statistics feature를 하나의 daily feature table로 통합한다.
2. 통합 feature 기준으로 07 train/test shift를 재실행한다.
3. PSI 상위 feature와 adversarial validation feature importance를 확인한다.
4. shift가 큰 feature는 제거, clipping, rank transform, subject z-score 중 하나로 처리한다.
5. 최종 feature set 확정 후 baseline 모델의 OOF log-loss와 함께 해석한다.
