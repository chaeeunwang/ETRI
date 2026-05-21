# Missing Analysis Summary

## 1. Overall Result

센서별 하루 단위 결측은 심각하지 않다.

- missing ratio가 0.5 이상인 센서는 없음
- train/test 간 결측률 차이가 큰 센서도 없음
- target class별로 결측이 크게 달라지는 패턴도 없음

따라서 현재 단계에서는 복잡한 결측치 보간보다, 센서별/시간대별 coverage feature를 추가하는 방향이 적절하다.

---

## 2. Sensor Missing Ratio by Train/Test

### Sensors with missing ratio >= 0.5

- None.

### Interpretation

하루 단위로 보면 대부분의 센서가 안정적으로 존재한다.

따라서 특정 센서를 결측률 때문에 즉시 제거할 필요는 없다.

---

## 3. Train/Test Missing Shift

- No sensor has train/test missing-ratio gap >= 0.2.

### Interpretation

train과 test의 결측 패턴 차이가 크지 않다.

따라서 결측 패턴으로 인한 public/private 일반화 위험은 현재 기준으로 크지 않다.

---

## 4. Missingness Related to Target

- No strong target-dependent missingness gap >= 0.15.

### Interpretation

특정 target class에서만 센서 결측이 강하게 몰리는 현상은 보이지 않는다.

따라서 결측 자체가 강한 target leakage 신호로 보이지는 않는다.

---

## 5. Hour-level Missing Pattern

하루 단위 결측은 안정적이지만, 시간대별로 보면 센서별 차이가 존재한다.

### 5.1 Stable Sensors

아래 센서들은 대부분의 시간대에서 결측률이 낮다.

- `ch2025_mACStatus`
- `ch2025_mActivity`
- `ch2025_mAmbience`
- `ch2025_mLight`
- `ch2025_mScreenStatus`

### Recommended Usage

이 센서들은 시간대별 feature로 바로 사용해도 된다.

예시:

- 야간 screen-on duration
- 수면 전 charging duration
- 시간대별 activity ratio
- 야간 mLight mean

---

### 5.2 wHr

`ch2025_wHr`는 야간과 새벽 시간대 결측률이 매우 높다.

특히 00~07시, 21~23시 결측이 크다.

### Interpretation

수면 관련 예측에 중요한 심박 센서이지만, 수면 시간대 결측이 많기 때문에 `night_hr_mean`만 단독으로 사용하면 위험하다.

### Recommended Features

- `wHr_00_06_mean`
- `wHr_00_06_std`
- `wHr_00_06_present_ratio`
- `wHr_00_06_missing_ratio`
- `wHr_21_24_mean`
- `wHr_21_24_present_ratio`
- `wHr_21_24_missing_ratio`
- `wHr_daytime_mean`
- `wHr_daytime_std`

### Decision

`wHr`는 제거하지 않는다.

다만 반드시 coverage feature와 함께 사용한다.

---

### 5.3 mUsageStats

`ch2025_mUsageStats`는 새벽 02~06시 결측률이 높다.

### Interpretation

이 결측은 센서 오류라기보다 사용자가 자고 있어서 앱 사용 기록이 없는 패턴일 가능성이 있다.

따라서 무조건 결측으로만 처리하지 말고, “새벽 앱 사용 없음” 자체를 수면 관련 신호로 활용한다.

### Recommended Features

- `usage_00_06_total`
- `usage_00_06_present_ratio`
- `usage_00_06_missing_ratio`
- `usage_18_24_total`
- `usage_21_24_total`
- `usage_21_24_count`
- `usage_late_night_flag`

### Decision

`mUsageStats`는 핵심 센서로 유지한다.

0-fill과 missing indicator를 함께 사용한다.

---

### 5.4 mBle

`ch2025_mBle`는 전반적으로 결측률이 높고, 특히 새벽 시간대 결측이 매우 크다.

### Interpretation

현재 단계에서 정교하게 파싱할 우선순위는 낮다.

### Recommended Features

1차 실험에서는 아래 정도만 사용한다.

- `mBle_present_ratio`
- `mBle_missing_ratio`
- `mBle_count`

### Decision

`mBle`는 후순위 센서로 둔다.

1차 feature engineering에서는 제외하거나 단순 count/presence만 사용한다.

---

### 5.5 wPedo / wLight

`ch2025_wPedo`, `ch2025_wLight`는 사용할 수 있지만 새벽 시간대 결측이 일부 존재한다.

### Recommended Features

- `wPedo_day_sum`
- `wPedo_00_06_sum`
- `wPedo_00_06_missing_ratio`
- `wPedo_evening_sum`
- `wLight_00_06_mean`
- `wLight_00_06_missing_ratio`
- `wLight_21_24_mean`
- `wLight_night_bright_count`

### Decision

`wPedo`, `wLight`는 핵심 센서로 유지한다.

다만 시간대별 missing ratio를 함께 추가한다.

---

## 6. Feature Engineering Decision

### 1차 핵심 센서

아래 센서를 우선 사용한다.

- `mScreenStatus`
- `mACStatus`
- `mActivity`
- `mLight`
- `wPedo`
- `wLight`
- `mUsageStats`

### 주의해서 사용할 센서

- `wHr`

`wHr`는 수면 예측에 중요하지만 야간 결측이 크므로 coverage feature와 함께 사용한다.

### 후순위 센서

- `mBle`
- `mWifi`
- `mGps`
- `mAmbience`

후순위 센서는 1차 baseline 이후 성능 개선 실험에서 추가 여부를 판단한다.

---

## 7. Missing Value Strategy

### LightGBM / XGBoost / CatBoost

- 결측값은 모델이 처리하도록 NaN 유지 가능
- 단, 센서별 present/missing ratio feature는 추가
- count, duration 계열은 0-fill 후보
- long-tail feature는 06_sensor_statistics에서 log1p 여부 판단

### BiLSTM

- 단순 NaN 입력은 불가
- 0-fill + mask feature 조합 권장
- 시간대별 missing ratio를 별도 input feature로 추가

---

## 8. Recommended Actions

다음 실험에서 적용할 액션은 아래와 같다.

1. 센서별 `present_ratio`, `missing_ratio` 추가
2. `wHr`는 야간 평균값만 쓰지 말고 coverage feature와 함께 사용
3. `mUsageStats`는 0-fill + missing indicator 적용
4. `mBle`는 1차 실험에서 제외하거나 count/presence만 사용
5. 복잡한 interpolation은 아직 적용하지 않음
6. 06_sensor_statistics에서 long-tail, outlier, log1p 후보 확인

---

## 9. Final Conclusion

05_missing_analysis 결과, 전체 센서 결측은 심각하지 않다.

하지만 시간대별로는 센서별 결측 패턴이 뚜렷하다.

따라서 현재 최적 전략은 복잡한 보간이 아니라 다음과 같다.

- sensor-day 결측은 단순 처리
- hour/timeblock coverage feature 추가
- `wHr`, `mUsageStats`는 시간대별 결측 패턴을 feature로 활용
- `mBle`는 후순위 처리
- 다음 단계인 06_sensor_statistics에서 값의 분포와 log transform 후보를 확인
