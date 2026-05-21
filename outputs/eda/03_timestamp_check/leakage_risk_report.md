# 03 Timestamp Check & Leakage Risk Report

## 1. sleep_date - lifelog_date

| split   | status   |   diff_days |   count |   ratio |
|:--------|:---------|------------:|--------:|--------:|
| train   | ok       |           1 |     450 |       1 |
| test    | ok       |           1 |     250 |       1 |


## 2. Sensor timestamp summary

| status   |   count |
|:---------|--------:|
| ok       |      12 |


## 3. Invalid timestamp rows

- invalid timestamp sample rows: 0

## 4. Sensor boundary risk

- boundary risk sensor files: 0


## 5. Fold date range

|   fold | scope        |   row_count |   subject_count | lifelog_date_min    | lifelog_date_max    |
|-------:|:-------------|------------:|----------------:|:--------------------|:--------------------|
|      0 | all_subjects |          95 |              10 | 2024-06-03 00:00:00 | 2024-09-06 00:00:00 |
|      1 | all_subjects |          92 |              10 | 2024-06-13 00:00:00 | 2024-09-18 00:00:00 |
|      2 | all_subjects |          91 |              10 | 2024-06-26 00:00:00 | 2024-10-09 00:00:00 |
|      3 | all_subjects |          87 |              10 | 2024-07-10 00:00:00 | 2024-10-29 00:00:00 |
|      4 | all_subjects |          85 |              10 | 2024-07-31 00:00:00 | 2024-11-14 00:00:00 |


## 6. Leakage risk checklist

- [ ] sleep_date와 lifelog_date 차이가 모델링 의도와 일치하는가?
- [ ] 수면 이후 timestamp를 feature로 사용하고 있지 않은가?
- [ ] S1/S2/S3 계산 원천이 되는 raw sleep sensor를 그대로 feature로 쓰지 않는가?
- [ ] train+test 전체 통계로 normalization/scaling하지 않는가?
- [ ] random split이 아니라 subject/time-aware split을 쓰는가?
- [ ] fold별 날짜 범위가 의도한 순서를 보존하는가?


## 7. Current risk notes

- sleep_date - lifelog_date 값은 1일로 관측됩니다.
- 전체 subject를 합친 fold 범위는 서로 겹칠 수 있습니다. 다만 subject별 chronological split이면 정상일 수 있으므로 subject별 fold 범위를 확인하세요.

## 8. Next action

- 04_sensor_schema에서 센서별 timestamp column과 처리 방식을 확정하세요.
- 05_missing_analysis에서 timestamp coverage와 결측 패턴을 subject/hour 단위로 확인하세요.
- feature 생성 시 현재 날짜 이후 데이터가 섞이지 않도록 aggregation window를 명시하세요.