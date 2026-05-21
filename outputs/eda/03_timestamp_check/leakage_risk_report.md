# 03 Timestamp Check & Leakage Risk Report

## 1. Summary

- train/test 모두 `sleep_date - lifelog_date = 1일`로 일관적이다.
- 12개 센서 파일 모두 timestamp가 정상적으로 파싱되었다.
- invalid timestamp row는 없다.
- 센서 timestamp가 전체 lifelog_date 범위를 벗어나는 boundary risk는 없다.
- subject별 fold는 모두 시간 순서를 보존한다.
- 명확한 timestamp 기반 leakage 위험은 낮다.
- 다만 일부 subject에서 날짜 gap이 존재하므로, `LOOKBACK=14` sequence가 실제 연속 14일을 보장하는지는 추가 확인이 필요하다.

## 2. Date Relationship

| split | diff_days | count | ratio |
| ----- | --------: | ----: | ----: |
| train |         1 |   450 |   1.0 |
| test  |         1 |   250 |   1.0 |

해석:

- `lifelog_date`의 당일 라이프로그 데이터로 다음 날 `sleep_date`의 target을 예측하는 구조로 보인다.
- feature 생성 시 `lifelog_date` 이후 또는 `sleep_date` 기준 미래 데이터가 포함되지 않도록 aggregation window를 명시해야 한다.

## 3. Sensor Timestamp Check

| 항목                      | 결과 |
| ------------------------- | ---: |
| 정상 파싱된 센서 파일 수  |   12 |
| invalid timestamp row     |    0 |
| boundary risk sensor file |    0 |

해석:

- 센서 timestamp 형식 문제는 발견되지 않았다.
- 전체 train/test lifelog date 범위를 벗어나는 센서 timestamp도 발견되지 않았다.

## 4. Fold Date Range Check

전체 subject 기준 fold date range는 서로 겹쳐 보이지만, 이는 subject별 수집 기간이 다르기 때문에 발생한 현상이다.

subject별 fold range를 확인한 결과:

- 모든 subject에서 fold 0 → fold 4 순서로 `lifelog_date`가 증가한다.
- 따라서 subject 내부 chronological order는 보존된다.
- random split으로 인한 명시적 temporal leakage 위험은 낮다.

단, 일부 subject에서 fold 사이 날짜 gap이 존재한다.

예시:

- id04: `2024-09-10 → 2024-09-16`
- id05: `2024-10-09 → 2024-10-18`
- id06: `2024-07-05 → 2024-07-14`
- id08: `2024-08-04 → 2024-08-11`

따라서 BiLSTM의 `LOOKBACK=14`가 실제 연속 14일 sequence로 구성되는지는 별도 검증이 필요하다.

## 5. Leakage Risk Checklist

- [x] `sleep_date`와 `lifelog_date` 차이가 일관적인가?
- [x] sensor timestamp가 정상적으로 파싱되는가?
- [x] 전체 lifelog date 범위를 벗어나는 sensor timestamp가 없는가?
- [x] subject별 fold가 시간 순서를 보존하는가?
- [ ] 수면 이후 timestamp가 feature에 포함되지 않는지 feature 생성 코드에서 검증
- [ ] S1/S2/S3 계산 원천이 되는 raw sleep sensor를 직접 feature로 쓰지 않는지 검증
- [ ] train+test 전체 통계로 normalization/scaling하지 않는지 검증
- [ ] `LOOKBACK=14` window의 실제 관측일 수 검증

## 6. Conclusion

03 timestamp check 결과, 데이터의 날짜 관계와 센서 timestamp는 정상이며 명확한 leakage 위험은 발견되지 않았다.

현재 fold split도 subject 내부 시간 순서를 보존하므로 EDA 기준으로는 통과로 판단한다.

다음 단계에서는 `04_sensor_schema`를 수행하여 센서별 column, dtype, timestamp column, value column, continuous/discrete/object 처리 방식을 확정한다.
