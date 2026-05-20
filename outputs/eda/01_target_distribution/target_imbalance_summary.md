# Target Imbalance Summary

## 1. Overview

- Train rows: `450`
- Targets: `Q1, Q2, Q3, S1, S2, S3, S4`
- Target type: all binary classification targets (`0` or `1`)

## 2. Class Imbalance

| Target | Majority Class | Majority Ratio | Judgment |
| ------ | -------------: | -------------: | -------- |
| Q1     |              0 |          0.504 | OK       |
| Q2     |              1 |          0.562 | OK       |
| Q3     |              1 |          0.600 | OK       |
| S1     |              1 |          0.682 | OK       |
| S2     |              1 |          0.651 | OK       |
| S3     |              1 |          0.662 | OK       |
| S4     |              1 |          0.560 | OK       |

### Interpretation

80:20 이상으로 심하게 불균형한 target은 없다.

따라서 현재 단계에서는 `pos_weight`나 `focal loss`를 우선 적용하지 않는다.  
본 대회는 log-loss 기반이므로, 불필요한 loss weighting은 확률 calibration을 악화시킬 수 있다.

## 3. Subject-wise Prior Difference

| Target | Positive Rate Range | Min ~ Max     | Judgment |
| ------ | ------------------: | ------------- | -------- |
| Q1     |               0.703 | 0.146 ~ 0.848 | High     |
| Q2     |               0.422 | 0.396 ~ 0.818 | High     |
| Q3     |               0.333 | 0.439 ~ 0.772 | High     |
| S1     |               0.491 | 0.446 ~ 0.938 | High     |
| S2     |               0.667 | 0.250 ~ 0.917 | High     |
| S3     |               0.822 | 0.136 ~ 0.958 | High     |
| S4     |               0.703 | 0.152 ~ 0.854 | High     |

### Interpretation

전체 class imbalance는 크지 않지만, subject별 positive rate 차이는 매우 크다.

즉, 성능 병목은 class imbalance보다 개인별 기준선 차이일 가능성이 높다.  
이후 모델 개선에서는 subject-aware modeling을 우선 검토한다.

우선 적용 후보:

1. subject embedding
2. subject_id one-hot feature
3. subject별 target prior feature
4. subject별 sensor normalization
5. 개인 평균 대비 deviation feature

## 4. Fold Distribution

| Target | Max Fold Ratio Gap | Judgment |
| ------ | -----------------: | -------- |
| Q1     |              0.139 | OK       |
| Q2     |              0.211 | Check    |
| Q3     |              0.094 | OK       |
| S1     |              0.130 | OK       |
| S2     |              0.124 | OK       |
| S3     |              0.098 | OK       |
| S4     |              0.116 | OK       |

### Interpretation

대부분의 target은 fold별 label 분포가 크게 깨지지 않았다.

다만 Q2는 fold 간 class ratio gap이 `0.211`로 상대적으로 크다.  
바로 fold를 바꾸기보다는 Q2의 fold별 subject 구성과 날짜 범위를 추가 확인한다.

## 5. Modeling Decision

| Item                  | Decision             |
| --------------------- | -------------------- |
| pos_weight            | 보류                 |
| focal loss            | 보류                 |
| S1 multi-class head   | 불필요               |
| subject embedding     | 우선 검토            |
| subject prior feature | 우선 검토            |
| fold redesign         | Q2 추가 진단 후 판단 |

## 6. Next Action

다음 EDA는 `02_subject_distribution`을 진행한다.

확인할 항목:

1. subject별 row 수
2. subject별 date range
3. subject별 missing date 수
4. subject별 target prior 상세
5. Q1, S2, S3, S4에서 극단적인 subject 식별

## 7. Conclusion

전체 target imbalance는 심하지 않다.

하지만 모든 target에서 subject별 label prior 차이가 크다.  
따라서 다음 실험의 핵심은 loss weighting이 아니라 subject-aware modeling이다.
