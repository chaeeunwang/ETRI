# 07 Train/Test Distribution Shift Summary

## 핵심 산출물
- train/test subject 구성 차이
- train/test date range 차이
- sensor day-level missing ratio 차이
- numeric feature distribution shift
- adversarial validation AUC

## Subject Count
| split   | subject_id   |   row_count |
|:--------|:-------------|------------:|
| test    | id01         |          27 |
| test    | id02         |          32 |
| test    | id03         |          21 |
| test    | id04         |          27 |
| test    | id05         |          21 |
| test    | id06         |          24 |
| test    | id07         |          30 |
| test    | id08         |          19 |
| test    | id09         |          27 |
| test    | id10         |          22 |
| train   | id01         |          41 |
| train   | id02         |          48 |
| train   | id03         |          33 |
| train   | id04         |          57 |
| train   | id05         |          44 |
| train   | id06         |          48 |
| train   | id07         |          49 |
| train   | id08         |          56 |
| train   | id09         |          41 |
| train   | id10         |          33 |

## Date Range
| split   | subject_id   | min_date            | max_date            |   row_count |
|:--------|:-------------|:--------------------|:--------------------|------------:|
| test    | id01         | 2024-07-30 00:00:00 | 2024-09-14 00:00:00 |          27 |
| test    | id02         | 2024-08-25 00:00:00 | 2024-10-15 00:00:00 |          32 |
| test    | id03         | 2024-08-16 00:00:00 | 2024-10-09 00:00:00 |          21 |
| test    | id04         | 2024-09-09 00:00:00 | 2024-10-29 00:00:00 |          27 |
| test    | id05         | 2024-09-28 00:00:00 | 2024-11-19 00:00:00 |          21 |
| test    | id06         | 2024-07-06 00:00:00 | 2024-08-24 00:00:00 |          24 |
| test    | id07         | 2024-07-13 00:00:00 | 2024-09-01 00:00:00 |          30 |
| test    | id08         | 2024-07-31 00:00:00 | 2024-09-19 00:00:00 |          19 |
| test    | id09         | 2024-08-05 00:00:00 | 2024-09-21 00:00:00 |          27 |
| test    | id10         | 2024-08-04 00:00:00 | 2024-09-26 00:00:00 |          22 |
| train   | id01         | 2024-06-26 00:00:00 | 2024-08-31 00:00:00 |          41 |
| train   | id02         | 2024-07-17 00:00:00 | 2024-09-27 00:00:00 |          48 |
| train   | id03         | 2024-07-17 00:00:00 | 2024-09-12 00:00:00 |          33 |
| train   | id04         | 2024-07-31 00:00:00 | 2024-10-26 00:00:00 |          57 |
| train   | id05         | 2024-08-28 00:00:00 | 2024-11-14 00:00:00 |          44 |
| train   | id06         | 2024-06-03 00:00:00 | 2024-08-18 00:00:00 |          48 |
| train   | id07         | 2024-06-09 00:00:00 | 2024-08-13 00:00:00 |          49 |
| train   | id08         | 2024-06-25 00:00:00 | 2024-09-16 00:00:00 |          56 |
| train   | id09         | 2024-07-01 00:00:00 | 2024-09-03 00:00:00 |          41 |
| train   | id10         | 2024-07-06 00:00:00 | 2024-09-14 00:00:00 |          33 |

## Top Missing Shift
| sensor        |   test_missing_ratio |   train_missing_ratio |   missing_diff |
|:--------------|---------------------:|----------------------:|---------------:|
| mGps          |                0.06  |             0.0555556 |     0.00444444 |
| mACStatus     |                0     |             0         |     0          |
| mAmbience     |                0     |             0         |     0          |
| mActivity     |                0     |             0         |     0          |
| mScreenStatus |                0     |             0         |     0          |
| mLight        |                0     |             0         |     0          |
| mWifi         |                0.016 |             0.0244444 |    -0.00844444 |
| mBle          |                0.06  |             0.0755556 |    -0.0155556  |
| mUsageStats   |                0     |             0.0222222 |    -0.0222222  |
| wLight        |                0     |             0.08      |    -0.08       |

## Top PSI Features
| feature   | train_mean   | test_mean   | mean_diff   | train_std   | test_std   | std_diff   | train_missing_ratio   | test_missing_ratio   | missing_diff   | psi   |
|-----------|--------------|-------------|-------------|-------------|------------|------------|-----------------------|----------------------|----------------|-------|

## Adversarial Validation
| status   |   auc_mean |   auc_std | fold_scores                             |
|:---------|-----------:|----------:|:----------------------------------------|
| ok       |   0.567756 | 0.0354744 | 0.53167|0.57311|0.53933|0.63200|0.56267 |

## 판단 기준
- PSI >= 0.25: 강한 분포 차이 후보
- PSI >= 0.10: 중간 수준 분포 차이 후보
- missing_diff 절댓값이 큰 센서: missing feature 또는 feature 제거 검토
- adversarial AUC >= 0.70: train/test shift가 크므로 feature 안정성 재검토 필요
