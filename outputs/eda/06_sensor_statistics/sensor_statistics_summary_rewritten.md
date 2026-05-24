# Sensor Statistics Summary — 읽기 쉬운 재작성본

> 목적: `06_sensor_statistics` 결과를 사람이 해석하기 쉽고, AI가 후속 실험 액션으로 파싱하기 쉬운 형태로 재구성한 문서입니다. 원본 수치와 항목은 누락하지 않고 표 중심으로 정리했습니다.

## 0. 먼저 볼 결론

- **처리 결과는 정상입니다.** 센서 12개를 모두 처리했고, skipped sensor는 없습니다.
- **feature는 248개 생성되었습니다.** 이 중 실제 행동 feature는 75개, coverage/missing proxy feature는 61개, invalid/constant feature는 112개입니다.
- **모델링에서는 feature group을 분리해야 합니다.** `actual_behavior`와 `coverage_proxy`를 같은 의미로 해석하면 안 됩니다.
- **우선순위가 높은 축은 조도(light), 활동량(wPedo), 화면/충전 상태, activity code입니다.** 특히 target별로 유효 시간대가 다릅니다.
- **subject-adjusted correlation에서 유지되는 feature를 우선 feature engineering 대상으로 삼아야 합니다.** raw correlation만 보고 feature를 고르면 개인차에 과적합될 수 있습니다.
- **S2/S3는 일반 calendar day가 아니라 16:00~다음날 16:00 analysis-day 기준 실험이 필요합니다.**

## 1. 실행 개요

| 항목 | 값 | 해석 |
| --- | --- | --- |
| Sensors processed | 12 | 처리된 센서 수 |
| Skipped sensors | 0 | 실패/제외 센서 수 |
| Daily feature count | 248 | 하루 단위 feature 총수 |
| Actual behavior features | 75 | 실제 행동값 기반 feature |
| Coverage proxy features | 61 | 수집량/결측/센서 coverage proxy |
| Invalid/constant features | 112 | 상수 또는 무효 feature |

**판단:** 센서 처리 실패는 없으나, invalid/constant feature가 112개로 많습니다. 학습 입력에는 제거 또는 별도 검증 후 제외가 필요합니다.

## 2. Feature 그룹 정의

| 그룹 | 의미 | 예시 | 모델링 처리 |
| --- | --- | --- | --- |
| actual_behavior | 실제 행동/환경/생체 값 | light median, step sum, screen ratio | 메인 feature로 사용 |
| coverage_proxy | 센서 수집량, non-null count, object length 등 | row_count, nonnull_count, zero_count | missing indicator/coverage feature로 분리 |
| invalid/constant | 정보량이 없거나 상수인 feature | constant feature | 원칙적으로 제거 |

> `row_count`, `nonnull_count`, `object_len` 계열은 실제 행동이 아니라 센서가 얼마나 수집됐는지에 대한 proxy입니다. 제거 대상은 아니지만 실제 행동 feature와 섞어 해석하지 않습니다.

## 3. Log Transform 후보

아래 feature는 long-tail 성격이 있어 `log1p` 변환 후보입니다. 원본과 변환본을 모두 만들고 CV log-loss 기준으로 선택하는 방식이 안전합니다.

| feature | skew | zero_ratio | q99/median | group |
| --- | --- | --- | --- | --- |
| mLight_m_light_sum | 5.970 | 0.000 | 6.467 | light |
| wPedo_burned_calories_std | 2.556 | 0.067 | 13.836 | activity_pedo |
| wPedo_burned_calories_max | 2.539 | 0.067 | 50.351 | activity_pedo |
| wLight_w_light_sum | 2.395 | 0.011 | 6.281 | light |

## 4. 이상치 비율이 높은 Feature

아래 feature는 outlier ratio가 높은 순서입니다. clip/winsorize/log1p 여부를 CV 기준으로 비교해야 합니다.

| feature | outlier_ratio | q99 | max | type |
| --- | --- | --- | --- | --- |
| mActivity_m_activity_value_1_count | 0.201 | 13.019999999999982 | 24.0 | actual_behavior |
| mActivity_m_activity_value_4_ratio | 0.187 | 0.972410089367744 | 1.0 | actual_behavior |
| mActivity_m_activity_value_4_count | 0.181 | 1122.2799999999995 | 1440.0 | actual_behavior |
| wPedo_burned_calories_std | 0.159 | 8.28611815392152 | 11.312065288401593 | actual_behavior |
| wPedo_burned_calories_max | 0.152 | 274.9171264648438 | 310.6953430175781 | actual_behavior |
| mActivity_m_activity_value_8_ratio | 0.141 | 0.0132013888888888 | 0.0173611111111111 | actual_behavior |
| mActivity_m_activity_value_8_count | 0.140 | 19.00999999999999 | 25.0 | actual_behavior |
| wLight_w_light_q25 | 0.127 | 157.5100000000001 | 264.0 | actual_behavior |
| mLight_m_light_q99 | 0.116 | 12486.52049999998 | 55753.64999999976 | actual_behavior |
| mLight_m_light_std | 0.111 | 6798.959728846292 | 27843.59579826702 | actual_behavior |
| mActivity_m_activity_value_1_ratio | 0.111 | 0.0100212761492618 | 0.0209545983701979 | actual_behavior |
| mLight_m_light_max | 0.100 | 80002.79984374999 | 334306.0 | actual_behavior |
| wLight_w_light_median | 0.098 | 309.11 | 391.0 | actual_behavior |
| wLight_w_light_max | 0.098 | 99028.15 | 126950.0 | actual_behavior |
| mLight_m_light_zero_count | 0.089 | 131.01 | 134.0 | coverage_proxy |
| wLight_w_light_q95 | 0.078 | 4104.579500000003 | 7639.299999999997 | actual_behavior |
| wPedo_step_frequency_max | 0.077 | 3.8 | 5.283333333333333 | actual_behavior |
| wPedo_step_max | 0.077 | 228.0 | 317.0 | actual_behavior |
| wLight_w_light_std | 0.069 | 4839.495688725523 | 5509.503546098394 | actual_behavior |
| mLight_m_light_median | 0.067 | 249.56499999999997 | 635.0 | actual_behavior |
| mLight_m_light_q75 | 0.059 | 829.0 | 2999.025146484375 | actual_behavior |
| wPedo_distance_q95 | 0.057 | 71.3310393676758 | 104.81809082031248 | actual_behavior |
| wPedo_speed_q95 | 0.057 | 1.1888506561279302 | 1.7469681803385415 | actual_behavior |
| wLight_w_light_q99 | 0.054 | 18599.66800000001 | 28608.48999999994 | actual_behavior |
| wPedo_burned_calories_sum | 0.054 | 620.146774501801 | 853.7582092285156 | actual_behavior |
| wPedo_step_frequency_q95 | 0.054 | 1.5950666666666655 | 2.2 | actual_behavior |
| wPedo_step_q95 | 0.054 | 95.70399999999994 | 132.0 | actual_behavior |
| wPedo_burned_calories_mean | 0.054 | 0.6327266953024973 | 1.3340457938831938 | actual_behavior |

## 5. Target별 핵심 해석 요약

| Target | 일 단위 actual-behavior Top1 | subject-adjusted Top1 | timeblock Top1 | 실험 해석 |
| --- | --- | --- | --- | --- |
| Q1 | wLight_w_light_median (spearman=-0.2239, group=light, n=414) | wPedo_burned_calories_q75 (subject_adjusted_spearman=0.2271, group=activity_pedo, n=404) | mLight_m_light_04_08_median (spearman=-0.2070, group=light, n=431) | 일 단위에서는 조도(light) 계열이 가장 반복적으로 등장<br>subject-adjusted에서는 wPedo q75 계열 활동량도 강함<br>04–08 조도와 20–24 활동량 timeblock 우선 검토 |
| Q2 | mACStatus_m_charging_value_1_count (spearman=-0.1748, group=charging, n=450) | wPedo_distance_sum (subject_adjusted_spearman=-0.1358, group=activity_pedo, n=404) | mActivity_m_activity_08_12_value_7_count (spearman=-0.1869, group=activity_code, n=442) | 충전 상태, 활동 코드, screen 계열이 주요 후보<br>subject-adjusted에서는 wPedo 활동량 계열이 음의 방향으로 반복<br>08–12 activity, 20–24 light, 12–16 screen timeblock 확인 |
| Q3 | mLight_m_light_median (spearman=-0.2204, group=light, n=450) | wPedo_burned_calories_q75 (subject_adjusted_spearman=0.1695, group=activity_pedo, n=404) | mLight_m_light_12_16_q25 (spearman=-0.2076, group=light, n=447) | mLight/wLight 조도 계열이 가장 강하게 반복<br>subject-adjusted에서도 조도와 wPedo q75 계열 유지<br>12–16/00–04/20–24 light timeblock 우선 |
| S1 | mScreenStatus_m_screen_use_value_0_count (spearman=0.2722, group=screen, n=450) | wPedo_distance_q75 (subject_adjusted_spearman=0.1905, group=activity_pedo, n=404) | wPedo_distance_00_04_q99 (spearman=-0.2787, group=activity_pedo, n=322) | screen 상태가 일 단위 최상위<br>subject-adjusted에서는 wPedo q75 활동량과 screen ratio 유지<br>00–04 wPedo 활동량 timeblock이 가장 강한 축 |
| S2 | mScreenStatus_m_screen_use_value_0_count (spearman=0.2204, group=screen, n=450) | wPedo_distance_q75 (subject_adjusted_spearman=0.1335, group=activity_pedo, n=404) | mScreenStatus_m_screen_use_20_24_value_0_ratio (spearman=0.1588, group=screen, n=443) | screen 상태가 일 단위 및 timeblock에서 반복<br>coverage proxy에서는 BLE/WiFi/GPS 수집량 proxy가 등장<br>16–24 screen, 04–08/00–04 관련 야간 feature 검토 |
| S3 | mLight_m_light_median (spearman=0.1245, group=light, n=450) | wPedo_distance_q75 (subject_adjusted_spearman=0.1187, group=activity_pedo, n=404) | wLight_w_light_00_04_max (spearman=0.1470, group=light, n=345) | 전체 상관은 상대적으로 약한 편<br>wPedo coverage 및 light/charging/activity가 분산되어 등장<br>00–04 light, 04–08 light, 12–16 activity, 08–12/16–20 charging 검토 |
| S4 | mActivity_m_activity_value_8_count (spearman=0.1359, group=activity_code, n=450) | mActivity_m_activity_value_0_ratio (subject_adjusted_spearman=0.1173, group=activity_code, n=450) | mACStatus_m_charging_04_08_value_0_ratio (spearman=-0.1896, group=charging, n=432) | activity code 8, pedo, screen이 일 단위 후보<br>04–08 charging이 timeblock에서 가장 뚜렷<br>light 야간/새벽 구간과 screen 12–16도 함께 확인 |

## 6. Daily Feature-Target Top Signals 상세

각 target별로 세 종류를 분리했습니다.

- **actual-behavior:** 실제 행동/센서값 기반 상관
- **coverage-proxy:** 센서 수집량/결측 proxy 기반 상관
- **subject-adjusted actual-behavior:** 개인 평균 효과를 제거한 뒤에도 남는 행동 feature 상관

### Q1

#### Top actual-behavior correlation features

| feature | spearman | group | n |
| --- | --- | --- | --- |
| wLight_w_light_median | -0.2239 | light | 414 |
| mLight_m_light_median | -0.1837 | light | 450 |
| wLight_w_light_q75 | -0.1576 | light | 414 |
| wLight_w_light_q25 | -0.1547 | light | 414 |
| mLight_m_light_q25 | -0.1289 | light | 450 |
| mLight_m_light_q75 | -0.1285 | light | 450 |
| mActivity_m_activity_value_8_count | -0.1257 | activity_code | 450 |
| mActivity_m_activity_value_8_ratio | -0.1252 | activity_code | 450 |
| mActivity_m_activity_value_0_count | 0.1158 | activity_code | 450 |
| wPedo_burned_calories_q75 | 0.1093 | activity_pedo | 404 |

#### Top coverage-proxy correlation features

| feature | spearman | group | n |
| --- | --- | --- | --- |
| wLight_w_light_zero_count | 0.2040 | light | 414 |
| mLight_m_light_zero_count | 0.1989 | light | 450 |
| mLight_m_light_nonzero_count | -0.1477 | light | 450 |
| wLight_row_count | 0.1194 | light | 414 |
| wLight_w_light_nonnull_count | 0.1194 | light | 414 |

#### Top subject-adjusted actual-behavior correlation features

| feature | subject_adjusted_spearman | group | n |
| --- | --- | --- | --- |
| wPedo_burned_calories_q75 | 0.2271 | activity_pedo | 404 |
| mLight_m_light_q25 | -0.2223 | light | 450 |
| wPedo_step_frequency_q75 | 0.1991 | activity_pedo | 404 |
| wPedo_step_q75 | 0.1991 | activity_pedo | 404 |
| wPedo_distance_q75 | 0.1932 | activity_pedo | 404 |
| wPedo_speed_q75 | 0.1932 | activity_pedo | 404 |
| wLight_w_light_q25 | -0.1702 | light | 414 |
| wLight_w_light_median | -0.1590 | light | 414 |
| mLight_m_light_min | -0.1565 | light | 450 |
| mLight_m_light_median | -0.1188 | light | 450 |

### Q2

#### Top actual-behavior correlation features

| feature | spearman | group | n |
| --- | --- | --- | --- |
| mACStatus_m_charging_value_1_count | -0.1748 | charging | 450 |
| mACStatus_m_charging_value_0_ratio | 0.1685 | charging | 450 |
| mACStatus_m_charging_value_1_ratio | -0.1685 | charging | 450 |
| mActivity_m_activity_value_7_count | -0.1543 | activity_code | 450 |
| mActivity_m_activity_value_7_ratio | -0.1472 | activity_code | 450 |
| mActivity_m_activity_value_4_ratio | 0.1185 | activity_code | 450 |
| mActivity_m_activity_value_4_count | 0.1156 | activity_code | 450 |
| mScreenStatus_m_screen_use_value_0_count | -0.1143 | screen | 450 |
| mACStatus_m_charging_value_0_count | 0.1137 | charging | 450 |
| mLight_m_light_q25 | 0.1134 | light | 450 |

#### Top coverage-proxy correlation features

| feature | spearman | group | n |
| --- | --- | --- | --- |
| mScreenStatus_m_screen_use_nonnull_count | -0.1471 | screen | 450 |
| mScreenStatus_row_count | -0.1471 | screen | 450 |
| mACStatus_m_charging_nonnull_count | -0.1236 | charging | 450 |
| mACStatus_row_count | -0.1236 | charging | 450 |
| mActivity_m_activity_nonnull_count | -0.1072 | activity_code | 450 |

#### Top subject-adjusted actual-behavior correlation features

| feature | subject_adjusted_spearman | group | n |
| --- | --- | --- | --- |
| wPedo_distance_sum | -0.1358 | activity_pedo | 404 |
| wPedo_speed_sum | -0.1358 | activity_pedo | 404 |
| mActivity_m_activity_value_0_count | -0.1353 | activity_code | 450 |
| wLight_w_light_min | 0.1346 | light | 414 |
| wPedo_burned_calories_q95 | -0.1330 | activity_pedo | 404 |
| wPedo_step_frequency_sum | -0.1325 | activity_pedo | 404 |
| wPedo_step_sum | -0.1325 | activity_pedo | 404 |
| wPedo_distance_q95 | -0.1285 | activity_pedo | 404 |
| wPedo_speed_q95 | -0.1285 | activity_pedo | 404 |
| wPedo_step_frequency_q95 | -0.1265 | activity_pedo | 404 |

### Q3

#### Top actual-behavior correlation features

| feature | spearman | group | n |
| --- | --- | --- | --- |
| mLight_m_light_median | -0.2204 | light | 450 |
| wLight_w_light_q75 | -0.1884 | light | 414 |
| mLight_m_light_mean | -0.1744 | light | 450 |
| mLight_m_light_q75 | -0.1667 | light | 450 |
| mLight_m_light_sum | -0.1574 | light | 450 |
| wLight_w_light_q25 | -0.1509 | light | 414 |
| mLight_m_light_std | -0.1399 | light | 450 |
| mLight_m_light_max | -0.1352 | light | 450 |
| wLight_w_light_median | -0.1329 | light | 414 |
| wPedo_burned_calories_q99 | -0.1156 | activity_pedo | 404 |

#### Top coverage-proxy correlation features

| feature | spearman | group | n |
| --- | --- | --- | --- |
| mLight_m_light_zero_count | 0.1845 | light | 450 |
| mLight_m_light_nonzero_count | -0.1308 | light | 450 |
| wHr_heart_rate_object_len_nonnull_count | -0.1166 | heart_rate | 391 |
| wHr_row_count | -0.1166 | heart_rate | 391 |
| mLight_m_light_nonnull_count | 0.1136 | light | 450 |

#### Top subject-adjusted actual-behavior correlation features

| feature | subject_adjusted_spearman | group | n |
| --- | --- | --- | --- |
| wPedo_burned_calories_q75 | 0.1695 | activity_pedo | 404 |
| wLight_w_light_q25 | -0.1599 | light | 414 |
| mLight_m_light_max | -0.1557 | light | 450 |
| mLight_m_light_q25 | -0.1502 | light | 450 |
| mLight_m_light_median | -0.1353 | light | 450 |
| mLight_m_light_std | -0.1340 | light | 450 |
| wPedo_step_frequency_q75 | 0.1214 | activity_pedo | 404 |
| wPedo_step_q75 | 0.1214 | activity_pedo | 404 |
| mActivity_m_activity_value_7_ratio | -0.1202 | activity_code | 450 |
| mLight_m_light_mean | -0.1177 | light | 450 |

### S1

#### Top actual-behavior correlation features

| feature | spearman | group | n |
| --- | --- | --- | --- |
| mScreenStatus_m_screen_use_value_0_count | 0.2722 | screen | 450 |
| mScreenStatus_m_screen_use_value_0_ratio | 0.2355 | screen | 450 |
| mScreenStatus_m_screen_use_value_1_ratio | -0.2355 | screen | 450 |
| mScreenStatus_m_screen_use_value_1_count | -0.2082 | screen | 450 |
| mLight_m_light_q25 | -0.1438 | light | 450 |
| mActivity_m_activity_value_4_ratio | -0.1381 | activity_code | 450 |
| mActivity_m_activity_value_4_count | -0.1365 | activity_code | 450 |
| wLight_w_light_q25 | -0.1257 | light | 414 |
| mLight_m_light_median | -0.1225 | light | 450 |
| wPedo_burned_calories_q99 | -0.1138 | activity_pedo | 404 |

#### Top coverage-proxy correlation features

| feature | spearman | group | n |
| --- | --- | --- | --- |
| mLight_m_light_zero_count | 0.2136 | light | 450 |
| mBle_m_ble_object_len_nonnull_count | -0.1855 | coverage_or_object_proxy | 416 |
| mBle_row_count | -0.1855 | coverage_or_object_proxy | 416 |
| mLight_m_light_nonzero_count | -0.1715 | light | 450 |
| mAmbience_m_ambience_object_len_nonnull_count | 0.1245 | coverage_or_object_proxy | 450 |

#### Top subject-adjusted actual-behavior correlation features

| feature | subject_adjusted_spearman | group | n |
| --- | --- | --- | --- |
| wPedo_distance_q75 | 0.1905 | activity_pedo | 404 |
| wPedo_speed_q75 | 0.1905 | activity_pedo | 404 |
| wPedo_step_frequency_q75 | 0.1656 | activity_pedo | 404 |
| wPedo_step_q75 | 0.1656 | activity_pedo | 404 |
| mLight_m_light_min | -0.1282 | light | 450 |
| mScreenStatus_m_screen_use_value_0_ratio | 0.1262 | screen | 450 |
| mScreenStatus_m_screen_use_value_1_ratio | -0.1262 | screen | 450 |
| mScreenStatus_m_screen_use_value_0_count | 0.1233 | screen | 450 |
| wPedo_burned_calories_q75 | 0.1182 | activity_pedo | 404 |
| mScreenStatus_m_screen_use_value_1_count | -0.1046 | screen | 450 |

### S2

#### Top actual-behavior correlation features

| feature | spearman | group | n |
| --- | --- | --- | --- |
| mScreenStatus_m_screen_use_value_0_count | 0.2204 | screen | 450 |
| mScreenStatus_m_screen_use_value_0_ratio | 0.1852 | screen | 450 |
| mScreenStatus_m_screen_use_value_1_ratio | -0.1852 | screen | 450 |
| mScreenStatus_m_screen_use_value_1_count | -0.1642 | screen | 450 |
| wPedo_burned_calories_q99 | -0.1182 | activity_pedo | 404 |
| mLight_m_light_max | -0.1156 | light | 450 |
| mLight_m_light_std | -0.1024 | light | 450 |
| wPedo_burned_calories_q95 | -0.1004 | activity_pedo | 404 |
| mLight_m_light_q99 | -0.0996 | light | 450 |
| mACStatus_m_charging_value_1_count | 0.0898 | charging | 450 |

#### Top coverage-proxy correlation features

| feature | spearman | group | n |
| --- | --- | --- | --- |
| mBle_m_ble_object_len_nonnull_count | -0.1743 | coverage_or_object_proxy | 416 |
| mBle_row_count | -0.1743 | coverage_or_object_proxy | 416 |
| mWifi_m_wifi_object_len_nonnull_count | 0.1671 | coverage_or_object_proxy | 439 |
| mWifi_row_count | 0.1671 | coverage_or_object_proxy | 439 |
| mGps_m_gps_object_len_nonnull_count | 0.1152 | coverage_or_object_proxy | 425 |

#### Top subject-adjusted actual-behavior correlation features

| feature | subject_adjusted_spearman | group | n |
| --- | --- | --- | --- |
| wPedo_distance_q75 | 0.1335 | activity_pedo | 404 |
| wPedo_speed_q75 | 0.1335 | activity_pedo | 404 |
| wPedo_burned_calories_q75 | 0.1280 | activity_pedo | 404 |
| wLight_w_light_min | 0.1193 | light | 414 |
| wPedo_step_frequency_q75 | 0.1103 | activity_pedo | 404 |
| wPedo_step_q75 | 0.1103 | activity_pedo | 404 |
| mActivity_m_activity_value_8_count | 0.0767 | activity_code | 450 |
| mActivity_m_activity_value_8_ratio | 0.0747 | activity_code | 450 |
| mActivity_m_activity_value_4_ratio | 0.0741 | activity_code | 450 |
| wLight_w_light_q25 | -0.0665 | light | 414 |

### S3

#### Top actual-behavior correlation features

| feature | spearman | group | n |
| --- | --- | --- | --- |
| mLight_m_light_median | 0.1245 | light | 450 |
| mScreenStatus_m_screen_use_value_0_count | 0.1146 | screen | 450 |
| wLight_w_light_median | 0.0929 | light | 414 |
| mActivity_m_activity_value_3_count | 0.0908 | activity_code | 450 |
| wLight_w_light_q25 | 0.0894 | light | 414 |
| mActivity_m_activity_value_4_ratio | -0.0882 | activity_code | 450 |
| mActivity_m_activity_value_4_count | -0.0864 | activity_code | 450 |
| mScreenStatus_m_screen_use_value_0_ratio | 0.0752 | screen | 450 |
| mScreenStatus_m_screen_use_value_1_ratio | -0.0752 | screen | 450 |
| wPedo_step_frequency_q75 | -0.0741 | activity_pedo | 404 |

#### Top coverage-proxy correlation features

| feature | spearman | group | n |
| --- | --- | --- | --- |
| wPedo_burned_calories_zero_count | 0.1824 | activity_pedo | 404 |
| wPedo_burned_calories_nonnull_count | 0.1807 | activity_pedo | 404 |
| wPedo_row_count | 0.1807 | activity_pedo | 404 |
| wLight_w_light_nonzero_count | 0.1414 | light | 414 |
| mWifi_m_wifi_object_len_nonnull_count | 0.1078 | coverage_or_object_proxy | 439 |

#### Top subject-adjusted actual-behavior correlation features

| feature | subject_adjusted_spearman | group | n |
| --- | --- | --- | --- |
| wPedo_distance_q75 | 0.1187 | activity_pedo | 404 |
| wPedo_speed_q75 | 0.1187 | activity_pedo | 404 |
| wPedo_step_frequency_q75 | 0.1036 | activity_pedo | 404 |
| wPedo_step_q75 | 0.1036 | activity_pedo | 404 |
| mACStatus_m_charging_value_0_ratio | 0.0909 | charging | 450 |
| mACStatus_m_charging_value_1_ratio | -0.0909 | charging | 450 |
| wPedo_burned_calories_max | 0.0906 | activity_pedo | 404 |
| wLight_w_light_q25 | 0.0894 | light | 414 |
| wPedo_burned_calories_sum | 0.0842 | activity_pedo | 404 |
| wLight_w_light_q75 | 0.0797 | light | 414 |

### S4

#### Top actual-behavior correlation features

| feature | spearman | group | n |
| --- | --- | --- | --- |
| mActivity_m_activity_value_8_count | 0.1359 | activity_code | 450 |
| mActivity_m_activity_value_8_ratio | 0.1348 | activity_code | 450 |
| wPedo_step_sum | -0.1265 | activity_pedo | 404 |
| wPedo_step_frequency_sum | -0.1265 | activity_pedo | 404 |
| wPedo_burned_calories_q95 | -0.1243 | activity_pedo | 404 |
| mScreenStatus_m_screen_use_value_1_count | -0.1221 | screen | 450 |
| mScreenStatus_m_screen_use_value_0_ratio | 0.1206 | screen | 450 |
| mScreenStatus_m_screen_use_value_1_ratio | -0.1206 | screen | 450 |
| wPedo_step_frequency_mean | -0.1202 | activity_pedo | 404 |
| wPedo_step_mean | -0.1202 | activity_pedo | 404 |

#### Top coverage-proxy correlation features

| feature | spearman | group | n |
| --- | --- | --- | --- |
| wPedo_burned_calories_nonzero_count | -0.1372 | activity_pedo | 404 |
| mUsageStats_m_usage_stats_object_len_nonnull_count | -0.1365 | app_usage | 440 |
| mUsageStats_row_count | -0.1365 | app_usage | 440 |
| mBle_m_ble_object_len_nonnull_count | -0.1349 | coverage_or_object_proxy | 416 |
| mBle_row_count | -0.1349 | coverage_or_object_proxy | 416 |

#### Top subject-adjusted actual-behavior correlation features

| feature | subject_adjusted_spearman | group | n |
| --- | --- | --- | --- |
| mActivity_m_activity_value_0_ratio | 0.1173 | activity_code | 450 |
| mActivity_m_activity_value_0_count | 0.1045 | activity_code | 450 |
| wLight_w_light_q99 | 0.0885 | light | 414 |
| wLight_w_light_min | 0.0787 | light | 414 |
| wLight_w_light_std | 0.0774 | light | 414 |
| wLight_w_light_max | 0.0711 | light | 414 |
| mActivity_m_activity_value_7_count | -0.0694 | activity_code | 450 |
| wLight_w_light_q75 | -0.0643 | light | 414 |
| mScreenStatus_m_screen_use_value_1_count | -0.0623 | screen | 450 |
| mLight_m_light_min | 0.0616 | light | 450 |

## 7. Timeblock Feature-Target Top Signals 상세

4시간 block 기준 feature입니다. 시간대별 signal은 target별 feature set 설계에 직접 사용합니다.

### Q1

| feature | spearman | group | n |
| --- | --- | --- | --- |
| mLight_m_light_04_08_median | -0.2070 | light | 431 |
| mLight_m_light_04_08_q75 | -0.1992 | light | 431 |
| mLight_m_light_04_08_sum | -0.1919 | light | 431 |
| mLight_m_light_04_08_mean | -0.1919 | light | 431 |
| mLight_m_light_04_08_q95 | -0.1862 | light | 431 |
| wLight_w_light_04_08_mean | -0.1830 | light | 338 |
| wLight_w_light_04_08_std | -0.1671 | light | 338 |
| wPedo_burned_calories_20_24_q95 | 0.1664 | activity_pedo | 355 |
| wLight_w_light_04_08_q99 | -0.1614 | light | 338 |
| mLight_m_light_04_08_std | -0.1607 | light | 431 |

### Q2

| feature | spearman | group | n |
| --- | --- | --- | --- |
| mActivity_m_activity_08_12_value_7_count | -0.1869 | activity_code | 442 |
| mActivity_m_activity_08_12_value_7_ratio | -0.1843 | activity_code | 442 |
| mLight_m_light_20_24_mean | 0.1812 | light | 442 |
| mActivity_m_activity_04_08_value_7_ratio | -0.1697 | activity_code | 432 |
| mActivity_m_activity_04_08_value_7_count | -0.1689 | activity_code | 432 |
| mScreenStatus_m_screen_use_12_16_value_0_count | -0.1664 | screen | 446 |
| mActivity_m_activity_08_12_value_4_ratio | 0.1656 | activity_code | 442 |
| mLight_m_light_20_24_sum | 0.1651 | light | 442 |
| mActivity_m_activity_08_12_value_4_count | 0.1634 | activity_code | 442 |
| wPedo_distance_04_08_q99 | -0.1627 | activity_pedo | 329 |

### Q3

| feature | spearman | group | n |
| --- | --- | --- | --- |
| mLight_m_light_12_16_q25 | -0.2076 | light | 447 |
| mLight_m_light_12_16_median | -0.1785 | light | 447 |
| mLight_m_light_00_04_q75 | -0.1747 | light | 432 |
| mLight_m_light_00_04_mean | -0.1742 | light | 432 |
| wLight_w_light_12_16_q25 | -0.1730 | light | 397 |
| mLight_m_light_00_04_q95 | -0.1719 | light | 432 |
| mLight_m_light_20_24_max | 0.1688 | light | 442 |
| mLight_m_light_00_04_std | -0.1685 | light | 430 |
| mLight_m_light_08_12_q25 | -0.1676 | light | 442 |
| mLight_m_light_20_24_q99 | 0.1657 | light | 442 |

### S1

| feature | spearman | group | n |
| --- | --- | --- | --- |
| wPedo_distance_00_04_q99 | -0.2787 | activity_pedo | 322 |
| wPedo_speed_00_04_q99 | -0.2787 | activity_pedo | 322 |
| wPedo_step_00_04_q99 | -0.2785 | activity_pedo | 322 |
| wPedo_step_frequency_00_04_q99 | -0.2785 | activity_pedo | 322 |
| wPedo_burned_calories_00_04_max | -0.2508 | activity_pedo | 322 |
| wPedo_burned_calories_00_04_std | -0.2503 | activity_pedo | 317 |
| wPedo_distance_00_04_max | -0.2496 | activity_pedo | 322 |
| wPedo_speed_00_04_max | -0.2496 | activity_pedo | 322 |
| wPedo_burned_calories_00_04_mean | -0.2495 | activity_pedo | 322 |
| wPedo_burned_calories_00_04_sum | -0.2495 | activity_pedo | 322 |

### S2

| feature | spearman | group | n |
| --- | --- | --- | --- |
| mScreenStatus_m_screen_use_20_24_value_0_ratio | 0.1588 | screen | 443 |
| mScreenStatus_m_screen_use_20_24_value_1_ratio | -0.1588 | screen | 443 |
| mScreenStatus_m_screen_use_20_24_value_1_count | -0.1565 | screen | 443 |
| mScreenStatus_m_screen_use_20_24_value_0_count | 0.1384 | screen | 443 |
| mScreenStatus_m_screen_use_08_12_value_0_count | 0.1279 | screen | 442 |
| mScreenStatus_m_screen_use_16_20_value_0_ratio | 0.1240 | screen | 444 |
| mScreenStatus_m_screen_use_16_20_value_1_ratio | -0.1240 | screen | 444 |
| mScreenStatus_m_screen_use_16_20_value_1_count | -0.1235 | screen | 444 |
| mScreenStatus_m_screen_use_16_20_value_0_count | 0.1208 | screen | 444 |
| wPedo_burned_calories_08_12_std | -0.1180 | activity_pedo | 382 |

### S3

| feature | spearman | group | n |
| --- | --- | --- | --- |
| wLight_w_light_00_04_max | 0.1470 | light | 345 |
| mLight_m_light_04_08_median | 0.1410 | light | 431 |
| mACStatus_m_charging_08_12_value_0_count | 0.1409 | charging | 442 |
| mActivity_m_activity_12_16_value_3_count | 0.1341 | activity_code | 446 |
| wLight_w_light_00_04_q99 | 0.1312 | light | 345 |
| mActivity_m_activity_12_16_value_3_ratio | 0.1306 | activity_code | 446 |
| wLight_w_light_00_04_std | 0.1281 | light | 345 |
| mACStatus_m_charging_16_20_value_1_count | -0.1165 | charging | 444 |
| mACStatus_m_charging_16_20_value_0_ratio | 0.1161 | charging | 444 |
| mACStatus_m_charging_16_20_value_1_ratio | -0.1161 | charging | 444 |

### S4

| feature | spearman | group | n |
| --- | --- | --- | --- |
| mACStatus_m_charging_04_08_value_0_ratio | -0.1896 | charging | 432 |
| mACStatus_m_charging_04_08_value_1_ratio | 0.1896 | charging | 432 |
| mACStatus_m_charging_04_08_value_1_count | 0.1889 | charging | 432 |
| mACStatus_m_charging_04_08_value_0_count | -0.1838 | charging | 432 |
| wLight_w_light_04_08_q75 | 0.1617 | light | 338 |
| mLight_m_light_00_04_max | 0.1597 | light | 432 |
| mLight_m_light_00_04_q99 | 0.1581 | light | 432 |
| wLight_w_light_04_08_q95 | 0.1560 | light | 338 |
| mLight_m_light_00_04_std | 0.1539 | light | 430 |
| mScreenStatus_m_screen_use_12_16_value_0_count | 0.1478 | screen | 446 |

## 8. 해석 규칙

1. `timestamp_object_len_*` 계열은 생성하지 않습니다. timestamp는 센서값이 아니라 메타데이터입니다.
2. `row_count`, `nonnull_count`, `object_len` 계열은 실제 행동값이 아니라 sensor coverage 또는 수집량 proxy입니다.
3. coverage proxy는 제거 대상이 아니라 별도 feature group으로 관리해야 합니다. 단, 실제 행동 feature와 섞어서 해석하지 않습니다.
4. `zero_ratio`가 1.0에 가까운 feature와 constant feature는 log-transform 후보에서 제외했습니다.
5. raw correlation보다 subject-adjusted correlation에서 유지되는 actual-behavior feature를 우선 feature engineering 대상으로 삼습니다.
6. timeblock feature는 4시간 단위 기준입니다. S2/S3는 20_24, 00_04, 04_08 구간을 우선 검토하십시오.

## 9. 다음 모델링 액션

1. actual_behavior feature와 coverage_proxy feature를 분리한 feature set으로 LightGBM CV 비교
2. log1p 후보 feature는 원본/변환 버전을 모두 만들고 CV log-loss 기준으로 선택
3. subject-adjusted correlation 상위 feature는 subject mean/deviation/z-score로 재생성
4. 야간/수면 전 timeblock feature는 target별 feature set에 우선 포함
5. coverage proxy feature는 missing indicator 그룹으로 분리해 SHAP/importance를 따로 해석
6. S2/S3는 16:00~다음날 16:00 analysis-day 기준 feature를 별도 실험

## 10. 실험 큐로 변환

| 우선순위 | 실험 | 사용 feature/대상 | 판단 기준 |
| --- | --- | --- | --- |
| P0 | actual_behavior vs coverage_proxy 분리 LightGBM CV | 전체 feature group 분리 | CV log-loss 개선 여부 |
| P0 | invalid/constant feature 제거 | 112개 invalid/constant | 성능/안정성 변화 |
| P1 | log1p 후보 원본+변환 비교 | mLight sum, wPedo burned_calories std/max, wLight sum | target별 CV log-loss |
| P1 | subject mean/deviation/z-score 생성 | subject-adjusted 상위 feature | subject별 log-loss 개선 |
| P1 | 야간/수면 전 timeblock feature 강화 | 20_24, 00_04, 04_08 | S2/S3 및 수면 target 개선 |
| P2 | coverage proxy를 missing indicator 그룹으로 별도 투입 | row_count, nonnull_count, object_len, zero_count | SHAP/importance 별도 해석 |
| P2 | S2/S3 analysis-day 재생성 | 16:00~다음날 16:00 | S2/S3 CV log-loss 개선 |

## 11. AI 파싱용 메타 요약

```yaml

run_status:

  sensors_processed: 12

  skipped_sensors: 0

  daily_feature_count: 248

  actual_behavior_features: 75

  coverage_proxy_features: 61

  invalid_constant_features: 112

priority_groups:

  - light
  - activity_pedo
  - screen
  - charging
  - activity_code

required_feature_separation:

  - actual_behavior
  - coverage_proxy
  - invalid_constant

recommended_experiments:

  - actual_behavior feature와 coverage_proxy feature를 분리한 feature set으로 LightGBM CV 비교

  - log1p 후보 feature는 원본/변환 버전을 모두 만들고 CV log-loss 기준으로 선택

  - subject-adjusted correlation 상위 feature는 subject mean/deviation/z-score로 재생성

  - 야간/수면 전 timeblock feature는 target별 feature set에 우선 포함

  - coverage proxy feature는 missing indicator 그룹으로 분리해 SHAP/importance를 따로 해석

  - S2/S3는 16:00~다음날 16:00 analysis-day 기준 feature를 별도 실험

```
