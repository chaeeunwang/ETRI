# Sensor Statistics Summary

## 1. Run Overview

- Sensors processed: 12
- Skipped sensors: 0
- Daily feature count: 248
- Actual behavior features: 75
- Coverage proxy features: 61
- Invalid/constant features: 112

## 2. Sensor Processing Status

- Skipped sensors: 0

## 3. Log Transform Candidates

- mLight_m_light_sum: skew=5.970, zero_ratio=0.000, q99/median=6.467, group=light
- wPedo_burned_calories_std: skew=2.556, zero_ratio=0.067, q99/median=13.836, group=activity_pedo
- wPedo_burned_calories_max: skew=2.539, zero_ratio=0.067, q99/median=50.351, group=activity_pedo
- wLight_w_light_sum: skew=2.395, zero_ratio=0.011, q99/median=6.281, group=light

## 4. High Outlier Features

- mActivity_m_activity_value_1_count: outlier_ratio=0.201, q99=13.019999999999982, max=24.0, type=actual_behavior
- mActivity_m_activity_value_4_ratio: outlier_ratio=0.187, q99=0.972410089367744, max=1.0, type=actual_behavior
- mActivity_m_activity_value_4_count: outlier_ratio=0.181, q99=1122.2799999999995, max=1440.0, type=actual_behavior
- wPedo_burned_calories_std: outlier_ratio=0.159, q99=8.28611815392152, max=11.312065288401593, type=actual_behavior
- wPedo_burned_calories_max: outlier_ratio=0.152, q99=274.9171264648438, max=310.6953430175781, type=actual_behavior
- mActivity_m_activity_value_8_ratio: outlier_ratio=0.141, q99=0.0132013888888888, max=0.0173611111111111, type=actual_behavior
- mActivity_m_activity_value_8_count: outlier_ratio=0.140, q99=19.00999999999999, max=25.0, type=actual_behavior
- wLight_w_light_q25: outlier_ratio=0.127, q99=157.5100000000001, max=264.0, type=actual_behavior
- mLight_m_light_q99: outlier_ratio=0.116, q99=12486.52049999998, max=55753.64999999976, type=actual_behavior
- mLight_m_light_std: outlier_ratio=0.111, q99=6798.959728846292, max=27843.59579826702, type=actual_behavior
- mActivity_m_activity_value_1_ratio: outlier_ratio=0.111, q99=0.0100212761492618, max=0.0209545983701979, type=actual_behavior
- mLight_m_light_max: outlier_ratio=0.100, q99=80002.79984374999, max=334306.0, type=actual_behavior
- wLight_w_light_median: outlier_ratio=0.098, q99=309.11, max=391.0, type=actual_behavior
- wLight_w_light_max: outlier_ratio=0.098, q99=99028.15, max=126950.0, type=actual_behavior
- mLight_m_light_zero_count: outlier_ratio=0.089, q99=131.01, max=134.0, type=coverage_proxy
- wLight_w_light_q95: outlier_ratio=0.078, q99=4104.579500000003, max=7639.299999999997, type=actual_behavior
- wPedo_step_frequency_max: outlier_ratio=0.077, q99=3.8, max=5.283333333333333, type=actual_behavior
- wPedo_step_max: outlier_ratio=0.077, q99=228.0, max=317.0, type=actual_behavior
- wLight_w_light_std: outlier_ratio=0.069, q99=4839.495688725523, max=5509.503546098394, type=actual_behavior
- mLight_m_light_median: outlier_ratio=0.067, q99=249.56499999999997, max=635.0, type=actual_behavior
- mLight_m_light_q75: outlier_ratio=0.059, q99=829.0, max=2999.025146484375, type=actual_behavior
- wPedo_distance_q95: outlier_ratio=0.057, q99=71.3310393676758, max=104.81809082031248, type=actual_behavior
- wPedo_speed_q95: outlier_ratio=0.057, q99=1.1888506561279302, max=1.7469681803385415, type=actual_behavior
- wLight_w_light_q99: outlier_ratio=0.054, q99=18599.66800000001, max=28608.48999999994, type=actual_behavior
- wPedo_burned_calories_sum: outlier_ratio=0.054, q99=620.146774501801, max=853.7582092285156, type=actual_behavior
- wPedo_step_frequency_q95: outlier_ratio=0.054, q99=1.5950666666666655, max=2.2, type=actual_behavior
- wPedo_step_q95: outlier_ratio=0.054, q99=95.70399999999994, max=132.0, type=actual_behavior
- wPedo_burned_calories_mean: outlier_ratio=0.054, q99=0.6327266953024973, max=1.3340457938831938, type=actual_behavior

## 5. Daily Feature-Target Top Signals

### Q1

#### Top actual-behavior correlation features

- wLight_w_light_median: spearman=-0.2239, group=light, n=414
- mLight_m_light_median: spearman=-0.1837, group=light, n=450
- wLight_w_light_q75: spearman=-0.1576, group=light, n=414
- wLight_w_light_q25: spearman=-0.1547, group=light, n=414
- mLight_m_light_q25: spearman=-0.1289, group=light, n=450
- mLight_m_light_q75: spearman=-0.1285, group=light, n=450
- mActivity_m_activity_value_8_count: spearman=-0.1257, group=activity_code, n=450
- mActivity_m_activity_value_8_ratio: spearman=-0.1252, group=activity_code, n=450
- mActivity_m_activity_value_0_count: spearman=0.1158, group=activity_code, n=450
- wPedo_burned_calories_q75: spearman=0.1093, group=activity_pedo, n=404

#### Top coverage-proxy correlation features

- wLight_w_light_zero_count: spearman=0.2040, group=light, n=414
- mLight_m_light_zero_count: spearman=0.1989, group=light, n=450
- mLight_m_light_nonzero_count: spearman=-0.1477, group=light, n=450
- wLight_row_count: spearman=0.1194, group=light, n=414
- wLight_w_light_nonnull_count: spearman=0.1194, group=light, n=414

#### Top subject-adjusted actual-behavior correlation features

- wPedo_burned_calories_q75: subject_adjusted_spearman=0.2271, group=activity_pedo, n=404
- mLight_m_light_q25: subject_adjusted_spearman=-0.2223, group=light, n=450
- wPedo_step_frequency_q75: subject_adjusted_spearman=0.1991, group=activity_pedo, n=404
- wPedo_step_q75: subject_adjusted_spearman=0.1991, group=activity_pedo, n=404
- wPedo_distance_q75: subject_adjusted_spearman=0.1932, group=activity_pedo, n=404
- wPedo_speed_q75: subject_adjusted_spearman=0.1932, group=activity_pedo, n=404
- wLight_w_light_q25: subject_adjusted_spearman=-0.1702, group=light, n=414
- wLight_w_light_median: subject_adjusted_spearman=-0.1590, group=light, n=414
- mLight_m_light_min: subject_adjusted_spearman=-0.1565, group=light, n=450
- mLight_m_light_median: subject_adjusted_spearman=-0.1188, group=light, n=450

### Q2

#### Top actual-behavior correlation features

- mACStatus_m_charging_value_1_count: spearman=-0.1748, group=charging, n=450
- mACStatus_m_charging_value_0_ratio: spearman=0.1685, group=charging, n=450
- mACStatus_m_charging_value_1_ratio: spearman=-0.1685, group=charging, n=450
- mActivity_m_activity_value_7_count: spearman=-0.1543, group=activity_code, n=450
- mActivity_m_activity_value_7_ratio: spearman=-0.1472, group=activity_code, n=450
- mActivity_m_activity_value_4_ratio: spearman=0.1185, group=activity_code, n=450
- mActivity_m_activity_value_4_count: spearman=0.1156, group=activity_code, n=450
- mScreenStatus_m_screen_use_value_0_count: spearman=-0.1143, group=screen, n=450
- mACStatus_m_charging_value_0_count: spearman=0.1137, group=charging, n=450
- mLight_m_light_q25: spearman=0.1134, group=light, n=450

#### Top coverage-proxy correlation features

- mScreenStatus_m_screen_use_nonnull_count: spearman=-0.1471, group=screen, n=450
- mScreenStatus_row_count: spearman=-0.1471, group=screen, n=450
- mACStatus_m_charging_nonnull_count: spearman=-0.1236, group=charging, n=450
- mACStatus_row_count: spearman=-0.1236, group=charging, n=450
- mActivity_m_activity_nonnull_count: spearman=-0.1072, group=activity_code, n=450

#### Top subject-adjusted actual-behavior correlation features

- wPedo_distance_sum: subject_adjusted_spearman=-0.1358, group=activity_pedo, n=404
- wPedo_speed_sum: subject_adjusted_spearman=-0.1358, group=activity_pedo, n=404
- mActivity_m_activity_value_0_count: subject_adjusted_spearman=-0.1353, group=activity_code, n=450
- wLight_w_light_min: subject_adjusted_spearman=0.1346, group=light, n=414
- wPedo_burned_calories_q95: subject_adjusted_spearman=-0.1330, group=activity_pedo, n=404
- wPedo_step_frequency_sum: subject_adjusted_spearman=-0.1325, group=activity_pedo, n=404
- wPedo_step_sum: subject_adjusted_spearman=-0.1325, group=activity_pedo, n=404
- wPedo_distance_q95: subject_adjusted_spearman=-0.1285, group=activity_pedo, n=404
- wPedo_speed_q95: subject_adjusted_spearman=-0.1285, group=activity_pedo, n=404
- wPedo_step_frequency_q95: subject_adjusted_spearman=-0.1265, group=activity_pedo, n=404

### Q3

#### Top actual-behavior correlation features

- mLight_m_light_median: spearman=-0.2204, group=light, n=450
- wLight_w_light_q75: spearman=-0.1884, group=light, n=414
- mLight_m_light_mean: spearman=-0.1744, group=light, n=450
- mLight_m_light_q75: spearman=-0.1667, group=light, n=450
- mLight_m_light_sum: spearman=-0.1574, group=light, n=450
- wLight_w_light_q25: spearman=-0.1509, group=light, n=414
- mLight_m_light_std: spearman=-0.1399, group=light, n=450
- mLight_m_light_max: spearman=-0.1352, group=light, n=450
- wLight_w_light_median: spearman=-0.1329, group=light, n=414
- wPedo_burned_calories_q99: spearman=-0.1156, group=activity_pedo, n=404

#### Top coverage-proxy correlation features

- mLight_m_light_zero_count: spearman=0.1845, group=light, n=450
- mLight_m_light_nonzero_count: spearman=-0.1308, group=light, n=450
- wHr_heart_rate_object_len_nonnull_count: spearman=-0.1166, group=heart_rate, n=391
- wHr_row_count: spearman=-0.1166, group=heart_rate, n=391
- mLight_m_light_nonnull_count: spearman=0.1136, group=light, n=450

#### Top subject-adjusted actual-behavior correlation features

- wPedo_burned_calories_q75: subject_adjusted_spearman=0.1695, group=activity_pedo, n=404
- wLight_w_light_q25: subject_adjusted_spearman=-0.1599, group=light, n=414
- mLight_m_light_max: subject_adjusted_spearman=-0.1557, group=light, n=450
- mLight_m_light_q25: subject_adjusted_spearman=-0.1502, group=light, n=450
- mLight_m_light_median: subject_adjusted_spearman=-0.1353, group=light, n=450
- mLight_m_light_std: subject_adjusted_spearman=-0.1340, group=light, n=450
- wPedo_step_frequency_q75: subject_adjusted_spearman=0.1214, group=activity_pedo, n=404
- wPedo_step_q75: subject_adjusted_spearman=0.1214, group=activity_pedo, n=404
- mActivity_m_activity_value_7_ratio: subject_adjusted_spearman=-0.1202, group=activity_code, n=450
- mLight_m_light_mean: subject_adjusted_spearman=-0.1177, group=light, n=450

### S1

#### Top actual-behavior correlation features

- mScreenStatus_m_screen_use_value_0_count: spearman=0.2722, group=screen, n=450
- mScreenStatus_m_screen_use_value_0_ratio: spearman=0.2355, group=screen, n=450
- mScreenStatus_m_screen_use_value_1_ratio: spearman=-0.2355, group=screen, n=450
- mScreenStatus_m_screen_use_value_1_count: spearman=-0.2082, group=screen, n=450
- mLight_m_light_q25: spearman=-0.1438, group=light, n=450
- mActivity_m_activity_value_4_ratio: spearman=-0.1381, group=activity_code, n=450
- mActivity_m_activity_value_4_count: spearman=-0.1365, group=activity_code, n=450
- wLight_w_light_q25: spearman=-0.1257, group=light, n=414
- mLight_m_light_median: spearman=-0.1225, group=light, n=450
- wPedo_burned_calories_q99: spearman=-0.1138, group=activity_pedo, n=404

#### Top coverage-proxy correlation features

- mLight_m_light_zero_count: spearman=0.2136, group=light, n=450
- mBle_m_ble_object_len_nonnull_count: spearman=-0.1855, group=coverage_or_object_proxy, n=416
- mBle_row_count: spearman=-0.1855, group=coverage_or_object_proxy, n=416
- mLight_m_light_nonzero_count: spearman=-0.1715, group=light, n=450
- mAmbience_m_ambience_object_len_nonnull_count: spearman=0.1245, group=coverage_or_object_proxy, n=450

#### Top subject-adjusted actual-behavior correlation features

- wPedo_distance_q75: subject_adjusted_spearman=0.1905, group=activity_pedo, n=404
- wPedo_speed_q75: subject_adjusted_spearman=0.1905, group=activity_pedo, n=404
- wPedo_step_frequency_q75: subject_adjusted_spearman=0.1656, group=activity_pedo, n=404
- wPedo_step_q75: subject_adjusted_spearman=0.1656, group=activity_pedo, n=404
- mLight_m_light_min: subject_adjusted_spearman=-0.1282, group=light, n=450
- mScreenStatus_m_screen_use_value_0_ratio: subject_adjusted_spearman=0.1262, group=screen, n=450
- mScreenStatus_m_screen_use_value_1_ratio: subject_adjusted_spearman=-0.1262, group=screen, n=450
- mScreenStatus_m_screen_use_value_0_count: subject_adjusted_spearman=0.1233, group=screen, n=450
- wPedo_burned_calories_q75: subject_adjusted_spearman=0.1182, group=activity_pedo, n=404
- mScreenStatus_m_screen_use_value_1_count: subject_adjusted_spearman=-0.1046, group=screen, n=450

### S2

#### Top actual-behavior correlation features

- mScreenStatus_m_screen_use_value_0_count: spearman=0.2204, group=screen, n=450
- mScreenStatus_m_screen_use_value_0_ratio: spearman=0.1852, group=screen, n=450
- mScreenStatus_m_screen_use_value_1_ratio: spearman=-0.1852, group=screen, n=450
- mScreenStatus_m_screen_use_value_1_count: spearman=-0.1642, group=screen, n=450
- wPedo_burned_calories_q99: spearman=-0.1182, group=activity_pedo, n=404
- mLight_m_light_max: spearman=-0.1156, group=light, n=450
- mLight_m_light_std: spearman=-0.1024, group=light, n=450
- wPedo_burned_calories_q95: spearman=-0.1004, group=activity_pedo, n=404
- mLight_m_light_q99: spearman=-0.0996, group=light, n=450
- mACStatus_m_charging_value_1_count: spearman=0.0898, group=charging, n=450

#### Top coverage-proxy correlation features

- mBle_m_ble_object_len_nonnull_count: spearman=-0.1743, group=coverage_or_object_proxy, n=416
- mBle_row_count: spearman=-0.1743, group=coverage_or_object_proxy, n=416
- mWifi_m_wifi_object_len_nonnull_count: spearman=0.1671, group=coverage_or_object_proxy, n=439
- mWifi_row_count: spearman=0.1671, group=coverage_or_object_proxy, n=439
- mGps_m_gps_object_len_nonnull_count: spearman=0.1152, group=coverage_or_object_proxy, n=425

#### Top subject-adjusted actual-behavior correlation features

- wPedo_distance_q75: subject_adjusted_spearman=0.1335, group=activity_pedo, n=404
- wPedo_speed_q75: subject_adjusted_spearman=0.1335, group=activity_pedo, n=404
- wPedo_burned_calories_q75: subject_adjusted_spearman=0.1280, group=activity_pedo, n=404
- wLight_w_light_min: subject_adjusted_spearman=0.1193, group=light, n=414
- wPedo_step_frequency_q75: subject_adjusted_spearman=0.1103, group=activity_pedo, n=404
- wPedo_step_q75: subject_adjusted_spearman=0.1103, group=activity_pedo, n=404
- mActivity_m_activity_value_8_count: subject_adjusted_spearman=0.0767, group=activity_code, n=450
- mActivity_m_activity_value_8_ratio: subject_adjusted_spearman=0.0747, group=activity_code, n=450
- mActivity_m_activity_value_4_ratio: subject_adjusted_spearman=0.0741, group=activity_code, n=450
- wLight_w_light_q25: subject_adjusted_spearman=-0.0665, group=light, n=414

### S3

#### Top actual-behavior correlation features

- mLight_m_light_median: spearman=0.1245, group=light, n=450
- mScreenStatus_m_screen_use_value_0_count: spearman=0.1146, group=screen, n=450
- wLight_w_light_median: spearman=0.0929, group=light, n=414
- mActivity_m_activity_value_3_count: spearman=0.0908, group=activity_code, n=450
- wLight_w_light_q25: spearman=0.0894, group=light, n=414
- mActivity_m_activity_value_4_ratio: spearman=-0.0882, group=activity_code, n=450
- mActivity_m_activity_value_4_count: spearman=-0.0864, group=activity_code, n=450
- mScreenStatus_m_screen_use_value_0_ratio: spearman=0.0752, group=screen, n=450
- mScreenStatus_m_screen_use_value_1_ratio: spearman=-0.0752, group=screen, n=450
- wPedo_step_frequency_q75: spearman=-0.0741, group=activity_pedo, n=404

#### Top coverage-proxy correlation features

- wPedo_burned_calories_zero_count: spearman=0.1824, group=activity_pedo, n=404
- wPedo_burned_calories_nonnull_count: spearman=0.1807, group=activity_pedo, n=404
- wPedo_row_count: spearman=0.1807, group=activity_pedo, n=404
- wLight_w_light_nonzero_count: spearman=0.1414, group=light, n=414
- mWifi_m_wifi_object_len_nonnull_count: spearman=0.1078, group=coverage_or_object_proxy, n=439

#### Top subject-adjusted actual-behavior correlation features

- wPedo_distance_q75: subject_adjusted_spearman=0.1187, group=activity_pedo, n=404
- wPedo_speed_q75: subject_adjusted_spearman=0.1187, group=activity_pedo, n=404
- wPedo_step_frequency_q75: subject_adjusted_spearman=0.1036, group=activity_pedo, n=404
- wPedo_step_q75: subject_adjusted_spearman=0.1036, group=activity_pedo, n=404
- mACStatus_m_charging_value_0_ratio: subject_adjusted_spearman=0.0909, group=charging, n=450
- mACStatus_m_charging_value_1_ratio: subject_adjusted_spearman=-0.0909, group=charging, n=450
- wPedo_burned_calories_max: subject_adjusted_spearman=0.0906, group=activity_pedo, n=404
- wLight_w_light_q25: subject_adjusted_spearman=0.0894, group=light, n=414
- wPedo_burned_calories_sum: subject_adjusted_spearman=0.0842, group=activity_pedo, n=404
- wLight_w_light_q75: subject_adjusted_spearman=0.0797, group=light, n=414

### S4

#### Top actual-behavior correlation features

- mActivity_m_activity_value_8_count: spearman=0.1359, group=activity_code, n=450
- mActivity_m_activity_value_8_ratio: spearman=0.1348, group=activity_code, n=450
- wPedo_step_sum: spearman=-0.1265, group=activity_pedo, n=404
- wPedo_step_frequency_sum: spearman=-0.1265, group=activity_pedo, n=404
- wPedo_burned_calories_q95: spearman=-0.1243, group=activity_pedo, n=404
- mScreenStatus_m_screen_use_value_1_count: spearman=-0.1221, group=screen, n=450
- mScreenStatus_m_screen_use_value_0_ratio: spearman=0.1206, group=screen, n=450
- mScreenStatus_m_screen_use_value_1_ratio: spearman=-0.1206, group=screen, n=450
- wPedo_step_frequency_mean: spearman=-0.1202, group=activity_pedo, n=404
- wPedo_step_mean: spearman=-0.1202, group=activity_pedo, n=404

#### Top coverage-proxy correlation features

- wPedo_burned_calories_nonzero_count: spearman=-0.1372, group=activity_pedo, n=404
- mUsageStats_m_usage_stats_object_len_nonnull_count: spearman=-0.1365, group=app_usage, n=440
- mUsageStats_row_count: spearman=-0.1365, group=app_usage, n=440
- mBle_m_ble_object_len_nonnull_count: spearman=-0.1349, group=coverage_or_object_proxy, n=416
- mBle_row_count: spearman=-0.1349, group=coverage_or_object_proxy, n=416

#### Top subject-adjusted actual-behavior correlation features

- mActivity_m_activity_value_0_ratio: subject_adjusted_spearman=0.1173, group=activity_code, n=450
- mActivity_m_activity_value_0_count: subject_adjusted_spearman=0.1045, group=activity_code, n=450
- wLight_w_light_q99: subject_adjusted_spearman=0.0885, group=light, n=414
- wLight_w_light_min: subject_adjusted_spearman=0.0787, group=light, n=414
- wLight_w_light_std: subject_adjusted_spearman=0.0774, group=light, n=414
- wLight_w_light_max: subject_adjusted_spearman=0.0711, group=light, n=414
- mActivity_m_activity_value_7_count: subject_adjusted_spearman=-0.0694, group=activity_code, n=450
- wLight_w_light_q75: subject_adjusted_spearman=-0.0643, group=light, n=414
- mScreenStatus_m_screen_use_value_1_count: subject_adjusted_spearman=-0.0623, group=screen, n=450
- mLight_m_light_min: subject_adjusted_spearman=0.0616, group=light, n=450

## 6. Timeblock Feature-Target Top Signals

### Q1

- mLight_m_light_04_08_median: spearman=-0.2070, group=light, n=431
- mLight_m_light_04_08_q75: spearman=-0.1992, group=light, n=431
- mLight_m_light_04_08_sum: spearman=-0.1919, group=light, n=431
- mLight_m_light_04_08_mean: spearman=-0.1919, group=light, n=431
- mLight_m_light_04_08_q95: spearman=-0.1862, group=light, n=431
- wLight_w_light_04_08_mean: spearman=-0.1830, group=light, n=338
- wLight_w_light_04_08_std: spearman=-0.1671, group=light, n=338
- wPedo_burned_calories_20_24_q95: spearman=0.1664, group=activity_pedo, n=355
- wLight_w_light_04_08_q99: spearman=-0.1614, group=light, n=338
- mLight_m_light_04_08_std: spearman=-0.1607, group=light, n=431

### Q2

- mActivity_m_activity_08_12_value_7_count: spearman=-0.1869, group=activity_code, n=442
- mActivity_m_activity_08_12_value_7_ratio: spearman=-0.1843, group=activity_code, n=442
- mLight_m_light_20_24_mean: spearman=0.1812, group=light, n=442
- mActivity_m_activity_04_08_value_7_ratio: spearman=-0.1697, group=activity_code, n=432
- mActivity_m_activity_04_08_value_7_count: spearman=-0.1689, group=activity_code, n=432
- mScreenStatus_m_screen_use_12_16_value_0_count: spearman=-0.1664, group=screen, n=446
- mActivity_m_activity_08_12_value_4_ratio: spearman=0.1656, group=activity_code, n=442
- mLight_m_light_20_24_sum: spearman=0.1651, group=light, n=442
- mActivity_m_activity_08_12_value_4_count: spearman=0.1634, group=activity_code, n=442
- wPedo_distance_04_08_q99: spearman=-0.1627, group=activity_pedo, n=329

### Q3

- mLight_m_light_12_16_q25: spearman=-0.2076, group=light, n=447
- mLight_m_light_12_16_median: spearman=-0.1785, group=light, n=447
- mLight_m_light_00_04_q75: spearman=-0.1747, group=light, n=432
- mLight_m_light_00_04_mean: spearman=-0.1742, group=light, n=432
- wLight_w_light_12_16_q25: spearman=-0.1730, group=light, n=397
- mLight_m_light_00_04_q95: spearman=-0.1719, group=light, n=432
- mLight_m_light_20_24_max: spearman=0.1688, group=light, n=442
- mLight_m_light_00_04_std: spearman=-0.1685, group=light, n=430
- mLight_m_light_08_12_q25: spearman=-0.1676, group=light, n=442
- mLight_m_light_20_24_q99: spearman=0.1657, group=light, n=442

### S1

- wPedo_distance_00_04_q99: spearman=-0.2787, group=activity_pedo, n=322
- wPedo_speed_00_04_q99: spearman=-0.2787, group=activity_pedo, n=322
- wPedo_step_00_04_q99: spearman=-0.2785, group=activity_pedo, n=322
- wPedo_step_frequency_00_04_q99: spearman=-0.2785, group=activity_pedo, n=322
- wPedo_burned_calories_00_04_max: spearman=-0.2508, group=activity_pedo, n=322
- wPedo_burned_calories_00_04_std: spearman=-0.2503, group=activity_pedo, n=317
- wPedo_distance_00_04_max: spearman=-0.2496, group=activity_pedo, n=322
- wPedo_speed_00_04_max: spearman=-0.2496, group=activity_pedo, n=322
- wPedo_burned_calories_00_04_mean: spearman=-0.2495, group=activity_pedo, n=322
- wPedo_burned_calories_00_04_sum: spearman=-0.2495, group=activity_pedo, n=322

### S2

- mScreenStatus_m_screen_use_20_24_value_0_ratio: spearman=0.1588, group=screen, n=443
- mScreenStatus_m_screen_use_20_24_value_1_ratio: spearman=-0.1588, group=screen, n=443
- mScreenStatus_m_screen_use_20_24_value_1_count: spearman=-0.1565, group=screen, n=443
- mScreenStatus_m_screen_use_20_24_value_0_count: spearman=0.1384, group=screen, n=443
- mScreenStatus_m_screen_use_08_12_value_0_count: spearman=0.1279, group=screen, n=442
- mScreenStatus_m_screen_use_16_20_value_0_ratio: spearman=0.1240, group=screen, n=444
- mScreenStatus_m_screen_use_16_20_value_1_ratio: spearman=-0.1240, group=screen, n=444
- mScreenStatus_m_screen_use_16_20_value_1_count: spearman=-0.1235, group=screen, n=444
- mScreenStatus_m_screen_use_16_20_value_0_count: spearman=0.1208, group=screen, n=444
- wPedo_burned_calories_08_12_std: spearman=-0.1180, group=activity_pedo, n=382

### S3

- wLight_w_light_00_04_max: spearman=0.1470, group=light, n=345
- mLight_m_light_04_08_median: spearman=0.1410, group=light, n=431
- mACStatus_m_charging_08_12_value_0_count: spearman=0.1409, group=charging, n=442
- mActivity_m_activity_12_16_value_3_count: spearman=0.1341, group=activity_code, n=446
- wLight_w_light_00_04_q99: spearman=0.1312, group=light, n=345
- mActivity_m_activity_12_16_value_3_ratio: spearman=0.1306, group=activity_code, n=446
- wLight_w_light_00_04_std: spearman=0.1281, group=light, n=345
- mACStatus_m_charging_16_20_value_1_count: spearman=-0.1165, group=charging, n=444
- mACStatus_m_charging_16_20_value_0_ratio: spearman=0.1161, group=charging, n=444
- mACStatus_m_charging_16_20_value_1_ratio: spearman=-0.1161, group=charging, n=444

### S4

- mACStatus_m_charging_04_08_value_0_ratio: spearman=-0.1896, group=charging, n=432
- mACStatus_m_charging_04_08_value_1_ratio: spearman=0.1896, group=charging, n=432
- mACStatus_m_charging_04_08_value_1_count: spearman=0.1889, group=charging, n=432
- mACStatus_m_charging_04_08_value_0_count: spearman=-0.1838, group=charging, n=432
- wLight_w_light_04_08_q75: spearman=0.1617, group=light, n=338
- mLight_m_light_00_04_max: spearman=0.1597, group=light, n=432
- mLight_m_light_00_04_q99: spearman=0.1581, group=light, n=432
- wLight_w_light_04_08_q95: spearman=0.1560, group=light, n=338
- mLight_m_light_00_04_std: spearman=0.1539, group=light, n=430
- mScreenStatus_m_screen_use_12_16_value_0_count: spearman=0.1478, group=screen, n=446

## 7. Interpretation Notes

- `timestamp_object_len_*` 계열은 생성하지 않습니다. timestamp는 센서값이 아니라 메타데이터입니다.
- `row_count`, `nonnull_count`, `object_len` 계열은 실제 행동값이 아니라 sensor coverage 또는 수집량 proxy입니다.
- coverage proxy는 제거 대상이 아니라 별도 feature group으로 관리해야 합니다. 단, 실제 행동 feature와 섞어서 해석하지 않습니다.
- `zero_ratio`가 1.0에 가까운 feature와 constant feature는 log-transform 후보에서 제외했습니다.
- raw correlation보다 subject-adjusted correlation에서 유지되는 actual-behavior feature를 우선 feature engineering 대상으로 삼습니다.
- timeblock feature는 4시간 단위 기준입니다. S2/S3는 20_24, 00_04, 04_08 구간을 우선 검토하십시오.

## 8. Next Modeling Actions

1. actual_behavior feature와 coverage_proxy feature를 분리한 feature set으로 LightGBM CV 비교
2. log1p 후보 feature는 원본/변환 버전을 모두 만들고 CV log-loss 기준으로 선택
3. subject-adjusted correlation 상위 feature는 subject mean/deviation/z-score로 재생성
4. 야간/수면 전 timeblock feature는 target별 feature set에 우선 포함
5. coverage proxy feature는 missing indicator 그룹으로 분리해 SHAP/importance를 따로 해석
6. S2/S3는 16:00~다음날 16:00 analysis-day 기준 feature를 별도 실험
