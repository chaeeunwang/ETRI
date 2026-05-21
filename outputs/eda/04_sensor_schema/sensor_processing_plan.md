# Step 04. Sensor Schema EDA Summary

## 1. Sensor File Overview

- 총 센서 파일 수: 12
- 읽기 성공 파일 수: 12
- 읽기 실패 파일 수: 0

## 2. Processing Type Counts

- object/list: 6
- discrete: 3
- continuous: 2
- continuous/multi-value: 1

## 3. High Priority Sensors

- mACStatus: discrete, rows=939896, freq=approx_1min
- mActivity: discrete, rows=961062, freq=approx_1min
- mLight: continuous, rows=96258, freq=approx_10min
- mScreenStatus: discrete, rows=939653, freq=approx_1min
- mUsageStats: object/list, rows=45197, freq=approx_10min
- wHr: object/list, rows=382918, freq=approx_1min
- wLight: continuous, rows=633741, freq=approx_1min
- wPedo: continuous/multi-value, rows=748100, freq=approx_1min

## 4. Low Priority / High Noise Sensors

- mAmbience: object/list, rows=476577, freq=approx_2min
- mBle: object/list, rows=21830, freq=approx_10min
- mGps: object/list, rows=800611, freq=approx_1min
- mWifi: object/list, rows=76336, freq=approx_10min

## 5. Recommended Initial Processing Plan

| Sensor | Raw File | Priority | Type | Value Columns | Recommended Processing |
|---|---|---:|---|---|---|
| mACStatus | ch2025_mACStatus.parquet | high-priority | discrete | m_charging | charging duration/count; night charging flag; timeblock ratio |
| mActivity | ch2025_mActivity.parquet | high-priority | discrete | m_activity | activity code count/duration/ratio by daily and time block |
| mLight | ch2025_mLight.parquet | high-priority | continuous | m_light | daily/timeblock mean/max; night light mean; bright event count |
| mScreenStatus | ch2025_mScreenStatus.parquet | high-priority | discrete | m_screen_use | screen-on duration/count; night screen use; pre-sleep screen use |
| mUsageStats | ch2025_mUsageStats.parquet | high-priority | object/list | m_usage_stats | parse app usage; total/night/pre-sleep usage; top-k app/category usage |
| wHr | ch2025_wHr.parquet | high-priority | object/list | heart_rate | parse heart-rate list; mean/std/min/max/night mean + missing indicator |
| wLight | ch2025_wLight.parquet | high-priority | continuous | w_light | daily/timeblock mean/max; night light mean; bright event count |
| wPedo | ch2025_wPedo.parquet | high-priority | continuous/multi-value | step, step_frequency, running_step, walking_step, distance, speed, burned_calories | aggregate steps/calories/distance/speed; daily/timeblock sum/mean/max |
| mAmbience | ch2025_mAmbience.parquet | low-priority/high-noise | object/list | m_ambience | defer initially; optionally top sound labels/count/probability summary |
| mBle | ch2025_mBle.parquet | low-priority/high-noise | object/list | m_ble | defer initially; optionally device count and RSSI summary |
| mGps | ch2025_mGps.parquet | low-priority/high-noise | object/list | m_gps | defer initially; optionally parse speed/altitude and movement coverage |
| mWifi | ch2025_mWifi.parquet | low-priority/high-noise | object/list | m_wifi | defer initially; optionally AP count and RSSI summary |

## 6. Manual Inspection Targets

- manual inspection required 센서 없음

## 7. Column Dictionary Notes

- `subject_id`: subject identifier
- `timestamp`: sensor observation time
- `m_charging`: smartphone charging status
- `m_activity`: smartphone activity code
- `m_screen_use`: smartphone screen-use status
- `m_usage_stats`: app usage list, not timestamp
- `heart_rate`: smartwatch heart-rate list
- `step`, `distance`, `speed`, `burned_calories`: pedometer values

## 8. Files Generated

- `sensor_file_list.csv`
- `sensor_column_dictionary.csv`
- `sensor_dtype_summary.csv`
- `sensor_row_count.csv`
- `sensor_processing_plan.md`