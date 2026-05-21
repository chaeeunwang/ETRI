# Step 04. Sensor Schema EDA Summary

## 핵심 결과

- 총 12개 센서 파일을 확인했으며, 전부 정상 로딩되었다.
- 센서 유형은 `object/list` 6개, `discrete` 3개, `continuous` 2개, `continuous/multi-value` 1개로 분류되었다.
- manual inspection이 필요한 센서는 없다.

## 1차 처리 대상 센서

| Sensor        | Type                   |  Freq | Value Column                              | Feature 방향                                       |
| ------------- | ---------------------- | ----: | ----------------------------------------- | -------------------------------------------------- |
| mACStatus     | discrete               |  1min | m_charging                                | 충전 시간, 야간 충전 여부                          |
| mActivity     | discrete               |  1min | m_activity                                | 활동 code별 count/duration/ratio                   |
| mScreenStatus | discrete               |  1min | m_screen_use                              | 화면 사용 시간, 야간/취침 전 screen-on             |
| mLight        | continuous             | 10min | m_light                                   | 조도 평균/최대, 야간 밝기                          |
| wLight        | continuous             |  1min | w_light                                   | 조도 평균/최대, 야간 밝기                          |
| wPedo         | continuous/multi-value |  1min | step, distance, speed, burned_calories 등 | 걸음수/거리/칼로리/속도 집계                       |
| wHr           | object/list            |  1min | heart_rate                                | 심박 mean/std/min/max, 야간 심박                   |
| mUsageStats   | object/list            | 10min | m_usage_stats                             | 앱 사용량, 야간/취침 전 사용량, top-k app/category |

## 후순위 센서

| Sensor    | Type        |  Freq | Value Column | 처리 방향                                           |
| --------- | ----------- | ----: | ------------ | --------------------------------------------------- |
| mAmbience | object/list |  2min | m_ambience   | 초기 제외, 필요 시 sound label count/probability    |
| mBle      | object/list | 10min | m_ble        | 초기 제외, 필요 시 device count/RSSI                |
| mGps      | object/list |  1min | m_gps        | 초기 제외, 필요 시 speed/altitude/movement coverage |
| mWifi     | object/list | 10min | m_wifi       | 초기 제외, 필요 시 AP count/RSSI                    |

## 결론

초기 feature engineering은 `wPedo`, `wHr`, `mUsageStats`, `mScreenStatus`, `mACStatus`, `mActivity`, `mLight`, `wLight` 중심으로 진행한다.

`mGps`, `mBle`, `mWifi`, `mAmbience`는 object/list 구조이고 noise 또는 sparsity 가능성이 높으므로 초기 모델에서는 후순위로 둔다.

다음 단계에서는 subject/date/hour 기준으로 센서별 missing ratio를 계산해 보간, masking, missing indicator 추가 여부를 결정한다.
