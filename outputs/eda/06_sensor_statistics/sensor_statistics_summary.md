# Sensor Statistics Summary

## continuous
- rows: 9

## discrete
- rows: 10

## object
- rows: 6

## outlier
- rows: 9

## log
- rows: 9

## Log Transform Candidates
- mLight.m_light: long_tail_q99_over_10x_median
- wLight.w_light: long_tail_q99_over_10x_median
- wPedo.step: high_skew
- wPedo.step_frequency: high_skew
- wPedo.distance: high_skew
- wPedo.speed: high_skew
- wPedo.burned_calories: high_skew

## Top Outlier Features
- mLight.m_light: outlier_ratio=0.1209
- wLight.w_light: outlier_ratio=0.1132
- wPedo.step: outlier_ratio=0.0000
- wPedo.step_frequency: outlier_ratio=0.0000
- wPedo.running_step: outlier_ratio=0.0000
- wPedo.walking_step: outlier_ratio=0.0000
- wPedo.distance: outlier_ratio=0.0000
- wPedo.speed: outlier_ratio=0.0000
- wPedo.burned_calories: outlier_ratio=0.0000
