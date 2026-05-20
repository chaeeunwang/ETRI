# Subject Distribution EDA Summary

## 1. 생성 목적

본 분석의 목적은 subject별 데이터 규모, 날짜 범위, 날짜 연속성, target prior를 확인하여 현재 sequence 기반 BiLSTM 구조가 적절한지 판단하는 것이다.

특히 현재 모델이 `lookback=14`를 사용하고 있으므로, 각 subject가 최소 14일 이상의 연속 관측 구간을 충분히 보유하고 있는지 확인하였다. 또한 subject별 target prior 차이를 분석하여 subject embedding, personalized feature, subject-aware validation의 필요성을 점검하였다.

---

## 2. Subject별 row 수

### 2.1 전체 요약

- train row 최소값: 33
- train row 최대값: 57
- train row 표준편차: 8.300

subject별 train row 수에는 일정 수준의 차이가 존재한다. 특히 일부 subject는 학습 데이터가 30일대에 불과하므로, 전체 모델이 데이터가 많은 subject 중심으로 학습될 가능성이 있다.

### 2.2 train row가 적은 subject TOP3

| subject_id | train_rows | test_rows |
| ---------- | ---------: | --------: |
| id03       |         33 |        21 |
| id10       |         33 |        22 |
| id09       |         41 |        27 |

### 2.3 판단

id03, id10은 train row가 33개로 가장 적다. 이들은 sequence 모델에서 학습 가능한 window 수 자체가 적을 가능성이 높다. 따라서 단순 K-Fold 또는 무작위 sampling만으로는 subject별 학습 안정성을 보장하기 어렵다.

후속 실험에서는 subject별 성능을 반드시 분리해서 확인해야 하며, 특정 subject에서만 성능이 급락하는지 확인할 필요가 있다.

---

## 3. 날짜 연속성

### 3.1 전체 요약

- train에서 날짜 gap이 있는 subject 수: 10
- train에서 `max_consecutive_days < 14`인 subject 수: 5

모든 subject에서 날짜 gap이 존재한다. 즉, lifelog_date가 완전히 연속적인 시계열로 제공되지 않는다.

### 3.2 lookback=14 위험 subject

| subject_id | max_consecutive_days | missing_days_count | max_gap_days |
| ---------- | -------------------: | -----------------: | -----------: |
| id03       |                    6 |                 25 |           14 |
| id05       |                   10 |                 35 |           12 |
| id06       |                   13 |                 29 |            9 |
| id09       |                   11 |                 24 |           17 |
| id10       |                    5 |                 38 |           25 |

### 3.3 판단

현재 `lookback=14`는 그대로 사용하기 어렵다.

전체 subject 10명 중 5명이 14일 연속 구간을 확보하지 못한다. 특히 id03과 id10은 최대 연속 구간이 각각 6일, 5일에 불과하다. 이 상태에서 14일 window를 강제로 구성하면 모델은 실제 14일 생활 패턴을 학습하는 것이 아니라, 불연속 날짜, padding, 결측 패턴을 함께 학습할 가능성이 높다.

따라서 현재 BiLSTM의 `lookback=14`는 과한 설정일 가능성이 크다.

### 3.4 후속 조치

lookback 길이를 고정하지 말고 아래 후보를 비교해야 한다.

| 실험 | lookback | 목적                              |
| ---- | -------: | --------------------------------- |
| A    |       14 | 기존 baseline 유지                |
| B    |        7 | 1주일 단위 생활 패턴 확인         |
| C    |        5 | id03, id10까지 고려한 현실적 후보 |
| D    |        3 | 짧은 시계열 기준 안전한 비교군    |

현재 EDA 결과 기준으로는 `lookback=5` 또는 `lookback=7`이 가장 현실적인 후보이다.

---

## 4. Sequence window 품질 검증 필요성

현재 분석은 subject별 최대 연속 일수와 missing date를 확인한 단계이다. 그러나 실제 BiLSTM 입력 window가 얼마나 유효한지는 아직 별도 검증이 필요하다.

따라서 다음 파일을 추가 생성해야 한다.

```text
outputs/eda/02_subject_distribution/
├── sequence_window_quality.csv
└── sequence_window_quality_summary.md
```
