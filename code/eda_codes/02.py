# src/eda/02_subject_distribution.py

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================
# Config
# =========================

DATA_DIR = Path("./data")
OUTPUT_DIR = Path("./outputs/eda/02_subject_distribution")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_PATH = DATA_DIR / "ch2026_metrics_train.csv"
TEST_PATH = DATA_DIR / "ch2026_submission_sample.csv"

SUBJECT_COL = "subject_id"
LIFELOG_DATE_COL = "lifelog_date"
SLEEP_DATE_COL = "sleep_date"

LOOKBACK = 14

# 실제 데이터 column 기준으로 자동 필터링합니다.
CANDIDATE_TARGETS = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]


# =========================
# Utility
# =========================

def load_data() -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    for df in [train_df, test_df]:
        for col in [LIFELOG_DATE_COL, SLEEP_DATE_COL]:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")

    targets = [col for col in CANDIDATE_TARGETS if col in train_df.columns]

    required_cols = [SUBJECT_COL, LIFELOG_DATE_COL]
    for col in required_cols:
        if col not in train_df.columns:
            raise ValueError(f"train_df에 필수 컬럼이 없습니다: {col}")
        if col not in test_df.columns:
            raise ValueError(f"test_df에 필수 컬럼이 없습니다: {col}")

    return train_df, test_df, targets


def calc_date_gaps(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    """
    subject별 날짜 연속성 확인:
    - observed_days: 실제 row가 있는 날짜 수
    - expected_days: min~max까지 모든 날짜 수
    - missing_days_count: 중간에 비어 있는 날짜 수
    - max_gap_days: 관측 날짜 사이 최대 gap
    - continuous_segments: 날짜가 끊길 때마다 segment 증가
    - max_consecutive_days: 가장 긴 연속 구간 길이
    """
    rows = []

    for subject, g in df.groupby(SUBJECT_COL):
        dates = (
            g[LIFELOG_DATE_COL]
            .dropna()
            .dt.normalize()
            .drop_duplicates()
            .sort_values()
            .reset_index(drop=True)
        )

        if len(dates) == 0:
            rows.append({
                "dataset": dataset_name,
                SUBJECT_COL: subject,
                "observed_days": 0,
                "date_min": pd.NaT,
                "date_max": pd.NaT,
                "expected_days": 0,
                "missing_days_count": np.nan,
                "missing_days_ratio": np.nan,
                "max_gap_days": np.nan,
                "gap_over_1day_count": np.nan,
                "continuous_segments": np.nan,
                "max_consecutive_days": 0,
                "lookback_14_feasible": False,
            })
            continue

        date_min = dates.min()
        date_max = dates.max()
        expected_range = pd.date_range(date_min, date_max, freq="D")
        observed_set = set(dates)
        missing_dates = [d for d in expected_range if d not in observed_set]

        diffs = dates.diff().dt.days.dropna()
        max_gap_days = int(diffs.max()) if len(diffs) > 0 else 0
        gap_over_1day_count = int((diffs > 1).sum()) if len(diffs) > 0 else 0
        continuous_segments = gap_over_1day_count + 1 if len(dates) > 0 else 0

        # 가장 긴 연속 일수 계산
        max_consecutive = 1
        current = 1

        for diff in diffs:
            if diff == 1:
                current += 1
            else:
                max_consecutive = max(max_consecutive, current)
                current = 1

        max_consecutive = max(max_consecutive, current)

        rows.append({
            "dataset": dataset_name,
            SUBJECT_COL: subject,
            "observed_days": int(len(dates)),
            "date_min": date_min.date(),
            "date_max": date_max.date(),
            "expected_days": int(len(expected_range)),
            "missing_days_count": int(len(missing_dates)),
            "missing_days_ratio": len(missing_dates) / len(expected_range) if len(expected_range) > 0 else np.nan,
            "max_gap_days": max_gap_days,
            "gap_over_1day_count": gap_over_1day_count,
            "continuous_segments": continuous_segments,
            "max_consecutive_days": int(max_consecutive),
            "lookback_14_feasible": bool(max_consecutive >= LOOKBACK),
        })

    return pd.DataFrame(rows)


def calc_missing_dates_detail(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    """
    subject별 실제로 빠진 날짜 목록을 long format으로 저장합니다.
    날짜 gap이 없다면 빈 CSV가 생성됩니다.
    """
    rows = []

    for subject, g in df.groupby(SUBJECT_COL):
        dates = (
            g[LIFELOG_DATE_COL]
            .dropna()
            .dt.normalize()
            .drop_duplicates()
            .sort_values()
        )

        if len(dates) == 0:
            continue

        expected_range = pd.date_range(dates.min(), dates.max(), freq="D")
        observed_set = set(dates)

        for d in expected_range:
            if d not in observed_set:
                rows.append({
                    "dataset": dataset_name,
                    SUBJECT_COL: subject,
                    "missing_date": d.date(),
                })

    return pd.DataFrame(rows)


def calc_row_count(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    train_count = (
        train_df.groupby(SUBJECT_COL)
        .size()
        .reset_index(name="train_rows")
    )

    test_count = (
        test_df.groupby(SUBJECT_COL)
        .size()
        .reset_index(name="test_rows")
    )

    row_count = pd.merge(train_count, test_count, on=SUBJECT_COL, how="outer").fillna(0)
    row_count["train_rows"] = row_count["train_rows"].astype(int)
    row_count["test_rows"] = row_count["test_rows"].astype(int)
    row_count["total_rows"] = row_count["train_rows"] + row_count["test_rows"]
    row_count["test_train_ratio"] = row_count["test_rows"] / row_count["train_rows"].replace(0, np.nan)

    return row_count.sort_values(SUBJECT_COL)


def calc_date_range(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    def summarize(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
        agg_dict = {
            LIFELOG_DATE_COL: ["min", "max", "nunique"],
        }

        if SLEEP_DATE_COL in df.columns:
            agg_dict[SLEEP_DATE_COL] = ["min", "max", "nunique"]

        out = df.groupby(SUBJECT_COL).agg(agg_dict)
        out.columns = ["_".join(col).strip() for col in out.columns.values]
        out = out.reset_index()
        out.insert(0, "dataset", dataset_name)

        return out

    return pd.concat(
        [
            summarize(train_df, "train"),
            summarize(test_df, "test"),
        ],
        axis=0,
        ignore_index=True,
    ).sort_values(["dataset", SUBJECT_COL])


def calc_subject_target_prior(train_df: pd.DataFrame, targets: list[str]) -> pd.DataFrame:
    """
    subject별 target prior 계산:
    - binary target: class별 비율 + mean
    - S1처럼 multiclass target: class별 비율 + mean
    """
    rows = []

    for subject, g in train_df.groupby(SUBJECT_COL):
        for target in targets:
            valid = g[target].dropna()
            total = len(valid)

            if total == 0:
                rows.append({
                    SUBJECT_COL: subject,
                    "target": target,
                    "n_valid": 0,
                    "target_mean": np.nan,
                    "target_std": np.nan,
                    "class": np.nan,
                    "class_count": np.nan,
                    "class_ratio": np.nan,
                })
                continue

            class_counts = valid.value_counts(dropna=False).sort_index()

            for cls, cnt in class_counts.items():
                rows.append({
                    SUBJECT_COL: subject,
                    "target": target,
                    "n_valid": int(total),
                    "target_mean": float(valid.mean()),
                    "target_std": float(valid.std()) if total > 1 else 0.0,
                    "class": cls,
                    "class_count": int(cnt),
                    "class_ratio": float(cnt / total),
                })

    prior_long = pd.DataFrame(rows)

    # long format 그대로 저장하기 좋게 반환
    return prior_long.sort_values([SUBJECT_COL, "target", "class"])


def calc_subject_target_prior_wide(subject_target_prior: pd.DataFrame) -> pd.DataFrame:
    """
    heatmap이나 빠른 확인용 wide format.
    class별 ratio를 column으로 펼칩니다.
    """
    if subject_target_prior.empty:
        return pd.DataFrame()

    wide = subject_target_prior.pivot_table(
        index=[SUBJECT_COL, "target"],
        columns="class",
        values="class_ratio",
        aggfunc="first",
    ).reset_index()

    wide.columns = [
        f"class_{c}_ratio" if isinstance(c, (int, float, np.integer, np.floating)) else str(c)
        for c in wide.columns
    ]

    return wide


def plot_subject_timeline(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """
    subject별 train/test lifelog_date 분포를 한 장에 저장합니다.
    """
    train_plot = train_df[[SUBJECT_COL, LIFELOG_DATE_COL]].copy()
    train_plot["dataset"] = "train"

    test_plot = test_df[[SUBJECT_COL, LIFELOG_DATE_COL]].copy()
    test_plot["dataset"] = "test"

    plot_df = pd.concat([train_plot, test_plot], ignore_index=True)
    plot_df = plot_df.dropna(subset=[LIFELOG_DATE_COL])

    subjects = sorted(plot_df[SUBJECT_COL].dropna().unique())
    subject_to_y = {s: i for i, s in enumerate(subjects)}

    plt.figure(figsize=(14, max(5, len(subjects) * 0.6)))

    for dataset_name, marker in [("train", "o"), ("test", "x")]:
        part = plot_df[plot_df["dataset"] == dataset_name]
        y = part[SUBJECT_COL].map(subject_to_y)
        plt.scatter(
            part[LIFELOG_DATE_COL],
            y,
            s=18,
            marker=marker,
            alpha=0.8,
            label=dataset_name,
        )

    plt.yticks(range(len(subjects)), subjects)
    plt.xlabel("lifelog_date")
    plt.ylabel("subject_id")
    plt.title("Subject Timeline by Dataset")
    plt.grid(axis="x", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "subject_timeline_plot.png", dpi=200)
    plt.close()


def write_summary(
    row_count: pd.DataFrame,
    gap_summary: pd.DataFrame,
    subject_target_prior: pd.DataFrame,
    targets: list[str],
) -> None:
    train_gap = gap_summary[gap_summary["dataset"] == "train"].copy()
    test_gap = gap_summary[gap_summary["dataset"] == "test"].copy()

    min_train_rows = row_count["train_rows"].min() if len(row_count) > 0 else np.nan
    max_train_rows = row_count["train_rows"].max() if len(row_count) > 0 else np.nan
    train_row_std = row_count["train_rows"].std() if len(row_count) > 1 else 0

    subjects_low_train = row_count.sort_values("train_rows").head(3)

    lookback_bad = train_gap[train_gap["max_consecutive_days"] < LOOKBACK]
    gap_problem = train_gap[train_gap["missing_days_count"] > 0]

    # subject별 target prior 변동성: target/class별 class_ratio 표준편차
    prior_variability = pd.DataFrame()
    if not subject_target_prior.empty:
        prior_variability = (
            subject_target_prior
            .groupby(["target", "class"])["class_ratio"]
            .agg(["mean", "std", "min", "max"])
            .reset_index()
            .sort_values(["target", "std"], ascending=[True, False])
        )

    lines = []
    lines.append("# Subject Distribution EDA Summary\n")
    lines.append("## 1. 생성 목적\n")
    lines.append(
        f"- subject별 데이터 수, 날짜 범위, 날짜 연속성, target prior를 확인했습니다.\n"
    )
    lines.append(
        f"- 현재 sequence 모델이 lookback={LOOKBACK}를 사용한다고 가정하고, subject별 최대 연속 일수가 {LOOKBACK}일 이상인지 점검했습니다.\n"
    )

    lines.append("\n## 2. Subject별 row 수\n")
    lines.append(f"- train row 최소값: {min_train_rows}\n")
    lines.append(f"- train row 최대값: {max_train_rows}\n")
    lines.append(f"- train row 표준편차: {train_row_std:.3f}\n")
    lines.append("- train row가 적은 subject TOP3:\n")
    for _, r in subjects_low_train.iterrows():
        lines.append(
            f"  - {r[SUBJECT_COL]}: train_rows={r['train_rows']}, test_rows={r['test_rows']}\n"
        )

    lines.append("\n## 3. 날짜 연속성\n")
    lines.append(f"- train에서 날짜 gap이 있는 subject 수: {gap_problem[SUBJECT_COL].nunique()}\n")
    lines.append(f"- train에서 max_consecutive_days < {LOOKBACK}인 subject 수: {lookback_bad[SUBJECT_COL].nunique()}\n")

    if len(lookback_bad) > 0:
        lines.append("- lookback 위험 subject:\n")
        for _, r in lookback_bad.iterrows():
            lines.append(
                f"  - {r[SUBJECT_COL]}: max_consecutive_days={r['max_consecutive_days']}, "
                f"missing_days_count={r['missing_days_count']}, max_gap_days={r['max_gap_days']}\n"
            )
    else:
        lines.append(f"- 모든 train subject가 최소 1개 이상의 {LOOKBACK}일 연속 구간을 가집니다.\n")

    lines.append("\n## 4. Target prior 차이\n")
    if prior_variability.empty:
        lines.append("- target prior를 계산할 수 없습니다. target column을 확인해야 합니다.\n")
    else:
        lines.append("- subject별 class ratio 변동성이 큰 target/class TOP10:\n")
        top_prior_var = prior_variability.dropna(subset=["std"]).sort_values("std", ascending=False).head(10)
        for _, r in top_prior_var.iterrows():
            lines.append(
                f"  - target={r['target']}, class={r['class']}: "
                f"mean={r['mean']:.3f}, std={r['std']:.3f}, min={r['min']:.3f}, max={r['max']:.3f}\n"
            )

    lines.append("\n## 5. 1차 판단\n")
    lines.append("- subject별 row 수 차이가 크면 subject별 sampling 또는 subject-aware validation을 검토해야 합니다.\n")
    lines.append(f"- max_consecutive_days가 {LOOKBACK}보다 짧은 subject가 있으면 lookback 축소, masking, sequence padding 정책을 재검토해야 합니다.\n")
    lines.append("- subject별 target prior 차이가 크면 subject prior feature, subject embedding, personalized normalization을 검토해야 합니다.\n")
    lines.append("- 날짜 gap이 크면 random split보다 time-aware split을 유지하는 것이 안전합니다.\n")

    lines.append("\n## 6. 생성 파일\n")
    lines.append("- subject_row_count.csv\n")
    lines.append("- subject_date_range.csv\n")
    lines.append("- subject_missing_dates.csv\n")
    lines.append("- subject_timeline_plot.png\n")
    lines.append("- subject_target_prior.csv\n")
    lines.append("- subject_target_prior_wide.csv\n")
    lines.append("- subject_summary.md\n")

    with open(OUTPUT_DIR / "subject_summary.md", "w", encoding="utf-8") as f:
        f.writelines(lines)


# =========================
# Main
# =========================

def main() -> None:
    print("[INFO] Loading data...")
    train_df, test_df, targets = load_data()

    print(f"[INFO] Targets found: {targets}")

    print("[INFO] Calculating subject row count...")
    row_count = calc_row_count(train_df, test_df)
    row_count.to_csv(OUTPUT_DIR / "subject_row_count.csv", index=False, encoding="utf-8-sig")

    print("[INFO] Calculating subject date range...")
    date_range = calc_date_range(train_df, test_df)
    date_range.to_csv(OUTPUT_DIR / "subject_date_range.csv", index=False, encoding="utf-8-sig")

    print("[INFO] Calculating subject missing dates...")
    train_missing_dates = calc_missing_dates_detail(train_df, "train")
    test_missing_dates = calc_missing_dates_detail(test_df, "test")
    missing_dates = pd.concat([train_missing_dates, test_missing_dates], ignore_index=True)
    missing_dates.to_csv(OUTPUT_DIR / "subject_missing_dates.csv", index=False, encoding="utf-8-sig")

    print("[INFO] Calculating date gap summary...")
    train_gap = calc_date_gaps(train_df, "train")
    test_gap = calc_date_gaps(test_df, "test")
    gap_summary = pd.concat([train_gap, test_gap], ignore_index=True)
    gap_summary.to_csv(OUTPUT_DIR / "subject_date_gap_summary.csv", index=False, encoding="utf-8-sig")

    print("[INFO] Calculating subject target prior...")
    subject_target_prior = calc_subject_target_prior(train_df, targets)
    subject_target_prior.to_csv(OUTPUT_DIR / "subject_target_prior.csv", index=False, encoding="utf-8-sig")

    subject_target_prior_wide = calc_subject_target_prior_wide(subject_target_prior)
    subject_target_prior_wide.to_csv(
        OUTPUT_DIR / "subject_target_prior_wide.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("[INFO] Plotting subject timeline...")
    plot_subject_timeline(train_df, test_df)

    print("[INFO] Writing summary...")
    write_summary(
        row_count=row_count,
        gap_summary=gap_summary,
        subject_target_prior=subject_target_prior,
        targets=targets,
    )

    print("[DONE] Subject distribution EDA completed.")
    print(f"[DONE] Outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()