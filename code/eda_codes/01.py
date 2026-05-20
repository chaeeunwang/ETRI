# eda_step1_target_distribution.py

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
OUTPUT_DIR = Path("./outputs/eda/01_target_distribution")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_PATH = DATA_DIR / "ch2026_metrics_train.csv"

# 실제 컬럼 존재 여부를 보고 자동 필터링합니다.
CANDIDATE_TARGETS = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]

SUBJECT_COL_CANDIDATES = ["subject_id", "subject", "id"]
DATE_COL_CANDIDATES = ["lifelog_date", "sleep_date", "date"]

IMBALANCE_THRESHOLD = 0.80
SUBJECT_PRIOR_RANGE_THRESHOLD = 0.30
N_FOLDS = 5


# =========================
# Utility
# =========================

def find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def get_targets(df: pd.DataFrame) -> list[str]:
    return [t for t in CANDIDATE_TARGETS if t in df.columns]


def safe_ratio_table(count_df: pd.DataFrame) -> pd.DataFrame:
    ratio_df = count_df.div(count_df.sum(axis=1), axis=0)
    return ratio_df.fillna(0.0)


def assign_chronological_folds(
    df: pd.DataFrame,
    subject_col: str,
    date_col: str,
    n_folds: int = 5,
) -> pd.Series:
    """
    subject별 시간 순서를 보존하면서 fold를 부여합니다.
    각 subject 내부에서 날짜순 rank를 만든 뒤 qcut으로 fold를 나눕니다.

    목적:
    - fold별 label 분포가 깨지는지 EDA용으로 확인
    - 실제 학습 fold를 확정하는 코드는 아님
    """
    fold = pd.Series(index=df.index, dtype="Int64")

    work = df[[subject_col, date_col]].copy()
    work[date_col] = pd.to_datetime(work[date_col], errors="coerce")

    for subject, idx in work.groupby(subject_col).groups.items():
        sub = work.loc[idx].sort_values(date_col)
        n = len(sub)

        if n < n_folds:
            # 데이터가 너무 적으면 round-robin으로라도 fold 부여
            assigned = np.arange(n) % n_folds
        else:
            rank = np.arange(n)
            assigned = pd.qcut(
                rank,
                q=n_folds,
                labels=False,
                duplicates="drop",
            )
            assigned = np.asarray(assigned, dtype=int)

        fold.loc[sub.index] = assigned

    return fold.astype(int)


def make_target_class_count(train_df: pd.DataFrame, targets: list[str]) -> pd.DataFrame:
    rows = []

    for target in targets:
        vc = train_df[target].value_counts(dropna=False).sort_index()
        for cls, count in vc.items():
            rows.append({
                "target": target,
                "class": cls,
                "count": int(count),
            })

    return pd.DataFrame(rows)


def make_target_class_ratio(class_count_df: pd.DataFrame) -> pd.DataFrame:
    df = class_count_df.copy()
    df["ratio"] = df.groupby("target")["count"].transform(lambda x: x / x.sum())
    return df


def make_subject_target_ratio(
    train_df: pd.DataFrame,
    targets: list[str],
    subject_col: str,
) -> pd.DataFrame:
    rows = []

    for target in targets:
        for subject, g in train_df.groupby(subject_col):
            values = g[target].dropna()
            total = len(values)

            if total == 0:
                continue

            class_counts = values.value_counts().sort_index().to_dict()
            unique_classes = sorted(values.unique())

            row = {
                "subject_id": subject,
                "target": target,
                "n": total,
                "unique_classes": ",".join(map(str, unique_classes)),
            }

            for cls in sorted(train_df[target].dropna().unique()):
                cnt = int(class_counts.get(cls, 0))
                row[f"class_{cls}_count"] = cnt
                row[f"class_{cls}_ratio"] = cnt / total

            # binary target이면 positive rate를 따로 기록
            non_null_classes = sorted(train_df[target].dropna().unique())
            if set(non_null_classes).issubset({0, 1}):
                row["positive_rate"] = float((values == 1).mean())
            else:
                row["positive_rate"] = np.nan

            rows.append(row)

    return pd.DataFrame(rows)


def make_fold_target_ratio(
    train_df: pd.DataFrame,
    targets: list[str],
    fold_col: str,
) -> pd.DataFrame:
    rows = []

    for target in targets:
        classes = sorted(train_df[target].dropna().unique())

        for fold, g in train_df.groupby(fold_col):
            values = g[target].dropna()
            total = len(values)

            row = {
                "fold": int(fold),
                "target": target,
                "n": total,
            }

            for cls in classes:
                cnt = int((values == cls).sum())
                row[f"class_{cls}_count"] = cnt
                row[f"class_{cls}_ratio"] = cnt / total if total > 0 else np.nan

            rows.append(row)

    return pd.DataFrame(rows)


def plot_target_distribution(
    target_ratio_df: pd.DataFrame,
    output_path: Path,
) -> None:
    pivot = target_ratio_df.pivot(
        index="target",
        columns="class",
        values="ratio",
    ).fillna(0.0)

    ax = pivot.plot(
        kind="bar",
        stacked=True,
        figsize=(12, 6),
    )

    ax.set_title("Target Class Ratio")
    ax.set_xlabel("Target")
    ax.set_ylabel("Class Ratio")
    ax.legend(title="Class", bbox_to_anchor=(1.02, 1), loc="upper left")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_subject_target_heatmap(
    subject_target_ratio_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    binary target은 positive_rate heatmap.
    multi-class target은 class별 ratio가 있으므로 여기서는 제외.
    """
    binary_df = subject_target_ratio_df.dropna(subset=["positive_rate"]).copy()

    if binary_df.empty:
        return

    pivot = binary_df.pivot(
        index="subject_id",
        columns="target",
        values="positive_rate",
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(pivot.values, aspect="auto")

    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)

    ax.set_title("Subject-wise Positive Rate Heatmap")
    ax.set_xlabel("Target")
    ax.set_ylabel("Subject")

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            value = pivot.iloc[i, j]
            if pd.notna(value):
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=8)

    fig.colorbar(im, ax=ax, label="Positive Rate")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def make_summary_markdown(
    train_df: pd.DataFrame,
    targets: list[str],
    target_ratio_df: pd.DataFrame,
    subject_target_ratio_df: pd.DataFrame,
    fold_target_ratio_df: pd.DataFrame,
    output_path: Path,
) -> None:
    lines = []

    lines.append("# Target Imbalance Summary\n")
    lines.append("## 1. Target Columns\n")
    lines.append(f"- Detected targets: `{targets}`\n")
    lines.append(f"- Train rows: `{len(train_df)}`\n")

    lines.append("\n## 2. Class Imbalance Check\n")

    imbalance_rows = []

    for target in targets:
        sub = target_ratio_df[target_ratio_df["target"] == target].copy()
        max_ratio = sub["ratio"].max()
        majority_class = sub.loc[sub["ratio"].idxmax(), "class"]
        n_classes = sub["class"].nunique()
        is_imbalanced = max_ratio >= IMBALANCE_THRESHOLD

        imbalance_rows.append({
            "target": target,
            "n_classes": n_classes,
            "majority_class": majority_class,
            "majority_ratio": max_ratio,
            "is_imbalanced_80_20": is_imbalanced,
        })

        flag = "⚠️" if is_imbalanced else "OK"
        lines.append(
            f"- {target}: majority class `{majority_class}` ratio `{max_ratio:.3f}` → {flag}"
        )

    imbalance_df = pd.DataFrame(imbalance_rows)

    lines.append("\n\n## 3. S1 Class Structure\n")
    if "S1" in targets:
        s1_classes = sorted(train_df["S1"].dropna().unique().tolist())
        lines.append(f"- S1 classes: `{s1_classes}`")
        if len(s1_classes) == 3 and set(s1_classes) == {0, 1, 2}:
            lines.append("- 판단: S1은 0/1/2 3-class 구조입니다.")
            lines.append("- 조치: S1을 binary sigmoid head로 처리하면 안 됩니다. multi-class head 또는 별도 loss가 필요합니다.")
        else:
            lines.append("- 판단: S1이 명확한 0/1/2 3-class인지 추가 확인이 필요합니다.")
    else:
        lines.append("- S1 column이 train에 없습니다.")

    lines.append("\n\n## 4. Subject-wise Prior Difference\n")

    subject_prior_rows = []

    for target in targets:
        sub = subject_target_ratio_df[
            (subject_target_ratio_df["target"] == target)
            & (subject_target_ratio_df["positive_rate"].notna())
        ]

        if sub.empty:
            continue

        min_rate = sub["positive_rate"].min()
        max_rate = sub["positive_rate"].max()
        rate_range = max_rate - min_rate
        need_subject_prior = rate_range >= SUBJECT_PRIOR_RANGE_THRESHOLD

        subject_prior_rows.append({
            "target": target,
            "min_positive_rate": min_rate,
            "max_positive_rate": max_rate,
            "range": rate_range,
            "need_subject_prior_feature": need_subject_prior,
        })

        flag = "⚠️ subject prior feature 검토" if need_subject_prior else "OK"
        lines.append(
            f"- {target}: subject positive rate range `{rate_range:.3f}` "
            f"({min_rate:.3f} ~ {max_rate:.3f}) → {flag}"
        )

    lines.append("\n\n## 5. Fold Distribution Check\n")

    fold_issue_rows = []

    for target in targets:
        class_ratio_cols = [
            c for c in fold_target_ratio_df.columns
            if c.startswith("class_") and c.endswith("_ratio")
        ]

        sub = fold_target_ratio_df[fold_target_ratio_df["target"] == target]

        max_gap = 0.0
        worst_col = None

        for col in class_ratio_cols:
            if col in sub.columns and sub[col].notna().any():
                gap = sub[col].max() - sub[col].min()
                if gap > max_gap:
                    max_gap = gap
                    worst_col = col

        fold_issue = max_gap >= 0.20

        fold_issue_rows.append({
            "target": target,
            "max_fold_ratio_gap": max_gap,
            "worst_class_ratio_col": worst_col,
            "fold_distribution_issue": fold_issue,
        })

        flag = "⚠️ fold 재설계 검토" if fold_issue else "OK"
        lines.append(
            f"- {target}: max fold class-ratio gap `{max_gap:.3f}` "
            f"({worst_col}) → {flag}"
        )

    lines.append("\n\n## 6. Recommended Actions\n")

    if imbalance_df["is_imbalanced_80_20"].any():
        bad_targets = imbalance_df.loc[
            imbalance_df["is_imbalanced_80_20"],
            "target"
        ].tolist()
        lines.append(f"- 불균형 target `{bad_targets}`: pos_weight 또는 focal loss 검토")
    else:
        lines.append("- 80:20 이상으로 심한 전체 class imbalance는 발견되지 않았습니다.")

    if subject_prior_rows:
        subject_prior_df = pd.DataFrame(subject_prior_rows)
        need_targets = subject_prior_df.loc[
            subject_prior_df["need_subject_prior_feature"],
            "target"
        ].tolist()
        if need_targets:
            lines.append(f"- subject별 prior 차이가 큰 target `{need_targets}`: subject prior feature 추가 검토")
        else:
            lines.append("- subject별 positive rate 차이는 큰 편이 아닙니다.")
    else:
        lines.append("- binary target positive rate 기준 subject prior 분석 대상이 없습니다.")

    fold_issue_df = pd.DataFrame(fold_issue_rows)
    fold_bad_targets = fold_issue_df.loc[
        fold_issue_df["fold_distribution_issue"],
        "target"
    ].tolist()
    if fold_bad_targets:
        lines.append(f"- fold 분포 차이가 큰 target `{fold_bad_targets}`: fold 설계 재검토")
    else:
        lines.append("- fold별 label ratio는 큰 붕괴 없이 유지되는 편입니다.")

    lines.append("\n\n## 7. Output Files\n")
    lines.append("- `target_class_count.csv`")
    lines.append("- `target_class_ratio.csv`")
    lines.append("- `subject_target_ratio.csv`")
    lines.append("- `fold_target_ratio.csv`")
    lines.append("- `target_distribution_bar.png`")
    lines.append("- `subject_target_heatmap.png`")
    lines.append("- `target_imbalance_summary.md`")

    output_path.write_text("\n".join(lines), encoding="utf-8")


# =========================
# Main
# =========================

def main() -> None:
    if not TRAIN_PATH.exists():
        raise FileNotFoundError(f"Train file not found: {TRAIN_PATH}")

    train_df = pd.read_csv(TRAIN_PATH)

    subject_col = find_col(train_df, SUBJECT_COL_CANDIDATES)
    date_col = find_col(train_df, DATE_COL_CANDIDATES)
    targets = get_targets(train_df)

    if subject_col is None:
        raise ValueError(
            f"Subject column not found. Tried: {SUBJECT_COL_CANDIDATES}\n"
            f"Available columns: {list(train_df.columns)}"
        )

    if date_col is None:
        raise ValueError(
            f"Date column not found. Tried: {DATE_COL_CANDIDATES}\n"
            f"Available columns: {list(train_df.columns)}"
        )

    if not targets:
        raise ValueError(
            f"No target columns found. Tried: {CANDIDATE_TARGETS}\n"
            f"Available columns: {list(train_df.columns)}"
        )

    print(f"[INFO] subject_col = {subject_col}")
    print(f"[INFO] date_col = {date_col}")
    print(f"[INFO] targets = {targets}")

    train_df[date_col] = pd.to_datetime(train_df[date_col], errors="coerce")

    # EDA용 chronological fold 생성
    train_df["eda_fold"] = assign_chronological_folds(
        train_df,
        subject_col=subject_col,
        date_col=date_col,
        n_folds=N_FOLDS,
    )

    # 1. target_class_count.csv
    target_class_count = make_target_class_count(train_df, targets)
    target_class_count.to_csv(
        OUTPUT_DIR / "target_class_count.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # 2. target_class_ratio.csv
    target_class_ratio = make_target_class_ratio(target_class_count)
    target_class_ratio.to_csv(
        OUTPUT_DIR / "target_class_ratio.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # 3. subject_target_ratio.csv
    subject_target_ratio = make_subject_target_ratio(
        train_df,
        targets,
        subject_col=subject_col,
    )
    subject_target_ratio.to_csv(
        OUTPUT_DIR / "subject_target_ratio.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # 4. fold_target_ratio.csv
    fold_target_ratio = make_fold_target_ratio(
        train_df,
        targets,
        fold_col="eda_fold",
    )
    fold_target_ratio.to_csv(
        OUTPUT_DIR / "fold_target_ratio.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # 5. target_distribution_bar.png
    plot_target_distribution(
        target_class_ratio,
        OUTPUT_DIR / "target_distribution_bar.png",
    )

    # 6. subject_target_heatmap.png
    plot_subject_target_heatmap(
        subject_target_ratio,
        OUTPUT_DIR / "subject_target_heatmap.png",
    )

    # 7. target_imbalance_summary.md
    make_summary_markdown(
        train_df=train_df,
        targets=targets,
        target_ratio_df=target_class_ratio,
        subject_target_ratio_df=subject_target_ratio,
        fold_target_ratio_df=fold_target_ratio,
        output_path=OUTPUT_DIR / "target_imbalance_summary.md",
    )

    print("[DONE] Step 1 target distribution EDA completed.")
    print(f"[DONE] Saved to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()