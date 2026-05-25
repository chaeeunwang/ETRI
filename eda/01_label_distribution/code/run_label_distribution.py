from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from eda.common.config import TRAIN_CSV, TARGETS
from eda.common.utils import ensure_dir, load_metrics_train, get_target_columns, save_plot, write_text, add_value_labels

BASE_DIR = Path(__file__).resolve().parents[1]
OUT_DIR = BASE_DIR / "outputs"
PLOT_DIR = ensure_dir(OUT_DIR / "plots")
CSV_DIR = ensure_dir(OUT_DIR / "csv")
REPORT_DIR = ensure_dir(OUT_DIR / "reports")

df = load_metrics_train(TRAIN_CSV)
targets = get_target_columns(df, TARGETS)
if not targets:
    raise ValueError(f"No target columns found. Expected one of: {TARGETS}\nActual columns: {list(df.columns)}")

summary_rows = []
long_rows = []
report_lines = ["# Label Distribution Summary", ""]

for target in targets:
    s = df[target]
    counts = s.value_counts(dropna=False).sort_index()
    ratios = s.value_counts(normalize=True, dropna=False).sort_index()

    dist = pd.DataFrame({
        "target": target,
        "class": [str(x) for x in counts.index],
        "count": counts.values,
        "ratio": ratios.values,
    })
    dist.to_csv(CSV_DIR / f"{target}_distribution.csv", index=False, encoding="utf-8-sig")
    long_rows.append(dist)

    # 1) target별 count plot
    plt.figure(figsize=(6.5, 4.2))
    order = sorted(s.dropna().unique())
    ax = sns.countplot(data=df, x=target, order=order)
    add_value_labels(ax, fmt="{:.0f}")
    plt.title(f"{target} Label Distribution")
    plt.xlabel(target)
    plt.ylabel("Count")
    save_plot(PLOT_DIR / f"{target}_count_distribution.png")

    # 2) target별 ratio plot
    ratio_df = dist[dist["class"] != "nan"].copy()
    plt.figure(figsize=(6.5, 4.2))
    ax = sns.barplot(data=ratio_df, x="class", y="ratio")
    add_value_labels(ax, fmt="{:.2f}")
    plt.ylim(0, 1)
    plt.title(f"{target} Label Ratio")
    plt.xlabel("Class")
    plt.ylabel("Ratio")
    save_plot(PLOT_DIR / f"{target}_ratio_distribution.png")

    majority_class = counts.idxmax()
    majority_ratio = float(ratios.max())
    class1_ratio = float(ratios.get(1, 0.0))
    n_missing = int(s.isna().sum())

    summary_rows.append({
        "target": target,
        "n_total_rows": len(df),
        "n_valid": int(s.notna().sum()),
        "n_missing": n_missing,
        "missing_ratio": n_missing / len(df),
        "n_classes": int(s.nunique(dropna=True)),
        "majority_class": str(majority_class),
        "majority_ratio": majority_ratio,
        "class1_ratio_if_exists": class1_ratio,
        "imbalance_gap_majority_minus_uniform": majority_ratio - (1 / max(int(s.nunique(dropna=True)), 1)),
    })

    report_lines.append(
        f"- {target}: valid={s.notna().sum()}, classes={s.nunique(dropna=True)}, "
        f"majority={majority_class} ({majority_ratio:.3f}), missing={n_missing}"
    )

summary = pd.DataFrame(summary_rows)
summary.to_csv(CSV_DIR / "overall_label_distribution_summary.csv", index=False, encoding="utf-8-sig")

all_dist = pd.concat(long_rows, ignore_index=True)
all_dist.to_csv(CSV_DIR / "all_targets_distribution_long.csv", index=False, encoding="utf-8-sig")

# 3) 전체 target별 majority ratio
plt.figure(figsize=(10, 4.8))
ax = sns.barplot(data=summary, x="target", y="majority_ratio")
add_value_labels(ax, fmt="{:.2f}")
plt.ylim(0, 1)
plt.title("Majority Class Ratio by Target")
plt.xlabel("Target")
plt.ylabel("Majority Ratio")
save_plot(PLOT_DIR / "majority_ratio_by_target.png")

# 4) class1 ratio, binary target 이해용
plt.figure(figsize=(10, 4.8))
ax = sns.barplot(data=summary, x="target", y="class1_ratio_if_exists")
add_value_labels(ax, fmt="{:.2f}")
plt.ylim(0, 1)
plt.title("Class 1 Ratio by Target")
plt.xlabel("Target")
plt.ylabel("Class 1 Ratio")
save_plot(PLOT_DIR / "class1_ratio_by_target.png")

# 5) target x class ratio heatmap
pivot = all_dist.pivot_table(index="target", columns="class", values="ratio", fill_value=0)
plt.figure(figsize=(max(7, len(pivot.columns) * 1.2), 5.2))
sns.heatmap(pivot, annot=True, fmt=".2f", cmap="Blues", vmin=0, vmax=1)
plt.title("Target-Class Ratio Heatmap")
plt.xlabel("Class")
plt.ylabel("Target")
save_plot(PLOT_DIR / "target_class_ratio_heatmap.png")

write_text(REPORT_DIR / "summary.txt", "\n".join(report_lines))
print(summary)
