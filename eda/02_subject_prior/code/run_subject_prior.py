from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from eda.common.config import TRAIN_CSV, TARGETS
from eda.common.utils import ensure_dir, load_metrics_train, get_target_columns, get_subject_col, save_plot, write_text, add_value_labels

BASE_DIR = Path(__file__).resolve().parents[1]
OUT_DIR = BASE_DIR / "outputs"
PLOT_DIR = ensure_dir(OUT_DIR / "plots")
CSV_DIR = ensure_dir(OUT_DIR / "csv")
REPORT_DIR = ensure_dir(OUT_DIR / "reports")

df = load_metrics_train(TRAIN_CSV)
targets = get_target_columns(df, TARGETS)
subject_col = get_subject_col(df)
if subject_col is None:
    raise ValueError(f"Subject column not found. Actual columns: {list(df.columns)}")

# subject별 sample count
sample_counts = df.groupby(subject_col).size().reset_index(name="n_days")
sample_counts.to_csv(CSV_DIR / "subject_sample_counts.csv", index=False, encoding="utf-8-sig")
plt.figure(figsize=(10, 4.5))
ax = sns.barplot(data=sample_counts, x=subject_col, y="n_days")
add_value_labels(ax, fmt="{:.0f}")
plt.title("Number of Labeled Days by Subject")
plt.xlabel("Subject")
plt.ylabel("Number of Days")
plt.xticks(rotation=45, ha="right")
save_plot(PLOT_DIR / "subject_sample_counts.png")

# subject별 평균 label, binary는 class1 prior, multiclass는 평균값 참고용
subject_mean = df.groupby(subject_col)[targets].mean(numeric_only=True).reset_index()
subject_mean.to_csv(CSV_DIR / "subject_target_mean_prior.csv", index=False, encoding="utf-8-sig")

plt.figure(figsize=(11, max(4.5, len(subject_mean) * 0.45)))
sns.heatmap(subject_mean.set_index(subject_col), annot=True, fmt=".2f", cmap="coolwarm", center=0.5)
plt.title("Subject-wise Target Mean/Prior Heatmap")
plt.xlabel("Target")
plt.ylabel("Subject")
save_plot(PLOT_DIR / "subject_target_mean_prior_heatmap.png")

# target별 subject prior barplot
for target in targets:
    plot_df = subject_mean[[subject_col, target]].copy().sort_values(target)
    plt.figure(figsize=(9.5, 4.5))
    ax = sns.barplot(data=plot_df, x=subject_col, y=target)
    add_value_labels(ax, fmt="{:.2f}")
    plt.ylim(0, max(1, plot_df[target].max() * 1.15 if pd.notna(plot_df[target].max()) else 1))
    plt.title(f"Subject-wise Mean/Prior: {target}")
    plt.xlabel("Subject")
    plt.ylabel(f"Mean of {target}")
    plt.xticks(rotation=45, ha="right")
    save_plot(PLOT_DIR / f"{target}_subject_prior_barplot.png")

# target별 class 분포를 subject heatmap으로 저장
for target in targets:
    ct = pd.crosstab(df[subject_col], df[target], normalize="index")
    ct.to_csv(CSV_DIR / f"{target}_subject_class_ratio.csv", encoding="utf-8-sig")
    plt.figure(figsize=(max(7, ct.shape[1] * 1.4), max(4.5, ct.shape[0] * 0.45)))
    sns.heatmap(ct, annot=True, fmt=".2f", cmap="Blues", vmin=0, vmax=1)
    plt.title(f"Subject-wise Class Ratio: {target}")
    plt.xlabel("Class")
    plt.ylabel("Subject")
    save_plot(PLOT_DIR / f"{target}_subject_class_ratio_heatmap.png")

# subject prior 변동성 요약
variability = []
for target in targets:
    vals = subject_mean[target].dropna()
    variability.append({
        "target": target,
        "subject_prior_min": vals.min(),
        "subject_prior_max": vals.max(),
        "subject_prior_range": vals.max() - vals.min(),
        "subject_prior_std": vals.std(),
    })
variability_df = pd.DataFrame(variability)
variability_df.to_csv(CSV_DIR / "subject_prior_variability.csv", index=False, encoding="utf-8-sig")

plt.figure(figsize=(10, 4.8))
ax = sns.barplot(data=variability_df, x="target", y="subject_prior_range")
add_value_labels(ax, fmt="{:.2f}")
plt.title("Subject Prior Range by Target")
plt.xlabel("Target")
plt.ylabel("Max-Min across Subjects")
save_plot(PLOT_DIR / "subject_prior_range_by_target.png")

lines = ["# Subject Prior Summary", "", f"Detected subject column: {subject_col}", ""]
for _, r in variability_df.iterrows():
    lines.append(f"- {r['target']}: subject prior range={r['subject_prior_range']:.3f}, std={r['subject_prior_std']:.3f}")
write_text(REPORT_DIR / "summary.txt", "\n".join(lines))
print(subject_mean)
