from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from eda.common.config import TRAIN_CSV, TARGETS
from eda.common.utils import ensure_dir, load_metrics_train, get_target_columns, save_plot, write_text

BASE_DIR = Path(__file__).resolve().parents[1]
OUT_DIR = BASE_DIR / "outputs"
PLOT_DIR = ensure_dir(OUT_DIR / "plots")
CSV_DIR = ensure_dir(OUT_DIR / "csv")
REPORT_DIR = ensure_dir(OUT_DIR / "reports")

df = load_metrics_train(TRAIN_CSV)
targets = get_target_columns(df, TARGETS)

# Pearson/Spearman correlations; labels are numeric class codes, so use as rough association reference.
pearson = df[targets].corr(method="pearson")
spearman = df[targets].corr(method="spearman")
pearson.to_csv(CSV_DIR / "target_pearson_correlation.csv", encoding="utf-8-sig")
spearman.to_csv(CSV_DIR / "target_spearman_correlation.csv", encoding="utf-8-sig")

plt.figure(figsize=(8, 6.5))
sns.heatmap(pearson, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, center=0)
plt.title("Target Pearson Correlation")
save_plot(PLOT_DIR / "target_pearson_correlation_heatmap.png")

plt.figure(figsize=(8, 6.5))
sns.heatmap(spearman, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, center=0)
plt.title("Target Spearman Correlation")
save_plot(PLOT_DIR / "target_spearman_correlation_heatmap.png")

# Pairwise joint class tables and heatmaps
pair_rows = []
for i, t1 in enumerate(targets):
    for t2 in targets[i+1:]:
        ct = pd.crosstab(df[t1], df[t2], normalize="all")
        ct.to_csv(CSV_DIR / f"joint_ratio_{t1}_{t2}.csv", encoding="utf-8-sig")
        plt.figure(figsize=(6.2, 5.2))
        sns.heatmap(ct, annot=True, fmt=".2f", cmap="Blues", vmin=0, vmax=max(0.01, ct.values.max()))
        plt.title(f"Joint Class Ratio: {t1} vs {t2}")
        plt.xlabel(t2)
        plt.ylabel(t1)
        save_plot(PLOT_DIR / f"joint_ratio_{t1}_{t2}.png")

        pair_rows.append({
            "target_1": t1,
            "target_2": t2,
            "pearson": pearson.loc[t1, t2],
            "spearman": spearman.loc[t1, t2],
            "n_pair_valid": int(df[[t1, t2]].dropna().shape[0]),
        })

pair_df = pd.DataFrame(pair_rows)
pair_df.to_csv(CSV_DIR / "target_pairwise_correlation_long.csv", index=False, encoding="utf-8-sig")

# top absolute correlations
pair_df["abs_spearman"] = pair_df["spearman"].abs()
top_pairs = pair_df.sort_values("abs_spearman", ascending=False).head(10)
top_pairs.to_csv(CSV_DIR / "top_target_correlations.csv", index=False, encoding="utf-8-sig")

plt.figure(figsize=(10, 5))
plot_df = top_pairs.copy()
plot_df["pair"] = plot_df["target_1"] + "-" + plot_df["target_2"]
sns.barplot(data=plot_df, x="pair", y="spearman")
plt.axhline(0, linewidth=1)
plt.title("Top Target Pair Correlations by Spearman")
plt.xlabel("Target Pair")
plt.ylabel("Spearman Correlation")
plt.xticks(rotation=45, ha="right")
save_plot(PLOT_DIR / "top_target_pair_correlations.png")

lines = ["# Target Correlation Summary", ""]
for _, r in top_pairs.iterrows():
    lines.append(f"- {r['target_1']} ↔ {r['target_2']}: spearman={r['spearman']:.3f}, pearson={r['pearson']:.3f}, n={r['n_pair_valid']}")
write_text(REPORT_DIR / "summary.txt", "\n".join(lines))
print(top_pairs)
