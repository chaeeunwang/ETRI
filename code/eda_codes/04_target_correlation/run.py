from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data"
TRAIN_CSV = DATA_DIR / "ch2026_metrics_train.csv"
TARGETS = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]

def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

def savefig(path):
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

def load_train():
    if not TRAIN_CSV.exists():
        raise FileNotFoundError(
            f"TRAIN_CSV not found: {TRAIN_CSV}\n"
            "Please place ch2026_metrics_train.csv under data/."
        )
    df = pd.read_csv(TRAIN_CSV)
    missing = [c for c in ["subject_id", "sleep_date", "lifelog_date"] + TARGETS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in train csv: {missing}")
    return df

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "eda" / "04_target_correlation"
PLOT_DIR = OUTPUT_DIR / "plots"
CSV_DIR = OUTPUT_DIR / "csv"
ensure_dir(PLOT_DIR)
ensure_dir(CSV_DIR)

df = load_train()

def plot_corr(corr, title, filename):
    plt.figure(figsize=(8, 6))
    plt.imshow(corr.values, vmin=-1, vmax=1)
    plt.xticks(np.arange(len(corr.columns)), corr.columns)
    plt.yticks(np.arange(len(corr.index)), corr.index)
    plt.colorbar(label="Correlation")
    plt.title(title)
    for i in range(corr.shape[0]):
        for j in range(corr.shape[1]):
            plt.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=9)
    savefig(PLOT_DIR / filename)

pearson = df[TARGETS].corr(method="pearson")
spearman = df[TARGETS].corr(method="spearman")

pearson.to_csv(CSV_DIR / "target_correlation_pearson.csv", encoding="utf-8-sig")
spearman.to_csv(CSV_DIR / "target_correlation_spearman.csv", encoding="utf-8-sig")

plot_corr(pearson, "Target Correlation - Pearson", "target_correlation_pearson.png")
plot_corr(spearman, "Target Correlation - Spearman", "target_correlation_spearman.png")

# pair joint ratio
pair_rows = []
for i, t1 in enumerate(TARGETS):
    for t2 in TARGETS[i+1:]:
        table = pd.crosstab(df[t1], df[t2], normalize="all")
        table.to_csv(CSV_DIR / f"joint_ratio_{t1}_{t2}.csv", encoding="utf-8-sig")
        pair_rows.append({
            "target_1": t1,
            "target_2": t2,
            "both_positive_ratio": float(((df[t1] == 1) & (df[t2] == 1)).mean())
        })

pd.DataFrame(pair_rows).to_csv(CSV_DIR / "target_pair_both_positive_ratio.csv", index=False, encoding="utf-8-sig")

plt.figure(figsize=(10, 5))
labels = [f"{r['target_1']}-{r['target_2']}" for r in pair_rows]
values = [r["both_positive_ratio"] for r in pair_rows]
plt.bar(labels, values)
plt.xticks(rotation=90)
plt.title("Target Pair Both Positive Ratio")
plt.ylabel("Ratio")
savefig(PLOT_DIR / "target_pair_both_positive_ratio.png")

print("[DONE] 04_target_correlation")
