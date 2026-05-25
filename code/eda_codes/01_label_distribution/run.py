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

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "eda" / "01_label_distribution"
PLOT_DIR = OUTPUT_DIR / "plots"
CSV_DIR = OUTPUT_DIR / "csv"
ensure_dir(PLOT_DIR)
ensure_dir(CSV_DIR)

df = load_train()

summary = []
for target in TARGETS:
    counts = df[target].value_counts(dropna=False).sort_index()
    ratios = df[target].value_counts(normalize=True, dropna=False).sort_index()

    out = pd.DataFrame({"label": counts.index.astype(str), "count": counts.values, "ratio": ratios.values})
    out.to_csv(CSV_DIR / f"{target}_distribution.csv", index=False, encoding="utf-8-sig")

    plt.figure(figsize=(6, 4))
    plt.bar(out["label"], out["count"])
    plt.title(f"{target} Label Count")
    plt.xlabel("Label")
    plt.ylabel("Count")
    savefig(PLOT_DIR / f"{target}_count.png")

    plt.figure(figsize=(6, 4))
    plt.bar(out["label"], out["ratio"])
    plt.title(f"{target} Label Ratio")
    plt.xlabel("Label")
    plt.ylabel("Ratio")
    savefig(PLOT_DIR / f"{target}_ratio.png")

    row = {"target": target, "n": len(df)}
    for _, r in out.iterrows():
        row[f"label_{r['label']}_count"] = r["count"]
        row[f"label_{r['label']}_ratio"] = r["ratio"]
    summary.append(row)

summary_df = pd.DataFrame(summary)
summary_df.to_csv(CSV_DIR / "label_distribution_summary.csv", index=False, encoding="utf-8-sig")

plt.figure(figsize=(10, 5))
x = np.arange(len(summary_df))
pos_ratio = [df[t].mean() for t in TARGETS]
plt.bar(TARGETS, pos_ratio)
plt.title("Positive Ratio by Target")
plt.xlabel("Target")
plt.ylabel("Positive Ratio")
savefig(PLOT_DIR / "positive_ratio_by_target.png")

print("[DONE] 01_label_distribution")
