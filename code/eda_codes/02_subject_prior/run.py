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

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "eda" / "02_subject_prior"
PLOT_DIR = OUTPUT_DIR / "plots"
CSV_DIR = OUTPUT_DIR / "csv"
ensure_dir(PLOT_DIR)
ensure_dir(CSV_DIR)

df = load_train()

subject_prior = df.groupby("subject_id")[TARGETS].mean().reset_index()
subject_count = df.groupby("subject_id").size().reset_index(name="n_days")
subject_prior = subject_count.merge(subject_prior, on="subject_id", how="left")
subject_prior.to_csv(CSV_DIR / "subject_prior.csv", index=False, encoding="utf-8-sig")

# Heatmap without seaborn dependency
heat = subject_prior.set_index("subject_id")[TARGETS]
plt.figure(figsize=(12, max(5, len(heat) * 0.4)))
plt.imshow(heat.values, aspect="auto")
plt.xticks(np.arange(len(TARGETS)), TARGETS)
plt.yticks(np.arange(len(heat.index)), heat.index)
plt.colorbar(label="Positive Ratio")
plt.title("Subject-wise Label Prior")
for i in range(heat.shape[0]):
    for j in range(heat.shape[1]):
        plt.text(j, i, f"{heat.values[i, j]:.2f}", ha="center", va="center", fontsize=8)
savefig(PLOT_DIR / "subject_prior_heatmap.png")

for target in TARGETS:
    plt.figure(figsize=(10, 4))
    plt.bar(subject_prior["subject_id"], subject_prior[target])
    plt.title(f"Subject-wise Prior: {target}")
    plt.xlabel("Subject")
    plt.ylabel("Positive Ratio")
    plt.xticks(rotation=45)
    savefig(PLOT_DIR / f"{target}_subject_prior.png")

print("[DONE] 02_subject_prior")
