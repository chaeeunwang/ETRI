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

from sklearn.metrics import log_loss, accuracy_score, f1_score

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "eda" / "03_naive_baseline"
PLOT_DIR = OUTPUT_DIR / "plots"
CSV_DIR = OUTPUT_DIR / "csv"
ensure_dir(PLOT_DIR)
ensure_dir(CSV_DIR)

df = load_train()

rows = []
eps = 1e-6
for target in TARGETS:
    y = df[target].astype(int).values
    p = float(np.clip(df[target].mean(), eps, 1 - eps))
    prob = np.full(len(y), p)
    pred = (prob >= 0.5).astype(int)

    rows.append({
        "target": target,
        "global_prior": p,
        "log_loss": log_loss(y, prob, labels=[0, 1]),
        "accuracy": accuracy_score(y, pred),
        "f1": f1_score(y, pred, zero_division=0),
        "n": len(y),
    })

results = pd.DataFrame(rows)
results.to_csv(CSV_DIR / "global_prior_baseline.csv", index=False, encoding="utf-8-sig")

for metric in ["log_loss", "accuracy", "f1", "global_prior"]:
    plt.figure(figsize=(8, 4))
    plt.bar(results["target"], results[metric])
    plt.title(f"Naive Baseline: {metric}")
    plt.xlabel("Target")
    plt.ylabel(metric)
    savefig(PLOT_DIR / f"{metric}.png")

# subject prior table
subject_rows = []
for target in TARGETS:
    tmp = df.groupby("subject_id")[target].mean().reset_index()
    tmp["target"] = target
    tmp = tmp.rename(columns={target: "subject_prior"})
    subject_rows.append(tmp)
pd.concat(subject_rows, ignore_index=True).to_csv(
    CSV_DIR / "subject_prior_baseline_table.csv",
    index=False,
    encoding="utf-8-sig"
)

print("[DONE] 03_naive_baseline")
