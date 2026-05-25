from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import log_loss, accuracy_score, f1_score

from eda.common.config import TRAIN_CSV, TARGETS
from eda.common.utils import ensure_dir, load_metrics_train, get_target_columns, get_subject_col, save_plot, write_text, safe_prob, add_value_labels

BASE_DIR = Path(__file__).resolve().parents[1]
OUT_DIR = BASE_DIR / "outputs"
PLOT_DIR = ensure_dir(OUT_DIR / "plots")
CSV_DIR = ensure_dir(OUT_DIR / "csv")
REPORT_DIR = ensure_dir(OUT_DIR / "reports")

df = load_metrics_train(TRAIN_CSV)
targets = get_target_columns(df, TARGETS)
subject_col = get_subject_col(df)

rows = []
per_class_rows = []

for target in targets:
    y = df[target].dropna()
    classes = sorted(y.unique())
    if len(classes) < 2:
        continue

    # Global prior probability for each class
    class_probs = y.value_counts(normalize=True).reindex(classes).fillna(0).values
    class_probs = np.clip(class_probs, 1e-6, 1 - 1e-6)
    class_probs = class_probs / class_probs.sum()
    proba = np.tile(class_probs, (len(y), 1))
    y_pred = np.repeat(classes[int(np.argmax(class_probs))], len(y))

    global_logloss = log_loss(y, proba, labels=classes)
    acc = accuracy_score(y, y_pred)
    f1_macro = f1_score(y, y_pred, average="macro", zero_division=0)

    rows.append({
        "target": target,
        "baseline": "global_prior",
        "n_valid": len(y),
        "n_classes": len(classes),
        "majority_class": classes[int(np.argmax(class_probs))],
        "majority_prob": class_probs.max(),
        "logloss": global_logloss,
        "accuracy": acc,
        "macro_f1": f1_macro,
    })

    for cls, p in zip(classes, class_probs):
        per_class_rows.append({"target": target, "class": cls, "global_prior_prob": p})

    # Subject prior baseline: train-set internal optimistic reference only
    if subject_col is not None:
        valid_df = df[[subject_col, target]].dropna().copy()
        pred_probs = []
        pred_labels = []
        global_map = {c: p for c, p in zip(classes, class_probs)}
        for _, r in valid_df.iterrows():
            sub = r[subject_col]
            sub_y = valid_df.loc[valid_df[subject_col] == sub, target]
            sub_probs = sub_y.value_counts(normalize=True).reindex(classes).fillna(0).values
            # smoothing: 80% subject prior + 20% global prior
            sub_probs = 0.8 * sub_probs + 0.2 * class_probs
            sub_probs = np.clip(sub_probs, 1e-6, 1 - 1e-6)
            sub_probs = sub_probs / sub_probs.sum()
            pred_probs.append(sub_probs)
            pred_labels.append(classes[int(np.argmax(sub_probs))])
        pred_probs = np.vstack(pred_probs)
        rows.append({
            "target": target,
            "baseline": "subject_prior_smoothed_in_sample_reference",
            "n_valid": len(valid_df),
            "n_classes": len(classes),
            "majority_class": "subject_specific",
            "majority_prob": np.nan,
            "logloss": log_loss(valid_df[target], pred_probs, labels=classes),
            "accuracy": accuracy_score(valid_df[target], pred_labels),
            "macro_f1": f1_score(valid_df[target], pred_labels, average="macro", zero_division=0),
        })

results = pd.DataFrame(rows)
results.to_csv(CSV_DIR / "naive_baseline_results.csv", index=False, encoding="utf-8-sig")
per_class = pd.DataFrame(per_class_rows)
per_class.to_csv(CSV_DIR / "global_prior_per_class.csv", index=False, encoding="utf-8-sig")

# 1) Global prior probability heatmap
if not per_class.empty:
    pivot = per_class.pivot_table(index="target", columns="class", values="global_prior_prob", fill_value=0)
    plt.figure(figsize=(max(7, pivot.shape[1] * 1.3), 5.2))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="Blues", vmin=0, vmax=1)
    plt.title("Global Prior Probability by Target/Class")
    plt.xlabel("Class")
    plt.ylabel("Target")
    save_plot(PLOT_DIR / "global_prior_probability_heatmap.png")

# 2) logloss barplot
plt.figure(figsize=(10.5, 5))
ax = sns.barplot(data=results, x="target", y="logloss", hue="baseline")
add_value_labels(ax, fmt="{:.3f}")
plt.title("Naive Baseline Log-Loss by Target")
plt.xlabel("Target")
plt.ylabel("Log-Loss")
plt.legend(loc="best")
save_plot(PLOT_DIR / "naive_baseline_logloss_by_target.png")

# 3) accuracy barplot
plt.figure(figsize=(10.5, 5))
ax = sns.barplot(data=results, x="target", y="accuracy", hue="baseline")
add_value_labels(ax, fmt="{:.2f}")
plt.ylim(0, 1)
plt.title("Naive Baseline Accuracy by Target")
plt.xlabel("Target")
plt.ylabel("Accuracy")
plt.legend(loc="best")
save_plot(PLOT_DIR / "naive_baseline_accuracy_by_target.png")

# 4) macro f1 barplot
plt.figure(figsize=(10.5, 5))
ax = sns.barplot(data=results, x="target", y="macro_f1", hue="baseline")
add_value_labels(ax, fmt="{:.2f}")
plt.ylim(0, 1)
plt.title("Naive Baseline Macro-F1 by Target")
plt.xlabel("Target")
plt.ylabel("Macro-F1")
plt.legend(loc="best")
save_plot(PLOT_DIR / "naive_baseline_macro_f1_by_target.png")

lines = [
    "# Naive Baseline Summary",
    "",
    "- global_prior: 전체 train label 분포만 사용한 가장 기본 baseline입니다.",
    "- subject_prior_smoothed_in_sample_reference: subject별 prior 효과 확인용입니다. 실제 성능 추정용으로는 chronological validation에서 다시 계산해야 합니다.",
    "",
]
for _, r in results.iterrows():
    lines.append(f"- {r['target']} / {r['baseline']}: logloss={r['logloss']:.4f}, acc={r['accuracy']:.3f}, macro_f1={r['macro_f1']:.3f}")
write_text(REPORT_DIR / "summary.txt", "\n".join(lines))
print(results)
