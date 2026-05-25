from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def detect_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand in df.columns:
            return cand
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


def load_metrics_train(train_csv: Path, date_cols=None) -> pd.DataFrame:
    if not train_csv.exists():
        raise FileNotFoundError(
            f"TRAIN_CSV does not exist: {train_csv}\n"
            "Please edit eda/common/config.py so PROJECT_ROOT points to your project folder, not a parquet file."
        )
    if date_cols is None:
        date_cols = ["sleep_date", "lifelog_date", "date", "timestamp"]
    df = pd.read_csv(train_csv)
    for col in date_cols:
        real_col = detect_column(df, [col])
        if real_col is not None:
            df[real_col] = pd.to_datetime(df[real_col], errors="coerce")
    return df


def get_target_columns(df: pd.DataFrame, default_targets: list[str]) -> list[str]:
    return [t for t in default_targets if t in df.columns]


def get_subject_col(df: pd.DataFrame) -> str | None:
    return detect_column(df, ["subject_id", "subject", "id", "user_id", "participant_id"])


def get_date_col(df: pd.DataFrame) -> str | None:
    return detect_column(df, ["sleep_date", "lifelog_date", "date", "timestamp"])


def save_plot(path: Path):
    ensure_dir(path.parent)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def write_text(path: Path, text: str):
    ensure_dir(path.parent)
    path.write_text(text, encoding="utf-8")


def safe_prob(p, eps=1e-6):
    return float(np.clip(p, eps, 1 - eps))


def add_value_labels(ax, fmt="{:.2f}"):
    for p in ax.patches:
        height = p.get_height()
        if np.isfinite(height):
            ax.annotate(
                fmt.format(height),
                (p.get_x() + p.get_width() / 2.0, height),
                ha="center",
                va="bottom",
                fontsize=8,
                xytext=(0, 2),
                textcoords="offset points",
            )
