"""
05_missing_analysis.py

Purpose:
- Analyze sensor-level missing patterns by subject/date/hour.
- Compare train/test missing patterns.
- Check whether missingness itself is related to targets.
- Save all artifacts under outputs/eda/05_missing_analysis/.

Expected project structure:
data/
├── ch2026_metrics_train.csv
├── ch2026_submission_sample.csv
└── ch2025_data_items/
    ├── sensor files...
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


TARGET_CANDIDATES = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]

SUBJECT_COL_CANDIDATES = [
    "subject_id", "subject", "id", "user_id", "participant_id", "pid"
]

DATE_COL_CANDIDATES = [
    "lifelog_date", "date", "dt", "sleep_date"
]

TIME_COL_CANDIDATES = [
    "timestamp", "time", "datetime", "date_time",
    "start_time", "end_time", "ts", "created_at", "event_time"
]


def find_col(columns: list[str], candidates: list[str]) -> Optional[str]:
    lower_map = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


def safe_sensor_name(path: Path) -> str:
    name = path.stem
    name = re.sub(r"[^0-9a-zA-Z가-힣_]+", "_", name)
    return name


def read_columns(path: Path) -> list[str]:
    if path.suffix.lower() == ".csv":
        return list(pd.read_csv(path, nrows=0).columns)
    if path.suffix.lower() in [".parquet", ".pq"]:
        return list(pd.read_parquet(path, columns=None).columns)
    raise ValueError(f"Unsupported file type: {path}")


def read_sensor_minimal(path: Path) -> tuple[pd.DataFrame, dict]:
    """
    Read only subject/date/time columns needed for missing analysis.
    """
    cols = read_columns(path)

    subject_col = find_col(cols, SUBJECT_COL_CANDIDATES)
    date_col = find_col(cols, DATE_COL_CANDIDATES)
    time_col = find_col(cols, TIME_COL_CANDIDATES)

    meta = {
        "file": str(path),
        "sensor": safe_sensor_name(path),
        "subject_col": subject_col,
        "date_col": date_col,
        "time_col": time_col,
        "status": "ok",
        "reason": "",
    }

    if subject_col is None:
        meta["status"] = "skip"
        meta["reason"] = "subject column not found"
        return pd.DataFrame(), meta

    usecols = [subject_col]
    if date_col is not None:
        usecols.append(date_col)
    if time_col is not None and time_col not in usecols:
        usecols.append(time_col)

    if date_col is None and time_col is None:
        meta["status"] = "skip"
        meta["reason"] = "date/time column not found"
        return pd.DataFrame(), meta

    try:
        if path.suffix.lower() == ".csv":
            df = pd.read_csv(path, usecols=usecols, low_memory=False)
        else:
            df = pd.read_parquet(path, columns=usecols)
    except Exception as e:
        meta["status"] = "error"
        meta["reason"] = str(e)
        return pd.DataFrame(), meta

    df = df.rename(columns={subject_col: "subject_id"})

    if date_col is not None:
        df["lifelog_date"] = pd.to_datetime(df[date_col], errors="coerce").dt.date
    else:
        df["lifelog_date"] = pd.NaT

    if time_col is not None:
        df["timestamp"] = pd.to_datetime(df[time_col], errors="coerce")
        if df["lifelog_date"].isna().all():
            df["lifelog_date"] = df["timestamp"].dt.date
        df["hour"] = df["timestamp"].dt.hour
    else:
        df["timestamp"] = pd.NaT
        df["hour"] = np.nan

    df["subject_id"] = df["subject_id"].astype(str)
    df["lifelog_date"] = pd.to_datetime(df["lifelog_date"], errors="coerce").dt.date
    df["sensor"] = meta["sensor"]

    return df[["subject_id", "lifelog_date", "hour", "sensor"]], meta


def load_key_frames(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    train_path = data_dir / "ch2026_metrics_train.csv"
    sub_path = data_dir / "ch2026_submission_sample.csv"

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(sub_path)

    for df in [train_df, test_df]:
        df["subject_id"] = df["subject_id"].astype(str)
        df["lifelog_date"] = pd.to_datetime(df["lifelog_date"], errors="coerce").dt.date

    target_cols = [c for c in TARGET_CANDIDATES if c in train_df.columns]

    train_keys = train_df[["subject_id", "lifelog_date"] + target_cols].copy()
    train_keys["split"] = "train"

    test_keys = test_df[["subject_id", "lifelog_date"]].copy()
    test_keys["split"] = "test"

    return train_keys, test_keys, target_cols


def discover_sensor_files(items_dir: Path) -> list[Path]:
    exts = ["*.csv", "*.parquet", "*.pq"]
    files: list[Path] = []
    for ext in exts:
        files.extend(items_dir.rglob(ext))

    ignore_patterns = [
        "metrics_train",
        "submission",
        "sample",
        "label",
        "target",
    ]

    filtered = []
    for f in files:
        lower = f.name.lower()
        if any(p in lower for p in ignore_patterns):
            continue
        filtered.append(f)

    return sorted(filtered)


def make_day_presence(
    all_keys: pd.DataFrame,
    sensor_daily_counts: pd.DataFrame,
    sensors: list[str],
) -> pd.DataFrame:
    base = all_keys[["split", "subject_id", "lifelog_date"]].drop_duplicates()

    sensor_frame = pd.DataFrame({"sensor": sensors})
    base["__key"] = 1
    sensor_frame["__key"] = 1

    full_index = base.merge(sensor_frame, on="__key").drop(columns="__key")

    out = full_index.merge(
        sensor_daily_counts,
        on=["subject_id", "lifelog_date", "sensor"],
        how="left",
    )
    out["row_count"] = out["row_count"].fillna(0).astype(int)
    out["present"] = (out["row_count"] > 0).astype(int)
    out["missing"] = 1 - out["present"]

    return out


def make_hour_presence(
    all_keys: pd.DataFrame,
    sensor_hour_counts: pd.DataFrame,
    sensors: list[str],
) -> pd.DataFrame:
    base = all_keys[["split", "subject_id", "lifelog_date"]].drop_duplicates()

    sensor_frame = pd.DataFrame({"sensor": sensors})
    hour_frame = pd.DataFrame({"hour": list(range(24))})

    base["__key"] = 1
    sensor_frame["__key"] = 1
    hour_frame["__key"] = 1

    full_index = (
        base.merge(sensor_frame, on="__key")
            .merge(hour_frame, on="__key")
            .drop(columns="__key")
    )

    out = full_index.merge(
        sensor_hour_counts,
        on=["subject_id", "lifelog_date", "sensor", "hour"],
        how="left",
    )
    out["row_count"] = out["row_count"].fillna(0).astype(int)
    out["present"] = (out["row_count"] > 0).astype(int)
    out["missing"] = 1 - out["present"]

    return out


def save_heatmap(
    pivot: pd.DataFrame,
    title: str,
    output_path: Path,
    xlabel: str,
    ylabel: str,
) -> None:
    if pivot.empty:
        return

    fig, ax = plt.subplots(figsize=(max(8, pivot.shape[1] * 0.6), max(4, pivot.shape[0] * 0.35)))
    im = ax.imshow(pivot.values, aspect="auto")

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right")

    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Missing Ratio")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def build_missing_by_target(
    day_presence: pd.DataFrame,
    train_keys: pd.DataFrame,
    target_cols: list[str],
) -> pd.DataFrame:
    if not target_cols:
        return pd.DataFrame()

    merged = day_presence[day_presence["split"] == "train"].merge(
        train_keys[["subject_id", "lifelog_date"] + target_cols],
        on=["subject_id", "lifelog_date"],
        how="left",
    )

    rows = []
    for target in target_cols:
        temp = merged.dropna(subset=[target]).copy()
        if temp.empty:
            continue

        grouped = (
            temp.groupby(["sensor", target], dropna=False)
            .agg(
                n=("missing", "size"),
                missing_ratio=("missing", "mean"),
                present_ratio=("present", "mean"),
                mean_row_count=("row_count", "mean"),
            )
            .reset_index()
            .rename(columns={target: "class"})
        )
        grouped["target"] = target
        rows.append(grouped)

    if not rows:
        return pd.DataFrame()

    return pd.concat(rows, ignore_index=True)[
        ["target", "sensor", "class", "n", "missing_ratio", "present_ratio", "mean_row_count"]
    ]


def write_missing_strategy(
    out_dir: Path,
    sensor_missing_train_test: pd.DataFrame,
    missing_by_target: pd.DataFrame,
    skipped_files: pd.DataFrame,
) -> None:
    lines = []
    lines.append("# Missing Analysis Summary\n")

    lines.append("## 1. Sensor Missing Ratio by Train/Test\n")
    if sensor_missing_train_test.empty:
        lines.append("- No sensor missing summary generated.\n")
    else:
        high_missing = sensor_missing_train_test[
            sensor_missing_train_test["missing_ratio"] >= 0.5
        ].sort_values(["split", "missing_ratio"], ascending=[True, False])

        lines.append("### Sensors with missing ratio >= 0.5\n")
        if high_missing.empty:
            lines.append("- None.\n")
        else:
            for _, row in high_missing.iterrows():
                lines.append(
                    f"- [{row['split']}] {row['sensor']}: "
                    f"missing_ratio={row['missing_ratio']:.4f}\n"
                )

    lines.append("\n## 2. Train/Test Missing Shift\n")
    if not sensor_missing_train_test.empty:
        pivot = sensor_missing_train_test.pivot(
            index="sensor",
            columns="split",
            values="missing_ratio",
        )
        if {"train", "test"}.issubset(pivot.columns):
            pivot["abs_diff"] = (pivot["test"] - pivot["train"]).abs()
            shifted = pivot[pivot["abs_diff"] >= 0.2].sort_values("abs_diff", ascending=False)

            if shifted.empty:
                lines.append("- No sensor has train/test missing-ratio gap >= 0.2.\n")
            else:
                for sensor, row in shifted.iterrows():
                    lines.append(
                        f"- {sensor}: train={row['train']:.4f}, "
                        f"test={row['test']:.4f}, abs_diff={row['abs_diff']:.4f}\n"
                    )

    lines.append("\n## 3. Missingness Related to Target\n")
    if missing_by_target.empty:
        lines.append("- missing_by_target.csv is empty.\n")
    else:
        rows = []
        for (target, sensor), g in missing_by_target.groupby(["target", "sensor"]):
            if g["class"].nunique() < 2:
                continue
            diff = g["missing_ratio"].max() - g["missing_ratio"].min()
            rows.append((target, sensor, diff))

        target_diff = pd.DataFrame(rows, columns=["target", "sensor", "missing_ratio_gap"])
        target_diff = target_diff[target_diff["missing_ratio_gap"] >= 0.15]
        target_diff = target_diff.sort_values("missing_ratio_gap", ascending=False)

        if target_diff.empty:
            lines.append("- No strong target-dependent missingness gap >= 0.15.\n")
        else:
            for _, row in target_diff.iterrows():
                lines.append(
                    f"- {row['target']} / {row['sensor']}: "
                    f"missing_ratio_gap={row['missing_ratio_gap']:.4f}\n"
                )

    lines.append("\n## 4. Recommended Actions\n")
    lines.append("- Add `sensor_present` and `sensor_missing` indicators for high-missing sensors.\n")
    lines.append("- Avoid unconditional interpolation before checking hour-level missing patterns.\n")
    lines.append("- For sensors with large train/test missing shift, use robust features or consider excluding unstable features.\n")
    lines.append("- If missingness differs by target class, treat missingness itself as a predictive feature.\n")

    lines.append("\n## 5. Skipped or Error Files\n")
    if skipped_files.empty:
        lines.append("- None.\n")
    else:
        for _, row in skipped_files.iterrows():
            lines.append(f"- {row['sensor']}: {row['status']} / {row['reason']}\n")

    (out_dir / "missing_strategy.md").write_text("".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--items_dir", type=str, default="data/ch2025_data_items")
    parser.add_argument("--out_dir", type=str, default="outputs/eda/05_missing_analysis")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    items_dir = Path(args.items_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_keys, test_keys, target_cols = load_key_frames(data_dir)
    all_keys = pd.concat(
        [
            train_keys[["split", "subject_id", "lifelog_date"]],
            test_keys[["split", "subject_id", "lifelog_date"]],
        ],
        ignore_index=True,
    )

    sensor_files = discover_sensor_files(items_dir)

    all_sensor_rows = []
    meta_rows = []

    print(f"[INFO] Found {len(sensor_files)} sensor files.")

    for path in sensor_files:
        print(f"[INFO] Reading sensor file: {path}")
        df, meta = read_sensor_minimal(path)
        meta_rows.append(meta)

        if meta["status"] != "ok" or df.empty:
            continue

        invalid_count = df["lifelog_date"].isna().sum()
        meta["invalid_date_rows"] = int(invalid_count)

        df = df.dropna(subset=["subject_id", "lifelog_date"])
        if not df.empty:
            all_sensor_rows.append(df)

    sensor_meta = pd.DataFrame(meta_rows)
    sensor_meta.to_csv(out_dir / "sensor_missing_read_status.csv", index=False, encoding="utf-8-sig")

    valid_meta = sensor_meta[sensor_meta["status"] == "ok"].copy()
    sensors = sorted(valid_meta["sensor"].dropna().unique().tolist())

    if not all_sensor_rows:
        print("[WARN] No valid sensor rows found. Check column names and data paths.")
        pd.DataFrame().to_csv(out_dir / "sensor_day_presence.csv", index=False)
        return

    sensor_events = pd.concat(all_sensor_rows, ignore_index=True)
    sensor_events["hour"] = pd.to_numeric(sensor_events["hour"], errors="coerce")

    sensor_daily_counts = (
        sensor_events
        .groupby(["subject_id", "lifelog_date", "sensor"])
        .size()
        .reset_index(name="row_count")
    )

    sensor_hour_counts = (
        sensor_events.dropna(subset=["hour"])
        .assign(hour=lambda x: x["hour"].astype(int))
        .groupby(["subject_id", "lifelog_date", "sensor", "hour"])
        .size()
        .reset_index(name="row_count")
    )

    day_presence = make_day_presence(all_keys, sensor_daily_counts, sensors)
    hour_presence = make_hour_presence(all_keys, sensor_hour_counts, sensors)

    day_presence.to_csv(
        out_dir / "sensor_day_presence.csv",
        index=False,
        encoding="utf-8-sig",
    )

    subject_missing = (
        day_presence
        .groupby(["split", "subject_id", "sensor"])
        .agg(
            n_days=("missing", "size"),
            missing_ratio=("missing", "mean"),
            present_ratio=("present", "mean"),
            mean_row_count=("row_count", "mean"),
        )
        .reset_index()
    )
    subject_missing.to_csv(
        out_dir / "sensor_missing_ratio_by_subject.csv",
        index=False,
        encoding="utf-8-sig",
    )

    hour_missing = (
        hour_presence
        .groupby(["split", "sensor", "hour"])
        .agg(
            n_slots=("missing", "size"),
            missing_ratio=("missing", "mean"),
            present_ratio=("present", "mean"),
            mean_row_count=("row_count", "mean"),
        )
        .reset_index()
    )
    hour_missing.to_csv(
        out_dir / "sensor_missing_ratio_by_hour.csv",
        index=False,
        encoding="utf-8-sig",
    )

    train_test_missing = (
        day_presence
        .groupby(["split", "sensor"])
        .agg(
            n_days=("missing", "size"),
            missing_ratio=("missing", "mean"),
            present_ratio=("present", "mean"),
            mean_row_count=("row_count", "mean"),
        )
        .reset_index()
    )
    train_test_missing.to_csv(
        out_dir / "sensor_missing_ratio_train_test.csv",
        index=False,
        encoding="utf-8-sig",
    )

    missing_by_target = build_missing_by_target(day_presence, train_keys, target_cols)
    missing_by_target.to_csv(
        out_dir / "missing_by_target.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # Heatmap 1: subject x sensor
    subject_pivot = (
        subject_missing[subject_missing["split"] == "train"]
        .pivot_table(
            index="subject_id",
            columns="sensor",
            values="missing_ratio",
            aggfunc="mean",
        )
        .fillna(1.0)
    )
    save_heatmap(
        subject_pivot,
        title="Train Missing Ratio by Subject and Sensor",
        output_path=out_dir / "missing_heatmap_subject_sensor.png",
        xlabel="Sensor",
        ylabel="Subject",
    )

    # Heatmap 2: hour x sensor
    hour_pivot = (
        hour_missing[hour_missing["split"] == "train"]
        .pivot_table(
            index="hour",
            columns="sensor",
            values="missing_ratio",
            aggfunc="mean",
        )
        .fillna(1.0)
        .sort_index()
    )
    save_heatmap(
        hour_pivot,
        title="Train Missing Ratio by Hour and Sensor",
        output_path=out_dir / "missing_heatmap_hour_sensor.png",
        xlabel="Sensor",
        ylabel="Hour",
    )

    skipped_files = sensor_meta[sensor_meta["status"] != "ok"].copy()
    write_missing_strategy(
        out_dir=out_dir,
        sensor_missing_train_test=train_test_missing,
        missing_by_target=missing_by_target,
        skipped_files=skipped_files,
    )

    print("[DONE] 05_missing_analysis artifacts saved to:", out_dir)


if __name__ == "__main__":
    main()