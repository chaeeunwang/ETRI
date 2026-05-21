# src/eda/eda_06_sensor_statistics.py

import ast
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


# =========================
# Config
# =========================

DATA_DIR = Path("./data")
ITEMS_DIR = DATA_DIR / "ch2025_data_items"

TRAIN_PATH = DATA_DIR / "ch2026_metrics_train.csv"
TEST_PATH = DATA_DIR / "ch2026_submission_sample.csv"

OUTPUT_DIR = Path("./outputs/eda/06_sensor_statistics")
PLOT_DIR = OUTPUT_DIR / "sensor_distribution_plots"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)

TARGETS = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]

# 시간대 버전 A: 기존 계획 기준
TIME_BLOCKS_A = [
    ("00_06", 0, 6),
    ("06_09", 6, 9),
    ("09_12", 9, 12),
    ("12_15", 12, 15),
    ("15_18", 15, 18),
    ("18_21", 18, 21),
    ("21_24", 21, 24),
]

# 시간대 버전 B: MIS-LSTM 참고 4시간 block
TIME_BLOCKS_B = [
    ("00_04", 0, 4),
    ("04_08", 4, 8),
    ("08_12", 8, 12),
    ("12_16", 12, 16),
    ("16_20", 16, 20),
    ("20_24", 20, 24),
]

HIGH_PRIORITY_SENSORS = {
    "wPedo",
    "wHr",
    "mUsageStats",
    "wLight",
    "mLight",
    "mActivity",
    "mScreenStatus",
    "mACStatus",
}

DISCRETE_HINTS = {
    "mActivity",
    "mScreenStatus",
    "mACStatus",
}

OBJECT_HINTS = {
    "mUsageStats",
    "mAmbience",
    "mWifi",
    "mBle",
    "mGps",
}


# =========================
# Utility
# =========================

def safe_read_table(path: Path) -> pd.DataFrame | None:
    try:
        if path.suffix.lower() == ".csv":
            return pd.read_csv(path)
        if path.suffix.lower() in [".parquet", ".pq"]:
            return pd.read_parquet(path)
    except Exception as e:
        print(f"[WARN] Failed to read {path}: {e}")
        return None
    return None


def find_sensor_files(items_dir: Path) -> list[Path]:
    files = []
    for ext in ["*.csv", "*.parquet", "*.pq"]:
        files.extend(items_dir.rglob(ext))
    return sorted(files)


def infer_sensor_name(path: Path) -> str:
    """
    파일명이 센서명인 경우를 우선 가정합니다.
    예: data/ch2025_data_items/wHr/*.parquet -> wHr
        data/ch2025_data_items/wHr.csv -> wHr
    """
    parent_name = path.parent.name
    stem = path.stem

    known_prefixes = [
        "mACStatus", "mActivity", "mAmbience", "mBle", "mGps",
        "mLight", "mScreenStatus", "mUsageStats", "mWifi",
        "wHr", "wLight", "wPedo"
    ]

    for name in known_prefixes:
        if name.lower() in parent_name.lower():
            return name
        if name.lower() in stem.lower():
            return name

    return parent_name if parent_name != "ch2025_data_items" else stem


def find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]

    for col in df.columns:
        col_lower = col.lower()
        for cand in candidates:
            if cand.lower() in col_lower:
                return col

    return None


def standardize_base_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    subject_col = find_col(df, ["subject_id", "subject", "user_id", "id"])
    timestamp_col = find_col(df, ["timestamp", "datetime", "time", "created_at", "ts"])
    date_col = find_col(df, ["lifelog_date", "date"])

    if subject_col and subject_col != "subject_id":
        df = df.rename(columns={subject_col: "subject_id"})

    if timestamp_col and timestamp_col != "timestamp":
        df = df.rename(columns={timestamp_col: "timestamp"})

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df["hour"] = df["timestamp"].dt.hour
        df["lifelog_date"] = df["timestamp"].dt.date.astype(str)
    elif date_col:
        if date_col != "lifelog_date":
            df = df.rename(columns={date_col: "lifelog_date"})
        df["lifelog_date"] = pd.to_datetime(df["lifelog_date"], errors="coerce").dt.date.astype(str)

    return df


def classify_sensor_type(sensor_name: str, df: pd.DataFrame) -> str:
    if sensor_name in OBJECT_HINTS:
        return "object"
    if sensor_name in DISCRETE_HINTS:
        return "discrete"

    object_cols = df.select_dtypes(include=["object"]).columns.tolist()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    base_cols = {"subject_id", "hour"}
    numeric_value_cols = [c for c in numeric_cols if c not in base_cols]

    if len(numeric_value_cols) >= 1:
        return "continuous"

    if len(object_cols) >= 1:
        return "object"

    return "unknown"


def get_value_columns(df: pd.DataFrame) -> list[str]:
    exclude = {
        "subject_id",
        "timestamp",
        "lifelog_date",
        "sleep_date",
        "hour",
        "date",
        "id",
    }
    return [c for c in df.columns if c not in exclude]


def numeric_value_columns(df: pd.DataFrame) -> list[str]:
    cols = get_value_columns(df)
    return [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]


def object_value_columns(df: pd.DataFrame) -> list[str]:
    cols = get_value_columns(df)
    return [c for c in cols if df[c].dtype == "object"]


def add_time_block(df: pd.DataFrame, blocks: list[tuple[str, int, int]], block_col: str) -> pd.DataFrame:
    df = df.copy()

    df[block_col] = pd.Series(pd.NA, index=df.index, dtype="object")

    if "hour" not in df.columns:
        return df

    hour = pd.to_numeric(df["hour"], errors="coerce")

    for name, start, end in blocks:
        mask = hour.ge(start) & hour.lt(end)
        df.loc[mask, block_col] = name

    return df


def iqr_outlier_count(series: pd.Series) -> int:
    x = series.dropna()
    if len(x) == 0:
        return 0

    q1 = x.quantile(0.25)
    q3 = x.quantile(0.75)
    iqr = q3 - q1

    if iqr == 0:
        return 0

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    return int(((x < lower) | (x > upper)).sum())


def safe_parse_object(value):
    # 1. 이미 list/dict/tuple/np.ndarray인 경우 먼저 처리
    if isinstance(value, (list, dict, tuple)):
        return value

    if isinstance(value, np.ndarray):
        return value.tolist()

    # 2. scalar 결측만 pd.isna로 처리
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    # 3. 문자열 처리
    text = str(value).strip()
    if not text or text.lower() in ["nan", "none", "null"]:
        return None

    try:
        return json.loads(text)
    except Exception:
        pass

    try:
        return ast.literal_eval(text)
    except Exception:
        return text


def object_length(value) -> int:
    parsed = safe_parse_object(value)

    if parsed is None:
        return 0

    if isinstance(parsed, np.ndarray):
        return len(parsed)

    if isinstance(parsed, (list, tuple)):
        return len(parsed)

    if isinstance(parsed, dict):
        return len(parsed.keys())

    if isinstance(parsed, str):
        return 1 if parsed else 0

    return 1


def save_distribution_plot(df: pd.DataFrame, sensor_name: str, col: str):
    values = df[col].replace([np.inf, -np.inf], np.nan).dropna()

    if len(values) == 0:
        return

    # 너무 크면 샘플링
    if len(values) > 100_000:
        values = values.sample(100_000, random_state=42)

    plt.figure(figsize=(8, 5))
    plt.hist(values, bins=50)
    plt.title(f"{sensor_name} - {col}")
    plt.xlabel(col)
    plt.ylabel("count")
    plt.tight_layout()

    safe_col = str(col).replace("/", "_").replace("\\", "_").replace(" ", "_")
    plt.savefig(PLOT_DIR / f"{sensor_name}__{safe_col}.png", dpi=150)
    plt.close()


def log_transform_decision(stats: dict) -> tuple[bool, str]:
    zero_ratio = stats.get("zero_ratio", np.nan)
    min_value = stats.get("min", np.nan)
    q50 = stats.get("q50", np.nan)
    q99 = stats.get("q99", np.nan)
    skew = stats.get("skew", np.nan)

    if pd.isna(min_value):
        return False, "invalid"

    if min_value < 0:
        return False, "negative_values_exist"

    if pd.notna(q50) and pd.notna(q99) and q50 > 0 and q99 / q50 >= 10:
        return True, "long_tail_q99_over_10x_median"

    if pd.notna(skew) and skew >= 2:
        return True, "high_skew"

    if pd.notna(zero_ratio) and zero_ratio >= 0.7 and pd.notna(q99) and q99 > 0:
        return True, "zero_inflated_positive"

    return False, "not_needed"


def load_train_targets() -> pd.DataFrame:
    train = pd.read_csv(TRAIN_PATH)
    train["lifelog_date"] = pd.to_datetime(train["lifelog_date"], errors="coerce").dt.date.astype(str)

    available_targets = [t for t in TARGETS if t in train.columns]
    keep_cols = ["subject_id", "lifelog_date"] + available_targets

    return train[keep_cols]


# =========================
# Main EDA
# =========================

def main():
    print("[INFO] Start 06_sensor_statistics EDA")

    sensor_files = find_sensor_files(ITEMS_DIR)
    if not sensor_files:
        raise FileNotFoundError(f"No sensor files found under {ITEMS_DIR}")

    train_targets = load_train_targets()
    available_targets = [t for t in TARGETS if t in train_targets.columns]

    continuous_rows = []
    discrete_rows = []
    object_rows = []
    outlier_rows = []
    log_candidate_rows = []

    timeblock_stat_rows = []
    timeblock_coverage_rows = []
    night_summary_rows = []
    pre_sleep_summary_rows = []

    daily_feature_frames = []

    for file_path in sensor_files:
        sensor_name = infer_sensor_name(file_path)
        print(f"[INFO] Processing {sensor_name}: {file_path}")

        df = safe_read_table(file_path)
        if df is None or df.empty:
            continue

        df = standardize_base_columns(df)

        if "lifelog_date" not in df.columns:
            print(f"[WARN] Skip {sensor_name}: no lifelog_date/timestamp")
            continue

        sensor_type = classify_sensor_type(sensor_name, df)
        n_rows = len(df)
        n_subjects = df["subject_id"].nunique() if "subject_id" in df.columns else np.nan

        num_cols = numeric_value_columns(df)
        obj_cols = object_value_columns(df)

        # -------------------------
        # Continuous statistics
        # -------------------------
        if sensor_type == "continuous" and num_cols:
            daily_parts = []

            for col in num_cols:
                s = df[col].replace([np.inf, -np.inf], np.nan)

                stats = {
                    "sensor": sensor_name,
                    "column": col,
                    "row_count": n_rows,
                    "subject_count": n_subjects,
                    "non_null_count": int(s.notna().sum()),
                    "missing_ratio": float(s.isna().mean()),
                    "mean": float(s.mean()) if s.notna().any() else np.nan,
                    "std": float(s.std()) if s.notna().any() else np.nan,
                    "min": float(s.min()) if s.notna().any() else np.nan,
                    "q25": float(s.quantile(0.25)) if s.notna().any() else np.nan,
                    "q50": float(s.quantile(0.50)) if s.notna().any() else np.nan,
                    "q75": float(s.quantile(0.75)) if s.notna().any() else np.nan,
                    "q95": float(s.quantile(0.95)) if s.notna().any() else np.nan,
                    "q99": float(s.quantile(0.99)) if s.notna().any() else np.nan,
                    "max": float(s.max()) if s.notna().any() else np.nan,
                    "zero_ratio": float((s == 0).mean()) if s.notna().any() else np.nan,
                    "negative_ratio": float((s < 0).mean()) if s.notna().any() else np.nan,
                    "unique_count": int(s.nunique(dropna=True)),
                    "skew": float(s.skew()) if s.notna().sum() > 2 else np.nan,
                }
                continuous_rows.append(stats)

                outlier_rows.append({
                    "sensor": sensor_name,
                    "column": col,
                    "outlier_count_iqr": iqr_outlier_count(s),
                    "outlier_ratio_iqr": iqr_outlier_count(s) / max(1, s.notna().sum()),
                    "min": stats["min"],
                    "max": stats["max"],
                    "q99": stats["q99"],
                })

                use_log, reason = log_transform_decision(stats)
                log_candidate_rows.append({
                    "sensor": sensor_name,
                    "column": col,
                    "log1p_candidate": use_log,
                    "reason": reason,
                    "min": stats["min"],
                    "q50": stats["q50"],
                    "q99": stats["q99"],
                    "skew": stats["skew"],
                    "zero_ratio": stats["zero_ratio"],
                })

                save_distribution_plot(df, sensor_name, col)

                # daily aggregate feature
                if "subject_id" in df.columns:
                    daily_agg = (
                        df.groupby(["subject_id", "lifelog_date"])[col]
                        .agg(["mean", "std", "min", "max", "sum", "median", "count"])
                        .reset_index()
                    )
                    daily_agg = daily_agg.rename(columns={
                        "mean": f"{sensor_name}_{col}_mean",
                        "std": f"{sensor_name}_{col}_std",
                        "min": f"{sensor_name}_{col}_min",
                        "max": f"{sensor_name}_{col}_max",
                        "sum": f"{sensor_name}_{col}_sum",
                        "median": f"{sensor_name}_{col}_median",
                        "count": f"{sensor_name}_{col}_count",
                    })
                    daily_parts.append(daily_agg)

            if daily_parts:
                merged = daily_parts[0]
                for part in daily_parts[1:]:
                    merged = merged.merge(part, on=["subject_id", "lifelog_date"], how="outer")
                daily_feature_frames.append(merged)

        # -------------------------
        # Discrete statistics
        # -------------------------
        elif sensor_type == "discrete" and num_cols:
            daily_parts = []

            for col in num_cols:
                vc = df[col].value_counts(dropna=False).reset_index()
                vc.columns = ["value", "count"]
                vc["sensor"] = sensor_name
                vc["column"] = col
                vc["ratio"] = vc["count"] / max(1, len(df))

                for _, row in vc.iterrows():
                    discrete_rows.append({
                        "sensor": sensor_name,
                        "column": col,
                        "value": row["value"],
                        "count": int(row["count"]),
                        "ratio": float(row["ratio"]),
                        "row_count": n_rows,
                        "subject_count": n_subjects,
                    })

                save_distribution_plot(df, sensor_name, col)

                if "subject_id" in df.columns:
                    # value별 하루 count
                    daily_count = (
                        df.groupby(["subject_id", "lifelog_date", col])
                        .size()
                        .reset_index(name="count")
                    )

                    pivot = daily_count.pivot_table(
                        index=["subject_id", "lifelog_date"],
                        columns=col,
                        values="count",
                        fill_value=0,
                    ).reset_index()

                    pivot.columns = [
                        "subject_id" if c == "subject_id" else
                        "lifelog_date" if c == "lifelog_date" else
                        f"{sensor_name}_{col}_value_{c}_count"
                        for c in pivot.columns
                    ]

                    daily_parts.append(pivot)

            if daily_parts:
                merged = daily_parts[0]
                for part in daily_parts[1:]:
                    merged = merged.merge(part, on=["subject_id", "lifelog_date"], how="outer")
                daily_feature_frames.append(merged)

        # -------------------------
        # Object/list statistics
        # -------------------------
        else:
            if obj_cols:
                daily_parts = []

                for col in obj_cols:
                    lengths = df[col].apply(object_length)

                    object_rows.append({
                        "sensor": sensor_name,
                        "column": col,
                        "row_count": n_rows,
                        "subject_count": n_subjects,
                        "non_null_count": int(df[col].notna().sum()),
                        "missing_ratio": float(df[col].isna().mean()),
                        "parsed_length_mean": float(lengths.mean()),
                        "parsed_length_std": float(lengths.std()),
                        "parsed_length_max": int(lengths.max()) if len(lengths) else 0,
                        "zero_length_ratio": float((lengths == 0).mean()),
                    })

                    if "subject_id" in df.columns:
                        temp = df[["subject_id", "lifelog_date"]].copy()
                        temp[f"{sensor_name}_{col}_object_len"] = lengths

                        daily_agg = (
                            temp.groupby(["subject_id", "lifelog_date"])[f"{sensor_name}_{col}_object_len"]
                            .agg(["mean", "max", "sum", "count"])
                            .reset_index()
                        )

                        daily_agg = daily_agg.rename(columns={
                            "mean": f"{sensor_name}_{col}_object_len_mean",
                            "max": f"{sensor_name}_{col}_object_len_max",
                            "sum": f"{sensor_name}_{col}_object_len_sum",
                            "count": f"{sensor_name}_{col}_object_len_count",
                        })

                        daily_parts.append(daily_agg)

                if daily_parts:
                    merged = daily_parts[0]
                    for part in daily_parts[1:]:
                        merged = merged.merge(part, on=["subject_id", "lifelog_date"], how="outer")
                    daily_feature_frames.append(merged)

        # -------------------------
        # Time block statistics
        # -------------------------
        if "hour" in df.columns and "subject_id" in df.columns:
            for block_version, blocks in [("A", TIME_BLOCKS_A), ("B_4hour", TIME_BLOCKS_B)]:
                temp = add_time_block(df, blocks, f"time_block_{block_version}")
                block_col = f"time_block_{block_version}"
                valid = temp[temp[block_col].notna()].copy()

                if valid.empty:
                    continue

                # coverage
                coverage = (
                    valid.groupby(["subject_id", "lifelog_date", block_col])
                    .size()
                    .reset_index(name="row_count")
                )
                coverage["sensor"] = sensor_name
                coverage["block_version"] = block_version

                timeblock_coverage_rows.extend(coverage.to_dict("records"))

                # numeric block stats
                for col in num_cols:
                    block_stats = (
                        valid.groupby(["subject_id", "lifelog_date", block_col])[col]
                        .agg(["mean", "std", "min", "max", "sum", "count"])
                        .reset_index()
                    )
                    block_stats["sensor"] = sensor_name
                    block_stats["column"] = col
                    block_stats["block_version"] = block_version

                    timeblock_stat_rows.extend(block_stats.to_dict("records"))

                # night/pre-sleep summary
                night = valid[valid["hour"].between(0, 5)]
                pre_sleep = valid[valid["hour"].between(21, 23)]

                for col in num_cols:
                    if not night.empty:
                        night_summary_rows.append({
                            "sensor": sensor_name,
                            "column": col,
                            "mean": night[col].mean(),
                            "std": night[col].std(),
                            "sum": night[col].sum(),
                            "count": night[col].count(),
                        })

                    if not pre_sleep.empty:
                        pre_sleep_summary_rows.append({
                            "sensor": sensor_name,
                            "column": col,
                            "mean": pre_sleep[col].mean(),
                            "std": pre_sleep[col].std(),
                            "sum": pre_sleep[col].sum(),
                            "count": pre_sleep[col].count(),
                        })

    # =========================
    # Save sensor-level outputs
    # =========================

    pd.DataFrame(continuous_rows).to_csv(
        OUTPUT_DIR / "continuous_sensor_stats.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(discrete_rows).to_csv(
        OUTPUT_DIR / "discrete_sensor_stats.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(object_rows).to_csv(
        OUTPUT_DIR / "object_sensor_parse_stats.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(outlier_rows).to_csv(
        OUTPUT_DIR / "sensor_outlier_summary.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(log_candidate_rows).to_csv(
        OUTPUT_DIR / "log_transform_candidates.csv", index=False, encoding="utf-8-sig"
    )

    pd.DataFrame(timeblock_coverage_rows).to_csv(
        OUTPUT_DIR / "timeblock_feature_coverage.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(timeblock_stat_rows).to_csv(
        OUTPUT_DIR / "timeblock_sensor_stats.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(night_summary_rows).to_csv(
        OUTPUT_DIR / "night_feature_summary.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(pre_sleep_summary_rows).to_csv(
        OUTPUT_DIR / "pre_sleep_feature_summary.csv", index=False, encoding="utf-8-sig"
    )

    # =========================
    # Daily feature table
    # =========================

    if daily_feature_frames:
        daily_features = daily_feature_frames[0]
        for part in daily_feature_frames[1:]:
            daily_features = daily_features.merge(
                part, on=["subject_id", "lifelog_date"], how="outer"
            )

        daily_features.to_csv(
            OUTPUT_DIR / "daily_sensor_features_for_analysis.csv",
            index=False,
            encoding="utf-8-sig",
        )

        merged = train_targets.merge(
            daily_features,
            on=["subject_id", "lifelog_date"],
            how="left",
        )

        feature_cols = [
            c for c in merged.columns
            if c not in ["subject_id", "lifelog_date"] + available_targets
            and pd.api.types.is_numeric_dtype(merged[c])
        ]

        # =========================
        # Feature-target correlation
        # =========================

        corr_rows = []
        effect_rows = []

        for target in available_targets:
            for feat in feature_cols:
                temp = merged[[target, feat]].dropna()
                if len(temp) < 10:
                    continue

                # correlation
                if temp[target].nunique() > 1 and temp[feat].nunique() > 1:
                    pearson = temp[target].corr(temp[feat], method="pearson")
                    spearman = temp[target].corr(temp[feat], method="spearman")
                else:
                    pearson = np.nan
                    spearman = np.nan

                corr_rows.append({
                    "target": target,
                    "feature": feat,
                    "n": len(temp),
                    "pearson": pearson,
                    "spearman": spearman,
                    "abs_pearson": abs(pearson) if pd.notna(pearson) else np.nan,
                    "abs_spearman": abs(spearman) if pd.notna(spearman) else np.nan,
                })

                # effect size for class difference
                class_values = sorted(temp[target].dropna().unique())

                if len(class_values) == 2:
                    c0, c1 = class_values[0], class_values[1]
                    x0 = temp.loc[temp[target] == c0, feat].dropna()
                    x1 = temp.loc[temp[target] == c1, feat].dropna()

                    pooled_std = temp[feat].std()
                    effect_size = (x1.mean() - x0.mean()) / pooled_std if pooled_std and pooled_std > 0 else np.nan

                    effect_rows.append({
                        "target": target,
                        "feature": feat,
                        "class_low": c0,
                        "class_high": c1,
                        "mean_low": x0.mean(),
                        "mean_high": x1.mean(),
                        "diff_high_minus_low": x1.mean() - x0.mean(),
                        "effect_size": effect_size,
                        "abs_effect_size": abs(effect_size) if pd.notna(effect_size) else np.nan,
                        "n_low": len(x0),
                        "n_high": len(x1),
                    })

                elif len(class_values) == 3:
                    grouped = temp.groupby(target)[feat].mean().to_dict()
                    max_mean = max(grouped.values())
                    min_mean = min(grouped.values())
                    pooled_std = temp[feat].std()

                    effect_size = (max_mean - min_mean) / pooled_std if pooled_std and pooled_std > 0 else np.nan

                    effect_rows.append({
                        "target": target,
                        "feature": feat,
                        "class_low": class_values[0],
                        "class_high": class_values[-1],
                        "mean_low": grouped.get(class_values[0], np.nan),
                        "mean_high": grouped.get(class_values[-1], np.nan),
                        "diff_high_minus_low": grouped.get(class_values[-1], np.nan) - grouped.get(class_values[0], np.nan),
                        "effect_size": effect_size,
                        "abs_effect_size": abs(effect_size) if pd.notna(effect_size) else np.nan,
                        "n_low": int((temp[target] == class_values[0]).sum()),
                        "n_high": int((temp[target] == class_values[-1]).sum()),
                    })

        corr_df = pd.DataFrame(corr_rows)
        effect_df = pd.DataFrame(effect_rows)

        corr_df.to_csv(
            OUTPUT_DIR / "feature_target_correlation.csv",
            index=False,
            encoding="utf-8-sig",
        )
        effect_df.to_csv(
            OUTPUT_DIR / "feature_target_effect_size.csv",
            index=False,
            encoding="utf-8-sig",
        )

        # timeblock correlation은 현재 daily feature 기반으로 대체 저장
        # 실제 timeblock-level feature를 wide table로 만들 경우 별도 확장 가능
        corr_df.to_csv(
            OUTPUT_DIR / "timeblock_target_correlation.csv",
            index=False,
            encoding="utf-8-sig",
        )

        top_rows = []

        if not corr_df.empty:
            corr_top = (
                corr_df.sort_values(["target", "abs_spearman"], ascending=[True, False])
                .groupby("target")
                .head(30)
            )
            corr_top["rank_type"] = "correlation"
            top_rows.append(corr_top)

        if not effect_df.empty:
            effect_top = (
                effect_df.sort_values(["target", "abs_effect_size"], ascending=[True, False])
                .groupby("target")
                .head(30)
            )
            effect_top["rank_type"] = "effect_size"
            top_rows.append(effect_top)

        if top_rows:
            top_features = pd.concat(top_rows, ignore_index=True, sort=False)
            top_features.to_csv(
                OUTPUT_DIR / "top_features_by_target.csv",
                index=False,
                encoding="utf-8-sig",
            )

        write_feature_target_summary(corr_df, effect_df)

    else:
        print("[WARN] No daily feature frames were generated.")

    write_sensor_statistics_summary()

    print(f"[DONE] 06_sensor_statistics outputs saved to: {OUTPUT_DIR}")


def write_feature_target_summary(corr_df: pd.DataFrame, effect_df: pd.DataFrame):
    lines = []
    lines.append("# Feature-Target Summary")
    lines.append("")

    if corr_df.empty and effect_df.empty:
        lines.append("- feature-target 관계를 계산할 수 있는 daily feature가 부족합니다.")
    else:
        for target in sorted(set(corr_df.get("target", [])) | set(effect_df.get("target", []))):
            lines.append(f"## {target}")
            lines.append("")

            if not corr_df.empty:
                top_corr = (
                    corr_df[corr_df["target"] == target]
                    .sort_values("abs_spearman", ascending=False)
                    .head(10)
                )
                lines.append("### Top correlation features")
                for _, row in top_corr.iterrows():
                    lines.append(
                        f"- {row['feature']}: spearman={row['spearman']:.4f}, pearson={row['pearson']:.4f}, n={int(row['n'])}"
                    )
                lines.append("")

            if not effect_df.empty:
                top_eff = (
                    effect_df[effect_df["target"] == target]
                    .sort_values("abs_effect_size", ascending=False)
                    .head(10)
                )
                lines.append("### Top effect-size features")
                for _, row in top_eff.iterrows():
                    lines.append(
                        f"- {row['feature']}: effect_size={row['effect_size']:.4f}, diff={row['diff_high_minus_low']:.4f}"
                    )
                lines.append("")

    (OUTPUT_DIR / "feature_target_summary.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def write_sensor_statistics_summary():
    lines = []
    lines.append("# Sensor Statistics Summary")
    lines.append("")

    paths = {
        "continuous": OUTPUT_DIR / "continuous_sensor_stats.csv",
        "discrete": OUTPUT_DIR / "discrete_sensor_stats.csv",
        "object": OUTPUT_DIR / "object_sensor_parse_stats.csv",
        "outlier": OUTPUT_DIR / "sensor_outlier_summary.csv",
        "log": OUTPUT_DIR / "log_transform_candidates.csv",
    }

    for name, path in paths.items():
        if path.exists():
            df = pd.read_csv(path)
            lines.append(f"## {name}")
            lines.append(f"- rows: {len(df)}")
            lines.append("")

    log_path = OUTPUT_DIR / "log_transform_candidates.csv"
    if log_path.exists():
        log_df = pd.read_csv(log_path)
        if "log1p_candidate" in log_df.columns:
            candidates = log_df[log_df["log1p_candidate"] == True]
            lines.append("## Log Transform Candidates")
            if candidates.empty:
                lines.append("- log1p 후보가 뚜렷하지 않습니다.")
            else:
                for _, row in candidates.head(30).iterrows():
                    lines.append(f"- {row['sensor']}.{row['column']}: {row['reason']}")
            lines.append("")

    outlier_path = OUTPUT_DIR / "sensor_outlier_summary.csv"
    if outlier_path.exists():
        out_df = pd.read_csv(outlier_path)
        if "outlier_ratio_iqr" in out_df.columns and not out_df.empty:
            top_outliers = out_df.sort_values("outlier_ratio_iqr", ascending=False).head(20)
            lines.append("## Top Outlier Features")
            for _, row in top_outliers.iterrows():
                lines.append(
                    f"- {row['sensor']}.{row['column']}: outlier_ratio={row['outlier_ratio_iqr']:.4f}"
                )
            lines.append("")

    (OUTPUT_DIR / "sensor_statistics_summary.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


if __name__ == "__main__":
    main()