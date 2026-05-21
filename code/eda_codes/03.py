"""
03_timestamp_check.py

Purpose:
- Check timestamp consistency and leakage risk.
- Validate relationship between lifelog_date and sleep_date.
- Inspect sensor timestamp ranges.
- Diagnose time-aware fold date ranges.

Expected output:
outputs/eda/03_timestamp_check/
├── sleep_lifelog_date_diff.csv
├── sensor_timestamp_minmax.csv
├── invalid_timestamp_rows.csv
├── fold_date_range.csv
└── leakage_risk_report.md
"""

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import json
import pandas as pd
import numpy as np


# =========================
# Config
# =========================

DATA_DIR = Path("./data")
ITEMS_DIR = DATA_DIR / "ch2025_data_items"
OUTPUT_DIR = Path("./outputs/eda/03_timestamp_check")

TRAIN_PATH = DATA_DIR / "ch2026_metrics_train.csv"
TEST_PATH = DATA_DIR / "ch2026_submission_sample.csv"

N_FOLDS = 5

DATE_COLUMNS = ["lifelog_date", "sleep_date"]
SUBJECT_COL_CANDIDATES = ["subject_id", "subject", "id", "user_id"]

TIMESTAMP_CANDIDATES = [
    "timestamp",
    "time",
    "datetime",
    "dateTime",
    "created_at",
    "start_time",
    "end_time",
    "ts",
]

DATE_CANDIDATES = [
    "lifelog_date",
    "sleep_date",
    "date",
    "day",
]


# =========================
# Utils
# =========================

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


def safe_to_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce")


def read_table(path: Path) -> pd.DataFrame | None:
    try:
        if path.suffix.lower() == ".csv":
            return pd.read_csv(path)
        if path.suffix.lower() in [".parquet", ".pq"]:
            return pd.read_parquet(path)
        if path.suffix.lower() == ".json":
            return pd.read_json(path)
        return None
    except Exception as e:
        print(f"[WARN] Failed to read {path}: {e}")
        return None


def list_sensor_files(items_dir: Path) -> list[Path]:
    if not items_dir.exists():
        print(f"[WARN] ITEMS_DIR does not exist: {items_dir}")
        return []

    exts = {".csv", ".parquet", ".pq", ".json"}
    files = [p for p in items_dir.rglob("*") if p.is_file() and p.suffix.lower() in exts]
    return sorted(files)


def infer_sensor_name(path: Path) -> str:
    """
    Example:
    data/ch2025_data_items/wHr/xxx.parquet -> wHr
    data/ch2025_data_items/mActivity.csv -> mActivity
    """
    try:
        rel = path.relative_to(ITEMS_DIR)
        if len(rel.parts) >= 2:
            return rel.parts[0]
        return path.stem
    except Exception:
        return path.stem


def load_main_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    for df in [train_df, test_df]:
        for col in DATE_COLUMNS:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")

    return train_df, test_df


# =========================
# 1. sleep_date - lifelog_date
# =========================

def analyze_sleep_lifelog_diff(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for split_name, df in [("train", train_df), ("test", test_df)]:
        if "lifelog_date" not in df.columns or "sleep_date" not in df.columns:
            rows.append({
                "split": split_name,
                "status": "missing lifelog_date or sleep_date",
                "diff_days": np.nan,
                "count": len(df),
                "ratio": 1.0,
            })
            continue

        temp = df.copy()
        temp["lifelog_date"] = pd.to_datetime(temp["lifelog_date"], errors="coerce")
        temp["sleep_date"] = pd.to_datetime(temp["sleep_date"], errors="coerce")
        temp["diff_days"] = (temp["sleep_date"] - temp["lifelog_date"]).dt.days

        vc = temp["diff_days"].value_counts(dropna=False).sort_index()
        for diff_days, count in vc.items():
            rows.append({
                "split": split_name,
                "status": "ok",
                "diff_days": diff_days,
                "count": int(count),
                "ratio": float(count / len(temp)) if len(temp) else np.nan,
            })

    return pd.DataFrame(rows)


# =========================
# 2. sensor timestamp min/max
# =========================

def analyze_sensor_timestamp_minmax(sensor_files: list[Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows = []
    invalid_rows = []

    for path in sensor_files:
        sensor_name = infer_sensor_name(path)
        df = read_table(path)

        if df is None or df.empty:
            summary_rows.append({
                "sensor": sensor_name,
                "file_path": str(path),
                "row_count": 0 if df is None else len(df),
                "status": "empty_or_read_failed",
                "subject_col": None,
                "timestamp_col": None,
                "date_col": None,
                "timestamp_min": None,
                "timestamp_max": None,
                "invalid_timestamp_count": None,
                "invalid_timestamp_ratio": None,
                "unique_subjects": None,
            })
            continue

        subject_col = find_col(df, SUBJECT_COL_CANDIDATES)
        timestamp_col = find_col(df, TIMESTAMP_CANDIDATES)
        date_col = find_col(df, DATE_CANDIDATES)

        if timestamp_col is None:
            summary_rows.append({
                "sensor": sensor_name,
                "file_path": str(path),
                "row_count": len(df),
                "status": "missing_timestamp_col",
                "subject_col": subject_col,
                "timestamp_col": None,
                "date_col": date_col,
                "timestamp_min": None,
                "timestamp_max": None,
                "invalid_timestamp_count": None,
                "invalid_timestamp_ratio": None,
                "unique_subjects": df[subject_col].nunique() if subject_col else None,
            })
            continue

        ts = safe_to_datetime(df[timestamp_col])
        invalid_mask = ts.isna()

        summary_rows.append({
            "sensor": sensor_name,
            "file_path": str(path),
            "row_count": len(df),
            "status": "ok",
            "subject_col": subject_col,
            "timestamp_col": timestamp_col,
            "date_col": date_col,
            "timestamp_min": ts.min(),
            "timestamp_max": ts.max(),
            "invalid_timestamp_count": int(invalid_mask.sum()),
            "invalid_timestamp_ratio": float(invalid_mask.mean()),
            "unique_subjects": df[subject_col].nunique() if subject_col else None,
        })

        if invalid_mask.any():
            invalid_sample = df.loc[invalid_mask].head(50).copy()
            invalid_sample.insert(0, "sensor", sensor_name)
            invalid_sample.insert(1, "file_path", str(path))
            invalid_rows.append(invalid_sample)

    sensor_summary = pd.DataFrame(summary_rows)

    if invalid_rows:
        invalid_df = pd.concat(invalid_rows, ignore_index=True)
    else:
        invalid_df = pd.DataFrame(columns=["sensor", "file_path", "reason"])

    return sensor_summary, invalid_df


# =========================
# 3. Check sensor timestamp against lifelog_date
# =========================

def analyze_sensor_date_boundary(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    sensor_files: list[Path],
) -> pd.DataFrame:
    """
    This checks whether sensor timestamps roughly fall within known lifelog_date ranges.
    It does not decide final leakage alone, but flags suspicious rows/files.
    """
    all_main = pd.concat(
        [
            train_df.assign(split="train"),
            test_df.assign(split="test"),
        ],
        ignore_index=True,
        sort=False,
    )

    subject_col_main = find_col(all_main, SUBJECT_COL_CANDIDATES)
    if subject_col_main is None or "lifelog_date" not in all_main.columns:
        return pd.DataFrame([{
            "status": "skipped",
            "reason": "main data missing subject_id or lifelog_date",
        }])

    all_main["lifelog_date"] = pd.to_datetime(all_main["lifelog_date"], errors="coerce")

    min_date = all_main["lifelog_date"].min()
    max_date = all_main["lifelog_date"].max()

    rows = []

    for path in sensor_files:
        sensor_name = infer_sensor_name(path)
        df = read_table(path)
        if df is None or df.empty:
            continue

        timestamp_col = find_col(df, TIMESTAMP_CANDIDATES)
        if timestamp_col is None:
            continue

        ts = safe_to_datetime(df[timestamp_col])

        # Allow 1-day margin because sleep/lifelog alignment may cross midnight.
        too_early = ts < (min_date - pd.Timedelta(days=1))
        too_late = ts > (max_date + pd.Timedelta(days=1))

        rows.append({
            "sensor": sensor_name,
            "file_path": str(path),
            "row_count": len(df),
            "global_lifelog_min": min_date,
            "global_lifelog_max": max_date,
            "sensor_timestamp_min": ts.min(),
            "sensor_timestamp_max": ts.max(),
            "too_early_count": int(too_early.sum()),
            "too_late_count": int(too_late.sum()),
            "too_early_ratio": float(too_early.mean()),
            "too_late_ratio": float(too_late.mean()),
            "boundary_risk": bool(too_early.any() or too_late.any()),
        })

    return pd.DataFrame(rows)


# =========================
# 4. Fold date range
# =========================

def make_time_aware_fold(df: pd.DataFrame, n_folds: int = 5) -> pd.DataFrame:
    """
    Creates diagnostic chronological folds per subject.
    This is for EDA validation only.
    If your actual training code already saves fold ids, compare with this output.
    """
    subject_col = find_col(df, SUBJECT_COL_CANDIDATES)

    if subject_col is None:
        raise ValueError("Cannot find subject column.")

    if "lifelog_date" not in df.columns:
        raise ValueError("Cannot find lifelog_date column.")

    temp = df.copy()
    temp["lifelog_date"] = pd.to_datetime(temp["lifelog_date"], errors="coerce")
    temp = temp.sort_values([subject_col, "lifelog_date"]).reset_index(drop=True)

    temp["eda_time_fold"] = -1

    for _, idx in temp.groupby(subject_col).groups.items():
        idx = list(idx)
        n = len(idx)

        if n == 0:
            continue

        # chronological split into n_folds chunks
        fold_ids = np.array_split(np.arange(n), n_folds)
        for fold, positions in enumerate(fold_ids):
            selected_idx = [idx[pos] for pos in positions]
            temp.loc[selected_idx, "eda_time_fold"] = fold

    return temp


def analyze_fold_date_range(train_df: pd.DataFrame) -> pd.DataFrame:
    subject_col = find_col(train_df, SUBJECT_COL_CANDIDATES)
    folded = make_time_aware_fold(train_df, n_folds=N_FOLDS)

    rows = []

    for fold, fold_df in folded.groupby("eda_time_fold"):
        rows.append({
            "fold": int(fold),
            "scope": "all_subjects",
            "row_count": len(fold_df),
            "subject_count": fold_df[subject_col].nunique(),
            "lifelog_date_min": fold_df["lifelog_date"].min(),
            "lifelog_date_max": fold_df["lifelog_date"].max(),
        })

    for (subject, fold), sub_df in folded.groupby([subject_col, "eda_time_fold"]):
        rows.append({
            "fold": int(fold),
            "scope": f"subject={subject}",
            "row_count": len(sub_df),
            "subject_count": 1,
            "lifelog_date_min": sub_df["lifelog_date"].min(),
            "lifelog_date_max": sub_df["lifelog_date"].max(),
        })

    return pd.DataFrame(rows)


# =========================
# 5. Report
# =========================

def build_leakage_risk_report(
    sleep_diff_df: pd.DataFrame,
    sensor_minmax_df: pd.DataFrame,
    invalid_ts_df: pd.DataFrame,
    boundary_df: pd.DataFrame,
    fold_range_df: pd.DataFrame,
) -> str:
    lines = []
    lines.append("# 03 Timestamp Check & Leakage Risk Report\n")

    lines.append("## 1. sleep_date - lifelog_date\n")
    if "status" in sleep_diff_df.columns:
        lines.append(sleep_diff_df.to_markdown(index=False))
    lines.append("\n")

    risk_notes = []

    if "diff_days" in sleep_diff_df.columns:
        valid_diff = sleep_diff_df.loc[sleep_diff_df["status"] == "ok", "diff_days"].dropna()
        unique_diffs = sorted(valid_diff.unique().tolist()) if len(valid_diff) else []
        if len(unique_diffs) > 1:
            risk_notes.append(
                f"- sleep_date - lifelog_date 값이 하나로 고정되어 있지 않습니다: {unique_diffs}"
            )
        elif len(unique_diffs) == 1:
            risk_notes.append(
                f"- sleep_date - lifelog_date 값은 {unique_diffs[0]}일로 관측됩니다."
            )

    lines.append("## 2. Sensor timestamp summary\n")
    if not sensor_minmax_df.empty:
        status_count = sensor_minmax_df["status"].value_counts(dropna=False).reset_index()
        status_count.columns = ["status", "count"]
        lines.append(status_count.to_markdown(index=False))
    else:
        lines.append("- No sensor files found.")
    lines.append("\n")

    lines.append("## 3. Invalid timestamp rows\n")
    invalid_count = len(invalid_ts_df)
    lines.append(f"- invalid timestamp sample rows: {invalid_count}\n")
    if invalid_count > 0:
        risk_notes.append(f"- invalid timestamp row가 존재합니다. invalid_timestamp_rows.csv 확인이 필요합니다.")

    lines.append("## 4. Sensor boundary risk\n")
    if not boundary_df.empty and "boundary_risk" in boundary_df.columns:
        boundary_risk_count = int(boundary_df["boundary_risk"].sum())
        lines.append(f"- boundary risk sensor files: {boundary_risk_count}\n")
        if boundary_risk_count > 0:
            risky = boundary_df.loc[boundary_df["boundary_risk"], ["sensor", "file_path", "too_early_ratio", "too_late_ratio"]]
            lines.append(risky.head(20).to_markdown(index=False))
            risk_notes.append("- 일부 센서 timestamp가 train/test lifelog_date 전체 범위를 벗어납니다.")
    else:
        lines.append("- Boundary check skipped or empty.\n")

    lines.append("\n## 5. Fold date range\n")
    if not fold_range_df.empty:
        all_scope = fold_range_df[fold_range_df["scope"] == "all_subjects"]
        lines.append(all_scope.to_markdown(index=False))
    lines.append("\n")

    # Fold order check
    if not fold_range_df.empty:
        all_scope = fold_range_df[fold_range_df["scope"] == "all_subjects"].sort_values("fold")
        if len(all_scope) > 1:
            overlap = False
            prev_max = None
            for _, row in all_scope.iterrows():
                cur_min = row["lifelog_date_min"]
                cur_max = row["lifelog_date_max"]
                if prev_max is not None and cur_min < prev_max:
                    overlap = True
                prev_max = cur_max

            if overlap:
                risk_notes.append(
                    "- 전체 subject를 합친 fold 범위는 서로 겹칠 수 있습니다. "
                    "다만 subject별 chronological split이면 정상일 수 있으므로 subject별 fold 범위를 확인하세요."
                )

    lines.append("## 6. Leakage risk checklist\n")
    checklist = [
        "- [ ] sleep_date와 lifelog_date 차이가 모델링 의도와 일치하는가?",
        "- [ ] 수면 이후 timestamp를 feature로 사용하고 있지 않은가?",
        "- [ ] S1/S2/S3 계산 원천이 되는 raw sleep sensor를 그대로 feature로 쓰지 않는가?",
        "- [ ] train+test 전체 통계로 normalization/scaling하지 않는가?",
        "- [ ] random split이 아니라 subject/time-aware split을 쓰는가?",
        "- [ ] fold별 날짜 범위가 의도한 순서를 보존하는가?",
    ]
    lines.extend(checklist)
    lines.append("\n")

    lines.append("## 7. Current risk notes\n")
    if risk_notes:
        lines.extend(risk_notes)
    else:
        lines.append("- 명확한 timestamp 위험 신호는 발견되지 않았습니다. 단, sensor-level feature 생성 로직에서 재검증이 필요합니다.")

    lines.append("\n## 8. Next action\n")
    lines.append("- 04_sensor_schema에서 센서별 timestamp column과 처리 방식을 확정하세요.")
    lines.append("- 05_missing_analysis에서 timestamp coverage와 결측 패턴을 subject/hour 단위로 확인하세요.")
    lines.append("- feature 생성 시 현재 날짜 이후 데이터가 섞이지 않도록 aggregation window를 명시하세요.")

    return "\n".join(lines)


# =========================
# Main
# =========================

def main() -> None:
    ensure_dir(OUTPUT_DIR)

    print("[INFO] Loading main data...")
    train_df, test_df = load_main_data()

    print("[INFO] Step 1: sleep_date - lifelog_date diff...")
    sleep_diff_df = analyze_sleep_lifelog_diff(train_df, test_df)
    sleep_diff_df.to_csv(OUTPUT_DIR / "sleep_lifelog_date_diff.csv", index=False, encoding="utf-8-sig")

    print("[INFO] Step 2: scanning sensor files...")
    sensor_files = list_sensor_files(ITEMS_DIR)
    print(f"[INFO] Found sensor files: {len(sensor_files)}")

    print("[INFO] Step 3: sensor timestamp min/max...")
    sensor_minmax_df, invalid_ts_df = analyze_sensor_timestamp_minmax(sensor_files)
    sensor_minmax_df.to_csv(OUTPUT_DIR / "sensor_timestamp_minmax.csv", index=False, encoding="utf-8-sig")
    invalid_ts_df.to_csv(OUTPUT_DIR / "invalid_timestamp_rows.csv", index=False, encoding="utf-8-sig")

    print("[INFO] Step 4: sensor date boundary check...")
    boundary_df = analyze_sensor_date_boundary(train_df, test_df, sensor_files)
    boundary_df.to_csv(OUTPUT_DIR / "sensor_date_boundary_check.csv", index=False, encoding="utf-8-sig")

    print("[INFO] Step 5: fold date range...")
    fold_range_df = analyze_fold_date_range(train_df)
    fold_range_df.to_csv(OUTPUT_DIR / "fold_date_range.csv", index=False, encoding="utf-8-sig")

    print("[INFO] Step 6: writing leakage risk report...")
    report = build_leakage_risk_report(
        sleep_diff_df=sleep_diff_df,
        sensor_minmax_df=sensor_minmax_df,
        invalid_ts_df=invalid_ts_df,
        boundary_df=boundary_df,
        fold_range_df=fold_range_df,
    )
    (OUTPUT_DIR / "leakage_risk_report.md").write_text(report, encoding="utf-8")

    print("[DONE] 03 timestamp check completed.")
    print(f"[DONE] Outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()