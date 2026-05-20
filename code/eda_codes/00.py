# scripts/eda_step0_data_overview.py

from pathlib import Path
import pandas as pd


# =========================
# 1. Path 설정
# =========================
DATA_DIR = Path("./data")
OUTPUT_DIR = Path("./outputs/eda/00_data_overview")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_PATH = DATA_DIR / "ch2026_metrics_train.csv"
TEST_PATH = DATA_DIR / "ch2026_submission_sample.csv"


# =========================
# 2. Utility 함수
# =========================
def save_shape(df: pd.DataFrame, name: str, output_path: Path) -> None:
    """DataFrame shape 저장"""
    shape_df = pd.DataFrame(
        {
            "dataset": [name],
            "rows": [df.shape[0]],
            "columns": [df.shape[1]],
        }
    )
    shape_df.to_csv(output_path, index=False, encoding="utf-8-sig")


def save_columns(df: pd.DataFrame, output_path: Path) -> None:
    """column명, dtype, null count 저장"""
    columns_df = pd.DataFrame(
        {
            "column": df.columns,
            "dtype": [str(df[col].dtype) for col in df.columns],
            "non_null_count": [df[col].notna().sum() for col in df.columns],
            "null_count": [df[col].isna().sum() for col in df.columns],
            "null_ratio": [df[col].isna().mean() for col in df.columns],
            "unique_count": [df[col].nunique(dropna=True) for col in df.columns],
        }
    )
    columns_df.to_csv(output_path, index=False, encoding="utf-8-sig")


def find_key_columns(df: pd.DataFrame) -> list[str]:
    """중복 확인에 사용할 key column 자동 탐색"""
    candidates = [
        ["subject_id", "lifelog_date"],
        ["subject", "lifelog_date"],
        ["user_id", "lifelog_date"],
        ["id", "lifelog_date"],
    ]

    for keys in candidates:
        if all(col in df.columns for col in keys):
            return keys

    return []


def save_duplicate_keys(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_path: Path,
) -> tuple[list[str], list[str], int, int]:
    """subject_id + lifelog_date 기준 중복 row 저장"""
    train_keys = find_key_columns(train_df)
    test_keys = find_key_columns(test_df)

    duplicate_frames = []

    train_dup_count = 0
    test_dup_count = 0

    if train_keys:
        train_dup = train_df[train_df.duplicated(subset=train_keys, keep=False)].copy()
        train_dup_count = len(train_dup)
        if not train_dup.empty:
            train_dup.insert(0, "dataset", "train")
            duplicate_frames.append(train_dup)

    if test_keys:
        test_dup = test_df[test_df.duplicated(subset=test_keys, keep=False)].copy()
        test_dup_count = len(test_dup)
        if not test_dup.empty:
            test_dup.insert(0, "dataset", "test")
            duplicate_frames.append(test_dup)

    if duplicate_frames:
        duplicate_df = pd.concat(duplicate_frames, ignore_index=True)
    else:
        duplicate_df = pd.DataFrame(
            columns=[
                "dataset",
                "message",
            ]
        )
        duplicate_df.loc[0] = [
            "train/test",
            "No duplicated rows found by detected key columns.",
        ]

    duplicate_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    return train_keys, test_keys, train_dup_count, test_dup_count


def get_subject_col(df: pd.DataFrame) -> str | None:
    """subject column 자동 탐색"""
    candidates = ["subject_id", "subject", "user_id", "id"]

    for col in candidates:
        if col in df.columns:
            return col

    return None


def get_date_summary(df: pd.DataFrame, date_col: str) -> dict:
    """날짜 column min/max/null 요약"""
    if date_col not in df.columns:
        return {
            f"{date_col}_exists": False,
            f"{date_col}_min": None,
            f"{date_col}_max": None,
            f"{date_col}_null_count": None,
        }

    parsed = pd.to_datetime(df[date_col], errors="coerce")

    return {
        f"{date_col}_exists": True,
        f"{date_col}_min": parsed.min(),
        f"{date_col}_max": parsed.max(),
        f"{date_col}_null_count": parsed.isna().sum(),
    }


def write_basic_summary(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_keys: list[str],
    test_keys: list[str],
    train_dup_count: int,
    test_dup_count: int,
    output_path: Path,
) -> None:
    """basic_summary.txt 생성"""
    target_candidates = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]
    train_targets = [col for col in target_candidates if col in train_df.columns]
    test_targets = [col for col in target_candidates if col in test_df.columns]

    train_subject_col = get_subject_col(train_df)
    test_subject_col = get_subject_col(test_df)

    train_subject_count = (
        train_df[train_subject_col].nunique(dropna=True)
        if train_subject_col is not None
        else None
    )
    test_subject_count = (
        test_df[test_subject_col].nunique(dropna=True)
        if test_subject_col is not None
        else None
    )

    train_lifelog_summary = get_date_summary(train_df, "lifelog_date")
    train_sleep_summary = get_date_summary(train_df, "sleep_date")
    test_lifelog_summary = get_date_summary(test_df, "lifelog_date")
    test_sleep_summary = get_date_summary(test_df, "sleep_date")

    sleep_lifelog_diff_summary = "Not available"
    if "sleep_date" in train_df.columns and "lifelog_date" in train_df.columns:
        sleep_date = pd.to_datetime(train_df["sleep_date"], errors="coerce")
        lifelog_date = pd.to_datetime(train_df["lifelog_date"], errors="coerce")
        diff_days = (sleep_date - lifelog_date).dt.days

        sleep_lifelog_diff_summary = (
            "\n"
            f"  - min diff days: {diff_days.min()}\n"
            f"  - max diff days: {diff_days.max()}\n"
            f"  - mean diff days: {diff_days.mean()}\n"
            f"  - diff value counts:\n{diff_days.value_counts(dropna=False).sort_index().to_string()}"
        )

    summary = f"""
# Step 0. Data Overview Summary

## 1. File Paths
- train path: {TRAIN_PATH}
- test/submission path: {TEST_PATH}
- output dir: {OUTPUT_DIR}

## 2. Shape
- train rows: {train_df.shape[0]}
- train columns: {train_df.shape[1]}
- test rows: {test_df.shape[0]}
- test columns: {test_df.shape[1]}

## 3. Subject
- train subject column: {train_subject_col}
- test subject column: {test_subject_col}
- train subject count: {train_subject_count}
- test subject count: {test_subject_count}

## 4. Target Columns
- target candidates: {target_candidates}
- train target columns: {train_targets}
- test/submission target columns: {test_targets}
- S4 in train: {"S4" in train_df.columns}
- S4 in test/submission: {"S4" in test_df.columns}

## 5. Key Columns for Duplicate Check
- train key columns: {train_keys}
- test key columns: {test_keys}
- train duplicated rows: {train_dup_count}
- test duplicated rows: {test_dup_count}

## 6. Train Date Summary
- lifelog_date exists: {train_lifelog_summary["lifelog_date_exists"]}
- lifelog_date min: {train_lifelog_summary["lifelog_date_min"]}
- lifelog_date max: {train_lifelog_summary["lifelog_date_max"]}
- lifelog_date null count: {train_lifelog_summary["lifelog_date_null_count"]}
- sleep_date exists: {train_sleep_summary["sleep_date_exists"]}
- sleep_date min: {train_sleep_summary["sleep_date_min"]}
- sleep_date max: {train_sleep_summary["sleep_date_max"]}
- sleep_date null count: {train_sleep_summary["sleep_date_null_count"]}

## 7. Test Date Summary
- lifelog_date exists: {test_lifelog_summary["lifelog_date_exists"]}
- lifelog_date min: {test_lifelog_summary["lifelog_date_min"]}
- lifelog_date max: {test_lifelog_summary["lifelog_date_max"]}
- lifelog_date null count: {test_lifelog_summary["lifelog_date_null_count"]}
- sleep_date exists: {test_sleep_summary["sleep_date_exists"]}
- sleep_date min: {test_sleep_summary["sleep_date_min"]}
- sleep_date max: {test_sleep_summary["sleep_date_max"]}
- sleep_date null count: {test_sleep_summary["sleep_date_null_count"]}

## 8. sleep_date - lifelog_date Difference in Train
{sleep_lifelog_diff_summary}

## 9. Immediate Checkpoints
- Check whether train/test subjects are identical.
- Check whether subject_id + lifelog_date is unique.
- Check whether sleep_date and lifelog_date have a consistent relationship.
- Check whether S4 is actually required by submission format.
""".strip()

    output_path.write_text(summary, encoding="utf-8")


# =========================
# 3. Main
# =========================
def main() -> None:
    if not TRAIN_PATH.exists():
        raise FileNotFoundError(f"Train file not found: {TRAIN_PATH}")

    if not TEST_PATH.exists():
        raise FileNotFoundError(f"Test/submission file not found: {TEST_PATH}")

    print("Loading data...")
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    print("Saving shape files...")
    save_shape(train_df, "train", OUTPUT_DIR / "train_shape.csv")
    save_shape(test_df, "test", OUTPUT_DIR / "test_shape.csv")

    print("Saving column files...")
    save_columns(train_df, OUTPUT_DIR / "train_columns.csv")
    save_columns(test_df, OUTPUT_DIR / "test_columns.csv")

    print("Checking duplicate keys...")
    train_keys, test_keys, train_dup_count, test_dup_count = save_duplicate_keys(
        train_df=train_df,
        test_df=test_df,
        output_path=OUTPUT_DIR / "duplicate_keys.csv",
    )

    print("Writing basic summary...")
    write_basic_summary(
        train_df=train_df,
        test_df=test_df,
        train_keys=train_keys,
        test_keys=test_keys,
        train_dup_count=train_dup_count,
        test_dup_count=test_dup_count,
        output_path=OUTPUT_DIR / "basic_summary.txt",
    )

    print("Done.")
    print(f"Saved files to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()