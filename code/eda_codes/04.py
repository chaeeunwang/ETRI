# 04.py
# Step 04. Sensor Schema EDA
#
# 목적:
# - 센서 파일 목록 확인
# - 센서별 row 수 / subject 수 / column / dtype 정리
# - timestamp column, value column 후보 추정
# - continuous / discrete / object / high-noise / high-priority 처리 계획 생성
#
# 실행:
# & c:\01_ETRI\.venv\Scripts\python.exe c:/01_ETRI/code/eda_codes/04.py

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# =========================
# 0. Path 설정
# =========================
DATA_DIR = Path("./data")
ITEMS_DIR = DATA_DIR / "ch2025_data_items"
OUTPUT_DIR = Path("./outputs/eda/04_sensor_schema")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# 1. 센서 우선순위/처리 규칙
# =========================
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

HIGH_NOISE_SENSORS = {
    "mGps",
    "mBle",
    "mWifi",
    "mAmbience",
}

DISCRETE_SENSORS = {
    "mActivity",
    "mScreenStatus",
    "mACStatus",
}

OBJECT_SENSORS = {
    "mUsageStats",
    "mAmbience",
    "mBle",
    "mWifi",
    "mGps",
    "wHr",
}

CONTINUOUS_SENSORS = {
    "mLight",
    "wLight",
    "wPedo",
}

CONTINUOUS_HINTS = {
    "hr",
    "heart",
    "light",
    "step",
    "steps",
    "distance",
    "speed",
    "calorie",
    "calories",
    "altitude",
    "latitude",
    "longitude",
    "rssi",
    "lux",
    "value",
    "charging",
    "screen",
    "activity",
}

# 주의:
# "ts"는 넣지 마세요.
# m_usage_stats 안의 "stats" 때문에 timestamp_candidate로 오탐됩니다.
TIMESTAMP_COLUMNS = {
    "timestamp",
    "datetime",
    "time",
}

DATE_COLUMNS = {
    "lifelog_date",
    "sleep_date",
    "date",
}

SUBJECT_COLUMNS = {
    "subject_id",
    "subject",
    "user_id",
    "user",
}


# =========================
# 2. 유틸 함수
# =========================
def read_any_table(path: Path, nrows: int | None = None) -> pd.DataFrame:
    """
    csv/parquet/json/jsonl 파일을 안전하게 읽습니다.
    """
    suffix = path.suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(path, nrows=nrows)

    if suffix == ".parquet":
        df = pd.read_parquet(path)
        if nrows is not None:
            return df.head(nrows)
        return df

    if suffix in [".json", ".jsonl"]:
        try:
            return pd.read_json(path, lines=(suffix == ".jsonl"))
        except ValueError:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return pd.json_normalize(data)

    raise ValueError(f"Unsupported file format: {path}")


def safe_sample_df(path: Path, sample_n: int = 5000) -> pd.DataFrame:
    """
    schema 확인용 sample dataframe 생성.
    parquet은 전체 로드 후 head를 사용합니다.
    """
    return read_any_table(path, nrows=sample_n)


def infer_sensor_name(path: Path) -> str:
    """
    파일명 기준 센서명 추정.

    예:
    ch2025_mActivity.parquet -> mActivity
    ch2026_wHr.parquet       -> wHr
    mActivity.parquet        -> mActivity
    """
    name = path.stem

    prefixes = [
        "ch2025_",
        "ch2026_",
        "ch2024_",
        "etri_",
        "ETRI_",
    ]

    for prefix in prefixes:
        if name.startswith(prefix):
            name = name.replace(prefix, "", 1)

    return name


def safe_to_string(x, max_len: int = 200) -> str:
    """
    ndarray/list/dict/object 값을 CSV에 저장 가능한 짧은 문자열로 변환합니다.
    """
    try:
        if isinstance(x, np.ndarray):
            text = str(x.tolist())
        elif isinstance(x, (list, dict, tuple, set)):
            text = str(x)
        else:
            text = str(x)

        if len(text) > max_len:
            return text[:max_len] + "..."

        return text

    except Exception:
        return "<unprintable>"


def safe_nunique(s: pd.Series, max_rows: int = 1000) -> int:
    """
    list, dict, numpy.ndarray처럼 unhashable한 값이 섞인 column도
    안전하게 unique count를 계산합니다.

    schema EDA 목적이므로 sample 기준으로만 계산합니다.
    """
    try:
        return int(s.nunique(dropna=True))

    except TypeError:
        sample = s.dropna().head(max_rows)
        return int(sample.map(lambda x: safe_to_string(x, max_len=500)).nunique(dropna=True))

    except Exception:
        return -1


def infer_column_role(col: str) -> str:
    """
    column 이름 기반 역할 추정.
    정확한 column명 우선으로 판정하여 m_usage_stats 같은 오탐을 방지합니다.
    """
    c = col.lower()

    if c in SUBJECT_COLUMNS:
        return "subject_candidate"

    if c in TIMESTAMP_COLUMNS:
        return "timestamp_candidate"

    if c in DATE_COLUMNS:
        return "date_candidate"

    if any(h in c for h in CONTINUOUS_HINTS):
        return "value_candidate"

    return "unknown"


def find_best_column(df: pd.DataFrame, role_keyword: str) -> str | None:
    """
    timestamp/subject 후보 column 중 가장 그럴듯한 column 선택.
    """
    candidates = []

    for col in df.columns:
        role = infer_column_role(col)
        if role_keyword in role:
            candidates.append(col)

    if not candidates:
        return None

    lowered = {c.lower(): c for c in candidates}

    if role_keyword == "subject":
        for key in ["subject_id", "subject", "user_id", "user"]:
            if key in lowered:
                return lowered[key]

    if role_keyword == "timestamp":
        for key in ["timestamp", "datetime", "time"]:
            if key in lowered:
                return lowered[key]

    return candidates[0]


def count_subjects(df: pd.DataFrame, subject_col: str | None) -> int | None:
    """
    sample 기준 subject 수 계산.
    Step 04에서는 schema 확인 목적이므로 sample 기준입니다.
    """
    if subject_col is None or subject_col not in df.columns:
        return None

    try:
        return int(df[subject_col].nunique(dropna=True))
    except Exception:
        return None


def infer_processing_type(sensor_name: str, df: pd.DataFrame) -> str:
    """
    센서명 우선으로 처리 유형 추정.

    메타 컬럼 때문에 dtype만 보면 mixed가 많이 나오므로,
    알려진 센서는 sensor_name 기준으로 강제 분류합니다.
    """
    if sensor_name in DISCRETE_SENSORS:
        return "discrete"

    if sensor_name in {"mLight", "wLight"}:
        return "continuous"

    if sensor_name == "wPedo":
        return "continuous/multi-value"

    if sensor_name == "wHr":
        return "object/list"

    if sensor_name in OBJECT_SENSORS:
        return "object/list"

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    object_cols = df.select_dtypes(include=["object"]).columns.tolist()

    non_meta_object_cols = [
        c for c in object_cols
        if infer_column_role(c) == "unknown"
    ]

    if len(numeric_cols) > 0 and len(non_meta_object_cols) == 0:
        return "continuous"

    if len(non_meta_object_cols) > 0 and len(numeric_cols) == 0:
        return "object/list"

    if len(numeric_cols) > 0 and len(non_meta_object_cols) > 0:
        return "mixed"

    return "unknown"


def infer_priority(sensor_name: str) -> str:
    """
    초기 feature engineering 우선순위 추정.
    """
    if sensor_name in HIGH_PRIORITY_SENSORS:
        return "high-priority"

    if sensor_name in HIGH_NOISE_SENSORS:
        return "low-priority/high-noise"

    return "normal"


def estimate_sampling_frequency(df: pd.DataFrame, timestamp_col: str | None) -> str:
    """
    timestamp 간격 기반 sampling frequency 대략 추정.
    """
    if timestamp_col is None or timestamp_col not in df.columns:
        return "unknown"

    try:
        ts = pd.to_datetime(df[timestamp_col], errors="coerce").dropna()

        if len(ts) < 3:
            return "unknown"

        ts = ts.sort_values()
        diffs = ts.diff().dropna().dt.total_seconds()
        diffs = diffs[(diffs > 0) & (diffs < 24 * 3600)]

        if len(diffs) == 0:
            return "unknown"

        median_sec = float(diffs.median())

        if median_sec <= 2:
            return "approx_1Hz"
        if median_sec <= 70:
            return "approx_1min"
        if median_sec <= 130:
            return "approx_2min"
        if median_sec <= 650:
            return "approx_10min"

        return f"approx_{round(median_sec, 1)}sec"

    except Exception:
        return "unknown"


def estimate_timestamp_minmax(
    df: pd.DataFrame,
    timestamp_col: str | None,
) -> tuple[str | None, str | None]:
    """
    timestamp min/max 추정.
    """
    if timestamp_col is None or timestamp_col not in df.columns:
        return None, None

    try:
        ts = pd.to_datetime(df[timestamp_col], errors="coerce").dropna()

        if len(ts) == 0:
            return None, None

        return str(ts.min()), str(ts.max())

    except Exception:
        return None, None


def get_value_columns(df: pd.DataFrame) -> list[str]:
    """
    subject/date/timestamp 후보를 제외한 실제 value 후보 컬럼 반환.
    """
    value_cols = []

    for col in df.columns:
        role = infer_column_role(col)

        if role in ["subject_candidate", "timestamp_candidate", "date_candidate"]:
            continue

        value_cols.append(col)

    return value_cols


def summarize_file(path: Path) -> tuple[dict, list[dict], list[dict]]:
    """
    단일 센서 파일 요약.

    return:
    - file_summary
    - column_dictionary_rows
    - dtype_summary_rows
    """
    sensor_name = infer_sensor_name(path)

    try:
        df_sample = safe_sample_df(path)
        df_full_for_len = read_any_table(path)

    except Exception as e:
        file_summary = {
            "sensor_name": sensor_name,
            "raw_file_stem": path.stem,
            "file_name": path.name,
            "file_path": str(path),
            "file_ext": path.suffix,
            "read_status": "failed",
            "error": str(e),
            "row_count": None,
            "col_count": None,
            "subject_col": None,
            "subject_count_sample": None,
            "timestamp_col": None,
            "timestamp_min_sample": None,
            "timestamp_max_sample": None,
            "sampling_frequency_est": "unknown",
            "processing_type": "unknown",
            "priority": infer_priority(sensor_name),
            "value_columns": None,
        }

        return file_summary, [], []

    subject_col = find_best_column(df_sample, "subject")
    timestamp_col = find_best_column(df_sample, "timestamp")
    processing_type = infer_processing_type(sensor_name, df_sample)
    priority = infer_priority(sensor_name)
    timestamp_min, timestamp_max = estimate_timestamp_minmax(df_sample, timestamp_col)
    value_columns = get_value_columns(df_sample)

    file_summary = {
        "sensor_name": sensor_name,
        "raw_file_stem": path.stem,
        "file_name": path.name,
        "file_path": str(path),
        "file_ext": path.suffix,
        "read_status": "success",
        "error": "",
        "row_count": len(df_full_for_len),
        "col_count": len(df_sample.columns),
        "subject_col": subject_col,
        "subject_count_sample": count_subjects(df_sample, subject_col),
        "timestamp_col": timestamp_col,
        "timestamp_min_sample": timestamp_min,
        "timestamp_max_sample": timestamp_max,
        "sampling_frequency_est": estimate_sampling_frequency(df_sample, timestamp_col),
        "processing_type": processing_type,
        "priority": priority,
        "value_columns": ", ".join(value_columns),
    }

    column_rows = []
    dtype_rows = []

    for col in df_sample.columns:
        s = df_sample[col]

        dtype = str(s.dtype)
        null_ratio = float(s.isna().mean())
        unique_count = safe_nunique(s)
        role = infer_column_role(col)

        example_values = (
            s.dropna()
            .head(3)
            .map(safe_to_string)
            .tolist()
        )

        column_rows.append({
            "sensor_name": sensor_name,
            "raw_file_stem": path.stem,
            "file_name": path.name,
            "column_name": col,
            "inferred_role": role,
            "dtype": dtype,
            "null_ratio_sample": null_ratio,
            "unique_count_sample": unique_count,
            "example_values": " | ".join(example_values),
        })

        dtype_rows.append({
            "sensor_name": sensor_name,
            "raw_file_stem": path.stem,
            "file_name": path.name,
            "dtype": dtype,
            "column_name": col,
        })

    return file_summary, column_rows, dtype_rows


def recommend_processing(sensor: str, ptype: str) -> str:
    """
    센서별 초기 처리 계획 추천.
    """
    if sensor == "mACStatus":
        return "charging duration/count; night charging flag; timeblock ratio"

    if sensor == "mActivity":
        return "activity code count/duration/ratio by daily and time block"

    if sensor == "mScreenStatus":
        return "screen-on duration/count; night screen use; pre-sleep screen use"

    if sensor == "mUsageStats":
        return "parse app usage; total/night/pre-sleep usage; top-k app/category usage"

    if sensor == "wHr":
        return "parse heart-rate list; mean/std/min/max/night mean + missing indicator"

    if sensor == "wPedo":
        return "aggregate steps/calories/distance/speed; daily/timeblock sum/mean/max"

    if sensor in {"mLight", "wLight"}:
        return "daily/timeblock mean/max; night light mean; bright event count"

    if sensor == "mGps":
        return "defer initially; optionally parse speed/altitude and movement coverage"

    if sensor == "mBle":
        return "defer initially; optionally device count and RSSI summary"

    if sensor == "mWifi":
        return "defer initially; optionally AP count and RSSI summary"

    if sensor == "mAmbience":
        return "defer initially; optionally top sound labels/count/probability summary"

    if ptype == "object/list":
        return "parse object/list; top-k count/summary; missing indicator"

    if ptype in ["continuous", "continuous/multi-value"]:
        return "daily/timeblock aggregation"

    if ptype == "discrete":
        return "count/duration/ratio by daily and time block"

    return "manual inspection required"


def write_processing_plan(
    sensor_file_list_df: pd.DataFrame,
    sensor_column_dict_df: pd.DataFrame,
) -> None:
    """
    sensor_processing_plan.md 생성.
    """
    lines = []

    lines.append("# Step 04. Sensor Schema EDA Summary\n")

    lines.append("## 1. Sensor File Overview\n")
    lines.append(f"- 총 센서 파일 수: {len(sensor_file_list_df)}")

    success_count = int((sensor_file_list_df["read_status"] == "success").sum())
    fail_count = int((sensor_file_list_df["read_status"] == "failed").sum())

    lines.append(f"- 읽기 성공 파일 수: {success_count}")
    lines.append(f"- 읽기 실패 파일 수: {fail_count}\n")

    lines.append("## 2. Processing Type Counts\n")

    if "processing_type" in sensor_file_list_df.columns:
        type_counts = sensor_file_list_df["processing_type"].value_counts(dropna=False)
        for k, v in type_counts.items():
            lines.append(f"- {k}: {v}")

    lines.append("")

    lines.append("## 3. High Priority Sensors\n")

    hp = sensor_file_list_df[sensor_file_list_df["priority"] == "high-priority"]

    if len(hp) == 0:
        lines.append("- 감지된 high-priority sensor 없음")
    else:
        for _, row in hp.sort_values("sensor_name").iterrows():
            lines.append(
                f"- {row['sensor_name']}: {row['processing_type']}, "
                f"rows={row['row_count']}, freq={row['sampling_frequency_est']}"
            )

    lines.append("")

    lines.append("## 4. Low Priority / High Noise Sensors\n")

    hn = sensor_file_list_df[sensor_file_list_df["priority"] == "low-priority/high-noise"]

    if len(hn) == 0:
        lines.append("- 감지된 high-noise sensor 없음")
    else:
        for _, row in hn.sort_values("sensor_name").iterrows():
            lines.append(
                f"- {row['sensor_name']}: {row['processing_type']}, "
                f"rows={row['row_count']}, freq={row['sampling_frequency_est']}"
            )

    lines.append("")

    lines.append("## 5. Recommended Initial Processing Plan\n")
    lines.append("| Sensor | Raw File | Priority | Type | Value Columns | Recommended Processing |")
    lines.append("|---|---|---:|---|---|---|")

    priority_order = {
        "high-priority": 0,
        "normal": 1,
        "low-priority/high-noise": 2,
    }

    tmp = sensor_file_list_df.copy()
    tmp["priority_order"] = tmp["priority"].map(priority_order).fillna(9)

    for _, row in tmp.sort_values(["priority_order", "sensor_name"]).iterrows():
        sensor = row["sensor_name"]
        raw_file = row["file_name"]
        ptype = row["processing_type"]
        priority = row["priority"]
        value_columns = row.get("value_columns", "")

        rec = recommend_processing(sensor, ptype)

        lines.append(
            f"| {sensor} | {raw_file} | {priority} | {ptype} | "
            f"{value_columns} | {rec} |"
        )

    lines.append("\n## 6. Manual Inspection Targets\n")

    manual_df = sensor_file_list_df[
        sensor_file_list_df["processing_type"].isin(["mixed", "unknown"])
    ]

    if len(manual_df) == 0:
        lines.append("- manual inspection required 센서 없음")
    else:
        for _, row in manual_df.sort_values("sensor_name").iterrows():
            lines.append(
                f"- {row['sensor_name']}: type={row['processing_type']}, "
                f"value_columns={row.get('value_columns', '')}"
            )

    lines.append("\n## 7. Column Dictionary Notes\n")
    lines.append("- `subject_id`: subject identifier")
    lines.append("- `timestamp`: sensor observation time")
    lines.append("- `m_charging`: smartphone charging status")
    lines.append("- `m_activity`: smartphone activity code")
    lines.append("- `m_screen_use`: smartphone screen-use status")
    lines.append("- `m_usage_stats`: app usage list, not timestamp")
    lines.append("- `heart_rate`: smartwatch heart-rate list")
    lines.append("- `step`, `distance`, `speed`, `burned_calories`: pedometer values")

    lines.append("\n## 8. Files Generated\n")
    lines.append("- `sensor_file_list.csv`")
    lines.append("- `sensor_column_dictionary.csv`")
    lines.append("- `sensor_dtype_summary.csv`")
    lines.append("- `sensor_row_count.csv`")
    lines.append("- `sensor_processing_plan.md`")

    with open(OUTPUT_DIR / "sensor_processing_plan.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# =========================
# 3. 메인 실행
# =========================
def main() -> None:
    if not ITEMS_DIR.exists():
        raise FileNotFoundError(f"ITEMS_DIR not found: {ITEMS_DIR.resolve()}")

    sensor_files = []

    for ext in ["*.csv", "*.parquet", "*.json", "*.jsonl"]:
        sensor_files.extend(ITEMS_DIR.rglob(ext))

    sensor_files = sorted(sensor_files)

    if len(sensor_files) == 0:
        raise FileNotFoundError(f"No sensor files found under: {ITEMS_DIR.resolve()}")

    file_summaries = []
    column_dictionary = []
    dtype_summary = []

    for idx, path in enumerate(sensor_files, start=1):
        print(f"[{idx}/{len(sensor_files)}] Processing: {path.name}")

        file_summary, col_rows, dtype_rows = summarize_file(path)

        file_summaries.append(file_summary)
        column_dictionary.extend(col_rows)
        dtype_summary.extend(dtype_rows)

    sensor_file_list_df = pd.DataFrame(file_summaries)
    sensor_column_dict_df = pd.DataFrame(column_dictionary)
    sensor_dtype_summary_df = pd.DataFrame(dtype_summary)

    if not sensor_dtype_summary_df.empty:
        dtype_count_df = (
            sensor_dtype_summary_df
            .groupby(["sensor_name", "dtype"])
            .size()
            .reset_index(name="column_count")
            .sort_values(["sensor_name", "dtype"])
        )
    else:
        dtype_count_df = pd.DataFrame(
            columns=["sensor_name", "dtype", "column_count"]
        )

    sensor_row_count_df = sensor_file_list_df[[
        "sensor_name",
        "raw_file_stem",
        "file_name",
        "row_count",
        "col_count",
        "subject_count_sample",
        "timestamp_col",
        "timestamp_min_sample",
        "timestamp_max_sample",
        "processing_type",
        "priority",
        "sampling_frequency_est",
        "value_columns",
    ]].copy()

    sensor_file_list_df.to_csv(
        OUTPUT_DIR / "sensor_file_list.csv",
        index=False,
        encoding="utf-8-sig",
    )

    sensor_column_dict_df.to_csv(
        OUTPUT_DIR / "sensor_column_dictionary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    dtype_count_df.to_csv(
        OUTPUT_DIR / "sensor_dtype_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    sensor_row_count_df.to_csv(
        OUTPUT_DIR / "sensor_row_count.csv",
        index=False,
        encoding="utf-8-sig",
    )

    write_processing_plan(sensor_file_list_df, sensor_column_dict_df)

    print("\nDone.")
    print(f"Saved to: {OUTPUT_DIR.resolve()}")

    print("\nProcessing Type Counts")
    print(sensor_file_list_df["processing_type"].value_counts(dropna=False))

    print("\nPriority Counts")
    print(sensor_file_list_df["priority"].value_counts(dropna=False))


if __name__ == "__main__":
    main()