import ast
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

DATA_DIR = Path("data")
ITEMS_DIR = DATA_DIR / "ch2025_data_items"
OUT_DIR = Path("outputs/eda/06_sensor_statistics")
TARGETS = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]
KEYS = ["subject_id", "lifelog_date"]
ID_COLS = {"subject_id", "lifelog_date", "sleep_date", "timestamp", "datetime", "date", "time"}
BLOCKS = [(0, 4), (4, 8), (8, 12), (12, 16), (16, 20), (20, 24)]

DISCRETE_HINTS = {
    "mACStatus": ["m_charging"],
    "mActivity": ["m_activity"],
    "mScreenStatus": ["m_screen_use"],
}

SENSOR_GROUP = {
    "mACStatus": "charging",
    "mActivity": "activity_code",
    "mAmbience": "ambience",
    "mBle": "ble",
    "mGps": "gps",
    "mLight": "light",
    "mScreenStatus": "screen",
    "mUsageStats": "usage",
    "mWifi": "wifi",
    "wHr": "heart_rate",
    "wLight": "light",
    "wPedo": "activity_pedo",
}


def read_table(path):
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)


def find_file(stem):
    hits = sorted(ITEMS_DIR.glob(f"*{stem}*.parquet")) + sorted(ITEMS_DIR.glob(f"*{stem}*.csv"))
    return hits[0] if hits else None


def normalize_keys(df):
    cols = {c.lower(): c for c in df.columns}
    sid = cols.get("subject_id") or cols.get("user_id") or cols.get("id")
    ts = cols.get("timestamp") or cols.get("datetime") or cols.get("time")
    ld = cols.get("lifelog_date") or cols.get("date")

    if sid and sid != "subject_id":
        df = df.rename(columns={sid: "subject_id"})
    if ts and ts != "timestamp":
        df = df.rename(columns={ts: "timestamp"})
    if ld and ld != "lifelog_date":
        df = df.rename(columns={ld: "lifelog_date"})

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    if "lifelog_date" in df.columns:
        df["lifelog_date"] = pd.to_datetime(df["lifelog_date"], errors="coerce").dt.date
    elif "timestamp" in df.columns:
        df["lifelog_date"] = df["timestamp"].dt.date

    if "subject_id" in df.columns:
        df["subject_id"] = df["subject_id"].astype(str)

    return df


def load_keys():
    train = pd.read_csv(DATA_DIR / "ch2026_metrics_train.csv")
    sub = pd.read_csv(DATA_DIR / "ch2026_submission_sample.csv")

    for df in [train, sub]:
        df["subject_id"] = df["subject_id"].astype(str)
        df["lifelog_date"] = pd.to_datetime(df["lifelog_date"], errors="coerce").dt.date

    targets = [t for t in TARGETS if t in train.columns]
    base = pd.concat(
        [
            train[KEYS + targets].assign(split="train"),
            sub[[c for c in KEYS if c in sub.columns]].assign(split="test"),
        ],
        ignore_index=True,
    ).drop_duplicates(KEYS)

    return train, sub, base, targets


def to_obj(x):
    if x is None:
        return None

    if isinstance(x, float) and np.isnan(x):
        return None

    if isinstance(x, (list, tuple, dict, set, np.ndarray)):
        return x.tolist() if isinstance(x, np.ndarray) else x

    try:
        if pd.isna(x):
            return None
    except Exception:
        pass

    if not isinstance(x, str):
        return x

    s = x.strip()
    if not s or s.lower() in {"nan", "none", "null"}:
        return None

    try:
        return json.loads(s)
    except Exception:
        try:
            return ast.literal_eval(s)
        except Exception:
            return s


def flatten_nums(x):
    out = []
    if x is None:
        return out

    if isinstance(x, np.ndarray):
        x = x.tolist()

    if isinstance(x, dict):
        for v in x.values():
            out.extend(flatten_nums(v))
    elif isinstance(x, (list, tuple, set)):
        for v in x:
            out.extend(flatten_nums(v))
    else:
        try:
            v = float(x)
            if np.isfinite(v):
                out.append(v)
        except Exception:
            pass
    return out

def obj_len(x):
    x = to_obj(x)
    if x is None:
        return np.nan
    if isinstance(x, dict):
        return len(x)
    if isinstance(x, (list, tuple, set)):
        return len(x)
    if isinstance(x, str):
        return np.nan
    return 1


def infer_value_cols(df):
    cols = []
    for c in df.columns:
        if c in ID_COLS:
            continue
        if c.startswith("Unnamed"):
            continue
        cols.append(c)
    return cols


def is_discrete(sensor, col, s):
    if any(h in col for h in DISCRETE_HINTS.get(sensor, [])):
        return True
    x = pd.to_numeric(s, errors="coerce").dropna()
    if len(x) == 0:
        return False
    return x.nunique() <= 20 and np.allclose(x, x.astype(int))


def agg_numeric(g, prefix):
    return pd.Series(
        {
            f"{prefix}_mean": g.mean(),
            f"{prefix}_std": g.std(),
            f"{prefix}_min": g.min(),
            f"{prefix}_q25": g.quantile(0.25),
            f"{prefix}_median": g.median(),
            f"{prefix}_q75": g.quantile(0.75),
            f"{prefix}_q95": g.quantile(0.95),
            f"{prefix}_q99": g.quantile(0.99),
            f"{prefix}_max": g.max(),
            f"{prefix}_sum": g.sum(),
            f"{prefix}_nonnull_count": g.notna().sum(),
            f"{prefix}_zero_ratio": (g == 0).mean(),
        }
    )


def make_daily_features(sensor, df):
    parts = []
    stat_rows = []
    disc_rows = []
    obj_rows = []

    val_cols = infer_value_cols(df)
    for col in val_cols:
        s_num = pd.to_numeric(df[col], errors="coerce")

        if s_num.notna().sum() > 0:
            prefix = f"{sensor}_{col}"
            if is_discrete(sensor, col, s_num):
                tmp = df[KEYS].copy()
                tmp[col] = s_num
                vc = tmp.dropna().groupby(KEYS + [col]).size().unstack(fill_value=0)
                vc.columns = [f"{prefix}_value_{int(c)}_count" for c in vc.columns]
                total = vc.sum(axis=1).replace(0, np.nan)
                ratio = vc.div(total, axis=0).add_suffix("_ratio")
                parts += [vc.reset_index(), ratio.reset_index()]

                for k, v in s_num.value_counts(dropna=True).sort_index().items():
                    disc_rows.append({"sensor": sensor, "column": col, "code": k, "count": v, "ratio": v / s_num.notna().sum()})
            else:
                agg = df.groupby(KEYS)[col].apply(lambda x: agg_numeric(pd.to_numeric(x, errors="coerce"), prefix)).unstack().reset_index()
                parts.append(agg)

                x = s_num.dropna()
                stat_rows.append(raw_stat(sensor, col, x, "continuous"))
            continue

        if df[col].dtype == "object" and col not in ID_COLS:
            parsed = df[col].map(to_obj)
            lens = parsed.map(obj_len)
            nums = parsed.map(flatten_nums)
            num_mean = nums.map(lambda z: np.mean(z) if len(z) else np.nan)
            num_sum = nums.map(lambda z: np.sum(z) if len(z) else np.nan)
            num_max = nums.map(lambda z: np.max(z) if len(z) else np.nan)

            tmp = df[KEYS].copy()
            tmp[f"{col}_object_len"] = lens
            tmp[f"{col}_object_num_mean"] = num_mean
            tmp[f"{col}_object_num_sum"] = num_sum
            tmp[f"{col}_object_num_max"] = num_max

            for oc in tmp.columns:
                if oc in KEYS:
                    continue
                prefix = f"{sensor}_{oc}"
                agg = tmp.groupby(KEYS)[oc].apply(lambda x: agg_numeric(pd.to_numeric(x, errors="coerce"), prefix)).unstack().reset_index()
                parts.append(agg)

            obj_rows.append(
                {
                    "sensor": sensor,
                    "column": col,
                    "nonnull": int(parsed.notna().sum()),
                    "parseable_object_ratio": float(parsed.notna().mean()),
                    "mean_object_len": float(pd.to_numeric(lens, errors="coerce").mean()),
                    "mean_numeric_items": float(nums.map(len).mean()),
                }
            )

    row_count = df.groupby(KEYS).size().rename(f"{sensor}_row_count").reset_index()
    parts.append(row_count)

    if not parts:
        return pd.DataFrame(columns=KEYS), stat_rows, disc_rows, obj_rows

    out = parts[0]
    for p in parts[1:]:
        out = out.merge(p, on=KEYS, how="outer")
    return out, stat_rows, disc_rows, obj_rows


def make_timeblock_features(sensor, df):
    if "timestamp" not in df.columns:
        return pd.DataFrame(columns=KEYS)

    tmp = df[df["timestamp"].notna()].copy()
    if tmp.empty:
        return pd.DataFrame(columns=KEYS)

    tmp["hour"] = tmp["timestamp"].dt.hour
    frames = []
    val_cols = infer_value_cols(tmp)

    for start, end in BLOCKS:
        b = tmp[(tmp["hour"] >= start) & (tmp["hour"] < end)]
        if b.empty:
            continue
        label = f"{start:02d}_{end:02d}"
        cnt = b.groupby(KEYS).size().rename(f"{sensor}_{label}_row_count").reset_index()
        frames.append(cnt)

        for col in val_cols:
            x = pd.to_numeric(b[col], errors="coerce")
            if x.notna().sum() == 0:
                continue

            if is_discrete(sensor, col, x):
                t = b[KEYS].copy()
                t[col] = x
                vc = t.dropna().groupby(KEYS + [col]).size().unstack(fill_value=0)
                vc.columns = [f"{sensor}_{col}_{label}_value_{int(c)}_count" for c in vc.columns]
                frames.append(vc.reset_index())
            else:
                t = b[KEYS].copy()
                t[col] = x
                prefix = f"{sensor}_{col}_{label}"
                agg = t.groupby(KEYS)[col].apply(lambda z: pd.Series({
                    f"{prefix}_mean": z.mean(),
                    f"{prefix}_std": z.std(),
                    f"{prefix}_min": z.min(),
                    f"{prefix}_median": z.median(),
                    f"{prefix}_max": z.max(),
                    f"{prefix}_sum": z.sum(),
                    f"{prefix}_nonnull_count": z.notna().sum(),
                })).unstack().reset_index()
                frames.append(agg)

    if not frames:
        return pd.DataFrame(columns=KEYS)

    out = frames[0]
    for f in frames[1:]:
        out = out.merge(f, on=KEYS, how="outer")
    return out


def raw_stat(sensor, col, x, kind):
    return {
        "sensor": sensor,
        "column": col,
        "kind": kind,
        "n": len(x),
        "mean": x.mean(),
        "std": x.std(),
        "min": x.min(),
        "q25": x.quantile(0.25),
        "median": x.median(),
        "q75": x.quantile(0.75),
        "q95": x.quantile(0.95),
        "q99": x.quantile(0.99),
        "max": x.max(),
        "zero_ratio": (x == 0).mean(),
        "negative_ratio": (x < 0).mean(),
        "unique_count": x.nunique(),
        "skew": x.skew(),
    }


def feature_group(col):
    if any(k in col for k in ["row_count", "nonnull_count", "object_len", "timestamp"]):
        return "coverage_proxy"
    for sensor, group in SENSOR_GROUP.items():
        if col.startswith(sensor + "_"):
            return group
    return "other"


def valid_feature(s):
    x = pd.to_numeric(s, errors="coerce").dropna()
    return len(x) >= 20 and x.nunique() > 1


def outlier_summary(df):
    rows = []
    for col in df.columns:
        if col in KEYS or not valid_feature(df[col]):
            continue
        x = pd.to_numeric(df[col], errors="coerce").dropna()
        q1, q3 = x.quantile([0.25, 0.75])
        iqr = q3 - q1
        if iqr == 0 or not np.isfinite(iqr):
            continue
        upper = q3 + 1.5 * iqr
        rows.append(
            {
                "feature": col,
                "group": feature_group(col),
                "outlier_ratio": float((x > upper).mean()),
                "q99": x.quantile(0.99),
                "max": x.max(),
            }
        )
    return pd.DataFrame(rows).sort_values("outlier_ratio", ascending=False)


def log_candidates(df):
    rows = []
    banned = ["_std", "_min", "_median", "_q25", "_zero_ratio", "object_len", "timestamp", "row_count", "nonnull_count"]
    for col in df.columns:
        if col in KEYS or any(b in col for b in banned) or not valid_feature(df[col]):
            continue
        x = pd.to_numeric(df[col], errors="coerce").dropna()
        if (x < 0).any():
            continue
        med = x[x > 0].median()
        q99 = x.quantile(0.99)
        zero_ratio = float((x == 0).mean())
        if not np.isfinite(med) or med <= 0 or zero_ratio >= 0.95:
            continue
        ratio = q99 / med if med else np.nan
        skew = x.skew()
        if np.isfinite(skew) and np.isfinite(ratio) and (skew >= 1.0 or ratio >= 10):
            rows.append({"feature": col, "group": feature_group(col), "skew": skew, "zero_ratio": zero_ratio, "q99_over_positive_median": ratio})
    return pd.DataFrame(rows).sort_values(["skew", "q99_over_positive_median"], ascending=False)


def corr_table(feat, train, targets):
    df = feat.merge(train[KEYS + targets], on=KEYS, how="inner")
    rows = []

    for target in targets:
        y = pd.to_numeric(df[target], errors="coerce")
        if y.notna().sum() < 30 or y.nunique(dropna=True) <= 1:
            continue

        y_adj = y - y.groupby(df["subject_id"]).transform("mean")

        for col in feat.columns:
            if col in KEYS or not valid_feature(df[col]):
                continue

            x = pd.to_numeric(df[col], errors="coerce")
            m = x.notna() & y.notna()
            if m.sum() < 30:
                continue

            r = x[m].corr(y[m], method="spearman")
            x_adj = x - x.groupby(df["subject_id"]).transform("mean")
            ra = x_adj[m].corr(y_adj[m], method="spearman")

            if pd.isna(r) and pd.isna(ra):
                continue

            rows.append(
                {
                    "target": target,
                    "feature": col,
                    "group": feature_group(col),
                    "n": int(m.sum()),
                    "spearman": r,
                    "abs_spearman": abs(r) if pd.notna(r) else np.nan,
                    "subject_adjusted_spearman": ra,
                    "abs_subject_adjusted_spearman": abs(ra) if pd.notna(ra) else np.nan,
                }
            )

    return pd.DataFrame(rows)


def top_lines(df, target, metric, n=10, exclude_proxy=False):
    if df.empty:
        return []
    d = df[df["target"] == target].copy()
    if exclude_proxy:
        d = d[d["group"] != "coverage_proxy"]
    d = d.sort_values(f"abs_{metric}", ascending=False).head(n)
    return [
        f"- {r.feature}: {metric}={getattr(r, metric):.4f}, group={r.group}, n={int(r.n)}"
        for r in d.itertuples()
        if pd.notna(getattr(r, metric))
    ]


def write_summary(sensors, daily, tb, cont, disc, obj, outlier, logs, corr, tcorr, targets):
    md = []
    md.append("# Sensor Statistics Summary\n")
    md.append("## 1. Run Overview")
    md.append(f"- Sensors processed: {len(sensors)}")
    md.append(f"- Daily feature count: {daily.shape[1] - len(KEYS)}")
    md.append(f"- Timeblock feature count: {tb.shape[1] - len(KEYS)}\n")

    md.append("## 2. Output Files")
    for name in [
        "daily_sensor_features.parquet",
        "timeblock_sensor_features.parquet",
        "continuous_sensor_stats.csv",
        "discrete_sensor_stats.csv",
        "object_sensor_parse_stats.csv",
        "sensor_outlier_summary.csv",
        "log_transform_candidates.csv",
        "daily_feature_target_signals.csv",
        "timeblock_feature_target_signals.csv",
    ]:
        md.append(f"- {name}")
    md.append("")

    md.append("## 3. Log Transform Candidates")
    if logs.empty:
        md.append("- No valid candidates")
    else:
        for r in logs.head(30).itertuples():
            md.append(f"- {r.feature}: skew={r.skew:.3f}, zero_ratio={r.zero_ratio:.3f}, q99/positive_median={r.q99_over_positive_median:.3f}, group={r.group}")
    md.append("")

    md.append("## 4. High Outlier Features")
    if outlier.empty:
        md.append("- No valid outlier features")
    else:
        for r in outlier.head(30).itertuples():
            md.append(f"- {r.feature}: outlier_ratio={r.outlier_ratio:.3f}, q99={r.q99}, max={r.max}, group={r.group}")
    md.append("")

    md.append("## 5. Daily Real-Signal Top Correlations")
    for t in targets:
        md.append(f"### {t}")
        lines = top_lines(corr, t, "spearman", 10, exclude_proxy=True)
        md.extend(lines if lines else ["- No valid signal"])
        md.append("")
        md.append("#### Subject-adjusted")
        lines = top_lines(corr, t, "subject_adjusted_spearman", 10, exclude_proxy=True)
        md.extend(lines if lines else ["- No valid signal"])
        md.append("")

    md.append("## 6. Daily Coverage Proxy Top Correlations")
    for t in targets:
        md.append(f"### {t}")
        d = corr[(corr["target"] == t) & (corr["group"] == "coverage_proxy")].sort_values("abs_spearman", ascending=False).head(10)
        md.extend([f"- {r.feature}: spearman={r.spearman:.4f}, n={int(r.n)}" for r in d.itertuples()] or ["- No valid proxy"])
        md.append("")

    md.append("## 7. Timeblock Real-Signal Top Correlations")
    for t in targets:
        md.append(f"### {t}")
        lines = top_lines(tcorr, t, "spearman", 10, exclude_proxy=True)
        md.extend(lines if lines else ["- No valid signal"])
        md.append("")

    md.append("## 8. Interpretation Notes")
    md.append("- timestamp/object_len/std/zero-only feature는 리포트 후보에서 제외했습니다.")
    md.append("- coverage_proxy는 결측/수집량 signal로 별도 관리하고, 실제 행동 feature와 분리해서 모델에 투입하십시오.")
    md.append("- log1p 후보는 non-negative, non-constant, positive median 기준을 통과한 feature만 남겼습니다.")
    md.append("- subject-adjusted correlation이 유지되는 feature를 subject deviation/z-score feature 후보로 우선 사용하십시오.")
    md.append("- timeblock feature는 4시간 단위입니다. 수면 관련 target은 20_24, 00_04, 04_08 구간을 우선 검토하십시오.")

    (OUT_DIR / "sensor_statistics_summary.md").write_text("\n".join(md), encoding="utf-8")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train, sub, base, targets = load_keys()

    sensors = []
    daily_parts = []
    tb_parts = []
    cont_rows, disc_rows, obj_rows = [], [], []

    files = sorted(ITEMS_DIR.glob("*.parquet")) + sorted(ITEMS_DIR.glob("*.csv"))

    for path in files:
        sensor = path.stem.replace("ch2025_", "").replace("ch2026_", "")
        print(f"[INFO] Processing {sensor}: {path}")

        df = normalize_keys(read_table(path))
        if not set(KEYS).issubset(df.columns):
            print(f"[WARN] Skip {sensor}: missing keys")
            continue

        df = df.dropna(subset=KEYS)
        if df.empty:
            print(f"[WARN] Skip {sensor}: empty after key normalization")
            continue

        sensors.append(sensor)

        daily, c, d, o = make_daily_features(sensor, df)
        tb = make_timeblock_features(sensor, df)

        daily_parts.append(daily)
        tb_parts.append(tb)
        cont_rows.extend(c)
        disc_rows.extend(d)
        obj_rows.extend(o)

    daily = base[KEYS].copy()
    for p in daily_parts:
        if not p.empty:
            daily = daily.merge(p, on=KEYS, how="left")

    tb = base[KEYS].copy()
    for p in tb_parts:
        if not p.empty:
            tb = tb.merge(p, on=KEYS, how="left")

    daily.to_parquet(OUT_DIR / "daily_sensor_features.parquet", index=False)
    tb.to_parquet(OUT_DIR / "timeblock_sensor_features.parquet", index=False)

    cont = pd.DataFrame(cont_rows)
    disc = pd.DataFrame(disc_rows)
    obj = pd.DataFrame(obj_rows)

    cont.to_csv(OUT_DIR / "continuous_sensor_stats.csv", index=False, encoding="utf-8-sig")
    disc.to_csv(OUT_DIR / "discrete_sensor_stats.csv", index=False, encoding="utf-8-sig")
    obj.to_csv(OUT_DIR / "object_sensor_parse_stats.csv", index=False, encoding="utf-8-sig")

    outlier = outlier_summary(daily)
    logs = log_candidates(daily)
    corr = corr_table(daily, train, targets)
    tcorr = corr_table(tb, train, targets)

    outlier.to_csv(OUT_DIR / "sensor_outlier_summary.csv", index=False, encoding="utf-8-sig")
    logs.to_csv(OUT_DIR / "log_transform_candidates.csv", index=False, encoding="utf-8-sig")
    corr.to_csv(OUT_DIR / "daily_feature_target_signals.csv", index=False, encoding="utf-8-sig")
    tcorr.to_csv(OUT_DIR / "timeblock_feature_target_signals.csv", index=False, encoding="utf-8-sig")

    write_summary(sensors, daily, tb, cont, disc, obj, outlier, logs, corr, tcorr, targets)

    print("[DONE] 06_sensor_statistics EDA completed")
    print(f"[OUT] {OUT_DIR}")


if __name__ == "__main__":
    main()