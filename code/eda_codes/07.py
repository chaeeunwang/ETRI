from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

DATA_DIR = Path("data")
ITEMS_DIR = DATA_DIR / "ch2025_data_items"
OUT_DIR = Path("outputs/eda/07_train_test_shift")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_PATH = DATA_DIR / "ch2026_metrics_train.csv"
TEST_PATH = DATA_DIR / "ch2026_submission_sample.csv"
KEYS = ["subject_id", "lifelog_date"]
TARGETS = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]


def save(df, name):
    df.to_csv(OUT_DIR / name, index=False, encoding="utf-8-sig")


def normalize_dates(df):
    for c in ["lifelog_date", "sleep_date"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    return df


def get_sensor_name(path):
    return path.stem.replace("ch2025_", "")


def find_col(cols, candidates):
    lower = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None


def sensor_day_presence(path):
    sensor = get_sensor_name(path)
    df = pd.read_parquet(path)

    subject_col = find_col(df.columns, ["subject_id", "user_id", "id"])
    date_col = find_col(df.columns, ["lifelog_date", "date", "sleep_date"])
    ts_col = find_col(df.columns, ["timestamp", "ts", "time", "datetime", "timestamp_ms"])

    if subject_col is None:
        return pd.DataFrame()

    if date_col is not None:
        d = pd.to_datetime(df[date_col], errors="coerce").dt.date
    elif ts_col is not None:
        ts = df[ts_col]
        if pd.api.types.is_numeric_dtype(ts):
            unit = "ms" if ts.dropna().astype(float).median() > 1e12 else "s"
            d = pd.to_datetime(ts, unit=unit, errors="coerce").dt.date
        else:
            d = pd.to_datetime(ts, errors="coerce").dt.date
    else:
        return pd.DataFrame()

    out = pd.DataFrame({
        "subject_id": df[subject_col].astype(str),
        "lifelog_date": pd.to_datetime(d, errors="coerce"),
        sensor: 1
    })

    out = out.dropna(subset=["lifelog_date"])
    out = out.groupby(KEYS, as_index=False)[sensor].max()
    return out


def build_sensor_presence(all_keys):
    merged = all_keys.copy()

    for path in sorted(ITEMS_DIR.glob("*.parquet")):
        try:
            p = sensor_day_presence(path)
            if p.empty:
                continue
            merged = merged.merge(p, on=KEYS, how="left")
            print(f"[OK] {get_sensor_name(path)}")
        except Exception as e:
            print(f"[SKIP] {path.name}: {type(e).__name__}: {e}")

    sensor_cols = [c for c in merged.columns if c not in KEYS + ["split"]]
    merged[sensor_cols] = merged[sensor_cols].fillna(0).astype(int)
    return merged


def psi(expected, actual, bins=10):
    e = pd.Series(expected).replace([np.inf, -np.inf], np.nan).dropna()
    a = pd.Series(actual).replace([np.inf, -np.inf], np.nan).dropna()

    if e.nunique() <= 1 or a.nunique() <= 1:
        return np.nan

    qs = np.linspace(0, 1, bins + 1)
    edges = np.unique(np.quantile(e, qs))

    if len(edges) <= 2:
        return np.nan

    e_cnt = pd.cut(e, edges, include_lowest=True).value_counts(normalize=True).sort_index()
    a_cnt = pd.cut(a, edges, include_lowest=True).value_counts(normalize=True).sort_index()

    e_pct = e_cnt.reindex(e_cnt.index, fill_value=0).clip(1e-6)
    a_pct = a_cnt.reindex(e_cnt.index, fill_value=0).clip(1e-6)

    return float(((a_pct - e_pct) * np.log(a_pct / e_pct)).sum())


def feature_shift(train_feat, test_feat, feature_cols):
    rows = []

    for c in feature_cols:
        tr = pd.to_numeric(train_feat[c], errors="coerce")
        te = pd.to_numeric(test_feat[c], errors="coerce")

        rows.append({
            "feature": c,
            "train_mean": tr.mean(),
            "test_mean": te.mean(),
            "mean_diff": te.mean() - tr.mean(),
            "train_std": tr.std(),
            "test_std": te.std(),
            "std_diff": te.std() - tr.std(),
            "train_missing_ratio": tr.isna().mean(),
            "test_missing_ratio": te.isna().mean(),
            "missing_diff": te.isna().mean() - tr.isna().mean(),
            "psi": psi(tr, te)
        })

    return pd.DataFrame(rows).sort_values(
        ["psi", "missing_diff"], ascending=False, na_position="last"
    )


def adversarial_validation(df, feature_cols):
    try:
        from sklearn.model_selection import StratifiedKFold, cross_val_score
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import make_pipeline
    except Exception as e:
        return pd.DataFrame([{"status": "skipped", "reason": str(e)}])

    x = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    y = (df["split"] == "test").astype(int)

    if len(feature_cols) == 0 or y.nunique() < 2:
        return pd.DataFrame([{"status": "skipped", "reason": "not enough features or labels"}])

    model = make_pipeline(
        SimpleImputer(strategy="median"),
        RandomForestClassifier(
            n_estimators=300,
            max_depth=4,
            min_samples_leaf=10,
            random_state=42,
            n_jobs=-1
        )
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, x, y, scoring="roc_auc", cv=cv, n_jobs=-1)

    return pd.DataFrame([{
        "status": "ok",
        "auc_mean": scores.mean(),
        "auc_std": scores.std(),
        "fold_scores": "|".join(f"{s:.5f}" for s in scores)
    }])


def write_summary(subject_count, date_range, missing_diff, shift, adv):
    top_missing = missing_diff.head(10)
    top_psi = shift.dropna(subset=["psi"]).head(10)
    adv_row = adv.iloc[0].to_dict()

    lines = [
        "# 07 Train/Test Distribution Shift Summary",
        "",
        "## 핵심 산출물",
        "- train/test subject 구성 차이",
        "- train/test date range 차이",
        "- sensor day-level missing ratio 차이",
        "- numeric feature distribution shift",
        "- adversarial validation AUC",
        "",
        "## Subject Count",
        subject_count.to_markdown(index=False),
        "",
        "## Date Range",
        date_range.to_markdown(index=False),
        "",
        "## Top Missing Shift",
        top_missing.to_markdown(index=False),
        "",
        "## Top PSI Features",
        top_psi.to_markdown(index=False),
        "",
        "## Adversarial Validation",
        pd.DataFrame([adv_row]).to_markdown(index=False),
        "",
        "## 판단 기준",
        "- PSI >= 0.25: 강한 분포 차이 후보",
        "- PSI >= 0.10: 중간 수준 분포 차이 후보",
        "- missing_diff 절댓값이 큰 센서: missing feature 또는 feature 제거 검토",
        "- adversarial AUC >= 0.70: train/test shift가 크므로 feature 안정성 재검토 필요",
        ""
    ]

    (OUT_DIR / "train_test_shift_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    train = normalize_dates(pd.read_csv(TRAIN_PATH))
    test = normalize_dates(pd.read_csv(TEST_PATH))

    train["subject_id"] = train["subject_id"].astype(str)
    test["subject_id"] = test["subject_id"].astype(str)

    train_keys = train[KEYS].copy()
    test_keys = test[KEYS].copy()
    train_keys["split"] = "train"
    test_keys["split"] = "test"
    all_keys = pd.concat([train_keys, test_keys], ignore_index=True)

    subject_count = all_keys.groupby(["split", "subject_id"]).size().reset_index(name="row_count")
    save(subject_count, "train_test_subject_count.csv")

    date_range = all_keys.groupby(["split", "subject_id"])["lifelog_date"].agg(
        min_date="min",
        max_date="max",
        row_count="count"
    ).reset_index()
    save(date_range, "train_test_date_range.csv")

    presence = build_sensor_presence(all_keys)
    sensor_cols = [c for c in presence.columns if c not in KEYS + ["split"]]
    save(presence, "sensor_day_presence_train_test.csv")

    miss = presence.groupby("split")[sensor_cols].apply(lambda x: 1 - x.mean()).T.reset_index()

    rename_map = {
        "index": "sensor",
        "train": "train_missing_ratio",
        "test": "test_missing_ratio",
    }
    miss = miss.rename(columns=rename_map)

    if "train_missing_ratio" not in miss.columns:
        miss["train_missing_ratio"] = np.nan
    if "test_missing_ratio" not in miss.columns:
        miss["test_missing_ratio"] = np.nan

    miss["missing_diff"] = miss["test_missing_ratio"] - miss["train_missing_ratio"]
    miss = miss.sort_values("missing_diff", ascending=False)
    save(miss, "train_test_missing_diff.csv")

    train_num = train.drop(columns=[c for c in TARGETS if c in train.columns], errors="ignore")
    test_num = test.copy()

    common_cols = sorted(set(train_num.columns) & set(test_num.columns))
    numeric_cols = [
        c for c in common_cols
        if c not in KEYS + ["sleep_date"]
        and pd.api.types.is_numeric_dtype(train_num[c])
        and pd.api.types.is_numeric_dtype(test_num[c])
    ]

    train_feat = train_num[KEYS + numeric_cols].merge(presence[presence["split"] == "train"].drop(columns="split"), on=KEYS, how="left")
    test_feat = test_num[KEYS + numeric_cols].merge(presence[presence["split"] == "test"].drop(columns="split"), on=KEYS, how="left")

    feature_cols = numeric_cols + sensor_cols
    shift = feature_shift(train_feat, test_feat, feature_cols)
    save(shift, "train_test_feature_shift.csv")
    save(shift[["feature", "psi"]].sort_values("psi", ascending=False), "psi_by_feature.csv")

    adv_df = pd.concat([
        train_feat.assign(split="train"),
        test_feat.assign(split="test")
    ], ignore_index=True)

    adv = adversarial_validation(adv_df, feature_cols)
    save(adv, "adversarial_validation_result.csv")

    write_summary(subject_count, date_range, miss, shift, adv)
    print(f"[DONE] saved to {OUT_DIR}")


if __name__ == "__main__":
    main()