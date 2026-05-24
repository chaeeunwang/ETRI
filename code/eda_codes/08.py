from pathlib import Path
import argparse
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

warnings.filterwarnings("ignore")

TARGETS_DEFAULT = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]
KEY_COLS = ["subject_id", "lifelog_date", "sleep_date", "fold"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--oof",
        type=str,
        default="C:/01_ETRI/outputs/MultiTaskBiLSTM산출물/oof_5fold_bilstm_multitask_safe.csv",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="C:/01_ETRI/outputs/eda/08_baseline_oof_analysis",
    )
    parser.add_argument(
        "--targets",
        nargs="*",
        default=TARGETS_DEFAULT,
    )
    parser.add_argument("--n_bins", type=int, default=10)
    parser.add_argument("--high_conf", type=float, default=0.9)
    parser.add_argument("--low_conf_min", type=float, default=0.4)
    parser.add_argument("--low_conf_max", type=float, default=0.6)
    return parser.parse_args()


def ensure_dir(path):
    path.mkdir(parents=True, exist_ok=True)


def find_prob_cols(df, target):
    patterns = [
        f"{target}_prob_",
        f"prob_{target}_",
        f"{target}_pred_",
        f"pred_{target}_",
        f"oof_{target}_",
    ]

    multi_cols = []
    for p in patterns:
        cols = [c for c in df.columns if c.startswith(p)]
        if cols:
            multi_cols = cols
            break

    if multi_cols:
        def cls_id(c):
            tail = c.split("_")[-1]
            return int(tail) if tail.isdigit() else tail

        return sorted(multi_cols, key=cls_id)

    single_candidates = [
        f"{target}_prob",
        f"prob_{target}",
        f"{target}_pred",
        f"pred_{target}",
        f"oof_{target}",
        f"{target}_oof",
    ]
    return [c for c in single_candidates if c in df.columns]


def get_target_info(df, target):
    if target not in df.columns:
        return None

    prob_cols = find_prob_cols(df, target)
    if not prob_cols:
        return None

    y = df[target].dropna()
    classes = sorted(y.unique().astype(int).tolist())

    return {
        "target": target,
        "classes": classes,
        "prob_cols": prob_cols,
        "is_binary_single_prob": len(prob_cols) == 1 and len(classes) == 2,
    }


def build_prob_matrix(df, info):
    cols = info["prob_cols"]
    classes = info["classes"]

    if info["is_binary_single_prob"]:
        p1 = df[cols[0]].astype(float).clip(1e-15, 1 - 1e-15)
        return np.vstack([1 - p1, p1]).T, [0, 1]

    prob = df[cols].astype(float).clip(1e-15, 1 - 1e-15)
    row_sum = prob.sum(axis=1).replace(0, np.nan)
    prob = prob.div(row_sum, axis=0).fillna(1 / len(cols))

    if len(cols) == len(classes):
        return prob.values, classes

    return prob.values, list(range(len(cols)))


def target_logloss(df, info):
    valid = df[[info["target"], *info["prob_cols"]]].dropna()
    if valid.empty:
        return np.nan

    prob, labels = build_prob_matrix(valid, info)
    y = valid[info["target"]].astype(int).values

    try:
        return log_loss(y, prob, labels=labels)
    except ValueError:
        return np.nan


def grouped_logloss(df, info, group_col):
    if group_col not in df.columns:
        return pd.DataFrame()

    rows = []
    for key, part in df.groupby(group_col):
        score = target_logloss(part, info)
        rows.append({
            group_col: key,
            "target": info["target"],
            "logloss": score,
            "n": len(part),
        })
    return pd.DataFrame(rows)


def prediction_distribution(df, info):
    valid = df[[info["target"], *info["prob_cols"]]].dropna()
    if valid.empty:
        return pd.DataFrame()

    prob, labels = build_prob_matrix(valid, info)
    pred_class = np.array(labels)[np.argmax(prob, axis=1)]
    confidence = prob.max(axis=1)

    rows = []
    for label in labels:
        mask = pred_class == label
        rows.append({
            "target": info["target"],
            "pred_class": label,
            "count": int(mask.sum()),
            "ratio": float(mask.mean()),
            "confidence_mean": float(confidence[mask].mean()) if mask.any() else np.nan,
            "confidence_median": float(np.median(confidence[mask])) if mask.any() else np.nan,
        })

    rows.append({
        "target": info["target"],
        "pred_class": "ALL",
        "count": len(valid),
        "ratio": 1.0,
        "confidence_mean": float(confidence.mean()),
        "confidence_median": float(np.median(confidence)),
    })
    return pd.DataFrame(rows)


def calibration_table(df, info, n_bins):
    valid = df[[info["target"], *info["prob_cols"]]].dropna()
    if valid.empty:
        return pd.DataFrame()

    prob, labels = build_prob_matrix(valid, info)
    y = valid[info["target"]].astype(int).values

    rows = []
    bins = np.linspace(0, 1, n_bins + 1)

    for idx, cls in enumerate(labels):
        p = prob[:, idx]
        y_bin = (y == cls).astype(int)
        bin_id = np.digitize(p, bins, right=True)
        bin_id = np.clip(bin_id, 1, n_bins)

        for b in range(1, n_bins + 1):
            mask = bin_id == b
            if not mask.any():
                continue
            rows.append({
                "target": info["target"],
                "class": cls,
                "bin": b,
                "bin_min": bins[b - 1],
                "bin_max": bins[b],
                "n": int(mask.sum()),
                "pred_mean": float(p[mask].mean()),
                "true_rate": float(y_bin[mask].mean()),
                "abs_gap": float(abs(p[mask].mean() - y_bin[mask].mean())),
            })

    return pd.DataFrame(rows)


def error_cases(df, info, high_conf, low_conf_min, low_conf_max):
    base_cols = [c for c in KEY_COLS if c in df.columns]
    valid = df[base_cols + [info["target"], *info["prob_cols"]]].dropna().copy()
    if valid.empty:
        return pd.DataFrame(), pd.DataFrame()

    prob, labels = build_prob_matrix(valid, info)
    pred_idx = np.argmax(prob, axis=1)
    pred_class = np.array(labels)[pred_idx]
    confidence = prob.max(axis=1)
    y = valid[info["target"]].astype(int).values

    valid["target"] = info["target"]
    valid["true_label"] = y
    valid["pred_label"] = pred_class
    valid["confidence"] = confidence
    valid["is_correct"] = y == pred_class

    high_wrong = valid[(~valid["is_correct"]) & (valid["confidence"] >= high_conf)]
    low_conf = valid[
        (valid["confidence"] >= low_conf_min)
        & (valid["confidence"] <= low_conf_max)
    ]

    keep = base_cols + ["target", "true_label", "pred_label", "confidence", "is_correct"]
    return high_wrong[keep], low_conf[keep]


def target_subject_matrix(df, infos):
    if "subject_id" not in df.columns:
        return pd.DataFrame()

    rows = []
    for info in infos:
        g = grouped_logloss(df, info, "subject_id")
        if not g.empty:
            rows.append(g)

    if not rows:
        return pd.DataFrame()

    long_df = pd.concat(rows, ignore_index=True)
    return long_df.pivot(index="subject_id", columns="target", values="logloss").reset_index()


def write_summary(out_dir, available_infos, by_target, by_subject, by_fold, high_wrong, low_conf):
    weak_target = by_target.sort_values("logloss", ascending=False).head(1)
    weak_subject = by_subject.sort_values("logloss", ascending=False).head(1) if not by_subject.empty else pd.DataFrame()
    weak_fold = by_fold.sort_values("logloss", ascending=False).head(1) if not by_fold.empty else pd.DataFrame()

    lines = [
        "# 08 Baseline OOF Analysis",
        "",
        "## 1. 분석 대상",
        f"- 사용 target: {', '.join([x['target'] for x in available_infos])}",
        f"- OOF target 수: {len(available_infos)}",
        "",
        "## 2. Target별 LogLoss",
    ]

    for _, r in by_target.iterrows():
        lines.append(f"- {r['target']}: {r['logloss']:.6f} (n={int(r['n'])})")

    lines.extend(["", "## 3. 주요 취약 구간"])

    if not weak_target.empty:
        r = weak_target.iloc[0]
        lines.append(f"- 가장 약한 target: {r['target']} / logloss={r['logloss']:.6f}")

    if not weak_subject.empty:
        r = weak_subject.iloc[0]
        lines.append(f"- 가장 약한 subject: {r['subject_id']} / target={r['target']} / logloss={r['logloss']:.6f}")

    if not weak_fold.empty:
        r = weak_fold.iloc[0]
        lines.append(f"- 가장 약한 fold: {r['fold']} / target={r['target']} / logloss={r['logloss']:.6f}")

    lines.extend([
        "",
        "## 4. Confidence 분석",
        f"- high confidence wrong cases: {len(high_wrong)}",
        f"- low confidence cases: {len(low_conf)}",
        "",
        "## 5. 다음 액션",
        "1. logloss가 큰 target부터 feature와 loss를 분리 검토",
        "2. 특정 subject에서만 성능이 낮으면 subject-aware feature 추가",
        "3. high-confidence wrong case는 calibration 또는 blending 우선 검토",
        "4. low-confidence case가 많으면 feature 부족 또는 모델 underfitting 가능성 점검",
    ])

    (out_dir / "bilstm_error_analysis.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    args = parse_args()
    oof_path = Path(args.oof)
    out_dir = Path(args.out)
    ensure_dir(out_dir)

    if not oof_path.exists():
        raise FileNotFoundError(f"OOF file not found: {oof_path}")

    df = pd.read_csv(oof_path)

    for c in ["lifelog_date", "sleep_date"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")

    infos = [get_target_info(df, t) for t in args.targets]
    infos = [x for x in infos if x is not None]

    if not infos:
        raise ValueError("No valid target/probability columns found in OOF file.")

    by_target = []
    by_subject = []
    by_fold = []
    pred_dist = []
    calib = []
    high_wrong_all = []
    low_conf_all = []

    for info in infos:
        by_target.append({
            "target": info["target"],
            "classes": json.dumps(info["classes"], ensure_ascii=False),
            "prob_cols": json.dumps(info["prob_cols"], ensure_ascii=False),
            "logloss": target_logloss(df, info),
            "n": int(df[[info["target"], *info["prob_cols"]]].dropna().shape[0]),
        })

        g_subject = grouped_logloss(df, info, "subject_id")
        g_fold = grouped_logloss(df, info, "fold")

        if not g_subject.empty:
            by_subject.append(g_subject)
        if not g_fold.empty:
            by_fold.append(g_fold)

        pred_dist.append(prediction_distribution(df, info))
        calib.append(calibration_table(df, info, args.n_bins))

        high_wrong, low_conf = error_cases(
            df,
            info,
            args.high_conf,
            args.low_conf_min,
            args.low_conf_max,
        )
        high_wrong_all.append(high_wrong)
        low_conf_all.append(low_conf)

    by_target = pd.DataFrame(by_target).sort_values("logloss", ascending=False)
    by_subject = pd.concat(by_subject, ignore_index=True) if by_subject else pd.DataFrame()
    by_fold = pd.concat(by_fold, ignore_index=True) if by_fold else pd.DataFrame()
    pred_dist = pd.concat(pred_dist, ignore_index=True) if pred_dist else pd.DataFrame()
    calib = pd.concat(calib, ignore_index=True) if calib else pd.DataFrame()
    high_wrong_all = pd.concat(high_wrong_all, ignore_index=True) if high_wrong_all else pd.DataFrame()
    low_conf_all = pd.concat(low_conf_all, ignore_index=True) if low_conf_all else pd.DataFrame()
    matrix = target_subject_matrix(df, infos)

    by_target.to_csv(out_dir / "oof_logloss_by_target.csv", index=False)
    by_subject.to_csv(out_dir / "oof_logloss_by_subject.csv", index=False)
    by_fold.to_csv(out_dir / "oof_logloss_by_fold.csv", index=False)
    matrix.to_csv(out_dir / "oof_logloss_target_subject_matrix.csv", index=False)
    pred_dist.to_csv(out_dir / "prediction_distribution.csv", index=False)
    calib.to_csv(out_dir / "calibration_by_target.csv", index=False)
    high_wrong_all.to_csv(out_dir / "high_confidence_wrong_cases.csv", index=False)
    low_conf_all.to_csv(out_dir / "low_confidence_cases.csv", index=False)

    write_summary(
        out_dir,
        infos,
        by_target,
        by_subject,
        by_fold,
        high_wrong_all,
        low_conf_all,
    )

    print(f"[DONE] Saved 08 baseline OOF analysis to: {out_dir}")


if __name__ == "__main__":
    main()