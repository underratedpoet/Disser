from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder


# ============================================================
# CONFIGURATION
# ============================================================

OUTPUT_DIR = Path("feature_experiment_results")
SAMPLE_PATH = OUTPUT_DIR / "sample.csv"

SEED = 42
N_ESTIMATORS = 100
N_FOLDS = 5

# (метка, путь_к_fixed, путь_к_mgd) — оба случая берутся из уже
# посчитанных прогонов, ничего заново не декодируется.
CONFIGS = [
    (
        "fmax=22050 (полный спектр)",
        OUTPUT_DIR / "fixed_diff__features.parquet",
        OUTPUT_DIR / "modified_group_delay__varying_features.parquet",
    ),
    (
        "fmax=16000",
        OUTPUT_DIR / "fixed_diff__fmax16000__features.parquet",
        OUTPUT_DIR / "modified_group_delay__fmax16000__features.parquet",
    ),
]


# ============================================================
# K-FOLD ПАРНОЕ СРАВНЕНИЕ (одни и те же фолды для fixed и combined)
# ============================================================

def kfold_compare(
    X_fixed: pd.DataFrame,
    X_combined: pd.DataFrame,
    y: pd.Series,
    n_folds: int = N_FOLDS,
    seed: int = SEED,
) -> pd.DataFrame:

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    rows = []

    for fold_index, (train_idx, test_idx) in enumerate(skf.split(X_fixed, y_encoded), 1):

        model_fixed = RandomForestClassifier(
            n_estimators=N_ESTIMATORS, random_state=seed, n_jobs=-1,
        )
        model_fixed.fit(X_fixed.iloc[train_idx], y_encoded[train_idx])

        proba_fixed = model_fixed.predict_proba(X_fixed.iloc[test_idx])[:, 1]
        pred_fixed = model_fixed.predict(X_fixed.iloc[test_idx])

        acc_fixed = accuracy_score(y_encoded[test_idx], pred_fixed)
        auc_fixed = roc_auc_score(y_encoded[test_idx], proba_fixed)

        model_combined = RandomForestClassifier(
            n_estimators=N_ESTIMATORS, random_state=seed, n_jobs=-1,
        )
        model_combined.fit(X_combined.iloc[train_idx], y_encoded[train_idx])

        proba_combined = model_combined.predict_proba(X_combined.iloc[test_idx])[:, 1]
        pred_combined = model_combined.predict(X_combined.iloc[test_idx])

        acc_combined = accuracy_score(y_encoded[test_idx], pred_combined)
        auc_combined = roc_auc_score(y_encoded[test_idx], proba_combined)

        rows.append({
            "fold": fold_index,
            "acc_fixed": acc_fixed,
            "acc_combined": acc_combined,
            "acc_diff": acc_combined - acc_fixed,
            "auc_fixed": auc_fixed,
            "auc_combined": auc_combined,
            "auc_diff": auc_combined - auc_fixed,
        })

        print(
            f"  fold {fold_index}/{n_folds}: "
            f"acc {acc_fixed:.4f} -> {acc_combined:.4f} ({acc_combined - acc_fixed:+.4f})  "
            f"auc {auc_fixed:.5f} -> {auc_combined:.5f} ({auc_combined - auc_fixed:+.5f})"
        )

    return pd.DataFrame(rows)


def summarize(results: pd.DataFrame, label: str) -> None:

    print()
    print(f"--- {label} ---")

    for metric in ("acc", "auc"):

        diffs = results[f"{metric}_diff"]

        t_stat, p_value = scipy_stats.ttest_1samp(diffs, 0.0)

        print(
            f"  {metric}_diff: mean={diffs.mean():+.5f}  std={diffs.std():.5f}  "
            f"paired t={t_stat:.2f}  p={p_value:.4f}"
            + ("  <- значимо (p<0.05)" if p_value < 0.05 else "  <- не значимо")
        )


# ============================================================
# MAIN
# ============================================================

def main():

    if not SAMPLE_PATH.exists():
        raise FileNotFoundError(f"Не найден {SAMPLE_PATH}")

    sample = pd.read_csv(SAMPLE_PATH)
    print(f"Sample size: {len(sample)}")

    all_results = {}

    for label, fixed_path, mgd_path in CONFIGS:

        if not fixed_path.exists():
            print(f"\nПропускаю '{label}' — не найден {fixed_path}")
            continue

        if not mgd_path.exists():
            print(f"\nПропускаю '{label}' — не найден {mgd_path}")
            continue

        print()
        print("=" * 80)
        print(label)
        print("=" * 80)

        X_fixed = pd.read_parquet(fixed_path)
        X_mgd = pd.read_parquet(mgd_path)

        X_combined = pd.concat(
            [X_fixed.reset_index(drop=True), X_mgd.reset_index(drop=True)], axis=1,
        )

        print(f"n_features: fixed={X_fixed.shape[1]}, combined={X_combined.shape[1]}")

        results = kfold_compare(X_fixed, X_combined, sample["label"])
        all_results[label] = results

        results.to_csv(
            OUTPUT_DIR / f"mgd_kfold__{label.split()[0].replace('=', '')}.csv",
            index=False,
        )

        summarize(results, label)

    print()
    print("=" * 80)
    print("ИТОГ")
    print("=" * 80)

    for label, results in all_results.items():
        acc_diff = results["acc_diff"]
        print(
            f"{label:30s} прирост accuracy от MGD: "
            f"{acc_diff.mean():+.4f} ± {acc_diff.std():.4f} по {len(results)} фолдам"
        )


if __name__ == "__main__":
    main()