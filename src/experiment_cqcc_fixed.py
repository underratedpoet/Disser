from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

import features as feat


# ============================================================
# CONFIGURATION
# ============================================================

OUTPUT_DIR = Path("feature_experiment_results")
SAMPLE_PATH = OUTPUT_DIR / "sample.csv"

# Старая (сломанная, internal_nfft=512, охват ~11930-22038 Гц) версия
# уже посчитана как cqcc__varying_features.parquet — не трогаем её,
# пишем исправленную версию под отдельным именем.
FIXED_CQCC_PATH = OUTPUT_DIR / "cqcc_fixed__varying_features.parquet"
FIXED_DIFF_PATH = OUTPUT_DIR / "fixed_diff__features.parquet"  # уже существует, полный спектр

SR = 44100
N_FFT = 2048
HOP_LENGTH = 512

N_CQCC = 20
INTERNAL_NFFT = 8192  # см. features.py::cqcc — новый дефолт, но указываем явно

SEED = 42
N_ESTIMATORS = 100
N_FOLDS = 5


# ============================================================
# СБОРКА ИСПРАВЛЕННЫХ ПРИЗНАКОВ CQCC ДЛЯ L/R
# ============================================================

def _ensure_stereo(y: np.ndarray) -> np.ndarray:
    if y.ndim == 1:
        return np.vstack([y, y])
    return y


def extract_cepstral_features(spectral, prefix: str) -> dict[str, float]:

    from scipy.stats import skew, kurtosis

    features = {}
    n_coeffs = spectral.matrix.shape[0]

    for coef_index in range(n_coeffs):

        values = np.asarray(spectral.matrix[coef_index, :], dtype=np.float64)
        values = values[np.isfinite(values)]

        if len(values) == 0:
            stats = {"mean": 0.0, "std": 0.0, "skew": 0.0, "kurtosis": 0.0}
        elif len(values) == 1:
            stats = {"mean": float(values[0]), "std": 0.0, "skew": 0.0, "kurtosis": 0.0}
        else:
            stats = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "skew": float(skew(values)),
                "kurtosis": float(kurtosis(values)),
            }

        for statistic_name, value in stats.items():
            features[f"{prefix}__coef_{coef_index:03d}__{statistic_name}"] = value

    return features


def build_fixed_cqcc_features(sample: pd.DataFrame) -> pd.DataFrame:

    rows = []

    for counter, row in enumerate(sample.itertuples(index=False), 1):

        print(
            f"\rcqcc (internal_nfft={INTERNAL_NFFT}): {counter}/{len(sample)}",
            end="",
            flush=True,
        )

        y, actual_sr = librosa.load(row.file, sr=SR, mono=False)
        y = _ensure_stereo(y)

        features = {}

        for channel_name, channel_signal in (("L", y[0]), ("R", y[1])):

            spectral = feat.cqcc(
                channel_signal,
                actual_sr,
                N_FFT,
                HOP_LENGTH,
                n_cqcc=N_CQCC,
                internal_nfft=INTERNAL_NFFT,
            )

            prefix = f"cqcc_fixed__{channel_name}_CQCC"
            features.update(extract_cepstral_features(spectral, prefix))

        rows.append(features)

    print()

    X = pd.DataFrame(rows)
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return X


# ============================================================
# K-FOLD ПАРНОЕ СРАВНЕНИЕ
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


def summarize(results: pd.DataFrame) -> None:

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

    if not FIXED_DIFF_PATH.exists():
        raise FileNotFoundError(
            f"Не найден {FIXED_DIFF_PATH} — нужен результат уже "
            f"отработавшего experiment_features.py."
        )

    sample = pd.read_csv(SAMPLE_PATH)
    print(f"Sample size: {len(sample)}")
    print(sample["label"].value_counts())

    # --------------------------------------------------------
    # Пересчёт CQCC с исправленным internal_nfft (резюмируемо)
    # --------------------------------------------------------

    if FIXED_CQCC_PATH.exists():
        print(f"\ncqcc (исправленный): уже на диске, загружаю {FIXED_CQCC_PATH}")
        X_cqcc = pd.read_parquet(FIXED_CQCC_PATH)
    else:
        print(f"\ncqcc (исправленный, internal_nfft={INTERNAL_NFFT}): считаю заново")
        X_cqcc = build_fixed_cqcc_features(sample)
        X_cqcc.to_parquet(FIXED_CQCC_PATH, index=False)

    X_fixed = pd.read_parquet(FIXED_DIFF_PATH)

    X_combined = pd.concat(
        [X_fixed.reset_index(drop=True), X_cqcc.reset_index(drop=True)], axis=1,
    )

    print(f"\nn_features: fixed={X_fixed.shape[1]}, cqcc={X_cqcc.shape[1]}, combined={X_combined.shape[1]}")

    # --------------------------------------------------------
    # Парный k-fold тест
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print(f"CQCC (internal_nfft={INTERNAL_NFFT}, охват ~746-22038 Гц) vs только IPD/MAG_DIFF")
    print("=" * 80)

    results = kfold_compare(X_fixed, X_combined, sample["label"])
    results.to_csv(OUTPUT_DIR / "cqcc_fixed_kfold.csv", index=False)

    print()
    summarize(results)

    print()
    print("Для сравнения (из прошлого прогона, старый CQCC с internal_nfft=512,")
    print("охват только ~11930-22038 Гц):")
    print("  cqcc (старый), одна оценка: accuracy=0.9956, auc=0.999844")
    print("  (полная k-fold проверка для старой версии не делалась)")


if __name__ == "__main__":
    main()