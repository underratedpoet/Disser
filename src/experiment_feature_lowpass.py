from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import features as feat
from bands import linear_bands
from stereo import left_right_plus_diff


# ============================================================
# CONFIGURATION
# ============================================================

SAMPLE_PATH = Path("feature_experiment_results/sample.csv")
OUTPUT_DIR = Path("feature_experiment_results")

SEED = 42
N_ESTIMATORS = 100
TEST_SIZE = 0.25

SR = 44100
N_FFT = 2048
HOP_LENGTH = 512

FMIN = 20.0
# Тот же консервативный потолок, что и в experiment_stereo_lowpass_check.py.
FMAX_SAFE = 16000.0

N_BANDS = 120

# Числа из полноспектрального прогона (feature_summary.csv +
# ablation_check.py) — печатаются для сравнения, ничего не пересчитывают.
FULL_SPECTRUM_FIXED_ACCURACY = 0.9904
FULL_SPECTRUM_FIXED_AUC = 0.99959
FULL_SPECTRUM_COMBINED_ACCURACY = 0.9984
FULL_SPECTRUM_COMBINED_AUC = 0.99988


# ============================================================
# СТАТИСТИКИ И ИЗВЛЕЧЕНИЕ ПРИЗНАКОВ
# ============================================================

def compute_statistics(values: np.ndarray) -> dict[str, float]:

    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return {"mean": 0.0, "std": 0.0, "skew": 0.0, "kurtosis": 0.0}

    if len(values) == 1:
        return {
            "mean": float(values[0]),
            "std": 0.0,
            "skew": 0.0,
            "kurtosis": 0.0,
        }

    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "skew": float(skew(values)),
        "kurtosis": float(kurtosis(values)),
    }


def _ensure_stereo(y: np.ndarray) -> np.ndarray:
    if y.ndim == 1:
        return np.vstack([y, y])
    return y


def extract_spectral_features(spectral, prefix: str, fmax: float) -> dict[str, float]:

    banded = linear_bands(spectral, n_bands=N_BANDS, fmin=FMIN, fmax=fmax)

    features = {}

    for band_index, bins in enumerate(banded.bins):

        if len(bins) == 0:
            continue

        values = spectral.matrix[bins, :].ravel()
        stats = compute_statistics(values)
        band_name = banded.names[band_index]

        for statistic_name, value in stats.items():
            features[f"{prefix}__{band_name}__{statistic_name}"] = value

    return features


# ============================================================
# ФИКСИРОВАННАЯ ЧАСТЬ (IPD/MAG_DIFF) ПРИ fmax=16000
# ============================================================

def build_fixed_diff_features_16k(sample: pd.DataFrame) -> pd.DataFrame:

    rows = []

    for counter, row in enumerate(sample.itertuples(index=False), 1):

        print(
            f"\rfixed IPD/MAG_DIFF (fmax={FMAX_SAFE:.0f}): {counter}/{len(sample)}",
            end="",
            flush=True,
        )

        y, actual_sr = librosa.load(row.file, sr=SR, mono=False)
        planes = left_right_plus_diff(y, actual_sr, N_FFT, HOP_LENGTH)

        features = {}
        for plane_name in ("IPD_COS", "IPD_SIN", "MAG_DIFF"):
            features.update(
                extract_spectral_features(planes[plane_name], plane_name, FMAX_SAFE)
            )

        rows.append(features)

    print()

    X = pd.DataFrame(rows)
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return X


# ============================================================
# ВАРЬИРУЕМАЯ ЧАСТЬ (MGD ДЛЯ L/R) ПРИ fmax=16000
# ============================================================

def build_mgd_features_16k(sample: pd.DataFrame) -> pd.DataFrame:

    rows = []

    for counter, row in enumerate(sample.itertuples(index=False), 1):

        print(
            f"\rMGD (fmax={FMAX_SAFE:.0f}): {counter}/{len(sample)}",
            end="",
            flush=True,
        )

        y, actual_sr = librosa.load(row.file, sr=SR, mono=False)
        y = _ensure_stereo(y)

        features = {}

        for channel_name, channel_signal in (("L", y[0]), ("R", y[1])):

            spectral = feat.modified_group_delay(channel_signal, actual_sr, N_FFT, HOP_LENGTH)
            prefix = f"modified_group_delay__{channel_name}_MGD"

            features.update(extract_spectral_features(spectral, prefix, FMAX_SAFE))

        rows.append(features)

    print()

    X = pd.DataFrame(rows)
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return X


# ============================================================
# ОЦЕНКА (только accuracy/auc — permutation importance тут не нужна)
# ============================================================

def evaluate_auc_accuracy(X: pd.DataFrame, y: pd.Series) -> tuple[float, float]:

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=TEST_SIZE, random_state=SEED, stratify=y_encoded,
    )

    model = RandomForestClassifier(n_estimators=N_ESTIMATORS, random_state=SEED, n_jobs=-1)
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]
    pred = model.predict(X_test)

    return accuracy_score(y_test, pred), roc_auc_score(y_test, proba)


# ============================================================
# MAIN
# ============================================================

def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not SAMPLE_PATH.exists():
        raise FileNotFoundError(f"Не найден {SAMPLE_PATH}")

    sample = pd.read_csv(SAMPLE_PATH)
    print(f"Sample size: {len(sample)}")
    print(sample["label"].value_counts())

    fixed_path = OUTPUT_DIR / f"fixed_diff__fmax{FMAX_SAFE:.0f}__features.parquet"

    if fixed_path.exists():
        print(f"\nfixed IPD/MAG_DIFF (fmax={FMAX_SAFE:.0f}): уже на диске, загружаю")
        X_fixed = pd.read_parquet(fixed_path)
    else:
        X_fixed = build_fixed_diff_features_16k(sample)
        X_fixed.to_parquet(fixed_path, index=False)

    mgd_path = OUTPUT_DIR / f"modified_group_delay__fmax{FMAX_SAFE:.0f}__features.parquet"

    if mgd_path.exists():
        print(f"MGD (fmax={FMAX_SAFE:.0f}): уже на диске, загружаю")
        X_mgd = pd.read_parquet(mgd_path)
    else:
        X_mgd = build_mgd_features_16k(sample)
        X_mgd.to_parquet(mgd_path, index=False)

    print()
    print("=" * 80)
    print(f"ОЦЕНКА ПРИ fmax={FMAX_SAFE:.0f} Гц (без верхних {22050 - FMAX_SAFE:.0f} Гц)")
    print("=" * 80)

    acc_fixed, auc_fixed = evaluate_auc_accuracy(X_fixed, sample["label"])
    print(f"IPD/MAG_DIFF только:      accuracy={acc_fixed:.4f}  auc={auc_fixed:.5f}  (n_features={X_fixed.shape[1]})")

    X_combined = pd.concat(
        [X_fixed.reset_index(drop=True), X_mgd.reset_index(drop=True)], axis=1,
    )
    acc_combined, auc_combined = evaluate_auc_accuracy(X_combined, sample["label"])
    print(f"IPD/MAG_DIFF + MGD(L,R):  accuracy={acc_combined:.4f}  auc={auc_combined:.5f}  (n_features={X_combined.shape[1]})")

    print()
    print(f"Прирост accuracy от MGD при fmax={FMAX_SAFE:.0f}: {acc_combined - acc_fixed:+.4f}")
    print(f"Прирост auc от MGD при fmax={FMAX_SAFE:.0f}:      {auc_combined - auc_fixed:+.5f}")

    print()
    print("Для сравнения — на полном спектре (fmax~22050, из предыдущих прогонов):")
    print(f"  IPD/MAG_DIFF только:      accuracy={FULL_SPECTRUM_FIXED_ACCURACY:.4f}  auc={FULL_SPECTRUM_FIXED_AUC:.5f}")
    print(f"  IPD/MAG_DIFF + MGD(L,R):  accuracy={FULL_SPECTRUM_COMBINED_ACCURACY:.4f}  auc={FULL_SPECTRUM_COMBINED_AUC:.5f}")
    print(f"  Прирост accuracy от MGD: {FULL_SPECTRUM_COMBINED_ACCURACY - FULL_SPECTRUM_FIXED_ACCURACY:+.4f}")


if __name__ == "__main__":
    main()