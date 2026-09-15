from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

import librosa
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis, mannwhitneyu
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from bands import linear_bands
from stereo import left_right_plus_diff


# ============================================================
# CONFIGURATION
# ============================================================

SAMPLE_PATH = Path("band_experiment_results_v2/sample.csv")
OUTPUT_DIR = Path("stereo_experiment_results")

SEED = 42
N_ESTIMATORS = 100
TEST_SIZE = 0.25

SR = 44100
N_FFT = 2048
HOP_LENGTH = 512

FMIN = 20.0

# Консервативный потолок — ниже него контент почти наверняка есть
# даже при низком битрейте MP3 (128 kbps даёт lowpass ~16 кГц, так
# что 16000 — граница "на грани", а не "с большим запасом"; если
# хотите перестраховаться сильнее, попробуйте 14000-15000).
FMAX_SAFE = 16000.0

N_BANDS = 120


# ============================================================
# СТАТИСТИКИ И ПРИЗНАКИ (как в experiment_stereo_upd.py, но fmax
# передаётся явно и осознанно занижен)
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


def extract_plane_features(
    spectral,
    plane_name: str,
    fmax: float,
) -> dict[str, float]:

    banded = linear_bands(
        spectral,
        n_bands=N_BANDS,
        fmin=FMIN,
        fmax=fmax,
    )

    features = {}

    for band_index, bins in enumerate(banded.bins):

        if len(bins) == 0:
            continue

        values = spectral.matrix[bins, :].ravel()
        stats = compute_statistics(values)

        band_name = banded.names[band_index]

        for statistic_name, value in stats.items():
            features[
                f"left_right_plus_diff__{plane_name}__{band_name}__{statistic_name}"
            ] = value

    return features


def build_features(sample: pd.DataFrame, fmax: float) -> tuple[pd.DataFrame, pd.Series]:

    rows = []
    labels = []

    for counter, row in enumerate(sample.itertuples(index=False), 1):

        print(f"\rfmax={fmax:.0f}Hz: {counter}/{len(sample)}", end="", flush=True)

        y, actual_sr = librosa.load(row.file, sr=SR, mono=False)

        planes = left_right_plus_diff(y, actual_sr, N_FFT, HOP_LENGTH)

        features = {}
        for plane_name, spectral in planes.items():
            features.update(extract_plane_features(spectral, plane_name, fmax))

        rows.append(features)
        labels.append(row.label)

    print()

    X = pd.DataFrame(rows)
    y_series = pd.Series(labels, name="label")
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return X, y_series


# ============================================================
# PERMUTATION IMPORTANCE С ПРОГРЕССОМ (как в experiment_stereo_upd.py)
# ============================================================

def permutation_importance_with_progress(
    model: RandomForestClassifier,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    n_repeats: int = 10,
    random_state: int = SEED,
    max_samples: float = 1.0,
    progress_every: int = 200,
) -> pd.Series:

    rng = np.random.default_rng(random_state)

    if max_samples < 1.0:
        n_sub = max(50, int(len(X_test) * max_samples))
        sub_idx = rng.choice(len(X_test), size=n_sub, replace=False)
        X_eval = X_test.iloc[sub_idx].reset_index(drop=True)
        y_eval = y_test[sub_idx]
    else:
        X_eval = X_test.reset_index(drop=True)
        y_eval = y_test

    columns = X_eval.columns
    values = X_eval.to_numpy().copy()
    n_features = values.shape[1]

    original_n_jobs = model.n_jobs
    model.n_jobs = 1

    try:
        baseline_score = accuracy_score(y_eval, model.predict(X_eval))
        importances = np.zeros(n_features)
        start = time.time()

        for feature_index in range(n_features):

            original_column = values[:, feature_index].copy()
            drops = np.empty(n_repeats)

            for repeat in range(n_repeats):
                rng.shuffle(values[:, feature_index])
                permuted_df = pd.DataFrame(values, columns=columns)
                permuted_score = accuracy_score(y_eval, model.predict(permuted_df))
                drops[repeat] = baseline_score - permuted_score

            values[:, feature_index] = original_column
            importances[feature_index] = drops.mean()

            done = feature_index + 1
            if done % progress_every == 0 or done == n_features:
                elapsed = time.time() - start
                eta = elapsed / done * (n_features - done)
                print(
                    f"\r  permutation importance: {done}/{n_features} "
                    f"признаков, {elapsed:.0f}с прошло, ~{eta:.0f}с осталось",
                    end="",
                    flush=True,
                )

        print()

    finally:
        model.n_jobs = original_n_jobs

    return pd.Series(importances, index=columns)


def evaluate_configuration(X: pd.DataFrame, y: pd.Series) -> dict:

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=TEST_SIZE, random_state=SEED, stratify=y_encoded,
    )

    model = RandomForestClassifier(n_estimators=N_ESTIMATORS, random_state=SEED, n_jobs=-1)
    model.fit(X_train, y_train)

    test_pred = model.predict(X_test)
    test_proba = model.predict_proba(X_test)[:, 1]

    test_accuracy = accuracy_score(y_test, test_pred)
    test_auc = roc_auc_score(y_test, test_proba)

    mdi_importance = pd.Series(
        model.feature_importances_, index=X.columns,
    ).sort_values(ascending=False)

    perm_n_repeats = 3 if X.shape[1] > 1000 else 10
    perm_max_samples = 0.4 if X.shape[1] > 1000 else 1.0

    perm_raw = permutation_importance_with_progress(
        model, X_test, y_test,
        n_repeats=perm_n_repeats, random_state=SEED, max_samples=perm_max_samples,
    )

    perm_clipped = perm_raw.clip(lower=0.0)
    perm_sum = perm_clipped.sum()
    perm_importance = (
        (perm_clipped / perm_sum).sort_values(ascending=False)
        if perm_sum > 0 else perm_clipped.sort_values(ascending=False)
    )

    return {
        "n_features": X.shape[1],
        "test_accuracy": test_accuracy,
        "test_auc": test_auc,
        "mdi_importance": mdi_importance,
        "perm_importance": perm_importance,
    }


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
    print(f"\nFMAX_SAFE = {FMAX_SAFE:.0f} Гц (вместо полного Найквиста ~22050 Гц)")

    features_path = OUTPUT_DIR / f"left_right_plus_diff__fmax{FMAX_SAFE:.0f}__features.parquet"

    if features_path.exists():
        print(f"\nПризнаки уже на диске: {features_path}")
        X = pd.read_parquet(features_path)
        y = sample["label"]
    else:
        X, y = build_features(sample, FMAX_SAFE)
        X.to_parquet(features_path, index=False)

    print(f"\nFeatures: {X.shape[1]}")

    evaluation = evaluate_configuration(X, y)

    evaluation["mdi_importance"].rename("importance").reset_index().rename(
        columns={"index": "feature"}
    ).to_csv(OUTPUT_DIR / f"left_right_plus_diff__fmax{FMAX_SAFE:.0f}__mdi_importance.csv", index=False)

    evaluation["perm_importance"].rename("importance").reset_index().rename(
        columns={"index": "feature"}
    ).to_csv(OUTPUT_DIR / f"left_right_plus_diff__fmax{FMAX_SAFE:.0f}__perm_importance.csv", index=False)

    print()
    print("=" * 80)
    print(f"РЕЗУЛЬТАТ ПРИ fmax={FMAX_SAFE:.0f} Гц (без верхних {22050-FMAX_SAFE:.0f} Гц)")
    print("=" * 80)
    print(f"test_accuracy = {evaluation['test_accuracy']:.4f}")
    print(f"test_auc      = {evaluation['test_auc']:.5f}")
    print()
    print("Полный спектр (для справки, из прошлого прогона):")
    print("test_accuracy = 0.9900")
    print("test_auc      = 0.99961")
    print()
    print("Топ-10 полос по MDI (без верхних частот — интересно, откуда")
    print("теперь берётся сигнал):")
    print(evaluation["mdi_importance"].head(10).to_string())


if __name__ == "__main__":
    main()