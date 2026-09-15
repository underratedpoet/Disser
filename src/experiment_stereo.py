from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

import librosa
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from bands import linear_bands
from stereo import STEREO_METHODS

# Единственная зависимость от experiment_bands.py — две константы
# частотного диапазона, чтобы полосы строились так же, как в первом
# эксперименте. Никакие функции оттуда не импортируются — этот файл
# самодостаточен, чтобы не расходиться с experiment_bands.py по
# версии при правках.
from experiment_bands import FMIN, FMAX


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

# Фиксировано — сравниваем только способ объединения каналов, не
# частотное разбиение.
N_BANDS = 120


# ============================================================
# СТАТИСТИКИ ПО ЗНАЧЕНИЯМ ВНУТРИ ОДНОЙ ПОЛОСЫ
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


# ============================================================
# СТАТИСТИКИ ДЛЯ ОДНОЙ "ПЛОСКОСТИ" (L, R, M, S, IPD_COS, ...)
# ============================================================

def extract_plane_features(
    spectral,
    plane_name: str,
    stereo_method: str,
) -> dict[str, float]:

    banded = linear_bands(
        spectral,
        n_bands=N_BANDS,
        fmin=FMIN,
        fmax=FMAX,
    )

    features = {}

    for band_index, bins in enumerate(banded.bins):

        if len(bins) == 0:
            # При linear_bands и n_bands=120 пустых полос не бывает
            # (см. первый эксперимент) — защита на случай ручного
            # изменения N_BANDS/FMIN.
            continue

        values = spectral.matrix[bins, :].ravel()
        stats = compute_statistics(values)

        band_name = banded.names[band_index]

        for statistic_name, value in stats.items():
            features[
                f"{stereo_method}__{plane_name}__{band_name}__{statistic_name}"
            ] = value

    return features


# ============================================================
# СБОРКА ПРИЗНАКОВ ДЛЯ ОДНОГО СПОСОБА ОБЪЕДИНЕНИЯ КАНАЛОВ
# ============================================================

def build_stereo_features(
    sample: pd.DataFrame,
    stereo_method: str,
    stereo_function: Callable,
) -> tuple[pd.DataFrame, pd.Series]:

    rows = []
    labels = []

    for counter, row in enumerate(sample.itertuples(index=False), 1):

        print(
            f"\r{stereo_method}: {counter}/{len(sample)}",
            end="",
            flush=True,
        )

        y, actual_sr = librosa.load(
            row.file,
            sr=SR,
            mono=False,
        )

        planes = stereo_function(y, actual_sr, N_FFT, HOP_LENGTH)

        features = {}

        for plane_name, spectral in planes.items():
            features.update(
                extract_plane_features(spectral, plane_name, stereo_method)
            )

        rows.append(features)
        labels.append(row.label)

    print()

    X = pd.DataFrame(rows)
    y_series = pd.Series(labels, name="label")

    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return X, y_series


# ============================================================
# PERMUTATION IMPORTANCE С ПРОГРЕССОМ
# ============================================================
#
# Собственная реализация вместо sklearn.inspection.permutation_importance:
#   1) у sklearn-версии нет прогресса — при 1000+ признаках вызов
#      может идти часами и снаружи неотличим от зависания;
#   2) max_samples позволяет считать на подвыборке test-части —
#      линейно снижает стоимость там, где признаков много;
#   3) на время цикла явно выставляем model.n_jobs=1: тысячи
#      повторных вызовов predict() с n_jobs=-1 означают тысячи
#      пересозданий пула потоков подряд — на Windows (особенно под
#      отладчиком) это может не просто медленно работать, а
#      зависать по-настоящему (deadlock в joblib/loky).


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
    values = X_eval.to_numpy().copy()  # copy: to_numpy() может вернуть read-only view
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
                permuted_score = accuracy_score(
                    y_eval, model.predict(permuted_df)
                )

                drops[repeat] = baseline_score - permuted_score

            values[:, feature_index] = original_column
            importances[feature_index] = drops.mean()

            done = feature_index + 1

            if done % progress_every == 0 or done == n_features:
                elapsed = time.time() - start
                eta = elapsed / done * (n_features - done)
                print(
                    f"\r  permutation importance: {done}/{n_features} "
                    f"признаков, {elapsed:.0f}с прошло, "
                    f"~{eta:.0f}с осталось",
                    end="",
                    flush=True,
                )

        print()

    finally:
        model.n_jobs = original_n_jobs

    return pd.Series(importances, index=columns)


# ============================================================
# ОЦЕНКА ОДНОЙ КОНФИГУРАЦИИ: held-out accuracy/AUC + MDI + perm
# ============================================================

def evaluate_configuration(
    X: pd.DataFrame,
    y: pd.Series,
    n_estimators: int = N_ESTIMATORS,
    seed: int = SEED,
    test_size: float = TEST_SIZE,
) -> dict:

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=test_size,
        random_state=seed,
        stratify=y_encoded,
    )

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=seed,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    # --- held-out качество классификации ---

    test_pred = model.predict(X_test)
    test_proba = model.predict_proba(X_test)[:, 1]

    test_accuracy = accuracy_score(y_test, test_pred)
    test_auc = roc_auc_score(y_test, test_proba)

    # --- MDI importance ---

    mdi_importance = pd.Series(
        model.feature_importances_,
        index=X.columns,
    ).sort_values(ascending=False)

    # --- permutation importance на held-out части ---

    # Чем больше признаков, тем дороже линейно растёт полный перебор
    # (n_features x n_repeats x стоимость predict) — при больших
    # конфигурациях (>1000 признаков, как у left_right_plus_diff с
    # его 2400) снижаем repeats и считаем на подвыборке test-части.
    if X.shape[1] > 1000:
        perm_n_repeats = 3
        perm_max_samples = 0.4
    else:
        perm_n_repeats = 10
        perm_max_samples = 1.0

    perm_raw = permutation_importance_with_progress(
        model,
        X_test,
        y_test,
        n_repeats=perm_n_repeats,
        random_state=seed,
        max_samples=perm_max_samples,
    )

    # Отрицательные значения (признак хуже, чем шум) обнуляем и
    # нормируем к сумме 1, чтобы top_k читался в тех же единицах,
    # что и у MDI.
    perm_clipped = perm_raw.clip(lower=0.0)
    perm_sum = perm_clipped.sum()

    if perm_sum > 0:
        perm_importance = (perm_clipped / perm_sum).sort_values(
            ascending=False
        )
    else:
        perm_importance = perm_clipped.sort_values(ascending=False)

    return {
        "n_features": X.shape[1],
        "test_accuracy": test_accuracy,
        "test_auc": test_auc,
        "mdi_importance": mdi_importance,
        "perm_importance": perm_importance,
    }


def make_summary_row(
    method: str,
    parameter_value,
    evaluation: dict,
) -> dict:

    mdi = evaluation["mdi_importance"]
    perm = evaluation["perm_importance"]

    return {
        "method": method,
        "parameter": "n_bands",
        "parameter_value": parameter_value,

        "n_features": evaluation["n_features"],

        "test_accuracy": evaluation["test_accuracy"],
        "test_auc": evaluation["test_auc"],

        "mdi_top_10": mdi.head(10).sum(),
        "mdi_top_25": mdi.head(25).sum(),

        "perm_top_10": perm.head(10).sum(),
        "perm_top_25": perm.head(25).sum(),
    }


# ============================================================
# RESUME: переиспользовать уже посчитанные признаки, если есть
# ============================================================

def get_features(
    sample: pd.DataFrame,
    stereo_method: str,
    stereo_function: Callable,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.Series]:

    features_path = output_dir / f"{stereo_method}__features.parquet"

    if features_path.exists():
        print(f"{stereo_method}: признаки уже на диске, загружаю {features_path}")

        X = pd.read_parquet(features_path)
        # Порядок строк в features.parquet соответствует порядку
        # sample.itertuples() в build_stereo_features — тот же
        # порядок, что и в sample.csv.
        y = sample["label"]

        return X, y

    print(f"{stereo_method}: признаков на диске нет, считаю из аудио заново")

    X, y = build_stereo_features(sample, stereo_method, stereo_function)

    # Сохраняем СРАЗУ после извлечения, до evaluate_configuration —
    # это самая дорогая по времени часть (аудио с диска на 2000
    # файлов), и если evaluate_configuration зависнет или упадёт,
    # эта работа не должна теряться.
    X.to_parquet(features_path, index=False)

    return X, y


# ============================================================
# MAIN
# ============================================================

def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not SAMPLE_PATH.exists():
        raise FileNotFoundError(
            f"Не найден {SAMPLE_PATH}. Укажите путь к уже готовой "
            f"sample.csv — список файлов не зависит от того, моно или "
            f"стерео их потом загружать."
        )

    sample = pd.read_csv(SAMPLE_PATH)

    print(f"Sample size: {len(sample)}")
    print(sample["label"].value_counts())

    summary = []

    for stereo_method, stereo_function in STEREO_METHODS.items():

        print()
        print("=" * 80)
        print(stereo_method)
        print("=" * 80)

        X, y = get_features(sample, stereo_method, stereo_function, OUTPUT_DIR)

        print(f"Features: {X.shape[1]}")

        evaluation = evaluate_configuration(X, y)

        evaluation["mdi_importance"].rename("importance").reset_index().rename(
            columns={"index": "feature"}
        ).to_csv(
            OUTPUT_DIR / f"{stereo_method}__mdi_importance.csv",
            index=False,
        )

        evaluation["perm_importance"].rename("importance").reset_index().rename(
            columns={"index": "feature"}
        ).to_csv(
            OUTPUT_DIR / f"{stereo_method}__perm_importance.csv",
            index=False,
        )

        row = make_summary_row(
            method=stereo_method,
            parameter_value=N_BANDS,
            evaluation=evaluation,
        )

        summary.append(row)

    summary_df = pd.DataFrame(summary)

    summary_df = summary_df.sort_values(
        ["test_auc", "test_accuracy"],
        ascending=False,
    ).reset_index(drop=True)

    summary_df.to_csv(OUTPUT_DIR / "stereo_summary.csv", index=False)

    print()
    print("=" * 80)
    print("СРАВНЕНИЕ СПОСОБОВ ОБЪЕДИНЕНИЯ КАНАЛОВ (по test_auc)")
    print("=" * 80)
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()