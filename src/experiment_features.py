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

import features as feat
from bands import linear_bands
from stereo import left_right_plus_diff


# ============================================================
# CONFIGURATION
# ============================================================

REAL_DIR = Path(r"D:\Study\NIR\Project\real")
SYNTHETIC_DIR = Path(r"D:\Study\NIR\Project\neuro")

# Файлы, уже использованные в предыдущих экспериментах — исключаем,
# чтобы новая выборка с ними не пересекалась (та же дисциплина, что
# и в experiment_bands_v2.py).
OLD_SAMPLE_PATHS = [
    Path("band_experiment_results/sample.csv"),
    Path("band_experiment_results_v2/sample.csv"),
]

OUTPUT_DIR = Path("feature_experiment_results")

N_PER_CLASS = 5000  # итого 10000 файлов
SEED = 42

SR = 44100
N_FFT = 2048
HOP_LENGTH = 512

FMIN = 20.0
FMAX = None  # -> Nyquist

N_BANDS = 120  # для частотно-индексированных признаков (см. пояснение ниже)

N_ESTIMATORS = 100
TEST_SIZE = 0.25


# ============================================================
# РЕЕСТР ТИПОВ ПРИЗНАКОВ
# ============================================================
#
# kind определяет, как именно превращать 2D-матрицу (или 1D-ряд) в
# набор (имя_признака -> число):
#
#   "spectral" — частотно-индексированный план (freqs = реальные Гц).
#                Режется той же сеткой linear_bands(n_bands=120), что
#                и IPD_COS/IPD_SIN/MAG_DIFF — "полоса №i" означает
#                одну и ту же область спектра у всех spectral-типов.
#   "cepstral" — коэффициент-индексированный план (freqs = номер
#                коэффициента). "Полоса" = один коэффициент, никакого
#                деления на частотные диапазоны — LFCC/MFCC/CQCC
#                физически не про конкретные Гц (см. features.py).
#   "timeseries" — план без частотной оси вообще (RMS, crest_factor).
#                Один набор из 4 статистик на весь файл, без полос.


def _magnitude_planes(y, sr, n_fft, hop):
    return {"MAG": feat.magnitude(y, sr, n_fft, hop)}


def _real_imag_planes(y, sr, n_fft, hop):
    return {
        "REAL": feat.real_part(y, sr, n_fft, hop),
        "IMAG": feat.imag_part(y, sr, n_fft, hop),
    }


def _phase_cos_sin_planes(y, sr, n_fft, hop):
    cos_plane, sin_plane = feat.phase_cos_sin(y, sr, n_fft, hop)
    return {"PHASE_COS": cos_plane, "PHASE_SIN": sin_plane}


def _group_delay_planes(y, sr, n_fft, hop):
    return {"GD": feat.group_delay(y, sr, n_fft, hop)}


def _modified_group_delay_planes(y, sr, n_fft, hop):
    return {"MGD": feat.modified_group_delay(y, sr, n_fft, hop)}


def _lfcc_planes(y, sr, n_fft, hop):
    return {"LFCC": feat.lfcc(y, sr, n_fft, hop)}


def _mfcc_planes(y, sr, n_fft, hop):
    return {"MFCC": feat.mfcc(y, sr, n_fft, hop)}


def _cqcc_planes(y, sr, n_fft, hop):
    return {"CQCC": feat.cqcc(y, sr, n_fft, hop)}


def _rms_series(y, sr, n_fft, hop):
    return {"RMS": feat.rms(y, n_fft, hop)}


def _crest_factor_series(y, sr, n_fft, hop):
    return {"CF": feat.crest_factor(y, n_fft, hop)}


# (kind, callable) — callable(channel_signal, sr, n_fft, hop) ->
# dict[plane_name, SpectralData | np.ndarray]
FEATURE_TYPES: dict[str, tuple[str, Callable]] = {
    "magnitude": ("spectral", _magnitude_planes),
    "real_imag": ("spectral", _real_imag_planes),
    "phase_cos_sin": ("spectral", _phase_cos_sin_planes),
    "group_delay": ("spectral", _group_delay_planes),
    "modified_group_delay": ("spectral", _modified_group_delay_planes),
    "lfcc": ("cepstral", _lfcc_planes),
    "mfcc": ("cepstral", _mfcc_planes),
    "cqcc": ("cepstral", _cqcc_planes),
    "rms": ("timeseries", _rms_series),
    "crest_factor": ("timeseries", _crest_factor_series),
}


# ============================================================
# СТАТИСТИКИ
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


# ============================================================
# ИЗВЛЕЧЕНИЕ ПРИЗНАКОВ ИЗ ОДНОГО "ПЛАНА" — ПО ТИПУ
# ============================================================

def extract_spectral_features(spectral, prefix: str) -> dict[str, float]:

    banded = linear_bands(spectral, n_bands=N_BANDS, fmin=FMIN, fmax=FMAX)

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


def extract_cepstral_features(spectral, prefix: str) -> dict[str, float]:

    features = {}
    n_coeffs = spectral.matrix.shape[0]

    for coef_index in range(n_coeffs):

        values = spectral.matrix[coef_index, :]
        stats = compute_statistics(values)

        for statistic_name, value in stats.items():
            features[f"{prefix}__coef_{coef_index:03d}__{statistic_name}"] = value

    return features


def extract_timeseries_features(values: np.ndarray, prefix: str) -> dict[str, float]:

    stats = compute_statistics(values)

    return {f"{prefix}__{statistic_name}": value for statistic_name, value in stats.items()}


# ============================================================
# ФИКСИРОВАННАЯ ЧАСТЬ: IPD_COS / IPD_SIN / MAG_DIFF
# ============================================================
#
# Не зависит от того, какой feature_type сейчас тестируется — считаем
# её ОДИН РАЗ на всю выборку и переиспользуем для всех 10 конфигураций,
# а не пересчитываем заново на каждый feature_type (это самая дорогая
# часть — полный проход по 10000 файлов).

def build_fixed_diff_features(sample: pd.DataFrame) -> pd.DataFrame:

    rows = []

    for counter, row in enumerate(sample.itertuples(index=False), 1):

        print(f"\rfixed IPD/MAG_DIFF: {counter}/{len(sample)}", end="", flush=True)

        y, actual_sr = librosa.load(row.file, sr=SR, mono=False)

        planes = left_right_plus_diff(y, actual_sr, N_FFT, HOP_LENGTH)

        features = {}
        for plane_name in ("IPD_COS", "IPD_SIN", "MAG_DIFF"):
            features.update(
                extract_spectral_features(planes[plane_name], plane_name)
            )

        rows.append(features)

    print()

    X = pd.DataFrame(rows)
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return X


# ============================================================
# ВАРЬИРУЕМАЯ ЧАСТЬ: L/R ДЛЯ КОНКРЕТНОГО feature_type
# ============================================================

def build_varying_features(sample: pd.DataFrame, feature_type_name: str) -> pd.DataFrame:

    kind, plane_fn = FEATURE_TYPES[feature_type_name]

    rows = []

    for counter, row in enumerate(sample.itertuples(index=False), 1):

        print(
            f"\r{feature_type_name}: {counter}/{len(sample)}",
            end="",
            flush=True,
        )

        y, actual_sr = librosa.load(row.file, sr=SR, mono=False)
        y = _ensure_stereo(y)

        features = {}

        for channel_name, channel_signal in (("L", y[0]), ("R", y[1])):

            planes = plane_fn(channel_signal, actual_sr, N_FFT, HOP_LENGTH)

            for plane_name, plane_data in planes.items():

                prefix = f"{feature_type_name}__{channel_name}_{plane_name}"

                if kind == "spectral":
                    features.update(extract_spectral_features(plane_data, prefix))
                elif kind == "cepstral":
                    features.update(extract_cepstral_features(plane_data, prefix))
                elif kind == "timeseries":
                    features.update(extract_timeseries_features(plane_data, prefix))
                else:
                    raise ValueError(f"Неизвестный kind: {kind}")

        rows.append(features)

    print()

    X = pd.DataFrame(rows)
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return X


# ============================================================
# RESUME: не пересчитывать то, что уже сохранено на диск
# ============================================================

def get_or_build(
    cache_path: Path,
    build_fn: Callable[[], pd.DataFrame],
    label: str,
) -> pd.DataFrame:

    if cache_path.exists():
        print(f"{label}: уже на диске, загружаю {cache_path}")
        return pd.read_parquet(cache_path)

    print(f"{label}: на диске нет, считаю заново")
    X = build_fn()

    # Сохраняем СРАЗУ после вычисления — самая дорогая часть не
    # должна теряться, если что-то упадёт на следующем шаге.
    X.to_parquet(cache_path, index=False)

    return X


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
# ОЦЕНКА ОДНОЙ КОНФИГУРАЦИИ
# ============================================================

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

    # Чем больше признаков, тем дороже permutation importance —
    # у cepstral/timeseries конфигураций признаков мало (в разы
    # меньше, чем у spectral), но фиксированная IPD/MAG_DIFF часть
    # (3 плоскости x 120 полос x 4 статистики = 1440) присутствует
    # everywhere, так что общий порог всё равно нужен.
    if X.shape[1] > 1500:
        perm_n_repeats = 3
        perm_max_samples = 0.3
    elif X.shape[1] > 800:
        perm_n_repeats = 5
        perm_max_samples = 0.6
    else:
        perm_n_repeats = 10
        perm_max_samples = 1.0

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


def make_summary_row(feature_type: str, kind: str, evaluation: dict) -> dict:

    mdi = evaluation["mdi_importance"]
    perm = evaluation["perm_importance"]

    return {
        "feature_type": feature_type,
        "kind": kind,
        "n_features": evaluation["n_features"],
        "test_accuracy": evaluation["test_accuracy"],
        "test_auc": evaluation["test_auc"],
        "mdi_top_10": mdi.head(10).sum(),
        "mdi_top_25": mdi.head(25).sum(),
        "perm_top_10": perm.head(10).sum(),
        "perm_top_25": perm.head(25).sum(),
    }


# ============================================================
# ВЫБОРКА
# ============================================================

def collect_audio_files(directory: Path) -> list[Path]:
    extensions = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}
    return sorted(
        path for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in extensions
    )


def sample_dataset(
    real_dir: Path,
    synthetic_dir: Path,
    n_per_class: int,
    seed: int,
    exclude_files: set[str],
) -> pd.DataFrame:

    rng = np.random.default_rng(seed)

    real_files = [p for p in collect_audio_files(real_dir) if str(p) not in exclude_files]
    synthetic_files = [p for p in collect_audio_files(synthetic_dir) if str(p) not in exclude_files]

    if len(real_files) < n_per_class:
        raise ValueError(f"Недостаточно real файлов: {len(real_files)} < {n_per_class}")
    if len(synthetic_files) < n_per_class:
        raise ValueError(f"Недостаточно synthetic файлов: {len(synthetic_files)} < {n_per_class}")

    real_indices = rng.choice(len(real_files), size=n_per_class, replace=False)
    synthetic_indices = rng.choice(len(synthetic_files), size=n_per_class, replace=False)

    rows = (
        [{"file": str(real_files[i]), "label": "real"} for i in real_indices]
        + [{"file": str(synthetic_files[i]), "label": "synthetic"} for i in synthetic_indices]
    )

    result = pd.DataFrame(rows).sample(frac=1.0, random_state=seed).reset_index(drop=True)

    return result


def load_excluded_files(paths: list[Path]) -> set[str]:

    excluded = set()

    for path in paths:
        if path.exists():
            old_sample = pd.read_csv(path)
            excluded.update(old_sample["file"].astype(str))
        else:
            print(f"Внимание: {path} не найден — пропускаю при формировании исключений.")

    return excluded


# ============================================================
# MAIN
# ============================================================

def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    excluded_files = load_excluded_files(OLD_SAMPLE_PATHS)
    print(f"Исключаем {len(excluded_files)} файлов из предыдущих экспериментов.")

    sample_path = OUTPUT_DIR / "sample.csv"

    if sample_path.exists():
        print("Загружаем существующую выборку:")
        print(sample_path)
        sample = pd.read_csv(sample_path)
    else:
        print("Создаём новую выборку (без пересечения с предыдущими)...")
        sample = sample_dataset(
            real_dir=REAL_DIR,
            synthetic_dir=SYNTHETIC_DIR,
            n_per_class=N_PER_CLASS,
            seed=SEED,
            exclude_files=excluded_files,
        )
        sample.to_csv(sample_path, index=False)

    print(f"Sample size: {len(sample)}")
    print(sample["label"].value_counts())

    overlap = set(sample["file"].astype(str)) & excluded_files
    if overlap:
        raise RuntimeError(
            f"Новая выборка пересекается со старыми в {len(overlap)} файлах."
        )

    # --------------------------------------------------------
    # Фиксированная часть (IPD_COS/IPD_SIN/MAG_DIFF) — один раз
    # --------------------------------------------------------

    X_fixed = get_or_build(
        OUTPUT_DIR / "fixed_diff__features.parquet",
        lambda: build_fixed_diff_features(sample),
        "fixed IPD/MAG_DIFF",
    )

    print(f"Fixed features: {X_fixed.shape[1]}")

    # --------------------------------------------------------
    # По каждому типу признака — варьируемая часть + оценка
    # --------------------------------------------------------

    summary = []

    for feature_type_name, (kind, _) in FEATURE_TYPES.items():

        print()
        print("=" * 80)
        print(feature_type_name)
        print("=" * 80)

        X_varying = get_or_build(
            OUTPUT_DIR / f"{feature_type_name}__varying_features.parquet",
            lambda name=feature_type_name: build_varying_features(sample, name),
            feature_type_name,
        )

        X = pd.concat(
            [X_fixed.reset_index(drop=True), X_varying.reset_index(drop=True)],
            axis=1,
        )

        print(f"Features: {X.shape[1]} (fixed={X_fixed.shape[1]}, varying={X_varying.shape[1]})")

        evaluation = evaluate_configuration(X, sample["label"])

        evaluation["mdi_importance"].rename("importance").reset_index().rename(
            columns={"index": "feature"}
        ).to_csv(
            OUTPUT_DIR / f"{feature_type_name}__mdi_importance.csv", index=False,
        )

        evaluation["perm_importance"].rename("importance").reset_index().rename(
            columns={"index": "feature"}
        ).to_csv(
            OUTPUT_DIR / f"{feature_type_name}__perm_importance.csv", index=False,
        )

        row = make_summary_row(feature_type_name, kind, evaluation)
        summary.append(row)

    # --------------------------------------------------------
    # Итог
    # --------------------------------------------------------

    summary_df = pd.DataFrame(summary)

    summary_df = summary_df.sort_values(
        ["test_auc", "test_accuracy"], ascending=False,
    ).reset_index(drop=True)

    summary_df.to_csv(OUTPUT_DIR / "feature_summary.csv", index=False)

    print()
    print("=" * 80)
    print("СРАВНЕНИЕ ТИПОВ ПРИЗНАКОВ (по test_auc)")
    print("=" * 80)
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()