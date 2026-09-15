from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Callable

import librosa
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from bands import (
    linear_bands,
    log_bands,
    mel_bands,
    erb_bands,
    bark_bands,
    octave_bands,
    fixed_hz_bands,
)


# ============================================================
# CONFIGURATION
# ============================================================

REAL_DIR = Path(r"D:\Study\NIR\Project\real")
SYNTHETIC_DIR = Path(r"D:\Study\NIR\Project\neuro")

OUTPUT_DIR = Path("band_experiment_results")

N_PER_CLASS = 500
SEED = 42

SR = 44100
N_FFT = 2048
HOP_LENGTH = 512

FMIN = 20.0
FMAX = None  # -> Nyquist

# Доля выборки, откладываемая под честную (held-out) оценку.
# Именно на ней считаются accuracy/ROC-AUC и permutation importance,
# чтобы не смешивать "информативность на трейне" с "переобучением на трейне".
TEST_SIZE = 0.25

N_ESTIMATORS = 100

# Верхняя граница поиска максимального n_bands "без пустых полос"
# (нужна только для перебора при определении порога, самого разбиения
# на такое число полос не будет).
MAX_SEARCH_N_BANDS = 500

# Глобальный потолок n_bands. У каждого метода собственный потолок —
# min(GLOBAL_N_BANDS_CEILING, его максимум без пустых полос).
GLOBAL_N_BANDS_CEILING = 120

# Внутри своего потолка каждый метод тестируется в трёх точках —
# ceiling//3, ceiling//2, ceiling (при ceiling=120 это даёт 40, 60, 120).
RESOLUTION_DIVISORS = (3, 2, 1)

# Для fixed-Hz потолок не завязан на n_bands, оставляем как контрольную группу
FIXED_HZ_WIDTHS = [100.0, 250.0, 500.0, 1000.0]


# ============================================================
# BAND METHODS
# ============================================================

BAND_METHODS: dict[str, Callable] = {
    "linear": linear_bands,
    "log": log_bands,
    "mel": mel_bands,
    "erb": erb_bands,
    "bark": bark_bands,
}


# ============================================================
# ПОИСК ГРАНИЦЫ "НЕТ ПУСТЫХ ПОЛОС" ДЛЯ КАЖДОГО МЕТОДА
# ============================================================
#
# Идея: при fmin=20 Гц полосы некоторых шкал (в первую очередь log,
# в меньшей степени octave/erb) на нижнем крае диапазона становятся
# уже, чем частотный шаг FFT (Δf = sr / n_fft ≈ 21.53 Гц при текущих
# параметрах), и остаются без единого FFT-бина. Такие полосы молча
# выбрасываются из признакового пространства.
#
# В отличие от первой версии этого скрипта, здесь НЕ используется один
# общий n_bands для всех методов (это резко ограничивало бы диапазон —
# у log потолок всего 17). Вместо этого у каждого метода свой потолок
# min(GLOBAL_N_BANDS_CEILING, его собственный максимум без пустых
# полос), и внутри этого потолка берутся три точки (см.
# RESOLUTION_DIVISORS). Это значит, что "одинаковая относительная
# позиция" (например, "на всю катушку", 1/1) у разных методов
# соответствует РАЗНОМУ абсолютному числу признаков — см. пояснение
# у test_auc в evaluate_configuration() и итоговую сортировку в main().


def _dummy_spectral(sr: float = SR, n_fft: int = N_FFT) -> SimpleNamespace:
    freqs = np.linspace(0, sr / 2, n_fft // 2 + 1)
    return SimpleNamespace(freqs=freqs, sr=sr)


def max_n_bands_without_empty(
    band_function: Callable,
    fmin: float,
    fmax: float | None,
    search_max: int = MAX_SEARCH_N_BANDS,
) -> int:
    """Наибольший n_bands, при котором ни одна полоса не пуста.

    Полагается на то, что при увеличении n_bands (сужении полос)
    появление пустых полос необратимо в исследуемом диапазоне —
    это выполняется для всех текущих шкал при fmin=20 Гц (проверено
    отдельно), поэтому останавливаемся на первом же n, где появилась
    хотя бы одна пустая полоса.
    """

    spectral = _dummy_spectral()

    best = 0

    for n in range(1, search_max + 1):
        banded = band_function(
            spectral,
            n_bands=n,
            fmin=fmin,
            fmax=fmax,
        )

        has_empty = any(len(b) == 0 for b in banded.bins)

        if has_empty:
            break

        best = n

    return best


def max_octave_fraction_without_empty(
    fmin: float,
    fmax: float | None,
    search_max: int = 50,
) -> int:

    spectral = _dummy_spectral()

    best = 0

    for fraction in range(1, search_max + 1):
        banded = octave_bands(
            spectral,
            fmin=fmin,
            fmax=fmax,
            fraction=fraction,
        )

        has_empty = any(len(b) == 0 for b in banded.bins)

        if has_empty:
            break

        best = fraction

    return best


def levels_for_ceiling(
    own_ceiling: int,
    divisors: tuple[int, ...] = RESOLUTION_DIVISORS,
) -> list[tuple[str, int]]:
    """Три точки внутри own_ceiling: ceiling//3, ceiling//2, ceiling//1.

    Возвращает пары (метка_уровня, n). Если два делителя дают одно и
    то же n (маленький ceiling, как у octave), дубликат не повторяется —
    остаётся первое (более "грубое") вхождение.
    """

    pairs: list[tuple[str, int]] = []
    seen: set[int] = set()

    for divisor in divisors:
        n = max(1, own_ceiling // divisor)

        if n in seen:
            continue

        # "1-3" вместо "1/3" — метка идёт прямо в имя выходного файла
        # (см. run_one_experiment), а "/" на Windows читается как
        # разделитель пути и ломает сохранение.
        pairs.append((f"1-{divisor}", n))
        seen.add(n)

    return pairs


def determine_per_method_n_bands() -> tuple[
    dict[str, list[tuple[str, int]]],
    dict[str, int],
]:

    per_method_max = {
        name: max_n_bands_without_empty(fn, FMIN, FMAX)
        for name, fn in BAND_METHODS.items()
    }

    per_method_levels = {
        name: levels_for_ceiling(min(own_max, GLOBAL_N_BANDS_CEILING))
        for name, own_max in per_method_max.items()
    }

    return per_method_levels, per_method_max


# ============================================================
# DATASET SAMPLING
# ============================================================

def collect_audio_files(directory: Path) -> list[Path]:
    extensions = {
        ".wav",
        ".mp3",
        ".flac",
        ".ogg",
        ".m4a",
        ".aac",
    }

    return sorted(
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in extensions
    )


def sample_dataset(
    real_dir: Path,
    synthetic_dir: Path,
    n_per_class: int,
    seed: int,
    exclude_files: set[str] | None = None,
) -> pd.DataFrame:

    rng = np.random.default_rng(seed)

    real_files = collect_audio_files(real_dir)
    synthetic_files = collect_audio_files(synthetic_dir)

    if exclude_files:
        real_files = [
            path for path in real_files if str(path) not in exclude_files
        ]
        synthetic_files = [
            path for path in synthetic_files if str(path) not in exclude_files
        ]

    if len(real_files) < n_per_class:
        raise ValueError(
            f"Недостаточно real файлов: "
            f"{len(real_files)} < {n_per_class}"
        )

    if len(synthetic_files) < n_per_class:
        raise ValueError(
            f"Недостаточно synthetic файлов: "
            f"{len(synthetic_files)} < {n_per_class}"
        )

    real_indices = rng.choice(
        len(real_files),
        size=n_per_class,
        replace=False,
    )

    synthetic_indices = rng.choice(
        len(synthetic_files),
        size=n_per_class,
        replace=False,
    )

    rows = []

    for index in real_indices:
        rows.append({
            "file": str(real_files[index]),
            "label": "real",
        })

    for index in synthetic_indices:
        rows.append({
            "file": str(synthetic_files[index]),
            "label": "synthetic",
        })

    result = pd.DataFrame(rows)

    result = result.sample(
        frac=1.0,
        random_state=seed,
    ).reset_index(drop=True)

    return result


# ============================================================
# STFT
# ============================================================

def compute_spectral_data(
    path: str | Path,
    sr: int = SR,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
):
    """
    Фиксированное спектральное представление для всех экспериментов.

    Меняется только способ разбиения частотных bins на полосы.
    """

    y, actual_sr = librosa.load(
        path,
        sr=sr,
        mono=True,
    )

    stft = librosa.stft(
        y,
        n_fft=n_fft,
        hop_length=hop_length,
    )

    magnitude = np.abs(stft)

    freqs = librosa.fft_frequencies(
        sr=actual_sr,
        n_fft=n_fft,
    )

    spectral = SimpleNamespace(
        data=magnitude,
        freqs=freqs,
        sr=actual_sr,
    )

    return spectral


# ============================================================
# STATISTICS
# ============================================================

def compute_statistics(values: np.ndarray) -> dict[str, float]:
    """
    Одинаковая статистика для всех band representations.
    """

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
# FEATURE EXTRACTION
# ============================================================

def extract_band_features(
    spectral,
    banded,
    method_name: str,
) -> dict[str, float]:

    features = {}
    magnitude = spectral.data

    for band_index, bins in enumerate(banded.bins):

        if len(bins) == 0:
            # При выбранных n_bands/fraction такого быть не должно
            # (см. determine_shared_n_bands_values), но оставляем
            # защиту на случай ручного запуска с другими параметрами.
            continue

        values = magnitude[bins, :].ravel()
        stats = compute_statistics(values)
        band_name = banded.names[band_index]

        for statistic_name, value in stats.items():
            features[
                f"{method_name}__{band_name}__{statistic_name}"
            ] = value

    return features


# ============================================================
# BUILD DATASET FOR ONE BAND CONFIGURATION
# ============================================================

def build_features(
    sample: pd.DataFrame,
    band_function: Callable,
    method_name: str,
    band_kwargs: dict,
) -> tuple[pd.DataFrame, pd.Series]:

    rows = []
    labels = []

    for counter, row in enumerate(sample.itertuples(index=False), 1):

        print(
            f"\r{method_name}: {counter}/{len(sample)}",
            end="",
            flush=True,
        )

        spectral = compute_spectral_data(row.file)

        banded = band_function(
            spectral,
            **band_kwargs,
        )

        features = extract_band_features(
            spectral,
            banded,
            method_name,
        )

        rows.append(features)
        labels.append(row.label)

    print()

    X = pd.DataFrame(rows)
    y = pd.Series(labels, name="label")

    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return X, y


# ============================================================
# EVALUATION: held-out accuracy/AUC + два вида importance
# ============================================================
#
# Раньше ранжирование строилось на top_k от MDI (Gini) importance,
# посчитанной на тех же данных, на которых модель обучалась. У этого
# два независимых источника смещения:
#
#   1) MDI по построению даёт сумму важностей = 1 для ЛЮБОГО набора
#      признаков, поэтому чем больше скоррелированных (соседних)
#      признаков, тем сильнее эта единица размазывается между ними —
#      top_k механически падает с ростом n_features независимо от
#      реальной информативности представления.
#   2) Importance считалась на обучающих данных — это склонно
#      переоценивать "выученный шум", особенно при небольшом числе
#      сэмплов (n=1000) и умеренно большом числе признаков.
#
# Здесь для каждой конфигурации:
#   - обучаем RF только на train-части;
#   - на test-части считаем accuracy и ROC-AUC — это ГЛАВНЫЙ критерий
#     сравнения представлений, он не завязан на число признаков
#     механически, только через реальную полезность/переобучение;
#   - как и раньше, сохраняем MDI (для сопоставимости и диагностики);
#   - дополнительно считаем permutation importance на test-части —
#     она меньше подвержена искусственному "разбавлению" среди
#     скоррелированных признаков, чем MDI, потому что измеряет
#     персональный вклад каждого признака при фиксированных остальных.


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

    # --- MDI importance (для сопоставимости со старыми результатами) ---

    mdi_importance = pd.Series(
        model.feature_importances_,
        index=X.columns,
    ).sort_values(ascending=False)

    # --- permutation importance на held-out части ---

    # n_jobs=1 здесь намеренно: RandomForestClassifier уже
    # распараллелен (n_jobs=-1 у самой модели), и повторное
    # распараллеливание снаружи (вложенный parallelism) не
    # ускоряет счёт, а только плодит лишние потоки/предупреждения
    # sklearn о невозможности прокинуть конфиг в joblib-воркеры.
    perm_result = permutation_importance(
        model,
        X_test,
        y_test,
        n_repeats=10,
        random_state=seed,
        n_jobs=1,
    )

    perm_raw = pd.Series(
        perm_result.importances_mean,
        index=X.columns,
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


# ============================================================
# SUMMARY
# ============================================================

def make_summary_row(
    method: str,
    parameter_name: str,
    parameter_value,
    resolution_level: str | None,
    evaluation: dict,
) -> dict:

    mdi = evaluation["mdi_importance"]
    perm = evaluation["perm_importance"]

    return {
        "method": method,
        "parameter": parameter_name,
        "parameter_value": parameter_value,
        # "1/3", "1/2", "1/1" — позиция относительно СОБСТВЕННОГО
        # потолка этого метода, не абсолютное число полос. Одинаковый
        # resolution_level у разных методов НЕ значит одинаковое
        # n_features — сверяйтесь с этой колонкой, а не с parameter_value,
        # когда группируете "сопоставимые по грубости" конфигурации.
        "resolution_level": resolution_level,

        "n_features": evaluation["n_features"],

        # Главный критерий сравнения — держим впереди для читаемости.
        "test_accuracy": evaluation["test_accuracy"],
        "test_auc": evaluation["test_auc"],

        # MDI: сопоставимо со старым скриптом, но помним про смещение.
        "mdi_top_10": mdi.head(10).sum(),
        "mdi_top_25": mdi.head(25).sum(),

        # Permutation importance: менее подвержена разбавлению среди
        # скоррелированных признаков.
        "perm_top_10": perm.head(10).sum(),
        "perm_top_25": perm.head(25).sum(),
    }


# ============================================================
# ONE EXPERIMENT
# ============================================================

def run_one_experiment(
    sample: pd.DataFrame,
    method: str,
    band_function: Callable,
    parameter_name: str,
    parameter_value,
    band_kwargs: dict,
    output_dir: Path,
    resolution_level: str | None = None,
) -> dict:

    level_suffix = f"__{resolution_level}" if resolution_level else ""
    experiment_name = f"{method}__{parameter_name}_{parameter_value}{level_suffix}"

    print()
    print("=" * 80)
    print(experiment_name)
    print("=" * 80)

    X, y = build_features(
        sample=sample,
        band_function=band_function,
        method_name=method,
        band_kwargs=band_kwargs,
    )

    print(f"Features: {X.shape[1]}")

    evaluation = evaluate_configuration(X, y)

    evaluation["mdi_importance"].rename("importance").reset_index().rename(
        columns={"index": "feature"}
    ).to_csv(
        output_dir / f"{experiment_name}__mdi_importance.csv",
        index=False,
    )

    evaluation["perm_importance"].rename("importance").reset_index().rename(
        columns={"index": "feature"}
    ).to_csv(
        output_dir / f"{experiment_name}__perm_importance.csv",
        index=False,
    )

    X.to_parquet(
        output_dir / f"{experiment_name}__features.parquet",
        index=False,
    )

    row = make_summary_row(
        method=method,
        parameter_name=parameter_name,
        parameter_value=parameter_value,
        resolution_level=resolution_level,
        evaluation=evaluation,
    )

    return row


# ============================================================
# MAIN
# ============================================================

def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # 0. Определяем per-method потолок n_bands без пустых полос
    #    и три рабочие точки внутри него (1/3, 1/2, 1/1)
    # --------------------------------------------------------

    per_method_levels, per_method_max = determine_per_method_n_bands()

    octave_max_fraction = max_octave_fraction_without_empty(FMIN, FMAX)
    octave_levels = levels_for_ceiling(octave_max_fraction)

    print("=" * 80)
    print(
        "ГРАНИЦЫ 'БЕЗ ПУСТЫХ ПОЛОС' (fmin=%.1f, n_fft=%d, sr=%d, "
        "глобальный потолок=%d)" % (FMIN, N_FFT, SR, GLOBAL_N_BANDS_CEILING)
    )
    print("=" * 80)

    for name in BAND_METHODS:
        own_max = per_method_max[name]
        own_ceiling = min(own_max, GLOBAL_N_BANDS_CEILING)
        levels = per_method_levels[name]
        print(
            f"  {name:8s}: собственный максимум={own_max:4d}  "
            f"собственный потолок={own_ceiling:4d}  "
            f"n_bands={[n for _, n in levels]}"
        )

    print(
        f"  {'octave':8s}: собственный максимум fraction={octave_max_fraction:4d}  "
        f"fraction={[n for _, n in octave_levels]}"
    )
    print()

    # --------------------------------------------------------
    # 1. Один и тот же sample для ВСЕХ экспериментов
    # --------------------------------------------------------

    sample_path = OUTPUT_DIR / "sample.csv"

    if sample_path.exists():
        print("Загружаем существующую выборку:")
        print(sample_path)
        sample = pd.read_csv(sample_path)
    else:
        print("Создаём выборку...")
        sample = sample_dataset(
            real_dir=REAL_DIR,
            synthetic_dir=SYNTHETIC_DIR,
            n_per_class=N_PER_CLASS,
            seed=SEED,
        )
        sample.to_csv(sample_path, index=False)

    print(f"Sample size: {len(sample)}")
    print(sample["label"].value_counts())

    # --------------------------------------------------------
    # 2. Запускаем эксперименты
    # --------------------------------------------------------

    summary = []

    # ========================================================
    # LINEAR / LOG / MEL / ERB / BARK — каждый в своих 1/3, 1/2, 1/1
    # ========================================================

    for method_name, band_function in BAND_METHODS.items():

        for level_label, n_bands in per_method_levels[method_name]:

            row = run_one_experiment(
                sample=sample,
                method=method_name,
                band_function=band_function,
                parameter_name="n_bands",
                parameter_value=n_bands,
                band_kwargs={
                    "n_bands": n_bands,
                    "fmin": FMIN,
                    "fmax": FMAX,
                },
                output_dir=OUTPUT_DIR,
                resolution_level=level_label,
            )

            summary.append(row)

    # ========================================================
    # FRACTIONAL OCTAVE — те же 1/3, 1/2, 1/1, но относительно
    # собственного потолка по fraction (обычно даёт только fraction=1,
    # т.к. уже fraction=2 создаёт пустые полосы при fmin=20 Гц)
    # ========================================================

    for level_label, fraction in octave_levels:

        row = run_one_experiment(
            sample=sample,
            method="octave",
            band_function=octave_bands,
            parameter_name="fraction",
            parameter_value=fraction,
            band_kwargs={
                "fmin": FMIN,
                "fmax": FMAX,
                "fraction": fraction,
            },
            output_dir=OUTPUT_DIR,
            resolution_level=level_label,
        )

        summary.append(row)

    # ========================================================
    # FIXED HZ — контрольная группа (пустых полос не бывает
    # при выбранных ширинах, т.к. width_hz >> Δf). У неё нет
    # "собственного потолка n_bands", поэтому resolution_level не
    # проставляется (остаётся пустым в итоговой таблице).
    # ========================================================

    for width_hz in FIXED_HZ_WIDTHS:

        row = run_one_experiment(
            sample=sample,
            method="fixed_hz",
            band_function=fixed_hz_bands,
            parameter_name="width_hz",
            parameter_value=width_hz,
            band_kwargs={
                "width_hz": width_hz,
                "fmin": FMIN,
                "fmax": FMAX,
            },
            output_dir=OUTPUT_DIR,
        )

        summary.append(row)

    # --------------------------------------------------------
    # 3. Общий результат
    # --------------------------------------------------------

    summary_df = pd.DataFrame(summary)

    # Главный критерий — held-out ROC-AUC, а не top_k importance.
    summary_df = summary_df.sort_values(
        ["test_auc", "test_accuracy"],
        ascending=False,
    ).reset_index(drop=True)

    summary_df.to_csv(OUTPUT_DIR / "band_summary.csv", index=False)

    print()
    print("=" * 80)
    print("BAND REPRESENTATION RANKING (по test_auc)")
    print("=" * 80)
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()