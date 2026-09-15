from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from bands import (
    linear_bands,
    importance_bands,
    importance_bands_soft,
    importance_bands_hybrid,
)

from experiment_bands import (
    REAL_DIR,
    SYNTHETIC_DIR,
    FMIN,
    FMAX,
    SEED,
    sample_dataset,
    run_one_experiment,
)


# ============================================================
# CONFIGURATION
# ============================================================

# Старая выборка (первый эксперимент) — её файлы исключаются из новой.
OLD_SAMPLE_PATH = Path("band_experiment_results/sample.csv")

# Кривая важности, посчитанная analyze_band_importance.py на РЕЗУЛЬТАТАХ
# первого эксперимента (т.е. на старой выборке) — не на этой, новой.
IMPORTANCE_CURVE_PATH = Path("band_experiment_results/importance_curve.csv")

OUTPUT_DIR = Path("band_experiment_results_v2")

# Итого 2000 файлов (1000 real + 1000 synthetic), не пересекающихся
# со старой sample.csv.
N_PER_CLASS = 1000

# Тот же бюджет признаков, что у лучшего baseline из первого
# эксперимента (linear_120) — сравнение идёт на равных по n_bands.
N_BANDS = 120

HYBRID_ALPHA = 0.5


# ============================================================
# ЗАГРУЗКА КРИВОЙ ВАЖНОСТИ И СПИСКА ИСКЛЮЧАЕМЫХ ФАЙЛОВ
# ============================================================

def load_importance_curve(path: Path) -> tuple[np.ndarray, np.ndarray]:

    if not path.exists():
        raise FileNotFoundError(
            f"Не найден {path}. Сначала запустите analyze_band_importance.py "
            f"на результатах первого эксперимента (band_experiment_results/)."
        )

    curve = pd.read_csv(path)

    return curve["freq_hz"].to_numpy(), curve["importance_density"].to_numpy()


def load_excluded_files(path: Path) -> set[str]:

    if not path.exists():
        print(
            f"Внимание: {path} не найден — исключать нечего, "
            f"новая выборка может пересечься со старой."
        )
        return set()

    old_sample = pd.read_csv(path)

    return set(old_sample["file"].astype(str))


# ============================================================
# MAIN
# ============================================================

def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    importance_freqs, importance_weights = load_importance_curve(
        IMPORTANCE_CURVE_PATH
    )

    excluded_files = load_excluded_files(OLD_SAMPLE_PATH)

    print(
        f"Исключаем {len(excluded_files)} файлов, "
        f"использованных в первом эксперименте."
    )

    # --------------------------------------------------------
    # Новая выборка: 1000 real + 1000 synthetic, без пересечения
    # со старой sample.csv
    # --------------------------------------------------------

    sample_path = OUTPUT_DIR / "sample.csv"

    if sample_path.exists():
        print("Загружаем существующую выборку v2:")
        print(sample_path)
        sample = pd.read_csv(sample_path)
    else:
        print("Создаём новую выборку (без пересечения со старой)...")

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

    # Явная проверка на пересечение — на случай, если пути в старой
    # sample.csv записаны иначе (другая нормализация слэшей и т.п.)
    # и фильтрация в sample_dataset могла промахнуться.
    overlap = set(sample["file"].astype(str)) & excluded_files

    if overlap:
        raise RuntimeError(
            f"Новая выборка пересекается со старой в {len(overlap)} "
            f"файлах — так быть не должно, проверьте exclude_files "
            f"и формат путей в sample.csv."
        )

    # --------------------------------------------------------
    # Конфигурации: linear-baseline + три закона importance-bands
    # --------------------------------------------------------

    shared_kwargs = {
        "importance_freqs": importance_freqs,
        "importance_weights": importance_weights,
        "n_bands": N_BANDS,
        "fmin": FMIN,
        "fmax": FMAX,
    }

    configs = [
        (
            "linear",
            linear_bands,
            {"n_bands": N_BANDS, "fmin": FMIN, "fmax": FMAX},
        ),
        (
            "importance_direct",
            importance_bands,
            dict(shared_kwargs),
        ),
        (
            "importance_soft",
            importance_bands_soft,
            dict(shared_kwargs),
        ),
        (
            "importance_hybrid",
            importance_bands_hybrid,
            {**shared_kwargs, "alpha": HYBRID_ALPHA},
        ),
    ]

    summary = []

    for method_name, band_function, band_kwargs in configs:

        row = run_one_experiment(
            sample=sample,
            method=method_name,
            band_function=band_function,
            parameter_name="n_bands",
            parameter_value=N_BANDS,
            band_kwargs=band_kwargs,
            output_dir=OUTPUT_DIR,
        )

        summary.append(row)

    # --------------------------------------------------------
    # Итог
    # --------------------------------------------------------

    summary_df = pd.DataFrame(summary)

    summary_df = summary_df.sort_values(
        ["test_auc", "test_accuracy"],
        ascending=False,
    ).reset_index(drop=True)

    summary_df.to_csv(OUTPUT_DIR / "band_summary_v2.csv", index=False)

    print()
    print("=" * 80)
    print("LINEAR vs IMPORTANCE-BASED РАЗБИЕНИЯ (по test_auc)")
    print("=" * 80)
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()