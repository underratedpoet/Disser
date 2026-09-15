from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from bands import linear_bands


SR = 44100
N_FFT = 2048
FMIN = 20.0
N_BANDS = 120


# ============================================================
# ЧАСТЬ 1: РАВНОМЕРНО ЛИ ЧИСЛО FFT-БИНОВ ПО ПОЛОСАМ
# ============================================================
#
# Это не требует ни аудио, ни уже посчитанных признаков — только
# самой геометрии bands.py при заданных sr/n_fft/fmin/fmax. Если тут
# есть перекос, он есть ВСЕГДА, для любого файла и любой конфигурации
# с такими же параметрами — то есть это структурная, а не случайная
# проблема.


def freqs_grid(sr: int = SR, n_fft: int = N_FFT) -> np.ndarray:
    return np.linspace(0, sr / 2, n_fft // 2 + 1)


def band_bin_counts(fmax: float, n_bands: int = N_BANDS) -> pd.DataFrame:

    spectral = SimpleNamespace(freqs=freqs_grid(), sr=SR)

    banded = linear_bands(spectral, n_bands=n_bands, fmin=FMIN, fmax=fmax)

    rows = [
        {
            "band_index": i,
            "band_name": name,
            "n_fft_bins": len(bins),
        }
        for i, (bins, name) in enumerate(zip(banded.bins, banded.names))
    ]

    return pd.DataFrame(rows)


def summarize_bin_counts(fmax: float, n_bands: int = N_BANDS) -> pd.DataFrame:

    df = band_bin_counts(fmax, n_bands)

    median_count = df["n_fft_bins"].median()

    print(f"fmax={fmax:.0f} Гц, n_bands={n_bands}")
    print(f"  медиана бинов на полосу: {median_count:.1f}")
    print(f"  первая полоса: {df['n_fft_bins'].iloc[0]} бинов")
    print(f"  последняя полоса: {df['n_fft_bins'].iloc[-1]} бинов")
    print(
        f"  разброс по всем полосам: "
        f"min={df['n_fft_bins'].min()}, max={df['n_fft_bins'].max()}"
    )

    deviation = (df["n_fft_bins"] - median_count).abs() / median_count
    anomalous = df.loc[deviation > 0.2]

    print(
        f"  полос с отклонением >20% от медианы: "
        f"{len(anomalous)} из {len(df)}"
    )

    if len(anomalous):
        print(
            anomalous[["band_index", "band_name", "n_fft_bins"]]
            .to_string(index=False)
        )

    print()

    return df


# ============================================================
# ЧАСТЬ 2: КОРРЕЛИРУЕТ ЛИ IMPORTANCE С ЧИСЛОМ БИНОВ В ПОЛОСЕ
# ============================================================
#
# Использует уже посчитанный *__mdi_importance.csv — новых вычислений
# по аудио не требует. Если суммарная importance полосы растёт/падает
# вместе с числом бинов в ней систематически (а не случайно) — это
# прямой признак артефакта биннинга, а не содержательного сигнала.


def extract_band_name(feature_name: str) -> str | None:

    parts = feature_name.split("__")

    if len(parts) < 3:
        return None

    # feature = "{method}__{plane}__{band_name}__{statistic}"
    return parts[-2]


def extract_plane(feature_name: str) -> str | None:

    parts = feature_name.split("__")

    if len(parts) < 3:
        return None

    return parts[-3]


def check_importance_correlation(
    bin_counts: pd.DataFrame,
    importance_csv: Path,
) -> None:

    if not importance_csv.exists():
        print(f"Пропускаю — не найден {importance_csv}")
        return

    importance = pd.read_csv(importance_csv)

    importance = importance.copy()
    importance["band_name"] = importance["feature"].apply(extract_band_name)
    importance["plane"] = importance["feature"].apply(extract_plane)

    merged = importance.merge(bin_counts, on="band_name", how="inner")

    if merged.empty:
        print(
            "Не удалось сопоставить признаки с band_name — проверьте, "
            "что fmax/n_bands совпадают с тем прогоном, из которого "
            "взят importance_csv."
        )
        return

    print(f"Источник: {importance_csv.name}")
    print()

    for plane in sorted(merged["plane"].dropna().unique()):

        subset = merged.loc[merged["plane"] == plane]

        agg = subset.groupby("band_name").agg(
            total_importance=("importance", "sum"),
            n_fft_bins=("n_fft_bins", "first"),
        )

        if agg["n_fft_bins"].nunique() < 2:
            print(f"  {plane:10s}: число бинов не варьируется — корреляцию считать не с чем")
            continue

        correlation = agg["total_importance"].corr(agg["n_fft_bins"])

        flag = "  ⚠ заметная корреляция" if abs(correlation) > 0.3 else ""

        print(f"  {plane:10s}: corr(importance, n_fft_bins) = {correlation:+.3f}{flag}")

    print()


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("ЧАСТЬ 1: равномерность числа FFT-бинов по полосам")
    print("=" * 80)
    print()

    df_full = summarize_bin_counts(fmax=22050.0)
    df_16k = summarize_bin_counts(fmax=16000.0)

    print("=" * 80)
    print("ЧАСТЬ 2: коррелирует ли importance с числом бинов в полосе")
    print("=" * 80)
    print()

    check_importance_correlation(
        df_16k,
        Path(
            "stereo_experiment_results/"
            "left_right_plus_diff__fmax16000__mdi_importance.csv"
        ),
    )

    check_importance_correlation(
        df_full,
        Path("stereo_experiment_results/left_right_plus_diff__mdi_importance.csv"),
    )


if __name__ == "__main__":
    main()