from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================
#
# Несколько независимых "зондов" из первого эксперимента — по
# возможности из разных базовых схем (linear/mel/bark/fixed_hz), а
# не одна. Усреднение по нескольким независимо устроенным сеткам
# гасит специфичный для одной конкретной сетки/выборки шум оценки
# важности отдельной полосы (мы уже видели, что MDI/perm сильно
# "гуляют" от конфигурации к конфигурации).
#
# Пути ниже соответствуют именам файлов, которые сохраняет
# run_one_experiment() в experiment_bands.py. Поправьте при
# необходимости под ваши реальные имена в band_experiment_results/.

RESULTS_DIR = Path("band_experiment_results")

IMPORTANCE_SOURCES = [
    RESULTS_DIR / "linear__n_bands_120__1-1__mdi_importance.csv",
    RESULTS_DIR / "mel__n_bands_120__1-1__mdi_importance.csv",
    RESULTS_DIR / "bark__n_bands_120__1-1__mdi_importance.csv",
    RESULTS_DIR / "fixed_hz__width_hz_250.0__mdi_importance.csv",
]

OUTPUT_CSV = RESULTS_DIR / "importance_curve.csv"

FMIN = 20.0
FMAX = 22050.0

# Разрешение общей сетки, на которую интерполируются все источники.
N_GRID = 4000


# ============================================================
# ПАРСИНГ ИМЁН ПРИЗНАКОВ
# ============================================================
#
# Имя признака имеет вид "{method}__{band_name}__{statistic}", а
# band_name всегда заканчивается на "{lo}-{hi}Hz" (см. _make_bands
# в bands.py) — независимо от конкретного префикса шкалы (lin_037,
# 500Hz_012, 1oct_005 и т.п.), поэтому парсим не жёстким regex на
# весь префикс, а вытаскиваем lo/hi из хвоста строки.

_STATS = {"mean", "std", "skew", "kurtosis"}
_EDGE_RE = re.compile(r"([\d.]+)-([\d.]+)Hz$")


def parse_feature_name(name: str) -> dict | None:

    parts = name.split("__")

    if len(parts) != 3:
        return None

    _, band_name, statistic = parts

    if statistic not in _STATS:
        return None

    match = _EDGE_RE.search(band_name)

    if match is None:
        return None

    return {
        "lo": float(match.group(1)),
        "hi": float(match.group(2)),
    }


def band_importance_from_csv(path: Path) -> pd.DataFrame:
    """Сворачивает 4 статистики на полосу в одну сумму важности на
    полосу, с её центральной частотой.
    """

    df = pd.read_csv(path)

    parsed = df["feature"].apply(parse_feature_name)
    valid = parsed.notna()

    if not valid.any():
        raise ValueError(
            f"Не удалось распарсить ни одного имени признака в {path}. "
            f"Проверьте формат колонки 'feature'."
        )

    df = df.loc[valid].copy()
    parsed = parsed.loc[valid]

    df["lo"] = parsed.apply(lambda d: d["lo"])
    df["hi"] = parsed.apply(lambda d: d["hi"])
    df["center_hz"] = (df["lo"] + df["hi"]) / 2.0

    grouped = (
        df.groupby("center_hz", as_index=False)["importance"]
        .sum()
        .sort_values("center_hz")
        .reset_index(drop=True)
    )

    return grouped


# ============================================================
# ПЕРЕВОД "ВАЖНОСТЬ ПО ПОЛОСАМ" -> "ПЛОТНОСТЬ НА ЕДИНОЙ СЕТКЕ"
# ============================================================
#
# Важность полосы сама по себе не сопоставима между сетками с разной
# шириной полос (широкая полоса естественно накапливает больше
# суммарной importance просто за счёт агрегирования по ней большего
# числа FFT-бинов). Поэтому делим важность каждой полосы на её
# ширину — получаем "важность на герц", которую уже можно
# интерполировать и сравнивать между разными сетками.


def to_density_on_grid(
    band_importance: pd.DataFrame,
    fmin: float,
    fmax: float,
    n_grid: int,
) -> tuple[np.ndarray, np.ndarray]:

    centers = band_importance["center_hz"].to_numpy()
    importance = band_importance["importance"].to_numpy()

    width = np.gradient(centers)
    density_at_centers = importance / np.maximum(width, 1e-9)

    grid = np.linspace(fmin, fmax, n_grid)

    density = np.interp(
        grid,
        centers,
        density_at_centers,
        left=density_at_centers[0],
        right=density_at_centers[-1],
    )

    # Нормируем к сумме 1, чтобы вклад разных источников (с разным
    # числом полос и разной суммарной importance) был сопоставим при
    # усреднении.
    density = density / density.sum()

    return grid, density


# ============================================================
# MAIN
# ============================================================

def main():

    grid = None
    densities = []

    for source in IMPORTANCE_SOURCES:

        if not source.exists():
            print(f"Пропускаю (файл не найден): {source}")
            continue

        band_importance = band_importance_from_csv(source)

        g, density = to_density_on_grid(
            band_importance,
            FMIN,
            FMAX,
            N_GRID,
        )

        grid = g
        densities.append(density)

        print(
            f"Учтён источник: {source.name} "
            f"({len(band_importance)} полос)"
        )

    if not densities:
        raise RuntimeError(
            "Не найдено ни одного файла importance. "
            "Проверьте IMPORTANCE_SOURCES — возможно, изменились имена "
            "файлов в band_experiment_results/."
        )

    combined_density = np.mean(densities, axis=0)

    curve = pd.DataFrame({
        "freq_hz": grid,
        "importance_density": combined_density,
    })

    curve.to_csv(OUTPUT_CSV, index=False)

    print()
    print(f"Источников усреднено: {len(densities)}")
    print(f"Сохранено: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()