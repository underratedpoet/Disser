from __future__ import annotations

from typing import Callable
import numpy as np

from core import SpectralData, BandedSpectralData


def _make_bands(
    spectral: SpectralData,
    edges: np.ndarray,
    prefix: str,
) -> BandedSpectralData:
    """Convert frequency edges into row-indexed bands.

    Empty bands are retained in the returned object so that the frequency
    definition itself is transparent. Feature functions may skip empty bands.
    """

    freqs = np.asarray(spectral.freqs)

    bins = []
    names = []

    for i in range(len(edges) - 1):
        lo = edges[i]
        hi = edges[i + 1]

        if i == len(edges) - 2:
            idx = np.where((freqs >= lo) & (freqs <= hi))[0]
        else:
            idx = np.where((freqs >= lo) & (freqs < hi))[0]

        bins.append(idx)
        names.append(
            f"{prefix}_{i:03d}_{lo:.1f}-{hi:.1f}Hz"
        )

    return BandedSpectralData(
        spectral=spectral,
        bins=bins,
        names=names,
        edges=edges,
    )


def linear_bands(
    spectral: SpectralData,
    n_bands: int = 40,
    fmin: float = 20.0,
    fmax: float | None = None,
) -> BandedSpectralData:
    """Uniform spacing in Hz."""

    nyquist = spectral.sr / 2
    fmax = nyquist if fmax is None else min(fmax, nyquist)

    if not 0 < fmin < fmax:
        raise ValueError("Require 0 < fmin < fmax <= Nyquist.")

    edges = np.linspace(fmin, fmax, n_bands + 1)
    return _make_bands(spectral, edges, "lin")


def log_bands(
    spectral: SpectralData,
    n_bands: int = 40,
    fmin: float = 20.0,
    fmax: float | None = None,
) -> BandedSpectralData:
    """Uniform spacing in log10(f)."""

    nyquist = spectral.sr / 2
    fmax = nyquist if fmax is None else min(fmax, nyquist)

    if not 0 < fmin < fmax:
        raise ValueError("Require 0 < fmin < fmax <= Nyquist.")

    edges = np.geomspace(fmin, fmax, n_bands + 1)
    return _make_bands(spectral, edges, "log")


def mel_bands(
    spectral: SpectralData,
    n_bands: int = 40,
    fmin: float = 20.0,
    fmax: float | None = None,
) -> BandedSpectralData:
    """Uniform spacing on the mel scale.

    This is useful as a third baseline between linear and log spacing.
    """

    nyquist = spectral.sr / 2
    fmax = nyquist if fmax is None else min(fmax, nyquist)

    if not 0 < fmin < fmax:
        raise ValueError("Require 0 < fmin < fmax <= Nyquist.")

    mel_min = 2595.0 * np.log10(1.0 + fmin / 700.0)
    mel_max = 2595.0 * np.log10(1.0 + fmax / 700.0)

    mel_edges = np.linspace(mel_min, mel_max, n_bands + 1)
    edges = 700.0 * (10.0 ** (mel_edges / 2595.0) - 1.0)

    return _make_bands(spectral, edges, "mel")


def erb_bands(
    spectral: SpectralData,
    n_bands: int = 40,
    fmin: float = 20.0,
    fmax: float | None = None,
) -> BandedSpectralData:
    """Uniform spacing on the ERB-rate scale."""

    nyquist = spectral.sr / 2
    fmax = nyquist if fmax is None else min(fmax, nyquist)

    if not 0 < fmin < fmax:
        raise ValueError("Require 0 < fmin < fmax <= Nyquist.")

    def hz_to_erb(f):
        return 21.4 * np.log10(1.0 + 0.00437 * f)

    def erb_to_hz(e):
        return (10.0 ** (e / 21.4) - 1.0) / 0.00437

    erb_min = hz_to_erb(fmin)
    erb_max = hz_to_erb(fmax)

    erb_edges = np.linspace(erb_min, erb_max, n_bands + 1)
    edges = erb_to_hz(erb_edges)

    return _make_bands(spectral, edges, "erb")


def bark_bands(
    spectral: SpectralData,
    n_bands: int = 24,
    fmin: float = 20.0,
    fmax: float | None = None,
) -> BandedSpectralData:
    """Uniform spacing on the Bark scale.

    24 bands is the classical Bark-band count; n_bands is configurable
    because our task is feature discovery rather than psychoacoustic modeling.
    """

    nyquist = spectral.sr / 2
    fmax = nyquist if fmax is None else min(fmax, nyquist)

    if not 0 < fmin < fmax:
        raise ValueError("Require 0 < fmin < fmax <= Nyquist.")

    def hz_to_bark(f):
        return (
            6.0 * np.arcsinh(f / 600.0)
        )

    def bark_to_hz(z):
        return 600.0 * np.sinh(z / 6.0)

    bark_min = hz_to_bark(fmin)
    bark_max = hz_to_bark(fmax)

    bark_edges = np.linspace(bark_min, bark_max, n_bands + 1)
    edges = bark_to_hz(bark_edges)

    return _make_bands(spectral, edges, "bark")


def octave_bands(
    spectral: SpectralData,
    fmin: float = 31.25,
    fmax: float | None = None,
    fraction: int = 1,
) -> BandedSpectralData:
    """Fractional-octave bands.

    fraction=1  -> octave bands
    fraction=3  -> one-third-octave bands
    fraction=6  -> one-sixth-octave bands
    """

    if fraction < 1:
        raise ValueError("fraction must be >= 1")

    nyquist = spectral.sr / 2
    fmax = nyquist if fmax is None else min(fmax, nyquist)

    if not 0 < fmin < fmax:
        raise ValueError("Require 0 < fmin < fmax <= Nyquist.")

    ratio = 2.0 ** (1.0 / fraction)

    edges = [fmin]
    while edges[-1] < fmax:
        edges.append(edges[-1] * ratio)

    edges[-1] = min(edges[-1], fmax)

    return _make_bands(
        spectral,
        np.asarray(edges),
        f"{fraction}oct",
    )


def fixed_hz_bands(
    spectral: SpectralData,
    width_hz: float = 500.0,
    fmin: float = 20.0,
    fmax: float | None = None,
) -> BandedSpectralData:
    """Equal-width bands with arbitrary width in Hz."""

    if width_hz <= 0:
        raise ValueError("width_hz must be positive.")

    nyquist = spectral.sr / 2
    fmax = nyquist if fmax is None else min(fmax, nyquist)

    if not 0 < fmin < fmax:
        raise ValueError("Require 0 < fmin < fmax <= Nyquist.")

    edges = np.arange(fmin, fmax, width_hz)
    edges = np.append(edges, fmax)

    return _make_bands(spectral, edges, f"{width_hz:g}Hz")

# ============================================================
# ГИБРИДНЫЕ ЗАКОНЫ РАЗБИЕНИЯ НА ПОЛОСЫ НА ОСНОВЕ ИЗМЕРЕННОЙ ВАЖНОСТИ
# ============================================================
#
#
# Общая идея (аналог того, как log/mel/erb/bark равномерно делят не
# саму частоту, а её "искажённую" версию): здесь роль искажающей
# функции играет не фиксированная психоакустическая формула, а
# эмпирическая плотность важности I(f) — границы полос расставляются
# так, чтобы между соседними границами накапливалась примерно равная
# "масса" плотности (а не равная ширина в Гц/mel/erb/...).
#
# Три закона отличаются тем, ЧТО именно берётся в качестве плотности:
#
#   importance_bands         — density = I(f)              (напрямую)
#   importance_bands_soft    — density = sqrt(I(f))         (смягчённо)
#   importance_bands_hybrid  — density = alpha*I(f) + (1-alpha)*(1/f)
#                               (регуляризовано физическим базовым
#                               законом — тем самым, что даёт log_bands)


def _importance_density_on_grid(
    importance_freqs: np.ndarray,
    importance_weights: np.ndarray,
    fmin: float,
    fmax: float,
    n_grid: int = 4000,
) -> tuple[np.ndarray, np.ndarray]:
    """Интерполирует дискретную кривую важности на равномерную сетку.

    importance_freqs/importance_weights — то, что лежит в
    importance_curve.csv (колонки freq_hz, importance_density),
    посчитанном analyze_band_importance.py на ДРУГОЙ выборке файлов,
    чем та, на которой тестируется само разбиение.
    """

    order = np.argsort(importance_freqs)
    freqs = np.asarray(importance_freqs)[order]
    weights = np.asarray(importance_weights)[order]

    grid = np.linspace(fmin, fmax, n_grid)

    density = np.interp(
        grid,
        freqs,
        weights,
        left=weights[0],
        right=weights[-1],
    )

    return grid, density


def _edges_from_density(
    freqs_grid: np.ndarray,
    density: np.ndarray,
    n_bands: int,
    fmin: float,
    fmax: float,
    min_bandwidth_hz: float | None = None,
    max_ceiling_search_iters: int = 40,
) -> np.ndarray:
    """n_bands+1 границ, между которыми накапливается примерно равная
    "масса" density(f) — обобщение того, как log/mel/erb/bark равномерно
    делят искажённую частоту, только здесь искажение — произвольная
    (в т.ч. эмпирическая) плотность, а не одна из типовых формул.

    min_bandwidth_hz — если задано (обычно фактическое разрешение FFT,
    Δf), функция гарантирует, что ни одна полоса не окажется у́же этого
    порога, а не молча оставляет её пустой (как происходит в
    log/erb/octave_bands при мелком дроблении). Достигается срезанием
    пиков плотности (ceiling), а не подъёмом дна (floor не спасает от
    СЛИШКОМ ВЫСОКОГО локального пика плотности — характерная проблема
    для законов, содержащих сингулярную у fmin компоненту вроде 1/f,
    см. importance_bands_hybrid).
    """

    density = np.clip(density, a_min=0.0, a_max=None)

    if density.sum() <= 0:
        raise ValueError(
            "density суммируется в ноль — нечего распределять между полосами."
        )

    def build(d: np.ndarray) -> np.ndarray:
        cumulative = np.cumsum(d)
        cumulative = cumulative / cumulative[-1]

        targets = np.linspace(0.0, 1.0, n_bands + 1)
        e = np.interp(targets, cumulative, freqs_grid)

        e[0] = fmin
        e[-1] = fmax

        return e

    edges = build(density)

    if min_bandwidth_hz is None:
        return edges

    ceiling_mult = None

    for _ in range(max_ceiling_search_iters):

        widths = np.diff(edges)

        if widths.min() >= min_bandwidth_hz:
            return edges

        current_ceiling = (
            density.max()
            if ceiling_mult is None
            else ceiling_mult * density.mean()
        )

        ceiling_mult = (current_ceiling / density.mean()) * 0.8

        clipped = np.minimum(density, ceiling_mult * density.mean())
        edges = build(clipped)

    raise RuntimeError(
        f"Не удалось подобрать разбиение без полос у́же {min_bandwidth_hz:.2f} Гц "
        f"(частотное разрешение FFT) за {max_ceiling_search_iters} итераций. "
        f"Попробуйте уменьшить n_bands, увеличить floor_fraction или (для "
        f"гибрида) уменьшить alpha."
    )


def importance_bands(
    spectral: SpectralData,
    importance_freqs: np.ndarray,
    importance_weights: np.ndarray,
    n_bands: int = 120,
    fmin: float = 20.0,
    fmax: float | None = None,
    floor_fraction: float = 0.05,
) -> BandedSpectralData:
    """Закон 1 — "прямой": плотность полос ∝ измеренной важности I(f).

    Максимально агрессивная концентрация разрешения там, где полосы
    оказались важны на пробной сетке. Как следствие — максимально
    чувствителен к шуму самой оценки I(f): она посчитана на конечной
    выборке файлов и сама по себе имеет разброс (мы это уже видели на
    MDI/perm importance в первом эксперименте).

    floor_fraction — минимальная плотность как доля от средней;
    не даёт совсем неважным участкам спектра схлопнуться в полосы
    шириной 0 (что превратило бы их в пустые полосы, см. bands.py).
    """

    nyquist = spectral.sr / 2
    fmax = nyquist if fmax is None else min(fmax, nyquist)

    if not 0 < fmin < fmax:
        raise ValueError("Require 0 < fmin < fmax <= Nyquist.")

    grid, density = _importance_density_on_grid(
        importance_freqs, importance_weights, fmin, fmax,
    )

    floor = floor_fraction * np.mean(density)
    density = np.maximum(density, floor)

    delta_f = float(np.median(np.diff(spectral.freqs)))

    edges = _edges_from_density(
        grid, density, n_bands, fmin, fmax, min_bandwidth_hz=delta_f,
    )

    return _make_bands(spectral, edges, "imp")


def importance_bands_soft(
    spectral: SpectralData,
    importance_freqs: np.ndarray,
    importance_weights: np.ndarray,
    n_bands: int = 120,
    fmin: float = 20.0,
    fmax: float | None = None,
    floor_fraction: float = 0.05,
) -> BandedSpectralData:
    """Закон 2 — "смягчённый": плотность ∝ sqrt(I(f)).

    Извлечение корня уменьшает контраст между "важными" и
    "неважными" участками по сравнению с importance_bands — меньше
    риск подстроиться под шум конкретной оценки I(f), но и меньше
    потенциальный выигрыш, если гипотеза о сильно неравномерной
    важности верна.
    """

    nyquist = spectral.sr / 2
    fmax = nyquist if fmax is None else min(fmax, nyquist)

    if not 0 < fmin < fmax:
        raise ValueError("Require 0 < fmin < fmax <= Nyquist.")

    grid, density = _importance_density_on_grid(
        importance_freqs, importance_weights, fmin, fmax,
    )

    floor = floor_fraction * np.mean(density)
    density = np.sqrt(np.maximum(density, floor))

    delta_f = float(np.median(np.diff(spectral.freqs)))

    edges = _edges_from_density(
        grid, density, n_bands, fmin, fmax, min_bandwidth_hz=delta_f,
    )

    return _make_bands(spectral, edges, "impsoft")


def importance_bands_hybrid(
    spectral: SpectralData,
    importance_freqs: np.ndarray,
    importance_weights: np.ndarray,
    n_bands: int = 120,
    fmin: float = 20.0,
    fmax: float | None = None,
    alpha: float = 0.5,
    floor_fraction: float = 0.05,
) -> BandedSpectralData:
    """Закон 3 — "гибридный": плотность = смесь I(f) и базового 1/f.

    density = alpha * I_normalized(f) + (1 - alpha) * (1/f)_normalized

    При alpha=1 эквивалентен importance_bands (без floor), при
    alpha=0 — эквивалентен log_bands (равномерное деление по log(f)
    соответствует плотности 1/f). Базовый закон 1/f работает
    регуляризатором: не даёт сетке полностью подстроиться под шум
    одной оценки I(f) на одной выборке, а тянет её к физически
    мотивированному распределению.
    """

    nyquist = spectral.sr / 2
    fmax = nyquist if fmax is None else min(fmax, nyquist)

    if not 0 < fmin < fmax:
        raise ValueError("Require 0 < fmin < fmax <= Nyquist.")

    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1].")

    grid, importance_density = _importance_density_on_grid(
        importance_freqs, importance_weights, fmin, fmax,
    )

    importance_density = np.maximum(
        importance_density,
        floor_fraction * np.mean(importance_density),
    )
    importance_density = importance_density / importance_density.sum()

    baseline_density = 1.0 / grid
    baseline_density = baseline_density / baseline_density.sum()

    density = alpha * importance_density + (1.0 - alpha) * baseline_density

    delta_f = float(np.median(np.diff(spectral.freqs)))

    edges = _edges_from_density(
        grid, density, n_bands, fmin, fmax, min_bandwidth_hz=delta_f,
    )

    return _make_bands(spectral, edges, f"imphyb{alpha:g}")