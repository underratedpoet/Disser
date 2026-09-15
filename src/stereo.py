from __future__ import annotations

from typing import Callable

import librosa
import numpy as np

from core import SpectralData


def _stft_magnitude(
    mono: np.ndarray,
    sr: int,
    n_fft: int,
    hop_length: int,
) -> SpectralData:

    stft = librosa.stft(mono, n_fft=n_fft, hop_length=hop_length)
    magnitude = np.abs(stft)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    return SpectralData(matrix=magnitude, freqs=freqs, sr=sr, signal=mono)


def _ensure_stereo(y: np.ndarray) -> np.ndarray:
    """Дублирует моно-сигнал в псевдо-стерео, если канал всего один.

    В датасете могут попасться моно-файлы — без этого left_right/
    mid_side/left_right_plus_diff упадут на y[1].
    """

    if y.ndim == 1:
        return np.vstack([y, y])

    return y


def left_right(
    y: np.ndarray,
    sr: int,
    n_fft: int = 2048,
    hop_length: int = 512,
) -> dict[str, SpectralData]:
    """Базовый вариант: L и R по отдельности, без явного объединения.

    Ровно то, что молчаливо использовалось в первой версии пайплайна
    признаков (каждый канал обрабатывался независимо друг от друга) —
    baseline для сравнения с mid_side и left_right_plus_diff.
    """

    y = _ensure_stereo(y)

    left = _stft_magnitude(y[0], sr, n_fft, hop_length)
    right = _stft_magnitude(y[1], sr, n_fft, hop_length)

    return {"L": left, "R": right}


def mid_side(
    y: np.ndarray,
    sr: int,
    n_fft: int = 2048,
    hop_length: int = 512,
) -> dict[str, SpectralData]:
    """M = (L+R)/2, S = (L-R)/2 — раскладываем сигнал на общую (M) и
    разностную (S) составляющие ДО STFT, объединение во временной
    области.

    M — почти моно-версия трека (то, что каналы разделяют между
    собой), S — то, что между ними асимметрично (стерео-ширина,
    разностные артефакты). В отличие от left_right, здесь L и R сами
    по себе до модели в явном виде не доходят — только их сумма и
    разность.
    """

    y = _ensure_stereo(y)

    mid = (y[0] + y[1]) / 2.0
    side = (y[0] - y[1]) / 2.0

    m = _stft_magnitude(mid, sr, n_fft, hop_length)
    s = _stft_magnitude(side, sr, n_fft, hop_length)

    return {"M": m, "S": s}


def left_right_plus_diff(
    y: np.ndarray,
    sr: int,
    n_fft: int = 2048,
    hop_length: int = 512,
) -> dict[str, SpectralData]:
    """L и R по отдельности (как в left_right) плюс явные межканальные
    величины, посчитанные напрямую из STFT(L) и STFT(R) — не через
    комбинирование сигналов во временной области, как в mid_side.

    IPD_COS, IPD_SIN
        Межканальная разность фаз angle(STFT(L)) - angle(STFT(R)),
        разложенная на cos/sin вместо использования сырого угла.
        Это принципиально: обычные mean/std от угла некорректны
        из-за разрыва на границе -pi/+pi (см. разбор PHASE-признака
        в статье) — cos/sin ограничены [-1, 1] и непрерывны, разрыва
        нет.

    MAG_DIFF
        |STFT(L)| - |STFT(R)| — разница амплитуд каналов на каждом
        бине/кадре, прокси межканального энергетического баланса.
        В отличие от старой формулы DR (см. разбор в статье), обе
        величины здесь честно посчитаны на уровне (бин, кадр), а не
        через один глобальный максимум на весь сегмент.

    Ни IPD, ни MAG_DIFF не заменяют L/R — они добавляются как
    дополнительные "плоскости" поверх них, аналогично тому, как
    MAG/PHASE/REAL/IMAG/MGD сосуществуют в основном пайплайне
    признаков.

    Внимание: это 5 плоскостей вместо 2 у left_right/mid_side — при
    n_bands=120 это 2400 признаков против 960. Держите в голове при
    сравнении test_auc: часть возможного выигрыша может объясняться
    просто бОльшей ёмкостью модели, а не новой информацией (см.
    stereo_experiment.py).
    """

    y = _ensure_stereo(y)

    stft_l = librosa.stft(y[0], n_fft=n_fft, hop_length=hop_length)
    stft_r = librosa.stft(y[1], n_fft=n_fft, hop_length=hop_length)

    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    mag_l = np.abs(stft_l)
    mag_r = np.abs(stft_r)

    phase_diff = np.angle(stft_l) - np.angle(stft_r)
    phase_diff = np.mod(phase_diff + np.pi, 2.0 * np.pi) - np.pi

    return {
        "L": SpectralData(matrix=mag_l, freqs=freqs, sr=sr),
        "R": SpectralData(matrix=mag_r, freqs=freqs, sr=sr),
        "IPD_COS": SpectralData(matrix=np.cos(phase_diff), freqs=freqs, sr=sr),
        "IPD_SIN": SpectralData(matrix=np.sin(phase_diff), freqs=freqs, sr=sr),
        "MAG_DIFF": SpectralData(matrix=mag_l - mag_r, freqs=freqs, sr=sr),
    }


STEREO_METHODS: dict[str, Callable] = {
    "left_right": left_right,
    "mid_side": mid_side,
    "left_right_plus_diff": left_right_plus_diff,
}