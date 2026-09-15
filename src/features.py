from __future__ import annotations

import librosa
import numpy as np
from scipy.fftpack import dct

from core import SpectralData
from spafe.features.cqcc import cqcc as _spafe_cqcc
from spafe.utils.preprocessing import SlidingWindow


# ============================================================
# ОБЩАЯ ИДЕЯ
# ============================================================
#
# Функции этого модуля делятся на три группы по смыслу оси "freqs" в
# возвращаемом SpectralData — это определяет, можно ли резать
# результат на полосы существующими linear_bands/log_bands/... из
# bands.py "как есть", без изменений:
#
#   1) ЧАСТОТНО-ИНДЕКСИРОВАННЫЕ (freqs — реальные Гц, та же сетка,
#      что у обычного STFT): magnitude, real_part, imag_part,
#      phase_cos_sin, group_delay, modified_group_delay.
#      Полностью совместимы с bands.py без каких-либо изменений —
#      просто передайте результат вместо "magnitude spectrogram".
#
#   2) КОЭФФИЦИЕНТ-ИНДЕКСИРОВАННЫЕ (freqs — просто номер коэффициента
#      0..N-1, НЕ герцы): lfcc, mfcc, cqcc. DCT перемешивает информацию
#      по всему диапазону в каждый коэффициент, поэтому привязать
#      коэффициент к одной физической частоте нельзя. bands.py
#      технически всё равно применим (он не проверяет единицы freqs),
#      но n_bands там осмысленно выставлять равным числу коэффициентов
#      (каждая "полоса" = один коэффициент) либо меньше (тогда
#      "полоса" = группа соседних коэффициентов, что менее
#      интерпретируемо, но не запрещено).
#
#   3) ВРЕМЕННЫЕ РЯДЫ БЕЗ ЧАСТОТНОЙ ОСИ ВООБЩЕ: rms, crest_factor.
#      Частотной оси нет в принципе — банднуть нечего, 4 статистики
#      считаются по ним напрямую, как и раньше.


def _stft(y: np.ndarray, n_fft: int, hop_length: int) -> np.ndarray:
    return librosa.stft(y, n_fft=n_fft, hop_length=hop_length)


def _center_pad(y: np.ndarray, n_fft: int) -> np.ndarray:
    """Дополняет сигнал так же, как librosa.stft(..., center=True)
    (padding по n_fft//2 отражением с каждой стороны).

    Нужно там, где кадры собираются вручную (group_delay,
    modified_group_delay, crest_factor) через librosa.util.frame —
    у неё, в отличие от librosa.stft, нет паддинга по умолчанию, и
    без этой поправки её кадры не совпадали бы по числу и выравниванию
    по времени с кадрами MAG/REAL/IMAG/RMS. Это важно, если позже вы
    будете подавать несколько таких "плоскостей" как разные каналы в
    одну модель по одной и той же временной оси.
    """
    pad_width = n_fft // 2
    return np.pad(y, pad_width, mode="reflect")


# ============================================================
# 1) ЧАСТОТНО-ИНДЕКСИРОВАННЫЕ ПРИЗНАКИ
# ============================================================

def magnitude(
    y: np.ndarray,
    sr: int,
    n_fft: int = 2048,
    hop_length: int = 512,
) -> SpectralData:
    """|STFT(y)| — амплитудный спектр. Эталон для сравнения с
    остальными признаками, без каких-либо изменений от стандартной
    формулы (Oppenheim & Schafer, "Discrete-Time Signal Processing").
    """

    X = _stft(y, n_fft, hop_length)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    return SpectralData(matrix=np.abs(X), freqs=freqs, sr=sr)


def real_part(
    y: np.ndarray,
    sr: int,
    n_fft: int = 2048,
    hop_length: int = 512,
) -> SpectralData:
    """Re(STFT(y))."""

    X = _stft(y, n_fft, hop_length)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    return SpectralData(matrix=np.real(X), freqs=freqs, sr=sr)


def imag_part(
    y: np.ndarray,
    sr: int,
    n_fft: int = 2048,
    hop_length: int = 512,
) -> SpectralData:
    """Im(STFT(y))."""

    X = _stft(y, n_fft, hop_length)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    return SpectralData(matrix=np.imag(X), freqs=freqs, sr=sr)


def phase_cos_sin(
    y: np.ndarray,
    sr: int,
    n_fft: int = 2048,
    hop_length: int = 512,
) -> tuple[SpectralData, SpectralData]:
    """cos(phase), sin(phase) — вместо сырой фазы одного канала.

    Сырая phase = angle(STFT) некорректна для линейных mean/std/skew/
    kurtosis из-за разрыва на границе -pi/+pi (см. разбор PHASE-
    признака в статье — это, вероятно, и есть причина его низкой
    информативности). cos/sin ограничены [-1, 1] и непрерывны — тот
    же приём, что уже применён к межканальной IPD в stereo.py.

    Внимание: это ФАЗА ОДНОГО канала (L или R по отдельности), не
    путать с IPD_COS/IPD_SIN из stereo.py — там разность фаз МЕЖДУ
    каналами. Это разные, дополняющие друг друга признаки.

    Источник по проблеме circular statistics: Fisher, N.I. (1993).
    "Statistical Analysis of Circular Data." Cambridge University Press.
    """

    X = _stft(y, n_fft, hop_length)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    phase = np.angle(X)

    return (
        SpectralData(matrix=np.cos(phase), freqs=freqs, sr=sr),
        SpectralData(matrix=np.sin(phase), freqs=freqs, sr=sr),
    )


def group_delay(
    y: np.ndarray,
    sr: int,
    n_fft: int = 2048,
    hop_length: int = 512,
) -> SpectralData:
    """Простая (без сглаживания выбросов) групповая задержка.

    τ(ω) = -dφ/dω = (X_I·∂X_R/∂ω - X_R·∂X_I/∂ω) / |X(ω)|²

    Знак минус — по стандартному определению group delay (Oppenheim &
    Schafer и вся последующая литература): τ(ω) = -dφ(ω)/dω, а не
    просто производная фазы без знака.

    ИСПРАВЛЕНО относительно исходного кода в двух местах:
      1) производная берётся ПО ОСИ ЧАСТОТЫ (axis=0 — в один момент
         времени, между соседними частотными бинами), а не по времени
         (было axis=1). Групповая задержка по определению — производная
         фазы ПО ЧАСТОТЕ; то, что считал исходный код (производная по
         времени), ближе к понятию мгновенной частоты — другая величина
         с другой физической интерпретацией.
      2) знак: у исходного кода (и в первой версии этого файла) не
         было минуса перед производной.

    Не устраняет выбросы вблизи нулей |X(ω)| — для этого см.
    modified_group_delay ниже.

    Источник: Yegnanarayana, B., & Murthy, H.A. (1992). "Significance
    of group delay functions in spectrum estimation." IEEE
    Transactions on Signal Processing, 40(9).
    """

    X = _stft(y, n_fft, hop_length)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    real = np.real(X)
    imag = np.imag(X)

    numerator = (
        imag * np.gradient(real, axis=0)
        - real * np.gradient(imag, axis=0)
    )
    denominator = real**2 + imag**2 + 1e-10

    return SpectralData(matrix=numerator / denominator, freqs=freqs, sr=sr)


def modified_group_delay(
    y: np.ndarray,
    sr: int,
    n_fft: int = 2048,
    hop_length: int = 512,
    alpha: float = 0.4,
    gamma: float = 0.9,
    lifter_cutoff: int = 30,
    clip_percentile: float = 1.0,
) -> SpectralData:
    """Модифицированная групповая задержка (MGDF).

    τ(ω) = [X_R(ω)Y_R(ω) + X_I(ω)Y_I(ω)] / |S(ω)|^(2γ)
    τ_m(ω) = sign(τ(ω)) · |τ(ω)|^α

    где Y(ω) = FFT(n·x_w(n)) — тот же оконный кадр, поэлементно
    умноженный на индекс отсчёта n перед FFT (стандартный приём для
    получения производной фазы без разворачивания/unwrap; выводится
    из dX/dω = -jY(ω), см. обсуждение), а |S(ω)| — кепстрально
    сглаженная версия |X(ω)| (лифтерование: обнуляем кепстральные
    коэффициенты выше lifter_cutoff, возвращаемся в частотную область)
    — устраняет выбросы group_delay() рядом с нулями |X(ω)|, которые
    иначе доминировали бы в статистике.

    alpha=0.4, gamma=0.9 — типичные значения из литературы (задача
    верификации диктора), не физические константы: гиперпараметры,
    которые в принципе можно подстраивать под задачу.

    clip_percentile: даже после сглаживания знаменателя и компрессии
    степенью alpha в редких кадрах (обычно тихие/шумные участки, где
    фазовая оценка нестабильна в принципе) встречаются выбросы на
    порядки больше типичных значений — из-за домножения на индекс
    отсчёта (до n_fft-1) в Y числитель может быть на 2-3 порядка
    больше, чем у X. Обрезаем по [clip_percentile, 100-clip_percentile]
    перцентилям ПЕРЕД тем, как считать mean/std/skew/kurtosis по
    полосам — иначе редкие выбросы искажают статистики высокого
    порядка так же, как это уже было замечено на std/skew/kurtosis в
    сырых пере-band статистиках MGD в статье.

    Источники:
    - Murthy, H.A., & Yegnanarayana, B. (2011). "Group delay functions
      and its applications to speech technology." Sadhana, 36(5).
    - Hegde, R.M., Murthy, H.A., & Rao, G.V.R. (2007). "Application of
      the Modified Group Delay Function to Speaker Identification and
      Discrimination." IEEE ICASSP.
    """

    y_padded = _center_pad(y, n_fft)
    frames = librosa.util.frame(y_padded, frame_length=n_fft, hop_length=hop_length)
    # frames.shape == (n_fft, n_frames)

    window = np.hanning(n_fft)[:, None]
    n_idx = np.arange(n_fft)[:, None]

    windowed = frames * window
    time_weighted = windowed * n_idx

    X = np.fft.rfft(windowed, axis=0)
    Y = np.fft.rfft(time_weighted, axis=0)

    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    # --- кепстральное сглаживание log|X| для устойчивого знаменателя ---
    log_mag = np.log(np.abs(X) + 1e-10)
    cepstrum = np.fft.irfft(log_mag, n=n_fft, axis=0)

    cutoff = min(lifter_cutoff, cepstrum.shape[0])
    liftered = np.zeros_like(cepstrum)
    liftered[:cutoff] = cepstrum[:cutoff]

    smoothed_log_mag = np.fft.rfft(liftered, axis=0).real
    smoothed_magnitude = np.exp(smoothed_log_mag)

    numerator = X.real * Y.real + X.imag * Y.imag
    denominator = smoothed_magnitude ** (2.0 * gamma) + 1e-10

    tau = numerator / denominator
    tau_modified = np.sign(tau) * np.abs(tau) ** alpha

    if clip_percentile > 0:
        lower = np.percentile(tau_modified, clip_percentile)
        upper = np.percentile(tau_modified, 100.0 - clip_percentile)
        tau_modified = np.clip(tau_modified, lower, upper)

    return SpectralData(matrix=tau_modified, freqs=freqs, sr=sr)


# ============================================================
# 2) КОЭФФИЦИЕНТ-ИНДЕКСИРОВАННЫЕ ПРИЗНАКИ (freqs = номер коэффициента)
# ============================================================

def _linear_filterbank(
    sr: int,
    n_fft: int,
    n_filters: int,
    fmin: float = 0.0,
    fmax: float | None = None,
) -> np.ndarray:
    """Треугольный ЛИНЕЙНЫЙ (не mel) фильтр-банк — аналог
    librosa.filters.mel, но с линейно (не по mel-шкале) расположенными
    центрами треугольников.
    """

    fmax = sr / 2 if fmax is None else fmax

    fft_freqs = np.linspace(0, sr / 2, n_fft // 2 + 1)
    edges = np.linspace(fmin, fmax, n_filters + 2)

    filterbank = np.zeros((n_filters, len(fft_freqs)))

    for i in range(n_filters):
        lo, center, hi = edges[i], edges[i + 1], edges[i + 2]

        rising = (fft_freqs - lo) / (center - lo)
        falling = (hi - fft_freqs) / (hi - center)

        filterbank[i] = np.clip(np.minimum(rising, falling), 0.0, None)

    return filterbank


def lfcc(
    y: np.ndarray,
    sr: int,
    n_fft: int = 2048,
    hop_length: int = 512,
    n_filters: int = 26,
    n_lfcc: int = 20,
) -> SpectralData:
    """Linear Frequency Cepstral Coefficients.

    |STFT|^2 -> линейный треугольный фильтр-банк -> log -> DCT-II.

    ИСПРАВЛЕНО относительно исходного кода: там "фильтр-банк" был
    единичной матрицей (np.eye(len(freqs))[:n_lfcc]) — это НЕ
    фильтр-банк, а просто выбор первых n_lfcc сырых FFT-бинов без
    какого-либо взвешивания или суммирования по соседним бинам.
    Здесь — настоящий треугольный линейный фильтр-банк (n_filters
    перекрывающихся треугольников, равномерных по герцам), как и
    полагается по определению LFCC.

    freqs в результате — номер коэффициента (0..n_lfcc-1), НЕ герцы:
    DCT перемешивает информацию по всему диапазону фильтр-банка в
    каждый коэффициент, привязать коэффициент к одной частоте нельзя.

    Источник: Sahidullah, M., Kinnunen, T., & Hanilçi, C. (2015). "A
    comparison of features for synthetic speech detection." Interspeech
    2015 — там же показано, что LFCC чаще превосходит MFCC именно в
    задаче различения синтетической и настоящей речи.
    """

    X = _stft(y, n_fft, hop_length)
    power = np.abs(X) ** 2

    filterbank = _linear_filterbank(sr, n_fft, n_filters)
    filtered = filterbank @ power
    log_filtered = np.log(filtered + 1e-10)

    coeffs = dct(log_filtered, axis=0, norm="ortho")[:n_lfcc]

    return SpectralData(matrix=coeffs, freqs=np.arange(n_lfcc), sr=sr)


def mfcc(
    y: np.ndarray,
    sr: int,
    n_fft: int = 2048,
    hop_length: int = 512,
    n_mfcc: int = 20,
) -> SpectralData:
    """Mel Frequency Cepstral Coefficients.

    Используем проверенную реализацию librosa (не переизобретаем
    mel-фильтр-банк вручную, в отличие от LFCC/CQCC — готовой
    корректной реализации под рукой не было, а для MFCC она есть и
    хорошо протестирована сообществом).

    freqs в результате — номер коэффициента (0..n_mfcc-1), НЕ герцы
    (см. пояснение в lfcc()).

    Источник: Davis, S., & Mermelstein, P. (1980). "Comparison of
    parametric representations for monosyllabic word recognition in
    continuously spoken sentences." IEEE Trans. ASSP, 28(4).
    """

    coeffs = librosa.feature.mfcc(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mfcc=n_mfcc,
    )

    return SpectralData(matrix=coeffs, freqs=np.arange(n_mfcc), sr=sr)


def cqcc(
    y: np.ndarray,
    sr: int,
    n_fft: int = 2048,
    hop_length: int = 512,
    n_cqcc: int = 20,
    bins_per_octave: int = 96,
    n_octaves: int = 8,
) -> SpectralData:
    """Constant-Q Cepstral Coefficients — через библиотеку spafe.

    ВАЖНО (после сверки с официальной эталонной MATLAB-реализацией
    авторов, публично доступной на asvspoof.org): первая версия этой
    функции в данном модуле была не просто без uniform resampling —
    она ещё и использовала неверные параметры относительно оригинала
    Todisco et al. (2016):
      - оригинал использует B=96 бинов на октаву (было 24);
      - оригинал берёт log ОТ МОЩНОСТИ log(|CQT|^2), а не от амплитуды
        log(|CQT|);
      - uniform resampling там - не простая линейная интерполяция, а
        специфичный для MATLAB resample() с кубическим сплайном и
        anti-aliasing фильтрацией, завязанный на параметр d=16 (число
        равномерных отсчётов в первой октаве);
      - сам CQT там считается другим бэкендом (nonstationary Gabor
        frames, Schorkhuber & Klapuri 2014, свой параметр "gamma" -
        не путать с gamma из modified_group_delay).

    Аккуратно воспроизвести именно эту цепочку вручную трудоёмко и
    легко сделать неточно. Вместо этого используем spafe - отдельную,
    протестированную и описанную в рецензируемой (JOSS) статье
    реализацию CQCC. Она НЕ гарантированно побитово совпадает с
    оригинальным MATLAB-кодом (свой бэкенд CQT, свои дефолты), но это
    надёжнее самодельной реконструкции по памяти неопубликованных
    деталей алгоритма.

    ОГРАНИЧЕНИЕ: число кадров у spafe не совпадает день-в-день с
    librosa.stft (разное соглашение о паддинге/центрировании,
    расхождение на практике небольшое, единицы процентов). Для
    агрегированных по всем кадрам статистик (как здесь) это не
    критично; если позже понадобится точное покадровое совмещение с
    другими признаками (для CNN), стоит будет отдельно проверить и
    при необходимости выровнять.

    freqs в результате - номер коэффициента (0..n_cqcc-1), НЕ герцы.

    Источники:
    - Todisco, M., Delgado, H., & Evans, N. (2016). "A New Feature for
      Automatic Speaker Verification Anti-Spoofing: Constant Q
      Cepstral Coefficients." Odyssey 2016.
    - Официальный код авторов (MATLAB), см. также зеркало:
      github.com/azraelkuan/asvspoof2017, baseline/CQCC_v1.0/cqcc.m -
      использовался для сверки параметров выше.
    - Malek, A. et al. "spafe: Simplified Python Audio Features
      Extraction." Journal of Open Source Software (используемая
      здесь библиотека, pip install spafe).
    """



    coeffs = _spafe_cqcc(
        y.astype(np.float64),
        fs=sr,
        num_ceps=n_cqcc,
        window=SlidingWindow(n_fft / sr, hop_length / sr, "hanning"),
        number_of_bins_per_octave=bins_per_octave,
        number_of_octaves=n_octaves,
    )

    # spafe возвращает (n_frames, n_ceps) - приводим к принятой в этом
    # модуле конвенции (n_ceps, n_frames), как у остальных функций.
    coeffs = coeffs.T

    return SpectralData(matrix=coeffs, freqs=np.arange(n_cqcc), sr=sr)



# ============================================================
# 3) ВРЕМЕННЫЕ РЯДЫ БЕЗ ЧАСТОТНОЙ ОСИ
# ============================================================

def rms(
    y: np.ndarray,
    frame_length: int = 2048,
    hop_length: int = 512,
) -> np.ndarray:
    """RMS-энергия по кадрам. Без изменений от исходной версии — она
    и была математически корректна.
    """

    return librosa.feature.rms(
        y=y, frame_length=frame_length, hop_length=hop_length,
    )[0]


def crest_factor(
    y: np.ndarray,
    frame_length: int = 2048,
    hop_length: int = 512,
) -> np.ndarray:
    """Crest factor = peak(кадр) / RMS(кадр), по кадрам.

    ИСПРАВЛЕНО относительно исходного "DR": там использовался
    ГЛОБАЛЬНЫЙ максимум по всему сегменту минус RMS кадра — из-за
    этого DR(кадр) вырождался в линейную функцию от RMS(кадр) (те же
    std/skew/kurtosis с точностью до знака), не измеряя транзиенты,
    как было заявлено. Здесь пик — локальный, свой для каждого кадра.

    "Crest factor" — общее понятие аудиоинженерии (отношение пика к
    RMS), устоявшегося единственного источника формулы нет.
    """

    y_padded = _center_pad(y, frame_length)
    frames = librosa.util.frame(
        y_padded, frame_length=frame_length, hop_length=hop_length,
    )

    peak = np.max(np.abs(frames), axis=0)
    frame_rms = np.sqrt(np.mean(frames**2, axis=0))

    return peak / (frame_rms + 1e-10)