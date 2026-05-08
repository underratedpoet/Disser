import numpy as np
import librosa
import scipy.stats
from scipy.fftpack import dct
import torch


# =========================
# BASIC STATS
# =========================
def compute_stats(x):
    return {
        "mean": np.mean(x),
        "std": np.std(x),
        "skew": scipy.stats.skew(x),
        "kurtosis": scipy.stats.kurtosis(x)
    }


# =========================
# LOG FREQUENCY BANDS
# =========================
def get_log_bands(sr, n_fft, n_bands=40, fmin=20, fmax=None):
    if fmax is None:
        fmax = sr // 2

    freqs = np.linspace(0, sr/2, n_fft//2 + 1)
    log_edges = np.logspace(np.log10(fmin), np.log10(fmax), n_bands+1)

    bands = []
    for i in range(n_bands):
        idx = np.where((freqs >= log_edges[i]) & (freqs < log_edges[i+1]))[0]
        bands.append(idx)

    return bands


# =========================
# BAND AGGREGATION
# =========================
def band_features(matrix, bands):
    feats = {}
    for i, band in enumerate(bands):
        if len(band) == 0:
            continue
        values = matrix[band, :].flatten()
        stats = compute_stats(values)
        for k, v in stats.items():
            feats[f"band_{i}_{k}"] = v
    return feats


# =========================
# LFCC
# =========================
def compute_lfcc(y, sr, n_fft=2048, n_lfcc=20):
    S = np.abs(librosa.stft(y, n_fft=n_fft))**2

    # linear filterbank
    freqs = np.linspace(0, sr/2, n_fft//2 + 1)
    filters = np.eye(len(freqs))[:n_lfcc]

    lfcc = np.dot(filters, S)
    lfcc = np.log(lfcc + 1e-10)
    lfcc = dct(lfcc, axis=0, norm='ortho')[:n_lfcc]

    return lfcc


# =========================
# CQCC
# =========================
def compute_cqcc(y, sr, n_cqcc=20):
    C = np.abs(librosa.cqt(y, sr=sr))
    C = np.log(C + 1e-10)
    cqcc = dct(C, axis=0, norm='ortho')[:n_cqcc]
    return cqcc


# =========================
# MGD (simplified)
# =========================
def compute_mgd(y, sr, n_fft=2048):
    X = librosa.stft(y, n_fft=n_fft)
    real = np.real(X)
    imag = np.imag(X)

    numerator = real * np.gradient(imag, axis=1) - imag * np.gradient(real, axis=1)
    denominator = real**2 + imag**2 + 1e-10

    mgd = numerator / denominator
    return mgd


# =========================
# RMS + DYNAMIC RANGE
# =========================
def compute_energy(y):
    rms = np.sqrt(np.mean(y**2))
    dr = np.max(y) - np.min(y)
    return rms, dr


def compute_interchannel_phase_diff(y, n_fft=2048, hop_length=512):
    # STFT для обоих каналов
    stft_0 = librosa.stft(y[0], n_fft=n_fft, hop_length=hop_length)
    stft_1 = librosa.stft(y[1], n_fft=n_fft, hop_length=hop_length)
    
    # Разница фаз
    phase_diff = np.angle(stft_0) - np.angle(stft_1)
    # Приводим к диапазону [-pi, pi]
    phase_diff = np.mod(phase_diff + np.pi, 2 * np.pi) - np.pi
    return phase_diff

def compute_temporal_connectivity(matrix):
    """
    Вычисляет корреляцию между соседними кадрами во времени.
    matrix: (freq_bins, frames)
    """
    # Корреляция между кадром i и i+1
    correlations = []
    for i in range(matrix.shape[1] - 1):
        col_t = matrix[:, i]
        col_t_plus_1 = matrix[:, i+1]
        
        # Если кадр пустой (тишина), корреляция 1 (они идентичны)
        if np.std(col_t) == 0 or np.std(col_t_plus_1) == 0:
            correlations.append(1.0)
        else:
            corr = np.corrcoef(col_t, col_t_plus_1)[0, 1]
            correlations.append(corr)
            
    return np.array(correlations)

def extract_features_40(path, target_frames=2584):
    # Загружаем стерео
    y, sr = librosa.load(path, sr=44100, mono=False)
    if y.ndim == 1: y = np.vstack([y, y])
    
    n_fft = 2048
    hop_length = 512
    bands = get_log_bands(sr, n_fft, n_bands=40)
    
    feature_matrix = []

    def get_band_series(matrix, band_idx):
        idx = bands[band_idx]
        return np.mean(matrix[idx, :], axis=0)

    for ch in [0, 1]:
        # 1. MGD (самые важные полосы из твоего ТОП-20) - 8 признаков
        mgd = compute_mgd(y[ch], sr, n_fft=n_fft)
        for b in [39, 38, 37, 33, 32, 31, 30, 28]: 
            feature_matrix.append(get_band_series(mgd, b))
        
        # 2. CQCC (коэффициенты 1 и 2) - 2 признака
        cqcc = compute_cqcc(y[ch], sr, n_cqcc=3)
        feature_matrix.append(cqcc[1, :])
        feature_matrix.append(cqcc[2, :])
        
        # 3. Энергия и Динамика - 2 признака
        rms = librosa.feature.rms(y=y[ch], hop_length=hop_length)[0]
        feature_matrix.append(rms)
        # Прокси для DR
        feature_matrix.append(np.abs(y[ch].max() - rms * np.ones_like(rms))) 

        # 4. Спектральные компоненты - 4 признака
        stft = librosa.stft(y[ch], n_fft=n_fft, hop_length=hop_length)
        mag = np.abs(stft)
        feature_matrix.append(get_band_series(mag, 39))
        feature_matrix.append(get_band_series(np.real(stft), 11))
        feature_matrix.append(get_band_series(np.real(stft), 0))
        feature_matrix.append(get_band_series(np.imag(stft), 8))
        
        # 5. Дополнительные полосы MGD для ровного счета - ТЕПЕРЬ 4 признака
        # Добавил полосу 29, чтобы в сумме было 20 на канал
        for b in [36, 35, 34, 29]:
            feature_matrix.append(get_band_series(mgd, b))

    # Итого: (8 + 2 + 2 + 4 + 4) * 2 канала = 40 признаков
    final_feats = np.array(feature_matrix)
    
    # Нормализация длины (делаем только если target_frames передан)
    if target_frames is not None:
        if final_feats.shape[1] < target_frames:
            final_feats = np.pad(final_feats, ((0, 0), (0, target_frames - final_feats.shape[1])), mode='constant')
        else:
            final_feats = final_feats[:, :target_frames]

    return torch.FloatTensor(final_feats)