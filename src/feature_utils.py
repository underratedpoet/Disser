import numpy as np
import librosa
import scipy.stats
from scipy.fftpack import dct


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