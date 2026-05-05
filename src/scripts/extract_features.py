import os
import numpy as np
import librosa
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool, Process, cpu_count
import pyarrow as pa

from explainable.feature_utils import *

# =========================
# CONFIG
# =========================
SR = 44100
N_FFT = 2048
HOP = 512
N_BANDS = 40
BATCH_SIZE = 500

# ГЛОБАЛЬНЫЙ КЭШ: Считаем полосы частот один раз для всех файлов!
# Это сэкономит кучу процессорного времени.
BANDS_CACHE = get_log_bands(SR, N_FFT, N_BANDS)

# =========================
# FEATURE EXTRACTION
# =========================
def process_file(path, label):
    try:
        y, sr = librosa.load(path, sr=SR, mono=False)

        if y.ndim == 1:
            y = np.vstack([y, y])

        # Используем предрассчитанные полосы
        bands = BANDS_CACHE

        all_feats = {}

        for ch, signal in enumerate(y):
            prefix = f"ch{ch}"

            # STFT
            X = librosa.stft(signal, n_fft=N_FFT, hop_length=HOP)
            mag = np.abs(X)
            phase = np.angle(X)

            # magnitude
            feats = band_features(mag, bands)
            for k, v in feats.items():
                all_feats[f"{prefix}_mag_{k}"] = v

            # phase
            feats = band_features(phase, bands)
            for k, v in feats.items():
                all_feats[f"{prefix}_phase_{k}"] = v

            # complex parts
            feats = band_features(np.real(X), bands)
            for k, v in feats.items():
                all_feats[f"{prefix}_real_{k}"] = v

            feats = band_features(np.imag(X), bands)
            for k, v in feats.items():
                all_feats[f"{prefix}_imag_{k}"] = v

            # MFCC
            mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=20)
            for i in range(mfcc.shape[0]):
                stats = compute_stats(mfcc[i])
                for k, v in stats.items():
                    all_feats[f"{prefix}_mfcc_{i}_{k}"] = v

            # LFCC
            lfcc = compute_lfcc(signal, sr)
            for i in range(lfcc.shape[0]):
                stats = compute_stats(lfcc[i])
                for k, v in stats.items():
                    all_feats[f"{prefix}_lfcc_{i}_{k}"] = v

            # CQCC
            cqcc = compute_cqcc(signal, sr)
            for i in range(cqcc.shape[0]):
                stats = compute_stats(cqcc[i])
                for k, v in stats.items():
                    all_feats[f"{prefix}_cqcc_{i}_{k}"] = v

            # MGD
            mgd = compute_mgd(signal, sr)
            feats = band_features(mgd, bands)
            for k, v in feats.items():
                all_feats[f"{prefix}_mgd_{k}"] = v

            # energy
            rms, dr = compute_energy(signal)
            all_feats[f"{prefix}_rms"] = rms
            all_feats[f"{prefix}_dr"] = dr

        all_feats["label"] = label
        all_feats["file"] = os.path.basename(path)

        return all_feats
    except Exception as e:
        # Если какой-то файл поврежден, скрипт не упадет
        print(f"\nОшибка при обработке {path}: {e}")
        return None

# =========================
# WRAPPER (multiprocessing)
# =========================
def process_wrapper(args):
    return process_file(*args)

# =========================
# СТРИМИНГОВАЯ ЗАПИСЬ БАТЧАМИ В ПАПКУ
# =========================
def process_folder_stream(folder, label, output_dir, n_workers=4, pos=0):
    if not os.path.exists(folder):
        print(f"Folder {folder} not found!")
        return

    # Создаем папку для сохранения батчей, если её нет
    os.makedirs(output_dir, exist_ok=True)

    files = [f for f in os.listdir(folder) if f.endswith(".flac")]
    paths = [(os.path.join(folder, f), label) for f in files]

    batch = []
    batch_idx = 0

    with Pool(processes=n_workers, maxtasksperchild=10) as pool:
        with tqdm(total=len(paths), desc=f"Processing {folder}", position=pos, leave=True) as pbar:
            # imap_unordered работает быстрее, так как не ждет завершения в строгом порядке
            for feats in pool.imap_unordered(process_wrapper, paths, chunksize=5):
                if feats is not None:
                    batch.append(feats)

                # Если накопили батч — записываем как отдельный файл
                if len(batch) >= BATCH_SIZE:
                    df_batch = pd.DataFrame(batch)
                    df_batch = df_batch.astype({"label": str, "file": str})
                    
                    # Имя файла: batch_0000.parquet, batch_0001.parquet и т.д.
                    batch_filename = os.path.join(output_dir, f"batch_{batch_idx:04d}.parquet")
                    df_batch.to_parquet(batch_filename, engine="pyarrow", compression="snappy")
                    
                    batch = [] # Очищаем батч
                    batch_idx += 1
                
                pbar.update(1)

            # Записываем остатки, если они есть
            if batch:
                df_batch = pd.DataFrame(batch)
                df_batch = df_batch.astype({"label": str, "file": str})
                batch_filename = os.path.join(output_dir, f"batch_{batch_idx:04d}.parquet")
                df_batch.to_parquet(batch_filename, engine="pyarrow", compression="snappy")

# =========================
# ЗАПУСК
# =========================
def run_parallel():
    # Теперь мы передаем не имя файла, а имя ПАПКИ для результатов (features_real_data)
    p_real = Process(
        target=process_folder_stream,
        args=("real", "real", "features_real_data", max(1, cpu_count() // 2), 0)
    )

    p_neuro = Process(
        target=process_folder_stream,
        args=("neuro", "neuro", "features_neuro_data", max(1, cpu_count() // 2), 1)
    )

    p_real.start()
    p_neuro.start()

    p_real.join()
    p_neuro.join()

    print("\nВсе задачи выполнены!")

if __name__ == "__main__":
    run_parallel()