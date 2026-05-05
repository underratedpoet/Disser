import pandas as pd
import os
import numpy as np
import librosa
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from explainable.feature_utils import * # Твои функции здесь

# Настройки те же
SR = 44100
N_FFT = 2048
BANDS = get_log_bands(SR, N_FFT, 40)

def patch_batch(batch_info):
    batch_path, audio_folder = batch_info
    batch_name = os.path.basename(batch_path)
    
    try:
        print(f"[START] {batch_name} - начало обработки")
        df = pd.read_parquet(batch_path)
        new_data = []

        for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"{batch_name}", leave=False):
            file_path = os.path.join(audio_folder, row['file'])
            
            try:
                y, _ = librosa.load(file_path, sr=SR, mono=False)
                if y.ndim == 1: y = np.vstack([y, y])
                
                feats = {}
                
                # 1. Разница фаз (ICPD) по полосам
                phase_diff_matrix = compute_interchannel_phase_diff(y)
                icpd_feats = band_features(phase_diff_matrix, BANDS)
                for k, v in icpd_feats.items():
                    feats[f"interchannel_phase_{k}"] = v
                
                # 2. Связность кадров (Temporal Connectivity) по полосам
                # Считаем для магнитуды первого канала (как пример)
                mag = np.abs(librosa.stft(y[0], n_fft=N_FFT))
                
                # Считаем корреляцию для каждой полосы отдельно
                for i, band in enumerate(BANDS):
                    if len(band) == 0: continue
                    band_data = mag[band, :] # Данные полосы во времени
                    connectivity = compute_temporal_connectivity(band_data)
                    
                    # Агрегируем связность (насколько стабильна корреляция)
                    stats = compute_stats(connectivity)
                    for k, v in stats.items():
                        feats[f"connectivity_band_{i}_{k}"] = v
                
                new_data.append(feats)
                
            except Exception as e:
                print(f"[ERROR] {batch_name}: {row['file']} - {e}")
                new_data.append({k: np.nan for k in feats.keys()} if 'feats' in locals() else {})

        # Создаем DF с новыми признаками и объединяем
        df_new = pd.DataFrame(new_data)
        # Важно! Используем индексы из исходного датафрейма
        df_new.index = df.index
        df_final = pd.concat([df, df_new], axis=1)
        
        # Перезаписываем батч
        df_final.to_parquet(batch_path, engine="pyarrow", compression="snappy")
        
        result = {
            'batch': batch_name,
            'status': 'OK',
            'rows': len(df),
            'features': len(df_new.columns)
        }
        print(f"[DONE] {batch_name} - ✓ сохранено {len(df)} записей с {len(df_new.columns)} признаками")
        return result
        
    except Exception as e:
        result = {
            'batch': batch_name,
            'status': 'ERROR',
            'error': str(e)
        }
        print(f"[FAIL] {batch_name} - ошибка: {e}")
        return result

# Запуск патча для одной из папок (пример)
if __name__ == "__main__":
    print("="*70)
    print("НАЧАЛО РАСПАРАЛЛЕЛЕННОЙ ОБРАБОТКИ БАТЧЕЙ")
    print("="*70)
    
    # Получаем список всех батчей
    batch_dir = "features_real_data"
    batches = sorted([f for f in os.listdir(batch_dir) if f.endswith(".parquet")])
    
    # Для тестирования - берем только первый батч
    test_batches = batches[:1]  # Измени это число для обработки большего количества батчей
    
    batch_paths = [(os.path.join(batch_dir, batch), "real") for batch in test_batches]
    
    print(f"[INFO] Найдено батчей для обработки: {len(batch_paths)}")
    print(f"[INFO] Использую {max(1, cpu_count() - 1)} процессов")
    print("="*70 + "\n")
    
    # Параллельная обработка батчей
    n_workers = max(1, cpu_count() - 1)
    
    with Pool(processes=n_workers) as pool:
        results = list(tqdm(
            pool.imap_unordered(patch_batch, batch_paths),
            total=len(batch_paths),
            desc="Общий прогресс",
            position=0
        ))
    
    print("\n" + "="*70)
    print("ИТОГИ ОБРАБОТКИ:")
    print("="*70)
    
    success_count = sum(1 for r in results if r['status'] == 'OK')
    error_count = sum(1 for r in results if r['status'] == 'ERROR')
    
    for result in results:
        if result['status'] == 'OK':
            print(f"✓ {result['batch']}: {result['rows']} записей, {result['features']} признаков")
        else:
            print(f"✗ {result['batch']}: {result['error']}")
    
    print("="*70)
    print(f"[SUMMARY] Успешно: {success_count}, Ошибок: {error_count}")
    print("="*70)