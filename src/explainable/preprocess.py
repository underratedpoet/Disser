import os
import torch
import numpy as np
from tqdm import tqdm
from feature_utils import extract_features_40 # Твоя исправленная функция

def preprocess_data(src_base, dst_base):
    for category in ['real', 'neuro']:
        src_dir = os.path.join(src_base, category)
        dst_dir = os.path.join(dst_base, category)
        
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)

        print(f"Обработка категории: {category}")
        files = [f for f in os.listdir(src_dir) if f.endswith('.flac')]
        
        for filename in tqdm(files):
            src_path = os.path.join(src_dir, filename)
            dst_path = os.path.join(dst_dir, filename.replace('.flac', '.pt'))
            
            if os.path.exists(dst_path): continue # Пропускаем уже готовые
            
            try:
                # Извлекаем признаки (уже тензор 40x2584)
                features = extract_features_40(src_path)
                # Сохраняем на диск
                torch.save(features, dst_path)
            except Exception as e:
                print(f"Ошибка в файле {filename}: {e}")

if __name__ == "__main__":
    SOURCE_PATH = r"D:\Study\NIR\Project" # Где лежат папки real и neuro
    DEST_PATH = r"D:\Study\NIR\Project\features_db" # Куда сохранить тензоры
    preprocess_data(SOURCE_PATH, DEST_PATH)