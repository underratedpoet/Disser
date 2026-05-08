import os
import torch
from tqdm import tqdm

FEAT_PATH = r"D:\Study\NIR\Project\features_db"

for category in ['real', 'neuro']:
    cat_dir = os.path.join(FEAT_PATH, category)
    if not os.path.exists(cat_dir): continue
    
    print(f"Проверка {category}...")
    for f in tqdm(os.listdir(cat_dir)):
        file_path = os.path.join(cat_dir, f)
        
        # 1. Удаляем если файл пустой
        if os.path.getsize(file_path) == 0:
            print(f"Удален пустой файл: {f}")
            os.remove(file_path)
            continue
            
        # 2. Пробуем загрузить (проверка на повреждение)
        try:
            _ = torch.load(file_path, weights_only=False)
        except Exception:
            print(f"Удален поврежденный файл: {f}")
            os.remove(file_path)