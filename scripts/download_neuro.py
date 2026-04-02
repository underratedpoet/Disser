import os
import requests
from datasets import load_dataset, Value  # Импортируем Value для "обмана" типов
from tqdm import tqdm
from huggingface_hub import login

# --- НАСТРОЙКИ ---
HF_TOKEN = "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" 
NUM_SONGS = 14000 
OUTPUT_DIR = "AI_Music_Dataset"

def download_suno():
    model_name = "Suno"
    repo_id = "nyuuzyou/suno"
    folder = os.path.join(OUTPUT_DIR, model_name)
    os.makedirs(folder, exist_ok=True)
    
    print(f"\n🔗 Подключение к {model_name}...")
    ds = load_dataset(repo_id, split="train", streaming=True)
    
    count = 0
    downloaded_urls = set() # Чтобы не качать дубликаты
    pbar = tqdm(total=NUM_SONGS, desc="Suno Download")
    
    for item in ds:
        if count >= NUM_SONGS: break
        
        url = item.get('audio_url')
        if url and url not in downloaded_urls:
            try:
                r = requests.get(url, timeout=15)
                if r.status_code == 200:
                    file_path = os.path.join(folder, f"suno_{count:04d}.mp3")
                    with open(file_path, 'wb') as f:
                        f.write(r.content)
                    downloaded_urls.add(url)
                    count += 1
                    pbar.update(1)
            except Exception:
                continue
    pbar.close()

def download_udio():
    model_name = "Udio"
    repo_id = "blanchon/udio_dataset"
    folder = os.path.join(OUTPUT_DIR, model_name)
    os.makedirs(folder, exist_ok=True)
    
    print(f"\n📦 Подключение к {model_name}...")
    
    try:
        ds = load_dataset(repo_id, split="train", streaming=True, token=HF_TOKEN)
        
        # СУПЕР-ХАК: Принудительно меняем тип колонки на двоичный (Binary).
        # Теперь datasets НЕ будет пытаться грузить torchcodec.
        ds = ds.cast_column("mp3", Value("binary"))
        
        count = 0
        pbar = tqdm(total=NUM_SONGS, desc="Udio Download")
        for item in ds:
            if count >= NUM_SONGS: break
            
            # Теперь в item['mp3'] лежат просто байты, а не объект Audio
            mp3_bytes = item.get('mp3')
            
            if mp3_bytes:
                try:
                    file_path = os.path.join(folder, f"udio_{count:04d}.mp3")
                    with open(file_path, 'wb') as f:
                        f.write(mp3_bytes['bytes'])
                    count += 1
                    pbar.update(1)
                except Exception as e:
                    continue
        pbar.close()
    except Exception as e:
        print(f"❌ Ошибка в блоке Udio: {e}")

if __name__ == "__main__":
    if HF_TOKEN and HF_TOKEN != "твой_токен_здесь":
        login(token=HF_TOKEN)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    download_suno()
    download_udio()
    
    print(f"\n✅ Готово! Файлы в: {os.path.abspath(OUTPUT_DIR)}")