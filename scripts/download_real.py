import requests
import os

def download_fma(filename):
    url = f"https://os.unil.cloud.switch.ch/fma/{filename}"
    print(f"Начинаю загрузку {filename}...")
    
    # Get total size
    with requests.head(url) as r:
        r.raise_for_status()
        total_size = int(r.headers.get('content-length', 0))
    
    # Check if file exists and get current size
    if os.path.exists(filename):
        current_size = os.path.getsize(filename)
        if current_size >= total_size:
            print(f"{filename} уже полностью загружен.")
            return
        headers = {'Range': f'bytes={current_size}-'}
        mode = 'ab'
        downloaded = current_size
        print(f"Возобновляю загрузку с {current_size} байт...")
    else:
        headers = {}
        mode = 'wb'
        downloaded = 0
    
    with requests.get(url, headers=headers, stream=True) as r:
        r.raise_for_status()
        with open(filename, mode) as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        progress = (downloaded / total_size) * 100
                        print(f"\r{filename}: {progress:.2f}% завершено", end='', flush=True)
    print(f"\nГотово: {filename}")

# Скачиваем метаданные и большой пак
download_fma("fma_metadata.zip")
download_fma("fma_large.zip")