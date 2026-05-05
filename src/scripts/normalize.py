from pydub import AudioSegment

AudioSegment.converter = r"D:\\Soft\\ffmpeg-8.1-full_build\\ffmpeg-8.1-full_build\\bin\\ffmpeg.exe"
AudioSegment.ffprobe = r"D:\\Soft\\ffmpeg-8.1-full_build\\ffmpeg-8.1-full_build\\bin\\ffprobe.exe"
import os
os.environ["PATH"] += os.pathsep + r"D:\\Soft\\ffmpeg-8.1-full_build\\ffmpeg-8.1-full_build\\bin"
import glob
from pydub import AudioSegment
from tqdm import tqdm
# --- НАСТРОЙКИ ---
# Флаг удаления оригиналов. Сначала тестируем на False!
DELETE_ORIGINAL = True 
# Исходные пути
FMA_DIR = r"D:\\Study\\NIR\\Project\\fma_large\\fma_large"
SUNO_DIR = r"D:\\Study\\NIR\\Project\\AI_Music_Dataset\\Suno"
UDIO_DIR = r"D:\\Study\\NIR\\Project\\AI_Music_Dataset\\Udio"
# Целевые пути
OUT_REAL_DIR = r"D:\\Study\\NIR\\Project\\real"
OUT_NEURO_DIR = r"D:\\Study\\NIR\\Project\\neuro"
# Целевые параметры аудио
TARGET_SR = 44100
TARGET_CHANNELS = 2
TARGET_BITRATE = "256k"
CHUNK_MS = 30 * 1000  # 30 секунд в миллисекундах

def process_file(file_path, output_dir, prefix_name):
    try:
        if not os.path.exists(file_path):
            print(f"\n⚠️ Файл не найден: {file_path}")
            return False

        audio = AudioSegment.from_mp3(file_path)

        # 💡 МОЖЕШЬ включить mono для экономии:
        # audio = audio.set_channels(1)

        audio = audio.set_frame_rate(TARGET_SR).set_channels(TARGET_CHANNELS)

        chunks_created = 0
        base_filename = os.path.splitext(os.path.basename(file_path))[0]

        for i in range(0, len(audio) - CHUNK_MS + 1, CHUNK_MS):
            chunk = audio[i : i + CHUNK_MS]

            out_name = f"{prefix_name}_{base_filename}_part{chunks_created:02d}.flac"
            out_path = os.path.join(output_dir, out_name)

            if os.path.exists(out_path) and os.path.getsize(out_path) > 1000:
                chunks_created += 1
                continue

            chunk.export(out_path, format="flac")
            chunks_created += 1

        return True

    except Exception as e:
        print(f"\n⚠️ Ошибка обработки {file_path}: {e}")
        return False
    
def process_directory(source_path, target_path, prefix, recursive=False):
    """
    Ищет все MP3 в папке (и подпапках), обрабатывает их и, 
    если DELETE_ORIGINAL == True, удаляет исходник.
    """
    os.makedirs(target_path, exist_ok=True)
    
    # Собираем все mp3 файлы
    if recursive:
        search_pattern = os.path.join(source_path, "**", "*.mp3")
        files = glob.glob(search_pattern, recursive=True)
    else:
        search_pattern = os.path.join(source_path, "*.mp3")
        files = glob.glob(search_pattern)
    if not files:
        print(f"Пустая папка или файлы не найдены: {source_path}")
        return
    print(f"\nОбработка {prefix}... Найдено файлов: {len(files)}")
    
    # Используем tqdm для прогресс-бара
    for file_path in tqdm(files, desc=f"Нормализация {prefix}"):
        success = process_file(file_path, target_path, prefix)
        
        # Удаляем оригинал ТОЛЬКО если обработка прошла без ошибок и флаг True
        if success and DELETE_ORIGINAL:
            try:
                os.remove(file_path)
            except Exception as e:
                print(f"\nНе удалось удалить оригинал {file_path}: {e}")

if __name__ == "__main__":
    print("🚀 Старт нормализации датасета...")
    print(f"Удаление оригиналов: {'ВКЛЮЧЕНО 🔴' if DELETE_ORIGINAL else 'ОТКЛЮЧЕНО 🟢'}\n")
    
    # 1. Обрабатываем REAL (fma_large разбит по подпапкам, поэтому recursive=True)
    #process_directory(FMA_DIR, OUT_REAL_DIR, prefix="real_fma", recursive=True)
    
    # 2. Обрабатываем NEURO (Suno)
    #process_directory(SUNO_DIR, OUT_NEURO_DIR, prefix="neuro_suno", recursive=False)
    
    # 3. Обрабатываем NEURO (Udio)
    process_directory(UDIO_DIR, OUT_NEURO_DIR, prefix="neuro_udio", recursive=False)
    
    print("\n✅ Все файлы успешно приведены к стандарту 44100Гц, Стерео, 256kbit/s, 30 сек.")