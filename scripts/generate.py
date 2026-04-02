import os
import torch
import numpy as np
import scipy.io.wavfile as wavfile
import librosa
from diffusers import StableAudioPipeline
from transformers import MusicgenForConditionalGeneration, MusicgenProcessor

# --- НАСТРОЙКИ ---
TARGET_GB_PER_MODEL = 1.0  
TRACK_DURATION_SEC = 47    
OUTPUT_DIR = "AI_Dataset"
GENRES = ["Rock", "Electronic", "Jazz", "Classical", "Hip-Hop", "Ambient", "Pop", "Metal"]

device = "cuda" if torch.cuda.is_available() else "cpu"
# На Python 3.13 и новых GPU лучше использовать float16 для экономии памяти
torch_dtype = torch.float16 if device == "cuda" else torch.float32

def save_wav(audio_data, rate, folder, filename):
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, filename)
    # Нормализация
    audio_max = np.abs(audio_data).max()
    if audio_max > 0:
        audio_data = audio_data / audio_max
    audio_int16 = (audio_data * 32767).astype(np.int16)
    wavfile.write(path, rate, audio_int16.T if audio_int16.shape[0] == 2 else audio_int16)

# --- МОДЕЛЬ 1: STABLE AUDIO OPEN (44.1 kHz) ---
def generate_stable_audio(num_tracks):
    print(f"--- Запуск Stable Audio Open ({num_tracks} треков) ---")
    pipe = StableAudioPipeline.from_pretrained("stabilityai/stable-audio-open-1.0", torch_dtype=torch_dtype, token="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")
    pipe.to(device)
    
    for i in range(num_tracks):
        genre = GENRES[i % len(GENRES)]
        prompt = f"High quality {genre} track, professional mastering, stereo, 44.1kHz"
        output = pipe(prompt, audio_end_in_s=TRACK_DURATION_SEC).audios[0] 
        save_wav(output.cpu().numpy(), 44100, f"{OUTPUT_DIR}/stable_audio", f"track_{i:04d}.wav")
    
    del pipe
    torch.cuda.empty_cache()

# --- МОДЕЛЬ 2: MUSICGEN (через Transformers) ---
def generate_musicgen_hf(num_tracks):
    print(f"--- Запуск MusicGen ({num_tracks} треков) ---")
    model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-medium", token="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")
    processor = MusicgenProcessor.from_pretrained("facebook/musicgen-medium", token="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")
    model.to(device)
    
    # MusicGen генерирует в 32кГц
    sampling_rate = model.config.audio_encoder.sampling_rate # обычно 32000
    max_tokens = int(TRACK_DURATION_SEC * 50) # 50 токенов в сек

    for i in range(num_tracks):
        genre = GENRES[i % len(GENRES)]
        inputs = processor(text=[f"Professional {genre} studio recording"], padding=True, return_tensors="pt").to(device)
        
        audio_values = model.generate(**inputs, max_new_tokens=max_tokens)
        audio_data = audio_values[0].cpu().numpy() # [channels, samples]

        # Ресемплинг в 44.1 кГц для консистентности датасета
        resampled = librosa.resample(audio_data, orig_sr=sampling_rate, target_sr=44100)
        save_wav(resampled, 44100, f"{OUTPUT_DIR}/musicgen", f"track_{i:04d}.wav")

if __name__ == "__main__":
    # Считаем количество треков
    BYTES_PER_SEC = 44100 * 2 * 2 
    TOTAL_BYTES = TARGET_GB_PER_MODEL * (1024**3)
    num_tracks = int(TOTAL_BYTES / (BYTES_PER_SEC * TRACK_DURATION_SEC))

    #generate_stable_audio(num_tracks)
    generate_musicgen_hf(num_tracks)