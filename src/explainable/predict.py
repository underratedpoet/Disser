import torch
import librosa
import numpy as np
from model import AudioTCN
from train import extract_top_20_features, MODEL_SAVE_PATH

def analyze_audio(file_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AudioTCN().to(device)
    
    checkpoint = torch.load(MODEL_SAVE_PATH, map_location=device)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()

    # Извлекаем признаки
    features = extract_top_20_features(file_path).unsqueeze(0).to(device)
    
    with torch.no_grad():
        verdict, timeline = model(features)
    
    score = verdict.item()
    timeline_np = timeline.squeeze().cpu().numpy()
    
    print(f"\nФайл: {file_path}")
    print(f"Вероятность нейросети: {score:.4f}")
    print(f"ВЕРДИКТ: {'NEURO (AI)' if score > 0.5 else 'REAL (HUMAN)'}")
    
    # Поиск сомнительных мест (где локальная вероятность > 0.7)
    print("\n--- Анализ временной шкалы ---")
    suspicious_zones = np.where(timeline_np > 0.7)[0]
    
    if len(suspicious_zones) == 0:
        print("Аномалий не обнаружено. Сигнал чист.")
    else:
        # Примерный расчет времени (зависит от количества MaxPool слоев в модели)
        # В нашей модели 3 слоя MaxPool по 2, итого сжатие в 8 раз.
        time_step = (512 * 8) / 44100 
        for idx in suspicious_zones:
            start_t = idx * time_step
            print(f"[!] Подозревается подделка: {start_t:.2f} сек. (Уверенность: {timeline_np[idx]:.2f})")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        analyze_audio(sys.argv[1])
    else:
        print("Использование: python predict.py your_audio_file.flac")