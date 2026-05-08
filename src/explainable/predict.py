import os
import torch
import numpy as np
from model import AudioTCN
from feature_utils import extract_features_40

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "tcn_audio_model.pth"
THRESHOLD = 0.8
COMPRESSION = 16 # Сжатие времени в модели
HOP_LENGTH = 512

def get_feature_detailed_names():
    names = []
    for ch in ["L", "R"]: # Каналы 0 и 1
        # 1. MGD (8 признаков)
        for b in [39, 38, 37, 33, 32, 31, 30, 28]:
            names.append(f"Ch_{ch}: MGD Phase (Band {b})")
        # 2. CQCC (2 признака)
        names.append(f"Ch_{ch}: CQCC Coeff 1")
        names.append(f"Ch_{ch}: CQCC Coeff 2")
        # 3. Энергия (2 признака)
        names.append(f"Ch_{ch}: RMS Energy")
        names.append(f"Ch_{ch}: Dynamic Range Proxy")
        # 4. Спектральные (4 признака)
        names.append(f"Ch_{ch}: Spectrogram Mag (39)")
        names.append(f"Ch_{ch}: Spectrogram Real (11)")
        names.append(f"Ch_{ch}: Spectrogram Real (0)")
        names.append(f"Ch_{ch}: Spectrogram Imag (8)")
        # 5. Доп MGD (4 признака)
        for b in [36, 35, 34, 29]:
            names.append(f"Ch_{ch}: MGD Phase (Band {b})")
    return names
FEATURE_NAMES = get_feature_detailed_names()

def xray_analyze(file_path):
    model = AudioTCN(num_features=40).to(DEVICE)
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()

    # Извлечение с поддержкой градиентов
    features = extract_features_40(file_path, target_frames=None)
    features = features.unsqueeze(0).to(DEVICE)
    features.requires_grad = True 

    # Прямой проход
    verdict, timeline = model(features)
    
    # Обратный проход для вычисления важности признаков
    model.zero_grad()
    verdict.backward()
    
    # Saliency Map: (40, Time)
    saliency = features.grad.data.abs().squeeze(0).cpu().numpy()
    
    score = verdict.item()
    timeline_np = timeline.detach().cpu().numpy().reshape(-1)
    
    print(f"\n" + "="*60)
    print(f"ЭКСПЕРТИЗА ФАЙЛА: {os.path.basename(file_path)}")
    print(f"ОБЩАЯ ВЕРОЯТНОСТЬ ПОДДЕЛКИ: {score:.4%}")
    print(f"СТАТУС: {'[!] ФАЛЬСИФИКАЦИЯ' if score > THRESHOLD else '[+] ОРИГИНАЛ'}")
    print("="*60)

    # Расчет временного шага
    time_step = (HOP_LENGTH * COMPRESSION) / 44100
    
    print("\n>>> ОБНАРУЖЕННЫЕ АНОМАЛИИ В СТРУКТУРЕ СИГНАЛА:")
    found = False
    
    for i, prob in enumerate(timeline_np):
        if prob > 0.7: # Порог для вывода детальной инфы
            found = True
            current_time = i * time_step
            
            # Определяем временной диапазон в исходных признаках
            t_start, t_end = i * COMPRESSION, (i + 1) * COMPRESSION
            
            # Считаем суммарную важность каждого из 40 признаков в этом окне
            feat_importance = saliency[:, t_start:t_end].sum(axis=1)
            
            # Находим ТОП-3 виновных признака
            top_indices = np.argsort(feat_importance)[-3:][::-1]
            
            print(f"\n[{current_time:6.2f} сек] Тревога: {prob:.2%}")
            print(f"    Ключевые дефекты:")
            for rank, idx in enumerate(top_indices):
                importance_share = feat_importance[idx] / feat_importance.sum()
                print(f"    {rank+1}. {FEATURE_NAMES[idx]:<30} (вклад: {importance_share:.1%})")

    if not found:
        print("    Критический анализ не выявил следов вмешательства нейросетей.")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    file = r'D:\Study\NIR\Project\test_neuro_4.flac'
    xray_analyze(file)