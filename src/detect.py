import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
import os

# ==========================================
# 1. АРХИТЕКТУРА НЕЙРОСЕТИ (DeepRawNet)
# ==========================================
# Это "тело" нашей модели. Оно должно быть точно таким же, как при обучении.

class ResBlock1d(nn.Module):
    """Остаточный блок: помогает сигналу проходить через глубокие слои без потерь."""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)

class DeepRawNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Начальный слой: сжимаем сырой звук
        self.stem = nn.Sequential(
            nn.Conv1d(2, 64, kernel_size=128, stride=32),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        # Глубокие слои масштабирования
        self.layer1 = self._make_layer(64, 128)
        self.layer2 = self._make_layer(128, 256)
        self.layer3 = self._make_layer(256, 512)
        
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def _make_layer(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(),
            ResBlock1d(out_ch),
            ResBlock1d(out_ch)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        return self.classifier(x)

# ==========================================
# 2. КЛАСС-ДЕТЕКТОР ДЛЯ ПРОВЕРКИ ФАЙЛОВ
# ==========================================

class SimpleAudioDetector:
    def __init__(self, weights_path):
        # Выбираем видеокарту, если она есть
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DeepRawNet().to(self.device)
        
        # Загружаем твои веса
        if os.path.exists(weights_path):
            self.model.load_state_dict(torch.load(weights_path, map_location=self.device, weights_only=True))
            self.model.eval() # Режим проверки (выключает Dropout)
            print(f"[*] Модель готова. Используем: {self.device}")
        else:
            raise FileNotFoundError(f"Файл {weights_path} не найден!")

    def analyze_file(self, file_path, threshold=0.8):
        """Разбивает файл на части по 30 сек и считает среднее."""
        
        # 1. Загрузка
        waveform, sr = torchaudio.load(file_path)
        
        # 2. Ресемплинг до 44100 (если нужно)
        if sr != 44100:
            resampler = torchaudio.transforms.Resample(sr, 44100)
            waveform = resampler(waveform)

        # 3. Приведение к стерео
        if waveform.shape[0] == 1:
            waveform = waveform.repeat(2, 1)
        elif waveform.shape[0] > 2:
            waveform = waveform[:2, :]

        # 4. Нарезка на блоки по 30 секунд
        seconds_per_block = 30
        samples_per_block = 44100 * seconds_per_block
        total_samples = waveform.shape[1]
        
        num_blocks = total_samples // samples_per_block
        
        if num_blocks == 0:
            return "Файл слишком короткий (меньше 30 сек)", 0

        all_probs = []

        # 5. Цикл проверки каждого блока
        with torch.no_grad(): # Отключаем расчет градиентов для скорости
            for i in range(num_blocks):
                start = i * samples_per_block
                end = start + samples_per_block
                
                # Вырезаем кусок
                chunk = waveform[:, start:end].unsqueeze(0).to(self.device)
                
                # Получаем ответ от нейросети
                prob = self.model(chunk).item()
                all_probs.append(prob)
                print(f"  > Блок {i+1}: {prob:.4f}")

        # 6. Считаем среднее арифметическое
        final_score = sum(all_probs) / len(all_probs)
        
        # 7. Финальный вердикт
        verdict = "NEURO (AI)" if final_score >= threshold else "REAL (HUMAN)"
        
        return verdict, final_score

# ==========================================
# 3. ЗАПУСК
# ==========================================
if __name__ == "__main__":
    # Укажи имя файла со своими весами
    MY_WEIGHTS = "raw_model_weights.pth"
    
    # Создаем детектор
    detector = SimpleAudioDetector(MY_WEIGHTS)
    
    # Файл для проверки (измени на свой)
    target_audio = "test_real_4.flac"
    
    if os.path.exists(target_audio):
        print(f"\nАнализируем: {target_audio}")
        result, score = detector.analyze_file(target_audio)
        
        print("-" * 30)
        print(f"ИТОГ: {result}")
        print(f"Средняя уверенность: {score:.4f}")
        print("-" * 30)
    else:
        print(f"Файл {target_audio} не найден. Положи его в папку со скриптом.")