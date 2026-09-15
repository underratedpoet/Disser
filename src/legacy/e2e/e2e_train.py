import os
import json
import random
import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
import soundfile as sf
from torch.utils.data import Dataset, DataLoader

os.environ["PATH"] += os.pathsep + r"D:\\Soft\\ffmpeg-8.1-full_build\\ffmpeg-8.1-full_build\\bin"
try:
    torchaudio.set_audio_backend("soundfile")
except Exception:
    pass

# ==========================================
# 1. МЕНЕДЖЕР ДАННЫХ (Остался без изменений)
# ==========================================
class DataManager:
    def __init__(self, real_dir, neuro_dir, state_file="dataset_state.json"):
        self.real_dir = real_dir
        self.neuro_dir = neuro_dir
        self.state_file = state_file
        self.db = self._load_or_create_db()

    def _load_or_create_db(self):
        if os.path.exists(self.state_file):
            with open(self.state_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        
        db = {"real": {}, "neuro": {}}
        for folder, label in [(self.real_dir, "real"), (self.neuro_dir, "neuro")]:
            if not os.path.exists(folder):
                continue
            for filename in os.listdir(folder):
                if filename.endswith(".flac"):
                    path = os.path.join(folder, filename)
                    db[label][path] = "unused"
        self._save_db(db)
        return db

    def _save_db(self, db_state=None):
        if db_state is None:
            db_state = self.db
        with open(self.state_file, 'w', encoding='utf-8') as f:
            json.dump(db_state, f, indent=4)

    def get_balanced_training_files(self, count_per_class):
        real_unused = [path for path, status in self.db["real"].items() if status == "unused"]
        neuro_unused = [path for path, status in self.db["neuro"].items() if status == "unused"]

        actual_count = min(count_per_class, len(real_unused), len(neuro_unused))
        if actual_count == 0:
            print("Нет доступных неиспользованных файлов для обучения!")
            return [], []

        selected_real = random.sample(real_unused, actual_count)
        selected_neuro = random.sample(neuro_unused, actual_count)
        return selected_real, selected_neuro

    def mark_as_trained(self, file_paths):
        for path in file_paths:
            if path in self.db["real"]:
                self.db["real"][path] = "trained"
            elif path in self.db["neuro"]:
                self.db["neuro"][path] = "trained"
        self._save_db()

# ==========================================
# 2. ЗАГРУЗЧИК СЫРЫХ АУДИОДАННЫХ
# ==========================================
class RawAudioDataset(Dataset):
    def __init__(self, real_files, neuro_files, sample_rate=44100, crop_seconds=4):
        self.files = real_files + neuro_files
        self.labels = [0] * len(real_files) + [1] * len(neuro_files)
        self.sample_rate = sample_rate
        # Считаем, сколько отсчетов (чисел) будет в нашем "срезe"
        self.crop_samples = sample_rate * crop_seconds

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # Загружаем сырое аудио напрямую
        waveform, sr = torchaudio.load(self.files[idx])
        
        # Если частота не 44100, приводим к ней
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)

        # ТРЕБОВАНИЕ: Оставляем стерео (2 канала)
        # Если захочешь перевести в моно, раскомментируй строку ниже:
        # waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Защита: если файл вдруг оказался моно, а мы ждем стерео, дублируем канал
        if waveform.shape[0] == 1:
             waveform = waveform.repeat(2, 1)

        # Вырезаем случайный кусок аудио (чтобы влезло в память видеокарты)
        audio_length = waveform.shape[1]
        if audio_length > self.crop_samples:
            # Выбираем случайную стартовую точку
            start = random.randint(0, audio_length - self.crop_samples)
            waveform = waveform[:, start : start + self.crop_samples]
        elif audio_length < self.crop_samples:
            # Если файл короче (что вряд ли, но защита нужна), добиваем нулями
            padding = self.crop_samples - audio_length
            waveform = torch.nn.functional.pad(waveform, (0, padding))

        return waveform, self.labels[idx]

# ==========================================
# 3. END-TO-END НЕЙРОСЕТЬ ДЛЯ СЫРОГО ЗВУКА
# ==========================================
class ResBlock1d(nn.Module):
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
        out += residual # Та самая "магия" глубоких сетей
        return F.relu(out)

class DeepRawNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Начальное сжатие: 2 канала стерео -> 64 признака
        self.stem = nn.Sequential(
            nn.Conv1d(2, 64, kernel_size=128, stride=32),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        
        # Основные блоки обработки (Масштабирование)
        self.layer1 = self._make_layer(64, 128)
        self.layer2 = self._make_layer(128, 256)
        self.layer3 = self._make_layer(256, 512)
        
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        
        # Мощный классификатор
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
            ResBlock1d(out_ch), # Добавляем глубину через Residual Block
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
# 4. ФУНКЦИЯ ОБУЧЕНИЯ
# ==========================================
def train_chunk(real_dir, neuro_dir, files_per_class=100, epochs=5, batch_size=16):
    print("Инициализация менеджера данных...")
    db_manager = DataManager(real_dir, neuro_dir)
    
    print(f"Выбираем по {files_per_class} файлов каждого класса...")
    real_files, neuro_files = db_manager.get_balanced_training_files(files_per_class)
    
    if not real_files:
        print("Обучение завершено, больше нет неиспользованных данных!")
        return

    # Создаем датасет (вырезаем по 4 секунды)
    dataset = RawAudioDataset(real_files, neuro_files, sample_rate=44100, crop_seconds=30)
    # Загрузчик с параметром batch_size. Для 8 ГБ видеопамяти 16 - хороший старт
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Принудительно ищем видеокарту NVIDIA (cuda)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Используемое устройство: {device}")
    
    model = DeepRawNet().to(device)
    
    # Загрузка весов для дообучения
    if os.path.exists("raw_model_weights.pth"):
        model.load_state_dict(torch.load("raw_model_weights.pth"))
        print("Найдены предыдущие веса. Продолжаем дообучение.")

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    criterion = nn.BCELoss()

    print(f"Начинаем обучение на {len(real_files) + len(neuro_files)} файлах ({epochs} эпох)...")
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        correct_predictions = 0
        total_samples = 0
        
        for waveforms, labels in dataloader:
            # Переносим данные на видеокарту
            waveforms = waveforms.to(device)
            labels = labels.float().unsqueeze(1).to(device)
            
            # Обучение
            optimizer.zero_grad()
            outputs = model(waveforms)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # Статистика
            total_loss += loss.item()
            # Если вероятность > 0.8, считаем, что нейросеть предсказала класс 1 (neuro)
            predictions = (outputs > 0.8).float()
            correct_predictions += (predictions == labels).sum().item()
            total_samples += labels.size(0)
            
        avg_loss = total_loss / len(dataloader)
        accuracy = (correct_predictions / total_samples) * 100
        print(f"Эпоха {epoch+1}/{epochs} | Ошибка (Loss): {avg_loss:.4f} | Точность (Accuracy): {accuracy:.2f}%")

    # Сохраняем модель
    torch.save(model.state_dict(), "raw_model_weights.pth")
    print("Модель сохранена.")
    
    # Отмечаем файлы
    db_manager.mark_as_trained(real_files + neuro_files)
    print("Файлы отмечены как использованные.\n")

# Пример запуска:
if __name__ == "__main__":
    # Укажи свои пути к папкам с реальными и нейросетевыми аудио
    REAL_DIR = r"D:\Study\NIR\Project\real"
    NEURO_DIR = r"D:\Study\NIR\Project\neuro"
    
    # Запускаем обучение. Можно повторять этот вызов, пока не закончится весь датасет.
    train_chunk(REAL_DIR, NEURO_DIR, files_per_class=5000, epochs=10, batch_size=16)
# train_chunk("путь/к/real", "путь/к/neuro", files_per_class=50, epochs=10, batch_size=16)

# 41K