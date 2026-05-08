import os
import json
import random
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from model import AudioTCN

# =========================
# CONFIG
# =========================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FEAT_PATH = r"D:\Study\NIR\Project\features_db"
MODEL_SAVE_PATH = "tcn_audio_model.pth"
STATE_FILE = "dataset_state.json"
BATCH_SIZE = 128 
LEARNING_RATE = 0.0004 # Чуть ниже для более глубокой сети
THRESHOLD = 0.8 # Твой порог для "фейка"

# =========================
# 1. МЕНЕДЖЕР ДАННЫХ (для .pt файлов)
# =========================
class DataManager:
    def __init__(self, root_dir, state_file=STATE_FILE):
        self.root_dir = root_dir
        self.state_file = state_file
        self.db = self._load_or_create_db()

    def _load_or_create_db(self):
        if os.path.exists(self.state_file):
            with open(self.state_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        
        db = {"real": {}, "neuro": {}}
        for category in ["real", "neuro"]:
            cat_path = os.path.join(self.root_dir, category)
            if not os.path.exists(cat_path): continue
            for filename in os.listdir(cat_path):
                if filename.endswith(".pt"):
                    path = os.path.abspath(os.path.join(cat_path, filename))
                    db[category][path] = "unused"
        self._save_db(db)
        return db

    def _save_db(self, db_state=None):
        if db_state is None: db_state = self.db
        with open(self.state_file, 'w', encoding='utf-8') as f:
            json.dump(db_state, f, indent=4)

    def get_balanced_training_files(self, count_per_class):
        # Обновляем базу, если появились новые файлы в папке
        self._refresh_db()
        
        real_unused = [p for p, s in self.db["real"].items() if s == "unused"]
        neuro_unused = [p for p, s in self.db["neuro"].items() if s == "unused"]

        actual_count = min(count_per_class, len(real_unused), len(neuro_unused))
        if actual_count == 0: return [], []

        return random.sample(real_unused, actual_count), random.sample(neuro_unused, actual_count)

    def _refresh_db(self):
        """Проверяет папку на наличие новых файлов, которых еще нет в JSON"""
        changed = False
        for category in ["real", "neuro"]:
            cat_path = os.path.join(self.root_dir, category)
            for filename in os.listdir(cat_path):
                if filename.endswith(".pt"):
                    path = os.path.abspath(os.path.join(cat_path, filename))
                    if path not in self.db[category]:
                        self.db[category][path] = "unused"
                        changed = True
        if changed: self._save_db()

    def mark_as_trained(self, file_paths):
        for path in file_paths:
            path_abs = os.path.abspath(path)
            if path_abs in self.db["real"]: self.db["real"][path_abs] = "trained"
            elif path_abs in self.db["neuro"]: self.db["neuro"][path_abs] = "trained"
        self._save_db()

# =========================
# 2. ДАТАСЕТ
# =========================
class FeatureDataset(Dataset):
    def __init__(self, real_files, neuro_files):
        self.files = real_files + neuro_files
        self.labels = [0.0] * len(real_files) + [1.0] * len(neuro_files)

    def __len__(self): return len(self.files)

    def __getitem__(self, idx):
        x = torch.load(self.files[idx], weights_only=False)
        y = torch.tensor([self.labels[idx]], dtype=torch.float32)
        return x, y

# =========================
# 3. ФУНКЦИЯ ОБУЧЕНИЯ ПОРЦИЕЙ
# =========================
def calculate_metrics(outputs, targets, threshold=THRESHOLD):
    preds = (outputs >= threshold).float()
    
    tp = ((preds == 1) & (targets == 1)).sum().item()
    tn = ((preds == 0) & (targets == 0)).sum().item()
    fp = ((preds == 1) & (targets == 0)).sum().item()
    fn = ((preds == 0) & (targets == 1)).sum().item()
    
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-7)
    precision = tp / (tp + fp + 1e-7)
    recall = tp / (tp + fn + 1e-7)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-7)
    
    return accuracy, precision, recall, f1

def train_chunk(files_per_class=10000, epochs=10):
    db_manager = DataManager(FEAT_PATH)
    real_files, neuro_files = db_manager.get_balanced_training_files(files_per_class)

    if not real_files:
        print(">>> Данные закончились.")
        return

    dataset = FeatureDataset(real_files, neuro_files)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

    model = AudioTCN(num_features=40).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCELoss()
    start_epoch = 0

    if os.path.exists(MODEL_SAVE_PATH):
        checkpoint = torch.load(MODEL_SAVE_PATH, weights_only=False)
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        start_epoch = checkpoint.get('total_epochs', 0)
        print(f">>> Модель загружена. Пробег: {start_epoch} эпох.")

    for epoch in range(epochs):
        model.train()
        all_outputs = []
        all_targets = []
        total_loss = 0
        
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            
            optimizer.zero_grad()
            output, _ = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            all_outputs.append(output.detach().cpu())
            all_targets.append(y.detach().cpu())
            
        # Считаем метрики за эпоху
        epoch_outputs = torch.cat(all_outputs)
        epoch_targets = torch.cat(all_targets)
        acc, prec, rec, f1 = calculate_metrics(epoch_outputs, epoch_targets)
        
        print(f"Эпоха {epoch+1}/{epochs} | Loss: {total_loss/len(loader):.4f} | "
              f"Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f}")

    torch.save({
        'total_epochs': start_epoch + epochs,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
    }, MODEL_SAVE_PATH)

    db_manager.mark_as_trained(real_files + neuro_files)

if __name__ == "__main__":
    train_chunk(files_per_class=11000, epochs=10)

# 42K на класс