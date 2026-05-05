import os
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from model import AudioTCN

# Настройки
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FEAT_PATH = r"D:\Study\NIR\Project\features_db"
BATCH_SIZE = 128 # Теперь можно ставить больше!
EPOCHS = 20

class FastFeatureDataset(Dataset):
    def __init__(self, root_dir):
        self.files = []
        self.labels = []
        
        for label, category in enumerate(['real', 'neuro']):
            cat_path = os.path.join(root_dir, category)
            for f in os.listdir(cat_path):
                if f.endswith('.pt'):
                    self.files.append(os.path.join(cat_path, f))
                    self.labels.append(float(label))

    def __len__(self): return len(self.files)

    def __getitem__(self, idx):
        # Просто загружаем готовый тензор
        x = torch.load(self.files[idx])
        y = torch.tensor([self.labels[idx]], dtype=torch.float32)
        return x, y

def train():
    dataset = FastFeatureDataset(FEAT_PATH)
    # num_workers=4 задействует многопоточность процессора для загрузки
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

    model = AudioTCN(num_features=40).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()

    print(f"Обучение на {len(dataset)} примерах. Устройство: {DEVICE}")

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            
            optimizer.zero_grad()
            output, _ = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Эпоха {epoch+1}/{EPOCHS}, Loss: {total_loss/len(loader):.4f}")
        
    torch.save(model.state_dict(), "fast_tcn_model.pth")

if __name__ == "__main__":
    train()