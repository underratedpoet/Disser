import torch
import torch.nn as nn
import torch.nn.functional as F

class AudioTCN(nn.Module):
    def __init__(self, num_features=40, num_channels=[64, 128, 256]):
        super(AudioTCN, self).__init__()
        layers = []
        in_ch = num_features
        
        # Строим блоки расширяющихся сверток
        for out_ch in num_channels:
            layers.append(nn.Conv1d(in_ch, out_ch, kernel_size=5, padding=2))
            layers.append(nn.BatchNorm1d(out_ch))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool1d(2)) # Сжимаем время для захвата контекста
            in_ch = out_ch
            
        self.network = nn.Sequential(*layers)
        
        # Слой интерпретации: превращает 256 признаков в 1 вероятность для каждого "окна"
        self.frame_classifier = nn.Conv1d(num_channels[-1], 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (Batch, 20, Time)
        features = self.network(x) 
        
        # Локальные предсказания (Timeline)
        frame_logits = self.frame_classifier(features)
        frame_probs = self.sigmoid(frame_logits) # (Batch, 1, Compressed_Time)
        
        # Глобальное решение через усреднение (Global Average Pooling)
        # Это заставляет модель искать аномалии по всей длине
        output = torch.mean(frame_probs, dim=2)
        
        return output, frame_probs