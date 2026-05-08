import torch
import torch.nn as nn

class AudioTCN(nn.Module):
    def __init__(self, num_features=40, num_channels=[64, 64, 128, 128, 256, 256, 512]):
        super(AudioTCN, self).__init__()
        layers = []
        in_ch = num_features
        
        # Строим блоки сверток
        for out_ch in num_channels:
            layers.append(nn.Conv1d(in_ch, out_ch, kernel_size=5, padding=2))
            layers.append(nn.BatchNorm1d(out_ch))
            layers.append(nn.ReLU())
            # Сжимаем время только на определенных этапах, чтобы не потерять разрешение слишком быстро
            if out_ch in [64, 128, 256, 512]:
                layers.append(nn.MaxPool1d(2))
            in_ch = out_ch
            
        self.network = nn.Sequential(*layers)
        
        # Слой интерпретации
        self.frame_classifier = nn.Conv1d(num_channels[-1], 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (Batch, 40, Time)
        features = self.network(x) 
        
        # Локальные предсказания
        frame_logits = self.frame_classifier(features)
        frame_probs = self.sigmoid(frame_logits) # (Batch, 1, Compressed_Time)
        
        # Глобальное решение через усреднение
        output = torch.mean(frame_probs, dim=2)
        
        return output, frame_probs