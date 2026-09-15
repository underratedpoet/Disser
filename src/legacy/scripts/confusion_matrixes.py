import os
import glob
import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import confusion_matrix, classification_report

# ──────────────────────────────────────────────
# НАСТРОЙКИ — меняй здесь
# ──────────────────────────────────────────────
REAL_DIR       = "real"          # папка с реальными треками
NEURO_DIR      = "neuro"         # папка с нейросетевыми треками
MAX_FILES      = 500             # максимум файлов из каждой папки
AUDIO_EXT      = ("*.flac", "*.wav", "*.mp3", "*.ogg")

# Модель 1 (DeepRawNet)
MODEL1_WEIGHTS = "D:\\Study\\NIR\\Project\\raw_model_weights.pth"
MODEL1_THRESH  = 0.8

# Модель 2 (AudioTCN)
MODEL2_WEIGHTS = "D:\\Study\\NIR\\Project\\tcn_audio_model.pth"
MODEL2_THRESH  = 0.8
MODEL2_FEATURES_PATH = "D:\\Study\\NIR\\Project\\src\\explainable\\feature_utils.py"   # должен лежать рядом
MODEL2_MODEL_PATH    = "D:\\Study\\NIR\\Project\\src\\explainable\\model.py"

OUTPUT_IMAGE   = "confusion_matrices.png"

# ──────────────────────────────────────────────
# АРХИТЕКТУРА 1: DeepRawNet (из скрипта detect1)
# ──────────────────────────────────────────────
class ResBlock1d(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm1d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)


class DeepRawNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(2, 64, kernel_size=128, stride=32),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
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


# ──────────────────────────────────────────────
# ИНФЕРЕНС ДЛЯ DeepRawNet
# ──────────────────────────────────────────────
def predict_deeprawnet(model, device, file_path):
    """Возвращает среднюю вероятность по блокам 30 сек. None если файл слишком короткий."""
    waveform, sr = torchaudio.load(file_path)

    if sr != 44100:
        waveform = torchaudio.transforms.Resample(sr, 44100)(waveform)

    if waveform.shape[0] == 1:
        waveform = waveform.repeat(2, 1)
    elif waveform.shape[0] > 2:
        waveform = waveform[:2, :]

    samples_per_block = 44100 * 30
    num_blocks = waveform.shape[1] // samples_per_block

    if num_blocks == 0:
        return None

    probs = []
    with torch.no_grad():
        for i in range(num_blocks):
            chunk = waveform[:, i * samples_per_block:(i + 1) * samples_per_block]
            chunk = chunk.unsqueeze(0).to(device)
            probs.append(model(chunk).item())

    return sum(probs) / len(probs)


# ──────────────────────────────────────────────
# ЗАГРУЗКА AudioTCN (динамически, как в оригинале)
# ──────────────────────────────────────────────
def load_tcn_model(device):
    """Импортирует AudioTCN и extract_features_40 из соседних файлов."""
    import importlib.util, sys

    def load_module(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        mod  = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod

    model_mod   = load_module("model",         MODEL2_MODEL_PATH)
    feature_mod = load_module("feature_utils", MODEL2_FEATURES_PATH)

    model = model_mod.AudioTCN(num_features=40).to(device)
    ckpt  = torch.load(MODEL2_WEIGHTS, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, feature_mod.extract_features_40


def predict_tcn(model, extract_fn, device, file_path):
    """Возвращает скалярную вероятность (verdict) для одного файла."""
    features = extract_fn(file_path, target_frames=None)
    features = features.unsqueeze(0).to(device)
    with torch.no_grad():
        verdict, _ = model(features)
    return verdict.item()


# ──────────────────────────────────────────────
# СБОРКА ДАТАСЕТА
# ──────────────────────────────────────────────
def collect_files(folder, max_n):
    files = []
    for ext in AUDIO_EXT:
        files.extend(glob.glob(os.path.join(folder, "**", ext), recursive=True))
        files.extend(glob.glob(os.path.join(folder, ext)))
    files = sorted(set(files))[:max_n]
    return files


# ──────────────────────────────────────────────
# ПОСТРОЕНИЕ И СОХРАНЕНИЕ МАТРИЦ
# ──────────────────────────────────────────────
def plot_cm(ax, cm, title, class_names, cmap="Blues"):
    im = ax.imshow(cm, interpolation="nearest", cmap=cmap)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks);  ax.set_xticklabels(class_names, fontsize=11)
    ax.set_yticks(tick_marks);  ax.set_yticklabels(class_names, fontsize=11)
    ax.set_ylabel("Истинный класс",    fontsize=11)
    ax.set_xlabel("Предсказанный класс", fontsize=11)

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]}",
                    ha="center", va="center", fontsize=14,
                    color="white" if cm[i, j] > thresh else "black")


def run_evaluation():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Устройство: {device}")

    real_files  = collect_files(REAL_DIR,  MAX_FILES)
    neuro_files = collect_files(NEURO_DIR, MAX_FILES)

    print(f"Найдено реальных треков:  {len(real_files)}")
    print(f"Найдено нейросетевых треков: {len(neuro_files)}")

    if not real_files or not neuro_files:
        raise RuntimeError("Не найдены аудиофайлы. Проверь пути REAL_DIR / NEURO_DIR.")

    all_files  = real_files  + neuro_files
    true_labels = [0] * len(real_files) + [1] * len(neuro_files)   # 0=real, 1=neuro

    results = {}   # { model_name: {"preds": [...], "scores": [...]} }

    # ── Модель 1: DeepRawNet ──
    if os.path.exists(MODEL1_WEIGHTS):# and False:
        print(f"\n[1/2] DeepRawNet — загружаем {MODEL1_WEIGHTS}")
        model1 = DeepRawNet().to(device)
        model1.load_state_dict(torch.load(MODEL1_WEIGHTS, map_location=device, weights_only=True))
        model1.eval()

        preds1, scores1, labels1 = [], [], []
        for idx, (fpath, label) in enumerate(zip(all_files, true_labels)):
            print(f"  [{idx+1}/{len(all_files)}] {os.path.basename(fpath)}", end=" ... ")
            try:
                score = predict_deeprawnet(model1, device, fpath)
                if score is None:
                    print("пропущен (< 30 сек)")
                    continue
                pred = 1 if score >= MODEL1_THRESH else 0
                preds1.append(pred)
                scores1.append(score)
                labels1.append(label)
                print(f"{score:.4f} → {'neuro' if pred else 'real'}")
            except Exception as e:
                print(f"ОШИБКА: {e}")

        results["DeepRawNet\n(raw waveform)"] = {
            "preds": preds1, "scores": scores1, "labels": labels1
        }
    else:
        print(f"[!] Файл весов {MODEL1_WEIGHTS} не найден, пропускаем DeepRawNet.")

    # ── Модель 2: AudioTCN ──
    if os.path.exists(MODEL2_WEIGHTS) and os.path.exists(MODEL2_MODEL_PATH) \
            and os.path.exists(MODEL2_FEATURES_PATH):
        print(f"\n[2/2] AudioTCN — загружаем {MODEL2_WEIGHTS}")
        try:
            model2, extract_fn = load_tcn_model(device)
            preds2, scores2, labels2 = [], [], []
            for idx, (fpath, label) in enumerate(zip(all_files, true_labels)):
                print(f"  [{idx+1}/{len(all_files)}] {os.path.basename(fpath)}", end=" ... ")
                try:
                    score = predict_tcn(model2, extract_fn, device, fpath)
                    pred  = 1 if score >= MODEL2_THRESH else 0
                    preds2.append(pred)
                    scores2.append(score)
                    labels2.append(label)
                    print(f"{score:.4f} → {'neuro' if pred else 'real'}")
                except Exception as e:
                    print(f"ОШИБКА: {e}")

            results["AudioTCN\n(feature-based)"] = {
                "preds": preds2, "scores": scores2, "labels": labels2
            }
        except Exception as e:
            print(f"[!] Не удалось загрузить AudioTCN: {e}")
    else:
        print(f"[!] Файлы для AudioTCN не найдены, пропускаем.")

    if not results:
        raise RuntimeError("Ни одна модель не была загружена.")

    # ── Рисуем матрицы ──
    n_models = len(results)
    fig, axes = plt.subplots(1, n_models, figsize=(7 * n_models, 6))
    if n_models == 1:
        axes = [axes]

    class_names = ["Real (human)", "Neuro (AI)"]
    cmaps       = ["Blues", "Oranges"]

    for ax, cmap, (model_name, data) in zip(axes, cmaps, results.items()):
        y_true = data["labels"]
        y_pred = data["preds"]
        cm = confusion_matrix(y_true, y_pred)

        # Метрики
        tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
        acc  = (tp + tn) / len(y_true) * 100
        prec = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
        rec  = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

        subtitle = (f"Accuracy: {acc:.1f}%   Precision: {prec:.1f}%\n"
                    f"Recall: {rec:.1f}%   F1: {f1:.1f}%   "
                    f"n={len(y_true)}")
        full_title = f"{model_name}\n{subtitle}"

        plot_cm(ax, cm, full_title, class_names, cmap=cmap)

        print(f"\n{'='*50}")
        print(f"Модель: {model_name.replace(chr(10), ' ')}")
        print(classification_report(y_true, y_pred,
                                    target_names=class_names, digits=4))

    plt.suptitle("Матрицы ошибок детекторов AI-аудио", fontsize=15,
                 fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_IMAGE, dpi=150, bbox_inches="tight")
    print(f"\nГотово! Матрицы сохранены в: {OUTPUT_IMAGE}")
    plt.show()


# ──────────────────────────────────────────────
if __name__ == "__main__":
    run_evaluation()