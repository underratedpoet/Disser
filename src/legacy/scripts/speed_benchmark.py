import os
import glob
import time
import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ──────────────────────────────────────────────
# НАСТРОЙКИ
# ──────────────────────────────────────────────
REAL_DIR       = "real"
NEURO_DIR      = "neuro"
MAX_FILES      = 1000          # сколько файлов брать на каждую модель
AUDIO_EXT      = ("*.flac", "*.wav", "*.mp3", "*.ogg")
WARMUP_RUNS    = 3             # прогревочные прогоны (не входят в замер)

MODEL1_WEIGHTS = "D:\\Study\\NIR\\Project\\raw_model_weights.pth"
MODEL2_WEIGHTS = "D:\\Study\\NIR\\Project\\tcn_audio_model.pth"
MODEL2_MODEL_PATH    = "D:\\Study\\NIR\\Project\\src\\explainable\\model.py"
MODEL2_FEATURES_PATH = "D:\\Study\\NIR\\Project\\src\\explainable\\feature_utils.py"

OUTPUT_IMAGE = "speed_benchmark.png"

# ──────────────────────────────────────────────
# DeepRawNet
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
            nn.BatchNorm1d(64), nn.ReLU()
        )
        self.layer1 = self._make_layer(64, 128)
        self.layer2 = self._make_layer(128, 256)
        self.layer3 = self._make_layer(256, 512)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(512, 256), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(256, 1), nn.Sigmoid()
        )
 
    def _make_layer(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(out_ch), nn.ReLU(),
            ResBlock1d(out_ch), ResBlock1d(out_ch)
        )
 
    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        return self.classifier(x)
 
 
# ──────────────────────────────────────────────
# Утилиты
# ──────────────────────────────────────────────
def collect_files(max_n):
    files = []
    for folder in (REAL_DIR, NEURO_DIR):
        for ext in AUDIO_EXT:
            files.extend(glob.glob(os.path.join(folder, "**", ext), recursive=True))
            files.extend(glob.glob(os.path.join(folder, ext)))
    files = sorted(set(files))
    rng = np.random.default_rng(42)
    rng.shuffle(files)
    return files[:max_n]
 
 
def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
 
 
def load_tcn_model(device):
    import importlib.util, sys
 
    def load_mod(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        mod  = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod
 
    model_mod   = load_mod("model",         MODEL2_MODEL_PATH)
    feature_mod = load_mod("feature_utils", MODEL2_FEATURES_PATH)
    model = model_mod.AudioTCN(num_features=40).to(device)
    ckpt  = torch.load(MODEL2_WEIGHTS, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, feature_mod.extract_features_40
 
 
# ──────────────────────────────────────────────
# Замеры: DeepRawNet
#   preproc = загрузка + ресемплинг + формирование тензора
#   infer   = только forward()
#   total   = preproc + infer
# ──────────────────────────────────────────────
def benchmark_deeprawnet(files, device):
    print(f"\n[DeepRawNet] загружаем веса...")
    model = DeepRawNet().to(device)
    model.load_state_dict(torch.load(MODEL1_WEIGHTS, map_location=device, weights_only=True))
    model.eval()
 
    samples = 44100 * 30
 
    def preprocess(f):
        waveform, sr = torchaudio.load(f)
        if sr != 44100:
            waveform = torchaudio.transforms.Resample(sr, 44100)(waveform)
        if waveform.shape[0] == 1:
            waveform = waveform.repeat(2, 1)
        elif waveform.shape[0] > 2:
            waveform = waveform[:2, :]
        waveform = waveform[:, :samples]
        if waveform.shape[1] < samples:
            waveform = F.pad(waveform, (0, samples - waveform.shape[1]))
        return waveform.unsqueeze(0).to(device)
 
    # Прогрев
    print(f"  Прогрев ({WARMUP_RUNS} файла)...")
    for f in files[:WARMUP_RUNS]:
        chunk = preprocess(f)
        with torch.no_grad():
            sync(device); _ = model(chunk); sync(device)
 
    t_pre, t_inf = [], []
    print(f"  Замер на {len(files)} файлах...")
    for i, f in enumerate(files):
        # preproc
        sync(device)
        t0 = time.perf_counter()
        chunk = preprocess(f)
        sync(device)
        t1 = time.perf_counter()
        t_pre.append((t1 - t0) * 1000)
 
        # infer
        with torch.no_grad():
            sync(device)
            t2 = time.perf_counter()
            _ = model(chunk)
            sync(device)
            t3 = time.perf_counter()
        t_inf.append((t3 - t2) * 1000)
 
        if (i + 1) % 200 == 0:
            print(f"    {i+1}/{len(files)}")
 
    t_pre = np.array(t_pre)
    t_inf = np.array(t_inf)
    return {
        "preproc": t_pre,
        "infer":   t_inf,
        "total":   t_pre + t_inf,
    }
 
 
# ──────────────────────────────────────────────
# Замеры: AudioTCN
#   preproc = extract_features_40 (DSP-пайплайн)
#   infer   = только forward()
#   total   = preproc + infer
# ──────────────────────────────────────────────
def benchmark_tcn(files, device):
    print(f"\n[AudioTCN] загружаем веса...")
    model, extract_fn = load_tcn_model(device)
 
    # Прогрев
    print(f"  Прогрев ({WARMUP_RUNS} файла)...")
    for f in files[:WARMUP_RUNS]:
        feats = extract_fn(f, target_frames=None).unsqueeze(0).to(device)
        with torch.no_grad():
            sync(device); _ = model(feats); sync(device)
 
    t_pre, t_inf = [], []
    print(f"  Замер на {len(files)} файлах...")
    for i, f in enumerate(files):
        # preproc (извлечение признаков — на CPU)
        t0 = time.perf_counter()
        feats = extract_fn(f, target_frames=None).unsqueeze(0).to(device)
        sync(device)
        t1 = time.perf_counter()
        t_pre.append((t1 - t0) * 1000)
 
        # infer
        with torch.no_grad():
            sync(device)
            t2 = time.perf_counter()
            _ = model(feats)
            sync(device)
            t3 = time.perf_counter()
        t_inf.append((t3 - t2) * 1000)
 
        if (i + 1) % 200 == 0:
            print(f"    {i+1}/{len(files)}")
 
    t_pre = np.array(t_pre)
    t_inf = np.array(t_inf)
    return {
        "preproc": t_pre,
        "infer":   t_inf,
        "total":   t_pre + t_inf,
    }
 
 
# ──────────────────────────────────────────────
# Диаграмма размаха
# ──────────────────────────────────────────────
STAGE_META = {
    "preproc": ("Загрузка и\nпредобработка", "#5B8DB8"),
    "infer":   ("Инференс\nсети",            "#E8833A"),
    "total":   ("Общее время",               "#6AAB69"),
}
 
def draw_one_box(ax, values, color, title):
    bp = ax.boxplot(
        values,
        vert=True,
        patch_artist=True,
        widths=0.5,
        showfliers=True,
        flierprops=dict(marker="o", markersize=2.5, alpha=0.3,
                        markerfacecolor=color, markeredgecolor=color),
        medianprops=dict(color="white", linewidth=2.5),
        whiskerprops=dict(color=color, linewidth=1.4, linestyle="--"),
        capprops=dict(color=color, linewidth=2),
        boxprops=dict(facecolor=color + "44", edgecolor=color, linewidth=1.5),
    )
 
    med  = np.median(values)
    mean = np.mean(values)
    q1, q3 = np.percentile(values, [25, 75])
    iqr  = q3 - q1
    p95  = np.percentile(values, 95)
    std  = np.std(values)
 
    # линия среднего
    ax.axhline(mean, color=color, linewidth=1.2, linestyle=":", alpha=0.75)
 
    # аннотации
    x_ann = 1.32
    ax.annotate(f"медиана {med:.1f} мс",
                xy=(1, med), xytext=(x_ann, med),
                fontsize=8, color="#333",
                arrowprops=dict(arrowstyle="-", color="#aaa", lw=0.7))
    ax.annotate(f"среднее {mean:.1f} мс",
                xy=(1, mean), xytext=(x_ann, mean + max(iqr * 0.25, std * 0.3)),
                fontsize=8, color=color,
                arrowprops=dict(arrowstyle="-", color=color, lw=0.7, linestyle=":"))
 
    # таблица метрик
    stats = (f"n = {len(values)}\n"
             f"мин  {np.min(values):.1f} мс\n"
             f"макс {np.max(values):.1f} мс\n"
             f"σ      {std:.1f} мс\n"
             f"IQR  {iqr:.1f} мс\n"
             f"P95  {p95:.1f} мс")
    ax.text(0.04, 0.03, stats, transform=ax.transAxes, fontsize=8,
            verticalalignment="bottom",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                      edgecolor="#ddd", alpha=0.9))
 
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    ax.set_ylabel("мс", fontsize=10)
    ax.set_xticks([])
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.spines[["top", "right"]].set_visible(False)
 
 
def plot_results(all_timing: dict):
    """
    all_timing = {
        "DeepRawNet": {"preproc": arr, "infer": arr, "total": arr},
        "AudioTCN":   {"preproc": arr, "infer": arr, "total": arr},
    }
    """
    model_names = list(all_timing.keys())
    n_models    = len(model_names)
    stages      = list(STAGE_META.keys())   # preproc, infer, total
    n_stages    = len(stages)
 
    # layout: строки = модели, столбцы = этапы
    fig, axes = plt.subplots(
        n_models, n_stages,
        figsize=(4.5 * n_stages, 5.5 * n_models),
        squeeze=False
    )
 
    for row, model_name in enumerate(model_names):
        timing = all_timing[model_name]
        for col, stage in enumerate(stages):
            ax = axes[row][col]
            label, color = STAGE_META[stage]
            vals = timing[stage]
            draw_one_box(ax, vals, color, label)
 
        # подпись модели слева
        axes[row][0].set_ylabel(
            f"{model_name}\n\nмс", fontsize=11, labelpad=10
        )
 
    # разделительные линии между строками
    for row in range(1, n_models):
        line = plt.Line2D([0, 1], [1 - row / n_models] * 2,
                          transform=fig.transFigure,
                          color="#ddd", linewidth=1)
        fig.add_artist(line)
 
    plt.suptitle(
        "Скорость обработки 30-секундного фрагмента по этапам",
        fontsize=14, fontweight="bold", y=1.01
    )
    plt.tight_layout()
    plt.savefig(OUTPUT_IMAGE, dpi=150, bbox_inches="tight")
    print(f"\nДиаграмма сохранена: {OUTPUT_IMAGE}")
    plt.show()
 
 
# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Устройство: {device}")
 
    files = collect_files(MAX_FILES)
    print(f"Файлов для замера: {len(files)}")
    if not files:
        raise RuntimeError("Не найдены аудиофайлы. Проверь REAL_DIR / NEURO_DIR.")
 
    all_timing = {}
 
    if os.path.exists(MODEL1_WEIGHTS):
        all_timing["DeepRawNet\n(raw waveform)"] = benchmark_deeprawnet(files, device)
    else:
        print(f"[!] {MODEL1_WEIGHTS} не найден, пропускаем DeepRawNet.")
 
    if (os.path.exists(MODEL2_WEIGHTS)
            and os.path.exists(MODEL2_MODEL_PATH)
            and os.path.exists(MODEL2_FEATURES_PATH)):
        all_timing["AudioTCN\n(feature-based)"] = benchmark_tcn(files, device)
    else:
        print(f"[!] Файлы AudioTCN не найдены, пропускаем.")
 
    if not all_timing:
        raise RuntimeError("Ни одна модель не загружена.")
 
    # Итоговая сводка в консоль
    for mname, timing in all_timing.items():
        print(f"\n{'='*50}")
        print(f"Модель: {mname.replace(chr(10), ' ')}")
        for stage, (label, _) in STAGE_META.items():
            v = timing[stage]
            print(f"  {label.replace(chr(10),' '):30s}  "
                  f"median={np.median(v):.2f}  mean={np.mean(v):.2f}  "
                  f"std={np.std(v):.2f}  P95={np.percentile(v,95):.2f}  мс")
 
    plot_results(all_timing)