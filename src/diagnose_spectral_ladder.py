from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu


# ============================================================
# CONFIGURATION
# ============================================================

SAMPLE_PATH = Path("band_experiment_results_v2/sample.csv")
OUTPUT_DIR = Path("stereo_experiment_results")

SR = 44100
N_FFT = 2048
HOP_LENGTH = 512

N_PER_CLASS = 150
SEED = 42

SOURCE_PATTERNS = {"suno": "suno", "udio": "udio"}

# Референсная полоса — где у музыки почти всегда много честного
# контента независимо от жанра/мастеринга, используется как база
# (0 дБ) для всех остальных полос этого же файла.
REF_BAND = (1000.0, 4000.0)

# "Лестница" полос от середины спектра до самого верха — вместе они
# трассируют форму спада спектра. Более мелкий шаг ближе к верху,
# где и ожидается возможный излом lowpass-фильтра.
LADDER_BANDS = [
    (4000.0, 8000.0),
    (8000.0, 11000.0),
    (11000.0, 13000.0),
    (13000.0, 15000.0),
    (15000.0, 16500.0),
    (16500.0, 18000.0),
    (18000.0, 19500.0),
    (19500.0, 21000.0),
    (21000.0, 22050.0),
]


def guess_source(path: str) -> str:

    lowered = path.lower()

    for name, pattern in SOURCE_PATTERNS.items():
        if pattern in lowered:
            return name

    return "unknown"


def band_level_db(magnitude: np.ndarray, freqs: np.ndarray, band: tuple[float, float]) -> float:

    lo, hi = band
    mask = (freqs >= lo) & (freqs < hi)

    if not mask.any():
        return float("nan")

    mean_magnitude = magnitude[mask, :].mean()

    return float(20.0 * np.log10(mean_magnitude + 1e-10))


def analyze_file(path: str) -> dict[str, float]:

    y, actual_sr = librosa.load(path, sr=SR, mono=True)

    stft = librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH)
    magnitude = np.abs(stft)
    freqs = librosa.fft_frequencies(sr=actual_sr, n_fft=N_FFT)

    ref_level = band_level_db(magnitude, freqs, REF_BAND)

    result = {}

    for lo, hi in LADDER_BANDS:
        level = band_level_db(magnitude, freqs, (lo, hi))
        label = f"{lo/1000:g}-{hi/1000:g}kHz"
        result[label] = level - ref_level  # относительно референса, дБ

    return result


def build_subset(sample: pd.DataFrame, n_per_class: int, seed: int) -> pd.DataFrame:

    parts = []

    for label in ("real", "synthetic"):
        pool = sample.loc[sample["label"] == label]
        n = min(n_per_class, len(pool))
        parts.append(pool.sample(n=n, random_state=seed))

    return pd.concat(parts).reset_index(drop=True)


def main():

    if not SAMPLE_PATH.exists():
        raise FileNotFoundError(f"Не найден {SAMPLE_PATH}")

    sample = pd.read_csv(SAMPLE_PATH)
    sample["source"] = sample["file"].astype(str).apply(guess_source)

    subset = build_subset(sample, N_PER_CLASS, SEED)

    print(f"Анализирую {len(subset)} файлов (до {N_PER_CLASS} на класс)")
    print(subset["label"].value_counts().to_string())
    print()

    rows = []

    for counter, row in enumerate(subset.itertuples(index=False), 1):

        print(f"\r{counter}/{len(subset)}", end="", flush=True)

        levels = analyze_file(row.file)
        levels["label"] = row.label
        levels["source"] = row.source
        levels["file"] = row.file

        rows.append(levels)

    print()

    result = pd.DataFrame(rows)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_DIR / "spectral_ladder_diagnostic.csv", index=False)

    band_labels = [f"{lo/1000:g}-{hi/1000:g}kHz" for lo, hi in LADDER_BANDS]

    print()
    print("=" * 80)
    print("МЕДИАННЫЙ УРОВЕНЬ ОТНОСИТЕЛЬНО РЕФЕРЕНСА (1-4 кГц), дБ")
    print("=" * 80)

    summary = result.groupby("label")[band_labels].median().T
    print(summary.to_string())

    print()
    print("По источникам (синтетика):")
    source_summary = result.groupby("source")[band_labels].median().T
    print(source_summary.to_string())

    print()
    print("=" * 80)
    print("МАНН-УИТНИ ПО КАЖДОЙ ПОЛОСЕ (real vs synthetic)")
    print("=" * 80)

    for label in band_labels:
        real_vals = result.loc[result["label"] == "real", label].dropna()
        synth_vals = result.loc[result["label"] == "synthetic", label].dropna()

        if len(real_vals) < 2 or len(synth_vals) < 2:
            continue

        _, p_value = mannwhitneyu(real_vals, synth_vals, alternative="two-sided")

        diff = real_vals.median() - synth_vals.median()

        flag = " ⚠" if p_value < 0.001 and abs(diff) > 3 else ""

        print(
            f"  {label:12s} real={real_vals.median():+6.1f}дБ  "
            f"synth={synth_vals.median():+6.1f}дБ  "
            f"diff={diff:+6.1f}дБ  p={p_value:.2e}{flag}"
        )

    print()
    print("Как читать:")
    print("  - Плавное, похожее по форме снижение у обоих классов до самого")
    print("    верха -> обреза нет, разница объясняется содержанием.")
    print("  - У одного класса уровень резко проваливается (на 15-30+ дБ)")
    print("    начиная с какой-то полосы и дальше остаётся на дне ('плато')")
    print("    -> это и есть характерный профиль lowpass-фильтра кодека.")
    print(f"\nСохранено: {OUTPUT_DIR / 'spectral_ladder_diagnostic.csv'}")


if __name__ == "__main__":
    main()