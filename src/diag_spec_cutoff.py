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

# Частота, ниже которой сосредоточена эта доля энергии. Внимание:
# 0.999 (изначально казалось логичным — "ловим самый край") на
# практике бесполезен — оконное просачивание STFT размазывает даже
# честно обрезанный сигнал, и крошечный хвost за пределами реального
# cutoff утаскивает rolloff почти к Найквисту независимо от истинной
# границы (проверено на синтетике с точно известным обрезом). 0.95
# гораздо устойчивее и корректно ловит разницу в разброс.
ROLL_PERCENT = 0.95

# Подвыборка вместо всех 2000 — для этой проверки не нужна вся
# выборка, а декодирование аудио — самая медленная часть.
N_PER_CLASS = 150

SEED = 42

# Те же паттерны, что и в diagnose_ipd.py.
SOURCE_PATTERNS = {"suno": "suno", "udio": "udio"}


def guess_source(path: str) -> str:

    lowered = path.lower()

    for name, pattern in SOURCE_PATTERNS.items():
        if pattern in lowered:
            return name

    return "unknown"


def effective_cutoff_hz(
    path: str,
    sr: int = SR,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
    roll_percent: float = ROLL_PERCENT,
) -> float:
    """Медиана по кадрам частоты spectral rolloff — ниже неё
    сосредоточена roll_percent доля энергии кадра. По сути,
    "эффективная верхняя граница" контента файла: у файла с MP3
    lowpass-фильтром (типично для битрейтов <256 kbps) она будет
    заметно ниже Найквиста и стабильна по кадрам; у файла без
    обрезки — будет держаться ближе к Найквисту.
    """

    y, actual_sr = librosa.load(path, sr=sr, mono=True)

    rolloff = librosa.feature.spectral_rolloff(
        y=y,
        sr=actual_sr,
        n_fft=n_fft,
        hop_length=hop_length,
        roll_percent=roll_percent,
    )[0]

    return float(np.median(rolloff))


def build_subset(sample: pd.DataFrame, n_per_class: int, seed: int) -> pd.DataFrame:

    rng_state = seed
    parts = []

    for label in ("real", "synthetic"):
        pool = sample.loc[sample["label"] == label]
        n = min(n_per_class, len(pool))
        parts.append(pool.sample(n=n, random_state=rng_state))

    return pd.concat(parts).reset_index(drop=True)


def main():

    if not SAMPLE_PATH.exists():
        raise FileNotFoundError(f"Не найден {SAMPLE_PATH}")

    sample = pd.read_csv(SAMPLE_PATH)
    sample["source"] = sample["file"].astype(str).apply(guess_source)

    subset = build_subset(sample, N_PER_CLASS, SEED)

    print(
        f"Проверяю спектральный потолок на {len(subset)} файлах "
        f"(до {N_PER_CLASS} на класс)"
    )
    print(subset["label"].value_counts().to_string())

    rows = []

    for counter, row in enumerate(subset.itertuples(index=False), 1):

        print(f"\r{counter}/{len(subset)}", end="", flush=True)

        cutoff = effective_cutoff_hz(row.file)

        rows.append({
            "file": row.file,
            "label": row.label,
            "source": row.source,
            "cutoff_hz": cutoff,
        })

    print()

    result = pd.DataFrame(rows)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_DIR / "spectral_cutoff_diagnostic.csv", index=False)

    print()
    print("=" * 80)
    print("МЕДИАННЫЙ ЭФФЕКТИВНЫЙ ПОТОЛОК СПЕКТРА, Гц")
    print("=" * 80)

    print("\nПо классу:")
    print(result.groupby("label")["cutoff_hz"].median().to_string())

    print("\nПо источнику:")
    print(result.groupby("source")["cutoff_hz"].median().to_string())

    real_vals = result.loc[result["label"] == "real", "cutoff_hz"]
    synth_vals = result.loc[result["label"] == "synthetic", "cutoff_hz"]

    _, p_value = mannwhitneyu(real_vals, synth_vals, alternative="two-sided")

    diff = real_vals.median() - synth_vals.median()

    print()
    print(
        f"Манн-Уитни real vs synthetic: p={p_value:.3e}, "
        f"медианы: real={real_vals.median():.0f} Гц, "
        f"synthetic={synth_vals.median():.0f} Гц, "
        f"разница={diff:.0f} Гц"
    )

    print()
    print("Ориентир (типичный lowpass у LAME MP3 по битрейту):")
    print("  128 kbps ~16 кГц | 160-192 kbps ~18-19 кГц | 256-320 kbps ~19-22 кГц")

    if abs(diff) > 500:
        print(
            "\n⚠ Разница медиан больше 500 Гц — стоит рассматривать как "
            "вероятную причину (или соавтора) высокочастотной находки "
            "MGD/IPD, а не только содержательный эффект синтеза."
        )
    else:
        print(
            "\nРазница медиан небольшая — прямое объяснение через разный "
            "lowpass-обрез маловероятно, но стоит также взглянуть на "
            "разброс (не только медиану) в spectral_cutoff_diagnostic.csv."
        )

    print(f"\nСохранено: {OUTPUT_DIR / 'spectral_cutoff_diagnostic.csv'}")


if __name__ == "__main__":
    main()