from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from itertools import product
from typing import Any, Callable, Protocol, Sequence
import hashlib
import json
import pickle

import librosa
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder


Array = np.ndarray


@dataclass(frozen=True)
class AudioFile:
    path: Path
    label: str


@dataclass
class SpectralData:
    """One transformed signal.

    matrix:
        Complex or real-valued time-frequency representation, shape
        (frequency_bins, frames).
    freqs:
        Center frequency of every row in matrix.
    sr:
        Sampling rate.
    signal:
        Original (or represented) 1-D signal.
    times:
        Frame times, if available.
    meta:
        Optional transform metadata.
    """
    matrix: Array
    freqs: Array
    sr: int
    signal: Array | None = None
    times: Array | None = None
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class BandedSpectralData:
    """Spectral data plus a list of frequency bands.

    bins[i] contains row indices of the spectral representation
    belonging to band i.
    """
    spectral: SpectralData
    bins: list[Array]
    names: list[str]
    edges: Array


@dataclass(frozen=True)
class Representation:
    name: str
    fn: Callable[[Array], dict[str, Array]]


@dataclass(frozen=True)
class Transform:
    name: str
    fn: Callable[[Array, int], SpectralData]


@dataclass(frozen=True)
class Feature:
    name: str
    fn: Callable[[BandedSpectralData], dict[str, Array]]


@dataclass(frozen=True)
class Statistic:
    name: str
    fn: Callable[[Array], float]


@dataclass(frozen=True)
class Experiment:
    representations: Sequence[Representation]
    transforms: Sequence[Transform]
    bands: Sequence[Callable[[SpectralData], BandedSpectralData]]
    features: Sequence[Feature]
    statistics: Sequence[Statistic]


class Cache:
    """Small local disk cache for intermediate numpy/dataclass objects.

    The cache is intentionally simple. Delete the cache directory whenever
    the implementation of a stage changes substantially.
    """

    def __init__(self, directory: str | Path = ".experiment_cache"):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def _key(
        self,
        audio: AudioFile,
        stage: str,
        name: str,
        sr: int,
    ) -> str:
        stat = audio.path.stat()
        payload = {
            "path": str(audio.path.resolve()),
            "mtime_ns": stat.st_mtime_ns,
            "size": stat.st_size,
            "stage": stage,
            "name": name,
            "sr": sr,
        }
        raw = json.dumps(payload, sort_keys=True).encode()
        return hashlib.sha256(raw).hexdigest()

    def load(
        self,
        audio: AudioFile,
        stage: str,
        name: str,
        sr: int,
    ) -> Any | None:
        key = self._key(audio, stage, name, sr)
        path = self.directory / f"{key}.pkl"
        if not path.exists():
            return None
        with path.open("rb") as f:
            return pickle.load(f)

    def save(
        self,
        audio: AudioFile,
        stage: str,
        name: str,
        sr: int,
        value: Any,
    ) -> None:
        key = self._key(audio, stage, name, sr)
        path = self.directory / f"{key}.pkl"
        tmp = path.with_suffix(".tmp")
        with tmp.open("wb") as f:
            pickle.dump(value, f, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(path)

    def clear(self) -> None:
        for p in self.directory.glob("*.pkl"):
            p.unlink()


def sample_dataset(
    real_dir: str | Path,
    synthetic_dir: str | Path,
    n_per_class: int,
    seed: int = 42,
    extensions: tuple[str, ...] = (".flac", ".wav", ".mp3"),
) -> list[AudioFile]:
    rng = np.random.default_rng(seed)

    def find_files(directory: Path) -> list[Path]:
        return [
            p for p in directory.rglob("*")
            if p.is_file() and p.suffix.lower() in extensions
        ]

    real_files = find_files(Path(real_dir))
    synthetic_files = find_files(Path(synthetic_dir))

    if len(real_files) < n_per_class:
        raise ValueError(f"Not enough real files: {len(real_files)} < {n_per_class}")
    if len(synthetic_files) < n_per_class:
        raise ValueError(
            f"Not enough synthetic files: {len(synthetic_files)} < {n_per_class}"
        )

    real = rng.choice(real_files, n_per_class, replace=False)
    synthetic = rng.choice(synthetic_files, n_per_class, replace=False)

    items = [
        *(AudioFile(Path(p), "real") for p in real),
        *(AudioFile(Path(p), "synthetic") for p in synthetic),
    ]
    rng.shuffle(items)
    return items


def save_sample(sample: Sequence[AudioFile], path: str | Path) -> None:
    pd.DataFrame({
        "path": [str(x.path) for x in sample],
        "label": [x.label for x in sample],
    }).to_csv(path, index=False)


def load_sample(path: str | Path) -> list[AudioFile]:
    df = pd.read_csv(path)
    return [AudioFile(Path(r.path), r.label) for r in df.itertuples()]


def make_experiment_combinations(experiment: Experiment):
    return product(
        experiment.representations,
        experiment.transforms,
        experiment.bands,
        experiment.features,
        experiment.statistics,
    )


def _load_audio(audio: AudioFile, sr: int) -> Array:
    y, _ = librosa.load(audio.path, sr=sr, mono=False)
    if y.ndim == 1:
        y = np.vstack([y, y])
    return y


def _safe_name(*parts: str) -> str:
    return "__".join(parts).replace("/", "_").replace("\\", "_")


def _stat_features(
    feature_values: dict[str, Array],
    statistic: Statistic,
) -> dict[str, float]:
    result = {}
    for feature_name, values in feature_values.items():
        values = np.asarray(values)
        result[f"{feature_name}__{statistic.name}"] = float(
            statistic.fn(values)
        )
    return result


def run_configuration(
    sample: Sequence[AudioFile],
    representation: Representation,
    transform: Transform,
    bands: Callable[[SpectralData], BandedSpectralData],
    feature: Feature,
    statistic: Statistic,
    sr: int = 44100,
    cache: Cache | None = None,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.Series, str]:
    """Compute one exact combination.

    Pipeline:
        audio -> representation -> transform -> bands -> feature -> statistic

    Intermediate representation and transform results are cached.
    Band definitions are cheap and are recomputed.
    """

    name = _safe_name(
        representation.name,
        transform.name,
        getattr(bands, "__name__", bands.__class__.__name__),
        feature.name,
        statistic.name,
    )

    rows: list[dict[str, float]] = []
    labels: list[str] = []

    # Grouping is intentionally kept simple. If several configurations use
    # the same representation/transform, the disk cache prevents recomputing.
    for i, audio in enumerate(sample, 1):
        if verbose:
            print(f"[{i}/{len(sample)}] {audio.label}: {audio.path.name}")

        # Representation is cached separately.
        representation_name = representation.name
        represented = None if cache is None else cache.load(
            audio, "representation", representation_name, sr
        )

        if represented is None:
            y = _load_audio(audio, sr)
            represented = representation.fn(y)
            if cache is not None:
                cache.save(
                    audio, "representation", representation_name, sr, represented
                )

        # The selected representation may produce multiple streams (L/R, M/S).
        # We concatenate feature results from every stream.
        row: dict[str, float] = {}

        for stream_name, signal in represented.items():
            transform_name = f"{representation.name}__{transform.name}__{stream_name}"

            spectral = None if cache is None else cache.load(
                audio, "transform", transform_name, sr
            )

            if spectral is None:
                spectral = transform.fn(signal, sr)
                if cache is not None:
                    cache.save(
                        audio, "transform", transform_name, sr, spectral
                    )

            banded = bands(spectral)
            feature_values = feature.fn(banded)

            # Prefix every feature with its stream and band name.
            for key, values in feature_values.items():
                # Feature functions should return 1-D temporal sequences.
                values = np.asarray(values)

                if values.ndim == 1:
                    stats = {key: values}
                elif values.ndim == 2:
                    # Convention: first axis = bands, second = time.
                    stats = {
                        f"{key}__{banded.names[j]}": values[j]
                        for j in range(values.shape[0])
                    }
                else:
                    raise ValueError(
                        f"Feature '{feature.name}' returned array with "
                        f"shape {values.shape}; expected 1-D or 2-D."
                    )

                stream_stats = _stat_features(stats, statistic)

                for k, v in stream_stats.items():
                    row[f"{stream_name}__{k}"] = v

        rows.append(row)
        labels.append(audio.label)

    X = pd.DataFrame(rows)
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    y = pd.Series(labels, name="label")

    return X, y, name


def evaluate_random_forest(
    X: pd.DataFrame,
    y: pd.Series,
    n_estimators: int = 100,
    seed: int = 42,
) -> pd.DataFrame:
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(X, y_encoded)

    result = pd.DataFrame({
        "feature": X.columns,
        "importance": model.feature_importances_,
    })
    return result.sort_values(
        "importance", ascending=False
    ).reset_index(drop=True)


def run_experiment(
    sample: Sequence[AudioFile],
    experiment: Experiment,
    output_dir: str | Path = "experiment_results",
    sr: int = 44100,
    cache_dir: str | Path = ".experiment_cache",
    n_estimators: int = 100,
    seed: int = 42,
    verbose: bool = True,
) -> pd.DataFrame:
    """Run all Cartesian combinations and return a summary table."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache = Cache(cache_dir)

    combinations = list(make_experiment_combinations(experiment))
    if not combinations:
        raise RuntimeError(
            "make_experiment_combinations() не создал ни одной комбинации. "
            "Проверь списки в Experiment."
        )
    summary = []

    print(f"Experiments: {len(combinations)}")

    for index, (
        representation,
        transform,
        bands,
        feature,
        statistic,
    ) in enumerate(combinations, 1):

        print()
        print("=" * 80)
        print(f"Experiment {index}/{len(combinations)}")
        print(
            representation.name,
            transform.name,
            getattr(bands, "__name__", bands.__class__.__name__),
            feature.name,
            statistic.name,
            sep=" | ",
        )
        print("=" * 80)

        X, y, name = run_configuration(
            sample=sample,
            representation=representation,
            transform=transform,
            bands=bands,
            feature=feature,
            statistic=statistic,
            sr=sr,
            cache=cache,
            verbose=verbose,
        )

        importance = evaluate_random_forest(
            X, y,
            n_estimators=n_estimators,
            seed=seed,
        )

        importance.to_csv(
            output_dir / f"{name}__importance.csv",
            index=False,
        )

        X.to_parquet(
            output_dir / f"{name}__features.parquet",
            index=False,
        )

        summary.append({
            "experiment": name,
            "n_features": len(importance),
            "top_1": importance.head(1)["importance"].sum(),
            "top_5": importance.head(5)["importance"].sum(),
            "top_10": importance.head(10)["importance"].sum(),
            "top_25": importance.head(25)["importance"].sum(),
        })

    if not summary:
        raise RuntimeError(
            "Не выполнено ни одного эксперимента: summary пуст. "
            "Проверь содержимое Experiment и make_experiment_combinations()."
        )

    summary_df = (
        pd.DataFrame(summary)
        .sort_values("top_10", ascending=False)
        .reset_index(drop=True)
    )

    summary_df.to_csv(
        output_dir / "summary.csv",
        index=False,
    )

    return summary_df