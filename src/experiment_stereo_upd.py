from __future__ import annotations

from pathlib import Path

import pandas as pd

from stereo import STEREO_METHODS
from experiment_stereo import OUTPUT_DIR, SAMPLE_PATH, build_stereo_features

from experiment_bands import evaluate_configuration, make_summary_row


def get_features(
    sample: pd.DataFrame,
    stereo_method: str,
    stereo_function,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.Series]:

    features_path = output_dir / f"{stereo_method}__features.parquet"

    if features_path.exists():
        print(f"{stereo_method}: признаки уже на диске, загружаю {features_path}")

        X = pd.read_parquet(features_path)
        # Порядок строк в features.parquet соответствует порядку
        # sample.itertuples() в build_stereo_features — тот же
        # порядок, что и в sample.csv, поэтому метки берём оттуда,
        # не пересчитывая аудио.
        y = sample["label"]

        return X, y

    print(f"{stereo_method}: признаков на диске нет, считаю из аудио заново")

    X, y = build_stereo_features(sample, stereo_method, stereo_function)

    X.to_parquet(features_path, index=False)

    return X, y


def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    sample = pd.read_csv(SAMPLE_PATH)

    print(f"Sample size: {len(sample)}")
    print(sample["label"].value_counts())

    summary = []

    for stereo_method, stereo_function in STEREO_METHODS.items():

        print()
        print("=" * 80)
        print(stereo_method)
        print("=" * 80)

        X, y = get_features(sample, stereo_method, stereo_function, OUTPUT_DIR)

        print(f"Features: {X.shape[1]}")

        # Само по себе дёшево (RF fit + сильно ускоренная
        # permutation importance) — пересчитываем всегда, даже для
        # методов с уже готовыми признаками, чтобы гарантированно
        # получить test_auc/test_accuracy для итогового summary
        # (их первый прогон в консоль не печатал и на диск отдельно
        # не сохранял).
        evaluation = evaluate_configuration(X, y)

        evaluation["mdi_importance"].rename("importance").reset_index().rename(
            columns={"index": "feature"}
        ).to_csv(
            OUTPUT_DIR / f"{stereo_method}__mdi_importance.csv",
            index=False,
        )

        evaluation["perm_importance"].rename("importance").reset_index().rename(
            columns={"index": "feature"}
        ).to_csv(
            OUTPUT_DIR / f"{stereo_method}__perm_importance.csv",
            index=False,
        )

        row = make_summary_row(
            method=stereo_method,
            parameter_name="n_bands",
            parameter_value=120,
            resolution_level=None,
            evaluation=evaluation,
        )

        summary.append(row)

    summary_df = pd.DataFrame(summary)

    summary_df = summary_df.sort_values(
        ["test_auc", "test_accuracy"],
        ascending=False,
    ).reset_index(drop=True)

    summary_df.to_csv(OUTPUT_DIR / "stereo_summary.csv", index=False)

    print()
    print("=" * 80)
    print("СРАВНЕНИЕ СПОСОБОВ ОБЪЕДИНЕНИЯ КАНАЛОВ (по test_auc)")
    print("=" * 80)
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()