from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder


# ============================================================
# CONFIGURATION
# ============================================================

SAMPLE_PATH = Path("band_experiment_results_v2/sample.csv")
FEATURES_PATH = Path(
    "stereo_experiment_results/left_right_plus_diff__features.parquet"
)
OUTPUT_DIR = Path("stereo_experiment_results")

N_FOLDS = 5
SEED = 42
N_ESTIMATORS = 100

# Подстроки в пути файла, по которым пытаемся угадать источник
# synthetic-записи. Если структура папок другая — просто не найдётся,
# разбивка по источнику (часть 3) аккуратно пропустится, без ошибки.
# Поправьте под свою структуру, если имена не совпадают.
SOURCE_PATTERNS = {"suno": "suno", "udio": "udio"}


# ============================================================
# ЗАГРУЗКА (без декодирования аудио — только уже посчитанное)
# ============================================================

def guess_source(path: str) -> str:

    lowered = path.lower()

    for name, pattern in SOURCE_PATTERNS.items():
        if pattern in lowered:
            return name

    return "unknown"


def load_data() -> tuple[pd.DataFrame, pd.Series, pd.Series]:

    if not SAMPLE_PATH.exists():
        raise FileNotFoundError(f"Не найден {SAMPLE_PATH}")

    if not FEATURES_PATH.exists():
        raise FileNotFoundError(
            f"Не найден {FEATURES_PATH}. Нужен результат уже "
            f"отработавшего experiment_stereo_upd.py для "
            f"left_right_plus_diff (он сохраняет этот parquet сам)."
        )

    sample = pd.read_csv(SAMPLE_PATH)
    X = pd.read_parquet(FEATURES_PATH)

    if len(sample) != len(X):
        raise RuntimeError(
            f"sample.csv ({len(sample)} строк) и {FEATURES_PATH.name} "
            f"({len(X)} строк) не совпадают по размеру — похоже, это "
            f"признаки для другой выборки. Порядок строк в parquet "
            f"должен соответствовать порядку в sample.csv."
        )

    labels = sample["label"]
    sources = sample["file"].astype(str).apply(guess_source)

    return X, labels, sources


# ============================================================
# ЧАСТЬ 1: РАЗЛИЧАЕТСЯ ЛИ ДИСПЕРСИЯ IPD/MAG_DIFF МЕЖДУ КЛАССАМИ
# ============================================================
#
# Смотрим конкретно на "std" (а не mean/skew/kurtosis) по каждой
# полосе IPD_COS/IPD_SIN/MAG_DIFF — гипотеза именно про изменчивость
# межканальных величин во времени, а не про их средний уровень.
# Манна-Уитни — потому что не требует нормальности распределения
# (у std по частотным полосам она вряд ли есть).


def compare_ipd_variance(X: pd.DataFrame, labels: pd.Series) -> pd.DataFrame:

    std_columns = [
        col for col in X.columns
        if "__std" in col and (
            "IPD_COS" in col or "IPD_SIN" in col or "MAG_DIFF" in col
        )
    ]

    if not std_columns:
        raise RuntimeError(
            "Не найдено ни одной колонки IPD_COS/IPD_SIN/MAG_DIFF__std "
            "— проверьте, что FEATURES_PATH указывает на "
            "left_right_plus_diff, а не на left_right/mid_side."
        )

    rows = []

    for col in std_columns:

        real_values = X.loc[labels == "real", col]
        synth_values = X.loc[labels == "synthetic", col]

        try:
            _, p_value = mannwhitneyu(
                real_values, synth_values, alternative="two-sided"
            )
        except ValueError:
            # Все значения идентичны в обеих группах — сравнивать нечего.
            p_value = float("nan")

        rows.append({
            "feature": col,
            "median_real": real_values.median(),
            "median_synthetic": synth_values.median(),
            "median_diff": real_values.median() - synth_values.median(),
            "p_value": p_value,
        })

    return pd.DataFrame(rows).sort_values("p_value").reset_index(drop=True)


# ============================================================
# ЧАСТЬ 2: K-FOLD СТАБИЛЬНОСТЬ test_auc
# ============================================================


def kfold_auc(
    X: pd.DataFrame,
    labels: pd.Series,
    n_folds: int = N_FOLDS,
    seed: int = SEED,
) -> pd.DataFrame:

    encoder = LabelEncoder()
    y = encoder.fit_transform(labels)

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    rows = []

    for fold_index, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):

        model = RandomForestClassifier(
            n_estimators=N_ESTIMATORS,
            random_state=seed,
            n_jobs=-1,
        )

        model.fit(X.iloc[train_idx], y[train_idx])

        proba = model.predict_proba(X.iloc[test_idx])[:, 1]
        auc = roc_auc_score(y[test_idx], proba)

        rows.append({"fold": fold_index, "test_auc": auc})

        print(f"  fold {fold_index}/{n_folds}: test_auc={auc:.5f}")

    return pd.DataFrame(rows)


# ============================================================
# ЧАСТЬ 3: ДЕРЖИТСЯ ЛИ ЭФФЕКТ ОТДЕЛЬНО НА SUNO И ОТДЕЛЬНО НА UDIO
# ============================================================
#
# Если IPD-эффект специфичен для одного конкретного генератора (его
# способа делать стерео), а не для синтеза вообще — auc должен
# заметно проседать на изолированной паре real vs <один источник>.
# Держится примерно одинаково высоким на обоих по отдельности —
# куда более сильный аргумент в пользу общей находки.


def kfold_auc_by_source(
    X: pd.DataFrame,
    labels: pd.Series,
    sources: pd.Series,
    n_folds: int = N_FOLDS,
    seed: int = SEED,
) -> pd.DataFrame | None:

    known_sources = sorted(s for s in sources.unique() if s != "unknown")

    if len(known_sources) < 2:
        print(
            "Источник synthetic-файлов не удалось определить по пути "
            "(SOURCE_PATTERNS не совпали ни с чем) — пропускаю "
            "разбивку по источникам. Поправьте SOURCE_PATTERNS под "
            "вашу структуру папок, если хотите эту проверку."
        )
        return None

    rows = []

    for source in known_sources:

        mask = (labels == "real") | (sources == source)

        X_subset = X.loc[mask].reset_index(drop=True)
        labels_subset = labels.loc[mask].reset_index(drop=True)

        encoder = LabelEncoder()
        y = encoder.fit_transform(labels_subset)

        skf = StratifiedKFold(
            n_splits=n_folds, shuffle=True, random_state=seed,
        )

        fold_aucs = []

        for train_idx, test_idx in skf.split(X_subset, y):

            model = RandomForestClassifier(
                n_estimators=N_ESTIMATORS,
                random_state=seed,
                n_jobs=-1,
            )

            model.fit(X_subset.iloc[train_idx], y[train_idx])

            proba = model.predict_proba(X_subset.iloc[test_idx])[:, 1]
            fold_aucs.append(roc_auc_score(y[test_idx], proba))

        rows.append({
            "source": source,
            "n_files": int(mask.sum()),
            "auc_mean": float(np.mean(fold_aucs)),
            "auc_std": float(np.std(fold_aucs)),
        })

        print(
            f"  real vs {source}: auc={np.mean(fold_aucs):.5f} "
            f"± {np.std(fold_aucs):.5f} (n={int(mask.sum())})"
        )

    return pd.DataFrame(rows)


# ============================================================
# MAIN
# ============================================================

def main():

    X, labels, sources = load_data()

    print(
        f"Файлов: {len(X)}  "
        f"(real={int((labels == 'real').sum())}, "
        f"synthetic={int((labels == 'synthetic').sum())})"
    )

    print("Определённые источники (по пути к файлу):")
    print(sources.value_counts().to_string())

    print()
    print("=" * 80)
    print("ЧАСТЬ 1: std(IPD_COS/IPD_SIN/MAG_DIFF) — real vs synthetic по полосам")
    print("=" * 80)

    variance_report = compare_ipd_variance(X, labels)
    print(variance_report.head(20).to_string(index=False))

    n_significant = int((variance_report["p_value"] < 0.01).sum())
    print(
        f"\nПолос с p < 0.01 (Манн-Уитни, без поправки на "
        f"множественное сравнение — их тут {len(variance_report)}, "
        f"так что для строгого вывода стоит поправить Бонферрони/BH): "
        f"{n_significant} из {len(variance_report)}"
    )

    print()
    print("=" * 80)
    print("ЧАСТЬ 2: устойчивость test_auc по 5 фолдам (left_right_plus_diff)")
    print("=" * 80)

    fold_report = kfold_auc(X, labels)
    print()
    print(
        f"test_auc: {fold_report['test_auc'].mean():.5f} "
        f"± {fold_report['test_auc'].std():.5f}"
    )

    print()
    print("=" * 80)
    print("ЧАСТЬ 3: держится ли эффект отдельно на suno и отдельно на udio")
    print("=" * 80)

    source_report = kfold_auc_by_source(X, labels, sources)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    variance_report.to_csv(
        OUTPUT_DIR / "ipd_variance_diagnostic.csv", index=False
    )
    fold_report.to_csv(
        OUTPUT_DIR / "left_right_plus_diff_kfold.csv", index=False
    )

    if source_report is not None:
        source_report.to_csv(
            OUTPUT_DIR / "left_right_plus_diff_by_source.csv", index=False
        )

    print()
    print(f"Сохранено в {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()