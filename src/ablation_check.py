"""
Быстрая проверка: насколько IPD_COS/IPD_SIN/MAG_DIFF (fixed-часть)
достаточны САМИ ПО СЕБЕ, без какого-либо признака L/R сверху.

Не декодирует аудио заново - использует уже посчитанный
fixed_diff__features.parquet. Добавьте в experiment_features.py
или запустите как отдельный скрипт рядом с ним (нужен тот же
evaluate_configuration/OUTPUT_DIR/sample.csv).
"""
from pathlib import Path
import pandas as pd

from experiment_features import OUTPUT_DIR, evaluate_configuration

sample = pd.read_csv(OUTPUT_DIR / "sample.csv")
X_fixed = pd.read_parquet(OUTPUT_DIR / "fixed_diff__features.parquet")

print(f"Только IPD_COS/IPD_SIN/MAG_DIFF, без L/R: {X_fixed.shape[1]} признаков")

evaluation = evaluate_configuration(X_fixed, sample["label"])

print()
print(f"test_accuracy = {evaluation['test_accuracy']:.4f}")
print(f"test_auc      = {evaluation['test_auc']:.5f}")
print()
print("Для сравнения из feature_summary.csv:")
print("  лучший (modified_group_delay, с L/R): auc=0.99988")
print("  худший (group_delay, с L/R):          auc=0.99933")