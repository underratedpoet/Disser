import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# =========================
# 1. ЗАГРУЗКА С БАЛАНСИРОВКОЙ
# =========================
def load_balanced_data(real_dir, neuro_dir):
    print("--- Загрузка и балансировка данных ---")
    
    # 1. Загружаем всю реальную музыку (её меньше)
    print(f"Читаем реальные данные из {real_dir}...")
    df_real = pd.read_parquet(real_dir, engine="pyarrow")
    n_real = len(df_real)
    print(f"Найдено реальных записей: {n_real}")

    # 2. Загружаем нейромузыку
    print(f"Читаем нейросети из {neuro_dir}...")
    df_neuro_full = pd.read_parquet(neuro_dir, engine="pyarrow")
    n_neuro_total = len(df_neuro_full)
    print(f"Всего нейрозаписей в папке: {n_neuro_total}")

    # 3. Делаем случайную выборку (Sampling)
    if n_neuro_total > n_real:
        print(f"Балансировка: выбираем {n_real} случайных нейрозаписей из {n_neuro_total}...")
        df_neuro_sampled = df_neuro_full.sample(n=n_real, random_state=42) 
        # random_state=42 нужен, чтобы результат был воспроизводимым
    else:
        print("Внимание: нейрозаписей меньше или столько же, сколько реальных. Берем всё.")
        df_neuro_sampled = df_neuro_full

    # 4. Объединяем
    df_all = pd.concat([df_real, df_neuro_sampled], ignore_index=True)
    
    # Перемешиваем итоговый датасет, чтобы real и neuro не шли сплошными кусками
    df_all = df_all.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"Итоговый размер выборки для анализа: {len(df_all)} строк (баланс 1:1)")
    return df_all

# =========================
# 2. ОЧИСТКА (без изменений)
# =========================
def clean_data(df):
    print("\nОчистка от NaN и бесконечностей...")
    labels = df["label"]
    files = df["file"]
    
    # Оставляем только числа
    features = df.drop(columns=["label", "file"])
    features = features.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    return features, labels, files

# =========================
# 3. АНАЛИЗ (Random Forest)
# =========================
def find_best_features(X, y):
    print(f"\nАнализ {len(X.columns)} признаков. Поехали...")
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Используем n_jobs=-1 для максимальной скорости на всех ядрах
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X, y_encoded)
    
    importance_df = pd.DataFrame({
        "Feature": X.columns,
        "Importance": rf.feature_importances_
    }).sort_values(by="Importance", ascending=False).reset_index(drop=True)
    
    return importance_df

# =========================
# ИСПОЛНЕНИЕ
# =========================
if __name__ == "__main__":
    REAL_PATH = "features_real_data"
    NEURO_PATH = "features_neuro_data"

    if not os.path.exists(REAL_PATH) or not os.path.exists(NEURO_PATH):
        print("Ошибка: Папки с данными не найдены. Убедись, что основной скрипт создал их.")
    else:
        # 1. Грузим сбалансированно
        df = load_balanced_data(REAL_PATH, NEURO_PATH)
        
        # 2. Чистим
        X, y, _ = clean_data(df)
        
        # 3. Считаем важность
        report = find_best_features(X, y)
        
        # Результаты
        print("\n" + "="*30)
        print("ТОП-25 САМЫХ ВАЖНЫХ ПРИЗНАКОВ")
        print("="*30)
        print(report.head(25))
        
        report.to_csv("balanced_feature_importance.csv", index=False)
        print(f"\nГотово! Полный отчет: balanced_feature_importance.csv")