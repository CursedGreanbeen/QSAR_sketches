import pandas as pd
import numpy as np
from itertools import combinations
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


def calculate_optimal_threshold(corr_matrix):
    # верхний треугольник матрицы (без диагонали)
    upper_triangle = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)]

    # анализ распределения корреляций
    mean_corr = np.mean(np.abs(upper_triangle))
    std_corr = np.std(np.abs(upper_triangle))
    median_corr = np.median(np.abs(upper_triangle))

    print(f"Статистика корреляций:")
    print(f"  Средняя: {mean_corr:.3f}")
    print(f"  Медиана: {median_corr:.3f}")
    print(f"  Стандартное отклонение: {std_corr:.3f}")

    # Адаптивный порог: медиана + 1.5 * стандартное отклонение
    adaptive_threshold = min(0.8, median_corr + 1.5 * std_corr)
    adaptive_threshold = max(0.5, adaptive_threshold)  # Не ниже 0.5

    print(f"  Предлагаемый порог: {adaptive_threshold:.3f}")

    return adaptive_threshold


def select_optimal_combinations():
    df = pd.read_csv("data_descriptors_new.csv")
    # df = pd.read_excel("data_descriptors.xlsx", sheet_name="Descriptors pyrazoles new")
    df = df.drop(['ligand', 'mol'], axis=1, errors='ignore')

    df['Energy 6wha'] = df['Energy 6wha'].astype(str).str.replace(',', '.').astype(float)
    df = df.dropna(subset=['Energy 6wha'])
    df = df.dropna()

    X_df = df.drop(['SMILES', 'Energy 6wha'], axis=1)
    y = df['Energy 6wha']

    # Отбор признаков с вариативностью
    selector = VarianceThreshold(threshold=0.01)
    X_selected = selector.fit_transform(X_df)
    selected_features = X_df.columns[selector.get_support()].tolist()

    # Расчет корреляционной матрицы
    corr_matrix = pd.DataFrame(X_selected, columns=selected_features).corr().abs()
    optimal_threshold = calculate_optimal_threshold(corr_matrix)

    # вывод матрицы
    import seaborn as sns
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 10))
    sns.set(font_scale=0.8)

    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # Маска для верхнего треугольника
    heatmap = sns.heatmap(corr_matrix,
                          mask=mask,
                          annot=True,
                          cmap='RdBu_r',
                          center=0,
                          fmt='.2f',
                          square=True,
                          cbar_kws={'shrink': 0.8})

    # Настройки внешнего вида
    plt.title('Матрица корреляций Пирсона для молекулярных дескрипторов',
              fontsize=14, pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    # Сохранение и вывод
    plt.savefig('correlation_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()


    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y, test_size=0.3, random_state=32
    )

    results = []

    # Перебор комбинаций от 2 до 6 признаков
    for n_features in range(2, min(7, len(selected_features) + 1)):
        # Генерация комбинации
        for feature_indices in combinations(range(len(selected_features)), n_features):
            features = [selected_features[i] for i in feature_indices]

            # корреляция внутри комбинации
            combo_corr = corr_matrix.iloc[list(feature_indices), list(feature_indices)]
            max_corr = combo_corr.values[np.triu_indices_from(combo_corr.values, k=1)].max()

            # Если максимальная корреляция < 0.7, оцениваем комбинацию
            if max_corr < optimal_threshold:
                X_train_sub = X_train[:, feature_indices]
                X_test_sub = X_test[:, feature_indices]

                model = LinearRegression()
                model.fit(X_train_sub, y_train)
                y_pred = model.predict(X_test_sub)
                r2 = r2_score(y_test, y_pred)

                results.append({
                    'features': features,
                    'r2_score': r2,
                    'n_features': n_features,
                    'max_correlation': max_corr
                })

    # Сортировка по R²
    results.sort(key=lambda x: x['r2_score'], reverse=True)

    # Вывод топ-5 комбинаций
    print("Топ-5 комбинаций дескрипторов:")
    for i, result in enumerate(results[:5]):
        print(f"{i + 1}. R² = {result['r2_score']:.3f}, "
              f"Признаков: {result['n_features']}, "
              f"Макс. корреляция: {result['max_correlation']:.3f}")
        print(f"   Дескрипторы: {', '.join(result['features'])}\n")

    return results[0] if results else None


if __name__ == "__main__":
    best_combo = select_optimal_combinations()
    if best_combo:
        print(f"ЛУЧШАЯ КОМБИНАЦИЯ:")
        print(f"R²: {best_combo['r2_score']:.3f}")
        print(f"Дескрипторы: {', '.join(best_combo['features'])}")