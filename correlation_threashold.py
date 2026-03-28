import numpy as np


def calculate_optimal_threshold(corr_matrix):
    """Автоматический подбор порога корреляции"""
    # Берем верхний треугольник (без диагонали)
    upper_triangle = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)]

    # Анализируем распределение корреляций
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

