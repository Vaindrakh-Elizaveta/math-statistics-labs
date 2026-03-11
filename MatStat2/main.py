import numpy as np
from scipy.stats import norm, cauchy, laplace, poisson, uniform

# =========================================================
# НАСТРОЙКИ
# =========================================================

rng = np.random.default_rng(42)

sample_sizes = [10, 100, 1000]
num_repeats = 1000

sqrt2 = np.sqrt(2)
sqrt3 = np.sqrt(3)

# Распределения из задания
distributions = [
    ("Normal N(0,1)", norm(loc=0, scale=1)),
    ("Cauchy C(0,1)", cauchy(loc=0, scale=1)),
    ("Laplace L(0,1/sqrt(2))", laplace(loc=0, scale=1 / sqrt2)),
    ("Poisson P(5)", poisson(mu=5)),
    ("Uniform U(-sqrt(3), sqrt(3))", uniform(loc=-sqrt3, scale=2 * sqrt3)),
]

# =========================================================
# ФУНКЦИИ ДЛЯ ВЫЧИСЛЕНИЯ ХАРАКТЕРИСТИК
# =========================================================

def sample_mean(x):
    """Выборочное среднее"""
    return np.mean(x)


def sample_median(x):
    """Медиана"""
    return np.median(x)


def z_R(x):
    """Полусумма экстремальных элементов"""
    return (np.min(x) + np.max(x)) / 2


def z_Q(x):
    """Полусумма квартилей"""
    q1 = np.quantile(x, 0.25)
    q3 = np.quantile(x, 0.75)
    return (q1 + q3) / 2


def z_tr(x):
    """Усечённое среднее (отбрасываем 10% минимальных и 10% максимальных значений)"""
    x_sorted = np.sort(x)
    n = len(x_sorted)
    r = int(n * 0.1)

    # Защита на случай слишком маленькой выборки
    if 2 * r >= n:
        return np.mean(x_sorted)

    trimmed = x_sorted[r:n - r]
    return np.mean(trimmed)


# =========================================================
# ФУНКЦИЯ ДЛЯ ОЦЕНКИ E(z), D(z), sqrt(D(z))
# =========================================================

def calc_statistics(values):
    """
    По массиву из 1000 значений характеристики вычисляет:
    E(z)  - среднее
    D(z)  - дисперсию
    sqrt(D(z)) - стандартное отклонение
    """
    values = np.asarray(values)
    Ez = np.mean(values)
    Dz = np.var(values)          # D(z) = E(z^2) - E(z)^2
    sigma = np.sqrt(Dz)
    return Ez, Dz, sigma


# =========================================================
# ОСНОВНАЯ ЧАСТЬ
# =========================================================

for dist_name, dist_obj in distributions:
    print("=" * 90)
    print(dist_name)
    print("=" * 90)

    for n in sample_sizes:
        values_mean = []
        values_med = []
        values_zR = []
        values_zQ = []
        values_ztr = []

        # Повторяем эксперимент 1000 раз
        for _ in range(num_repeats):
            sample = dist_obj.rvs(size=n, random_state=rng)

            values_mean.append(sample_mean(sample))
            values_med.append(sample_median(sample))
            values_zR.append(z_R(sample))
            values_zQ.append(z_Q(sample))
            values_ztr.append(z_tr(sample))

        # Собираем характеристики в словарь
        characteristics = {
            "x̄": values_mean,
            "med": values_med,
            "z_R": values_zR,
            "z_Q": values_zQ,
            "z_tr": values_ztr,
        }

        print(f"\nn = {n}")
        print("-" * 90)
        print(f"{'Характеристика':<15}{'E(z)':>15}{'D(z)':>15}{'sqrt(D)':>15}{'E ± sqrt(D)':>25}")
        print("-" * 90)

        for char_name, values in characteristics.items():
            Ez, Dz, sigma = calc_statistics(values)

            print(
                f"{char_name:<15}"
                f"{Ez:>15.6f}"
                f"{Dz:>15.6f}"
                f"{sigma:>15.6f}"
                f"{(f'{Ez:.6f} ± {sigma:.6f}'):>25}"
            )

        print()
