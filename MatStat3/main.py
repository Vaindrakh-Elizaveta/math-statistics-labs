import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, cauchy, laplace, poisson, uniform
import pandas as pd

# -----------------------------
# ПАРАМЕТРЫ ЭКСПЕРИМЕНТА
# -----------------------------
rng = np.random.default_rng(42)

sample_sizes = [20, 100]
num_experiments = 1000

sqrt2 = np.sqrt(2)
sqrt3 = np.sqrt(3)

# -----------------------------
# ГЕНЕРАТОРЫ РАСПРЕДЕЛЕНИЙ
# -----------------------------
def generate_normal(size):
    return norm.rvs(loc=0, scale=1, size=size, random_state=rng)

def generate_cauchy(size):
    return cauchy.rvs(loc=0, scale=1, size=size, random_state=rng)

def generate_laplace(size):
    return laplace.rvs(loc=0, scale=1 / sqrt2, size=size, random_state=rng)

def generate_poisson(size):
    return poisson.rvs(mu=5, size=size, random_state=rng)

def generate_uniform(size):
    return uniform.rvs(loc=-sqrt3, scale=2 * sqrt3, size=size, random_state=rng)

distributions = [
    ("Normal N(0,1)", generate_normal),
    ("Cauchy C(0,1)", generate_cauchy),
    ("Laplace L(0, 1/sqrt(2))", generate_laplace),
    ("Poisson P(5)", generate_poisson),
    ("Uniform U(-sqrt(3), sqrt(3))", generate_uniform),
]

# -----------------------------
# ФУНКЦИЯ ДЛЯ ПОИСКА ВЫБРОСОВ ПО ТЬЮКИ
# -----------------------------
def tukey_outlier_fraction(sample):
    q1 = np.percentile(sample, 25)
    q3 = np.percentile(sample, 75)
    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    outliers = sample[(sample < lower_bound) | (sample > upper_bound)]
    return len(outliers) / len(sample)

# -----------------------------
# ПОСТРОЕНИЕ БОКСПЛОТОВ
# -----------------------------
def plot_boxplots():
    fig, axes = plt.subplots(len(distributions), len(sample_sizes), figsize=(12, 18))
    fig.suptitle("Боксплоты Тьюки для различных распределений", fontsize=16)

    for i, (dist_name, generator) in enumerate(distributions):
        for j, n in enumerate(sample_sizes):
            sample = generator(n)

            ax = axes[i, j]
            ax.boxplot(sample, vert=True)
            ax.set_title(f"{dist_name}\nn = {n}")
            ax.grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()

# -----------------------------
# ЭКСПЕРИМЕНТАЛЬНАЯ ОЦЕНКА СРЕДНЕЙ ДОЛИ ВЫБРОСОВ
# -----------------------------
def calculate_average_outlier_fractions():
    results = []

    for dist_name, generator in distributions:
        for n in sample_sizes:
            fractions = []

            for _ in range(num_experiments):
                sample = generator(n)
                fraction = tukey_outlier_fraction(sample)
                fractions.append(fraction)

            mean_fraction = np.mean(fractions)

            results.append({
                "Распределение": dist_name,
                "n": n,
                "Средняя доля выбросов": mean_fraction
            })

    return pd.DataFrame(results)

# -----------------------------
# ОСНОВНАЯ ЧАСТЬ ПРОГРАММЫ
# -----------------------------
if __name__ == "__main__":
    # 1. Строим боксплоты
    plot_boxplots()

    # 2. Считаем среднюю долю выбросов
    result_table = calculate_average_outlier_fractions()

    # Красивый вывод таблицы
    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)
    pd.set_option("display.colheader_justify", "center")

    print("\nСредняя доля выбросов по 1000 выборкам:\n")
    print(result_table.to_string(index=False))