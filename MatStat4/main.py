import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, cauchy, laplace, poisson, uniform, gaussian_kde

# ============================================
# Лабораторная работа №4
# Эмпирическая функция распределения
# и ядерные оценки плотности
#
# Комментарий преподавателя учтён:
# для распределения Пуассона берём параметр lambda = 5
# ============================================

# Для воспроизводимости результатов
rng = np.random.default_rng(42)

# Объёмы выборок по заданию
sample_sizes = [20, 60, 100]

# Границы отрезков по заданию
continuous_x = np.linspace(-4, 4, 1000)
poisson_x = np.linspace(6, 14, 1000)
poisson_k = np.arange(6, 15)  # целые точки для PMF и CDF Пуассона

sqrt2 = np.sqrt(2)
sqrt3 = np.sqrt(3)

# Описание распределений:
# name          - название
# dist          - объект scipy.stats
# kind          - "continuous" или "discrete"
# x_grid        - сетка для графиков
# support_int   - нужны ли целочисленные точки (для Пуассона)
distributions = [
    {
        "name": "Нормальное N(0, 1)",
        "dist": norm(loc=0, scale=1),
        "kind": "continuous",
        "x_grid": continuous_x
    },
    {
        "name": "Коши C(0, 1)",
        "dist": cauchy(loc=0, scale=1),
        "kind": "continuous",
        "x_grid": continuous_x
    },
    {
        "name": "Лапласа L(0, 1/sqrt(2))",
        "dist": laplace(loc=0, scale=1 / sqrt2),
        "kind": "continuous",
        "x_grid": continuous_x
    },
    {
        "name": "Пуассона P(5)",
        "dist": poisson(mu=5),   # здесь параметр 5, как просил преподаватель
        "kind": "discrete",
        "x_grid": poisson_x,
        "k_grid": poisson_k
    },
    {
        "name": "Равномерное U(-sqrt(3), sqrt(3))",
        "dist": uniform(loc=-sqrt3, scale=2 * sqrt3),
        "kind": "continuous",
        "x_grid": continuous_x
    }
]


def ecdf(sample):
    """
    Возвращает точки для построения эмпирической функции распределения.
    """
    x = np.sort(sample)
    y = np.arange(1, len(x) + 1) / len(x)
    return x, y


def make_kde(sample, kind):
    """
    Строит гауссову ядерную оценку плотности.
    Для дискретного распределения Пуассона добавляем очень маленький шум,
    чтобы gaussian_kde работал устойчивее.
    """
    sample_for_kde = np.asarray(sample, dtype=float)

    if kind == "discrete":
        # Небольшой шум для устойчивой KDE на дискретной выборке
        sample_for_kde = sample_for_kde + rng.normal(0, 0.15, size=len(sample_for_kde))

    try:
        kde = gaussian_kde(sample_for_kde)
    except np.linalg.LinAlgError:
        # Запасной вариант, если ковариационная матрица выродилась
        sample_for_kde = sample_for_kde + rng.normal(0, 1e-3, size=len(sample_for_kde))
        kde = gaussian_kde(sample_for_kde)

    return kde


def plot_distribution(dist_info):
    """
    Для одного распределения строит:
    1) Эмпирическую функцию распределения и теоретическую функцию распределения
    2) Ядерную оценку плотности и теоретическую плотность/функцию вероятностей
    для n = 20, 60, 100
    """
    name = dist_info["name"]
    dist = dist_info["dist"]
    kind = dist_info["kind"]
    x_grid = dist_info["x_grid"]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(name, fontsize=16)

    for j, n in enumerate(sample_sizes):
        # Генерация выборки
        sample = dist.rvs(size=n, random_state=rng)

        # ---------- 1. Эмпирическая функция распределения ----------
        ax_cdf = axes[0, j]
        x_ecdf, y_ecdf = ecdf(sample)

        # Эмпирическая функция распределения
        ax_cdf.step(x_ecdf, y_ecdf, where='post', label='Эмпирическая ФР', linewidth=2)

        # Теоретическая функция распределения
        if kind == "continuous":
            ax_cdf.plot(x_grid, dist.cdf(x_grid), label='Теоретическая ФР', linewidth=2)
            ax_cdf.set_xlim(x_grid[0], x_grid[-1])
        else:
            k_grid = dist_info["k_grid"]
            ax_cdf.step(k_grid, dist.cdf(k_grid), where='post',
                        label='Теоретическая ФР', linewidth=2)
            ax_cdf.set_xlim(k_grid[0], k_grid[-1])

        ax_cdf.set_title(f"Функция распределения, n = {n}")
        ax_cdf.set_xlabel("x")
        ax_cdf.set_ylabel("F(x)")
        ax_cdf.grid(True, alpha=0.3)
        ax_cdf.legend()

        # ---------- 2. Ядерная оценка плотности ----------
        ax_pdf = axes[1, j]
        kde = make_kde(sample, kind)
        kde_values = kde(x_grid)

        ax_pdf.plot(x_grid, kde_values, label='Ядерная оценка плотности', linewidth=2)

        if kind == "continuous":
            ax_pdf.plot(x_grid, dist.pdf(x_grid), label='Теоретическая плотность', linewidth=2)
            ax_pdf.set_xlim(x_grid[0], x_grid[-1])
        else:
            # Для Пуассона "теоретическая плотность" в дискретном случае — это PMF
            k_grid = dist_info["k_grid"]
            ax_pdf.plot(k_grid, dist.pmf(k_grid), 'o-', label='Теоретическая функция вероятностей',
                        linewidth=2)
            ax_pdf.set_xlim(x_grid[0], x_grid[-1])

        ax_pdf.set_title(f"Ядерная оценка плотности, n = {n}")
        ax_pdf.set_xlabel("x")
        ax_pdf.set_ylabel("f(x)")
        ax_pdf.grid(True, alpha=0.3)
        ax_pdf.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


def main():
    for dist_info in distributions:
        plot_distribution(dist_info)


if __name__ == "__main__":
    main()