import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, chi2

# =========================
# НАСТРОЙКИ
# =========================
RNG_SEED = 42
REPEATS = 1000
SAMPLE_SIZES = [20, 60, 100]
RHO_VALUES = [0.0, 0.5, 0.9]

np.random.seed(RNG_SEED)


# =========================
# ГЕНЕРАЦИЯ ВЫБОРОК
# =========================
def generate_bivariate_normal(n, rho):
    """
    Генерация выборки из двумерного нормального распределения
    N(0, 0, 1, 1, rho)
    """
    mean = np.array([0.0, 0.0])
    cov = np.array([
        [1.0, rho],
        [rho, 1.0]
    ])
    return np.random.multivariate_normal(mean, cov, size=n)


def generate_mixture_sample(n):
    """
    Генерация выборки из смеси:
    0.9 N(0,0,1,1,0.9) + 0.1 N(0,0,10,10,-0.9)
    """
    mean1 = np.array([0.0, 0.0])
    cov1 = np.array([
        [1.0, 0.9],
        [0.9, 1.0]
    ])

    mean2 = np.array([0.0, 0.0])
    cov2 = np.array([
        [10.0, -0.9 * np.sqrt(10.0 * 10.0)],
        [-0.9 * np.sqrt(10.0 * 10.0), 10.0]
    ])

    # Для каждой точки выбираем компоненту смеси
    components = np.random.choice([0, 1], size=n, p=[0.9, 0.1])

    sample = np.zeros((n, 2))
    count_0 = np.sum(components == 0)
    count_1 = np.sum(components == 1)

    if count_0 > 0:
        sample[components == 0] = np.random.multivariate_normal(mean1, cov1, size=count_0)
    if count_1 > 0:
        sample[components == 1] = np.random.multivariate_normal(mean2, cov2, size=count_1)

    return sample


# =========================
# КОЭФФИЦИЕНТЫ КОРРЕЛЯЦИИ
# =========================
def pearson_corr(x, y):
    return np.corrcoef(x, y)[0, 1]


def spearman_corr(x, y):
    return spearmanr(x, y).statistic


def quadrant_corr(x, y):
    """
    Квадрантный коэффициент корреляции:
    r_Q = ((n1 + n3) - (n2 + n4)) / n

    Квадранты считаются относительно медиан x и y.
    """
    x_med = np.median(x)
    y_med = np.median(y)

    signs = np.sign(x - x_med) * np.sign(y - y_med)

    # Точки, попавшие ровно на медиану, дают 0.
    # В знаменателе оставляем общее n, как обычно делают в лабораторных.
    return np.sum(signs) / len(x)


# =========================
# МНОГОКРАТНОЕ МОДЕЛИРОВАНИЕ
# =========================
def simulate_correlations(generator_func, n, repeats=1000, rho=None):
    pearson_values = []
    spearman_values = []
    quadrant_values = []

    for _ in range(repeats):
        if rho is None:
            sample = generator_func(n)
        else:
            sample = generator_func(n, rho)

        x = sample[:, 0]
        y = sample[:, 1]

        pearson_values.append(pearson_corr(x, y))
        spearman_values.append(spearman_corr(x, y))
        quadrant_values.append(quadrant_corr(x, y))

    return {
        "pearson_mean": np.mean(pearson_values),
        "pearson_var": np.var(pearson_values, ddof=1),
        "spearman_mean": np.mean(spearman_values),
        "spearman_var": np.var(spearman_values, ddof=1),
        "quadrant_mean": np.mean(quadrant_values),
        "quadrant_var": np.var(quadrant_values, ddof=1),
    }


# =========================
# ЭЛЛИПС РАВНОВЕРОЯТНОСТИ
# =========================
def get_ellipse_points(mean, cov, alpha=0.95, num_points=400):
    """
    Строим эллипс равновероятности уровня alpha
    для двумерного нормального распределения.
    """
    eigvals, eigvecs = np.linalg.eigh(cov)

    # Значение хи-квадрат для двумерного случая
    chi2_val = chi2.ppf(alpha, df=2)

    radii = np.sqrt(eigvals * chi2_val)

    t = np.linspace(0, 2 * np.pi, num_points)
    circle = np.vstack((np.cos(t), np.sin(t)))

    ellipse = eigvecs @ np.diag(radii) @ circle
    ellipse[0, :] += mean[0]
    ellipse[1, :] += mean[1]

    return ellipse


def plot_ellipse_only(ax, sample, title, alpha=0.95):
    """
    Строим только эллипс равновероятности.
    Диаграмму рассеяния не строим по корректировке преподавателя.
    """
    mean = np.mean(sample, axis=0)
    cov = np.cov(sample[:, 0], sample[:, 1])

    ellipse = get_ellipse_points(mean, cov, alpha=alpha)

    ax.plot(ellipse[0], ellipse[1], linewidth=2)
    ax.scatter(mean[0], mean[1], marker='x', s=70)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.axis("equal")
    ax.grid(True)


# =========================
# ОСНОВНОЙ РАСЧЁТ
# =========================
def run_experiment():
    results = []

    # 1. Нормальные распределения
    for rho in RHO_VALUES:
        for n in SAMPLE_SIZES:
            stats = simulate_correlations(
                generator_func=generate_bivariate_normal,
                n=n,
                repeats=REPEATS,
                rho=rho
            )

            results.append({
                "distribution": f"N(0,0,1,1,{rho})",
                "n": n,
                "pearson_mean": stats["pearson_mean"],
                "pearson_var": stats["pearson_var"],
                "spearman_mean": stats["spearman_mean"],
                "spearman_var": stats["spearman_var"],
                "quadrant_mean": stats["quadrant_mean"],
                "quadrant_var": stats["quadrant_var"],
            })

    # 2. Смесь
    for n in SAMPLE_SIZES:
        stats = simulate_correlations(
            generator_func=generate_mixture_sample,
            n=n,
            repeats=REPEATS,
            rho=None
        )

        results.append({
            "distribution": "0.9N(0,0,1,1,0.9)+0.1N(0,0,10,10,-0.9)",
            "n": n,
            "pearson_mean": stats["pearson_mean"],
            "pearson_var": stats["pearson_var"],
            "spearman_mean": stats["spearman_mean"],
            "spearman_var": stats["spearman_var"],
            "quadrant_mean": stats["quadrant_mean"],
            "quadrant_var": stats["quadrant_var"],
        })

    return pd.DataFrame(results)


# =========================
# ПОСТРОЕНИЕ ГРАФИКОВ
# =========================
def plot_all_ellipses():
    # 3*3 + 3 = 12 графиков
    fig, axes = plt.subplots(4, 3, figsize=(15, 18))
    axes = axes.ravel()

    idx = 0

    # Нормальные распределения
    for rho in RHO_VALUES:
        for n in SAMPLE_SIZES:
            sample = generate_bivariate_normal(n, rho)
            title = f"N(0,0,1,1,{rho}), n={n}"
            plot_ellipse_only(axes[idx], sample, title)
            idx += 1

    # Смеси
    for n in SAMPLE_SIZES:
        sample = generate_mixture_sample(n)
        title = f"Смесь, n={n}"
        plot_ellipse_only(axes[idx], sample, title)
        idx += 1

    # Если останутся пустые оси
    while idx < len(axes):
        axes[idx].axis("off")
        idx += 1

    plt.tight_layout()
    plt.show()


# =========================
# СОХРАНЕНИЕ РЕЗУЛЬТАТОВ
# =========================
def save_results(df, filename="lab5_results.xlsx"):
    df_to_save = df.copy()

    numeric_cols = [
        "pearson_mean", "pearson_var",
        "spearman_mean", "spearman_var",
        "quadrant_mean", "quadrant_var"
    ]
    df_to_save[numeric_cols] = df_to_save[numeric_cols].round(4)

    with pd.ExcelWriter(filename, engine="openpyxl") as writer:
        df_to_save.to_excel(writer, sheet_name="results", index=False)


# =========================
# ЗАПУСК
# =========================
if __name__ == "__main__":
    df_results = run_experiment()

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)

    print("ИТОГОВАЯ ТАБЛИЦА:")
    print(df_results.round(4))

    save_results(df_results, filename="lab5_results.xlsx")
    plot_all_ellipses()