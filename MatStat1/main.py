import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, cauchy, laplace, poisson, uniform

rng = np.random.default_rng(42)

Ns = [10, 100, 1000]

# Параметры из задания:
# 1) N(x;0,1)
# 2) C(x;0,1)
# 3) L(x;0, 1/sqrt(2))  <-- в scipy это scale=b, где Var=2b^2, значит b=1/sqrt(2)
# 4) P(k;5)            <-- lambda=5
# 5) U(x; -sqrt(3), sqrt(3))  <-- в scipy uniform(loc=a, scale=b-a)

sqrt2 = np.sqrt(2)
sqrt3 = np.sqrt(3)

dists = [
    ("Normal N(0,1)",        norm(loc=0, scale=1),                      "continuous"),
    ("Cauchy C(0,1)",        cauchy(loc=0, scale=1),                    "continuous"),
    ("Laplace L(0,1/sqrt2)", laplace(loc=0, scale=1/sqrt2),             "continuous"),
    ("Poisson P(5)",        poisson(mu=5),                            "discrete"),
    ("Uniform U(-sqrt3,sqrt3)", uniform(loc=-sqrt3, scale=2*sqrt3),     "continuous"),
]

def choose_bins(x: np.ndarray) -> int:
    """Нормальный выбор числа бинов: FD, а если не вышло — Стержеса."""
    x = np.asarray(x)
    if x.size < 2:
        return 1
    try:
        edges = np.histogram_bin_edges(x, bins="fd")
        k = max(5, min(60, len(edges) - 1))
        return k
    except Exception:
        k = int(np.ceil(np.log2(x.size) + 1)) #формула Стержеса
        return max(5, min(60, k))

def x_grid_from_data(x: np.ndarray, kind: str, dist_obj):
    """Сетка по оси X: аккуратно режем хвосты, чтобы график был читабельным."""
    x = np.asarray(x)

    if kind == "discrete":
        # для Пуассона: вокруг основной массы вероятности
        lo, hi = dist_obj.ppf([0.001, 0.999]) #нижний квантиль, верхний квантиль
        lo = int(max(0, np.floor(lo)))
        hi = int(np.ceil(hi))
        return np.arange(lo, hi + 1)

    # для непрерывных: ограничим хвосты, особенно для Коши
    if x.size >= 30:
        q1, q2 = np.quantile(x, [0.01, 0.99])
        pad = 0.15 * (q2 - q1) if q2 > q1 else 1.0 #берём запас 15% ширины
        xmin, xmax = q1 - pad, q2 + pad
    else:
        xmin, xmax = np.min(x), np.max(x)
        pad = 0.25 * (xmax - xmin) if xmax > xmin else 1.0
        xmin, xmax = xmin - pad, xmax + pad

    # защита от совсем “улетевших” хвостов
    if not np.isfinite([xmin, xmax]).all():
        xmin, xmax = -5, 5

    return np.linspace(xmin, xmax, 600)

# 5 распределений x 3 объёма = 15 графиков (сделаем в одной фигуре 5x3)
fig, axes = plt.subplots(len(dists), len(Ns), figsize=(15, 18), constrained_layout=True)

for i, (title, dist_obj, kind) in enumerate(dists):
    for j, n in enumerate(Ns):
        ax = axes[i, j]

        # Генерация выборки
        if kind == "discrete":
            sample = dist_obj.rvs(size=n, random_state=rng)
        else:
            sample = dist_obj.rvs(size=n, random_state=rng)

        # Построение
        if kind == "discrete":
            xs = x_grid_from_data(sample, kind, dist_obj)
            # "гистограмма" для дискретного: столбики относительных частот
            values, counts = np.unique(sample, return_counts=True)
            freqs = counts / n
            ax.bar(values, freqs, width=0.8, align="center", alpha=0.6, label="Empirical freq")

            pmf = dist_obj.pmf(xs)
            ax.plot(xs, pmf, marker="o", linewidth=1.5, label="Theoretical PMF")

            ax.set_xlim(xs.min() - 1, xs.max() + 1)
            ax.set_ylabel("Probability")
        else:
            # для Коши ограничиваем диапазон отображения по квантилям
            if "Cauchy" in title:
                left, right = dist_obj.ppf([0.01, 0.99])
                sample_plot = sample[(sample >= left) & (sample <= right)]
                bins = choose_bins(sample_plot)

                ax.hist(sample_plot, bins=bins, density=True, alpha=0.6, label="Histogram (density)")

                xs = np.linspace(left, right, 600)
                pdf = dist_obj.pdf(xs)
                ax.plot(xs, pdf, linewidth=2, label="Theoretical PDF")

                ax.set_xlim(left, right)
            else:
                bins = choose_bins(sample)
                ax.hist(sample, bins=bins, density=True, alpha=0.6, label="Histogram (density)")

                xs = x_grid_from_data(sample, kind, dist_obj)
                pdf = dist_obj.pdf(xs)
                ax.plot(xs, pdf, linewidth=2, label="Theoretical PDF")

            ax.set_ylabel("Density")

        ax.set_title(f"{title}\n n = {n}")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=9)

plt.show()