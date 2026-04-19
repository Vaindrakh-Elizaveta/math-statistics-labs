import numpy as np
from scipy.stats import t, chi2, f


# ------------------------------------------------------------
# Лабораторная работа №8
# Доверительные интервалы. Критерий Стьюдента и F-тест
#
# По методичке:
# 1) выборки n1 = 20, n2 = 100 из N(0, 1)
# 2) ДИ для математического ожидания:
#       x̄ ± s * t_(1-α/2, n-1) / sqrt(n-1)
#    где s^2 = (1/n) * Σ(x_i - x̄)^2
# 3) ДИ для σ:
#       s*sqrt(n)/sqrt(χ²_(1-α/2, n-1)) < σ < s*sqrt(n)/sqrt(χ²_(α/2, n-1))
#    тогда для дисперсии σ² просто возводим границы в квадрат
# 4) F-тест на равенство дисперсий двух независимых выборок
#
# Формулы соответствуют разделам 3.1.1, 3.1.2 и 4.2 методички. :contentReference[oaicite:0]{index=0}
# В самой постановке лабораторной именно это и требуется для ЛР №8. :contentReference[oaicite:1]{index=1}
# ------------------------------------------------------------


ALPHA = 0.05
GAMMA = 1 - ALPHA
SEED = 42

N1 = 20
N2 = 100
MU = 0
SIGMA = 1


def sample_mean(x):
    return np.mean(x)


def sample_variance_mle_style(x):
    """
    Выборочная дисперсия в обозначениях методички:
    s^2 = (1/n) * sum((x_i - x̄)^2)
    """
    n = len(x)
    x_bar = sample_mean(x)
    return np.sum((x - x_bar) ** 2) / n


def confidence_interval_mean_normal(x, alpha=0.05):
    """
    Доверительный интервал для математического ожидания m
    по формуле методички:
        x̄ ± s * t_(1-α/2, n-1) / sqrt(n-1)
    """
    n = len(x)
    x_bar = sample_mean(x)
    s2 = sample_variance_mle_style(x)
    s = np.sqrt(s2)

    t_crit = t.ppf(1 - alpha / 2, df=n - 1)
    margin = s * t_crit / np.sqrt(n - 1)

    return x_bar - margin, x_bar + margin


def confidence_interval_sigma_normal(x, alpha=0.05):
    """
    Доверительный интервал для sigma по формуле методички:
        s*sqrt(n)/sqrt(χ²_(1-α/2, n-1)) < sigma <
        s*sqrt(n)/sqrt(χ²_(α/2, n-1))
    """
    n = len(x)
    s2 = sample_variance_mle_style(x)
    s = np.sqrt(s2)

    chi2_left = chi2.ppf(alpha / 2, df=n - 1)
    chi2_right = chi2.ppf(1 - alpha / 2, df=n - 1)

    sigma_left = s * np.sqrt(n) / np.sqrt(chi2_right)
    sigma_right = s * np.sqrt(n) / np.sqrt(chi2_left)

    return sigma_left, sigma_right


def confidence_interval_variance_normal(x, alpha=0.05):
    """
    Доверительный интервал для дисперсии sigma^2:
    просто квадрат интервала для sigma.
    """
    sigma_left, sigma_right = confidence_interval_sigma_normal(x, alpha)
    return sigma_left ** 2, sigma_right ** 2

def sample_variance_unbiased(x):
    n = len(x)
    x_bar = np.mean(x)
    return np.sum((x - x_bar) ** 2) / (n - 1)


def fisher_f_test_equal_variances(x1, x2, alpha=0.05):
    n1 = len(x1)
    n2 = len(x2)

    s1_2 = sample_variance_unbiased(x1)
    s2_2 = sample_variance_unbiased(x2)

    if s1_2 >= s2_2:
        f_stat = s1_2 / s2_2
        df1 = n1 - 1
        df2 = n2 - 1
        bigger = "s1^2"
    else:
        f_stat = s2_2 / s1_2
        df1 = n2 - 1
        df2 = n1 - 1
        bigger = "s2^2"

    f_crit_right = f.ppf(1 - alpha / 2, df1, df2)
    cdf_value = f.cdf(f_stat, df1, df2)
    p_value = 2 * min(cdf_value, 1 - cdf_value)

    reject_h0 = f_stat > f_crit_right

    return {
        "s1^2": s1_2,
        "s2^2": s2_2,
        "F": f_stat,
        "df1": df1,
        "df2": df2,
        "F_crit_right": f_crit_right,
        "p_value": p_value,
        "reject_h0": reject_h0,
        "bigger_variance_in_numerator": bigger,
    }

def analyze_sample(x, name, alpha=0.05):
    n = len(x)
    x_bar = sample_mean(x)
    s2 = sample_variance_mle_style(x)
    s = np.sqrt(s2)

    mean_ci = confidence_interval_mean_normal(x, alpha)
    sigma_ci = confidence_interval_sigma_normal(x, alpha)
    variance_ci = confidence_interval_variance_normal(x, alpha)

    print("=" * 70)
    print(f"{name}")
    print("=" * 70)
    print(f"Объём выборки n = {n}")
    print(f"Выборочное среднее x̄ = {x_bar:.6f}")
    print(f"Выборочная дисперсия s^2 = {s2:.6f}")
    print(f"Выборочное стандартное отклонение s = {s:.6f}")
    print()
    print(f"{int((1 - alpha) * 100)}%-й ДИ для математического ожидания m:")
    print(f"({mean_ci[0]:.6f}; {mean_ci[1]:.6f})")
    print()
    print(f"{int((1 - alpha) * 100)}%-й ДИ для стандартного отклонения σ:")
    print(f"({sigma_ci[0]:.6f}; {sigma_ci[1]:.6f})")
    print()
    print(f"{int((1 - alpha) * 100)}%-й ДИ для дисперсии σ^2:")
    print(f"({variance_ci[0]:.6f}; {variance_ci[1]:.6f})")
    print()


def main():
    rng = np.random.default_rng(SEED)

    sample1 = rng.normal(loc=MU, scale=SIGMA, size=N1)
    sample2 = rng.normal(loc=MU, scale=SIGMA, size=N2)

    print(f"Уровень значимости alpha = {ALPHA}")
    print(f"Доверительная вероятность gamma = {GAMMA}")
    print(f"Генератор: N({MU}, {SIGMA**2})")
    print(f"Seed = {SEED}")
    print()

    analyze_sample(sample1, "Выборка 1", ALPHA)
    analyze_sample(sample2, "Выборка 2", ALPHA)

    f_test = fisher_f_test_equal_variances(sample1, sample2, ALPHA)

    print("=" * 70)
    print("F-тест на равенство дисперсий")
    print("=" * 70)
    print("H0: σ1^2 = σ2^2")
    print("H1: σ1^2 != σ2^2")
    print()
    print(f"s1^2 = {f_test['s1^2']:.6f}")
    print(f"s2^2 = {f_test['s2^2']:.6f}")
    print(f"В числителе стоит большая дисперсия: {f_test['bigger_variance_in_numerator']}")
    print(f"Fнабл = {f_test['F']:.6f}")
    print(f"Степени свободы: df1 = {f_test['df1']}, df2 = {f_test['df2']}")
    print(f"Критическое значение F_(1-alpha/2) = {f_test['F_crit_right']:.6f}")
    print(f"p-value = {f_test['p_value']:.6f}")
    print()

    if f_test["reject_h0"]:
        print("Вывод: нулевая гипотеза H0 отвергается.")
        print("Дисперсии статистически значимо различаются.")
    else:
        print("Вывод: нет оснований отвергнуть нулевую гипотезу H0.")
        print("Статистически значимых различий дисперсий не обнаружено.")


if __name__ == "__main__":
    main()