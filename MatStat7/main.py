import numpy as np
from scipy.stats import norm, chi2


def mle_normal_params(sample):
    mu_hat = np.mean(sample)
    sigma2_hat = np.mean((sample - mu_hat) ** 2)
    sigma_hat = np.sqrt(sigma2_hat)
    return mu_hat, sigma_hat, sigma2_hat


def sturges_bins(n):
    k = int(np.ceil(1 + 3.3 * np.log10(n)))
    return max(k, 2)


def initial_bin_edges(sample, k):
    mu_hat, sigma_hat, _ = mle_normal_params(sample)

    if sigma_hat < 1e-12:
        sigma_hat = 1e-12

    probs = np.linspace(0, 1, k + 1)
    edges = norm.ppf(probs, loc=mu_hat, scale=sigma_hat)
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges


def observed_counts(sample, edges):
    counts = np.zeros(len(edges) - 1, dtype=int)

    for x in sample:
        idx = np.searchsorted(edges, x, side='left') - 1
        if idx < 0:
            idx = 0
        if idx >= len(counts):
            idx = len(counts) - 1
        counts[idx] += 1

    return counts


def expected_probs_normal(edges, mu_hat, sigma_hat):
    cdf_vals = norm.cdf(edges, loc=mu_hat, scale=sigma_hat)
    probs = np.diff(cdf_vals)
    probs = np.clip(probs, 0, None)
    probs = probs / probs.sum()
    return probs


def merge_bins_until_expected_ge_5(counts, probs, min_expected=5, n_total=None):
    if n_total is None:
        n_total = counts.sum()

    counts = counts.astype(float).tolist()
    probs = probs.astype(float).tolist()

    while True:
        expected = [n_total * p for p in probs]
        bad = [i for i, e in enumerate(expected) if e < min_expected]

        if not bad:
            break

        i = bad[0]

        if len(counts) <= 1:
            break

        if i == 0:
            counts[1] += counts[0]
            probs[1] += probs[0]
            del counts[0]
            del probs[0]
        elif i == len(counts) - 1:
            counts[i - 1] += counts[i]
            probs[i - 1] += probs[i]
            del counts[i]
            del probs[i]
        else:
            left_exp = n_total * probs[i - 1]
            right_exp = n_total * probs[i + 1]

            if left_exp <= right_exp:
                counts[i - 1] += counts[i]
                probs[i - 1] += probs[i]
                del counts[i]
                del probs[i]
            else:
                counts[i + 1] += counts[i]
                probs[i + 1] += probs[i]
                del counts[i]
                del probs[i]

    return np.array(counts), np.array(probs)


def chi_square_test_normal(sample, alpha=0.05, verbose=True):
    n = len(sample)
    mu_hat, sigma_hat, sigma2_hat = mle_normal_params(sample)

    # Для малых выборок специально берём мало интервалов,
    # чтобы критерий chi^2 был применим после оценки mu и sigma.
    if n <= 30:
        k0 = 4
    else:
        k0 = sturges_bins(n)

    edges = initial_bin_edges(sample, k0)

    counts = observed_counts(sample, edges)
    probs = expected_probs_normal(edges, mu_hat, sigma_hat)

    counts_merged, probs_merged = merge_bins_until_expected_ge_5(
        counts, probs, min_expected=5, n_total=n
    )

    expected_merged = n * probs_merged
    m = len(counts_merged)
    r = 2
    df = m - 1 - r

    if df <= 0:
        result = {
            "applicable": False,
            "reason": "Слишком мало интервалов после объединения, критерий chi^2 неприменим.",
            "n": n,
            "alpha": alpha,
            "mu_hat": mu_hat,
            "sigma_hat": sigma_hat,
            "sigma2_hat": sigma2_hat,
            "k_initial": k0,
            "m_final": m,
            "df": df,
            "observed_counts": counts_merged,
            "expected_counts": expected_merged,
        }

        if verbose:
            print("=" * 70)
            print(f"Объём выборки n = {n}")
            print(f"ММП-оценка mu_hat = {mu_hat:.6f}")
            print(f"ММП-оценка sigma_hat = {sigma_hat:.6f}")
            print(f"Начальное число интервалов k = {k0}")
            print(f"Итоговое число интервалов m = {m}")
            print(f"Степени свободы df = {df}")
            print("Критерий chi^2 в этом прогоне неприменим.")
        return result

    chi2_obs = np.sum((counts_merged - expected_merged) ** 2 / expected_merged)
    chi2_crit = chi2.ppf(1 - alpha, df)
    p_value = 1 - chi2.cdf(chi2_obs, df)
    reject = chi2_obs > chi2_crit

    result = {
        "applicable": True,
        "n": n,
        "alpha": alpha,
        "mu_hat": mu_hat,
        "sigma_hat": sigma_hat,
        "sigma2_hat": sigma2_hat,
        "k_initial": k0,
        "m_final": m,
        "df": df,
        "observed_counts": counts_merged,
        "expected_counts": expected_merged,
        "chi2_obs": chi2_obs,
        "chi2_crit": chi2_crit,
        "p_value": p_value,
        "reject_h0": reject,
    }

    if verbose:
        print("=" * 70)
        print(f"Объём выборки n = {n}")
        print(f"Уровень значимости alpha = {alpha}")
        print(f"ММП-оценка mu_hat = {mu_hat:.6f}")
        print(f"ММП-оценка sigma_hat = {sigma_hat:.6f}")
        print(f"ММП-оценка sigma_hat^2 = {sigma2_hat:.6f}")
        print(f"Начальное число интервалов k = {k0}")
        print(f"Итоговое число интервалов после объединения m = {m}")
        print(f"Степени свободы df = {df}")
        print("\nНаблюдаемые частоты ni:")
        print(np.round(counts_merged, 4))
        print("Ожидаемые частоты n*pi:")
        print(np.round(expected_merged, 4))
        print(f"\nНаблюдаемое значение chi^2 = {chi2_obs:.6f}")
        print(f"Критическое значение chi^2_(1-alpha, df) = {chi2_crit:.6f}")
        print(f"p-value = {p_value:.6f}")

        if reject:
            print("РЕШЕНИЕ: H0 отвергается.")
        else:
            print("РЕШЕНИЕ: нет оснований отвергать H0.")

    return result

def sensitivity_experiment(generator, n=20, alpha=0.05, repeats=1000, seed=42):
    rng = np.random.default_rng(seed)
    rejections = 0
    applicable_runs = 0
    skipped_runs = 0

    for _ in range(repeats):
        sample = generator(rng, n)
        result = chi_square_test_normal(sample, alpha=alpha, verbose=False)

        if not result["applicable"]:
            skipped_runs += 1
            continue

        applicable_runs += 1
        if result["reject_h0"]:
            rejections += 1

    if applicable_runs == 0:
        return {
            "power_estimate": None,
            "applicable_runs": 0,
            "skipped_runs": skipped_runs,
        }

    return {
        "power_estimate": rejections / applicable_runs,
        "applicable_runs": applicable_runs,
        "skipped_runs": skipped_runs,
    }


def generate_normal_sample(rng, n):
    return rng.normal(loc=0.0, scale=1.0, size=n)


def generate_uniform_sample(rng, n):
    a = -np.sqrt(3)
    b = np.sqrt(3)
    return rng.uniform(a, b, size=n)


def generate_laplace_sample(rng, n):
    return rng.laplace(loc=0.0, scale=1 / np.sqrt(2), size=n)


def main():
    alpha = 0.05
    rng = np.random.default_rng(42)

    normal_sample = generate_normal_sample(rng, 100)

    print("\nПУНКТЫ 1-3: ПРОВЕРКА НОРМАЛЬНОЙ ВЫБОРКИ")
    normal_result = chi_square_test_normal(normal_sample, alpha=alpha, verbose=True)

    print("\n" + "=" * 70)
    print("ПУНКТ 4: ИССЛЕДОВАНИЕ ЧУВСТВИТЕЛЬНОСТИ КРИТЕРИЯ")
    print("Альтернативные распределения: равномерное и Лапласа")
    print("Объём выборки n = 20")
    print("Повторов = 1000")

    uniform_result = sensitivity_experiment(
        generate_uniform_sample, n=20, alpha=alpha, repeats=1000, seed=123
    )
    laplace_result = sensitivity_experiment(
        generate_laplace_sample, n=20, alpha=alpha, repeats=1000, seed=456
    )

    print("\nРавномерное распределение:")
    print(f"Число применимых прогонов: {uniform_result['applicable_runs']}")
    print(f"Число пропущенных прогонов: {uniform_result['skipped_runs']}")
    if uniform_result["power_estimate"] is not None:
        print(f"Оценка чувствительности: {uniform_result['power_estimate']:.4f}")
    else:
        print("Оценка чувствительности не вычислена.")

    print("\nРаспределение Лапласа:")
    print(f"Число применимых прогонов: {laplace_result['applicable_runs']}")
    print(f"Число пропущенных прогонов: {laplace_result['skipped_runs']}")
    if laplace_result["power_estimate"] is not None:
        print(f"Оценка чувствительности: {laplace_result['power_estimate']:.4f}")
    else:
        print("Оценка чувствительности не вычислена.")

    print("\nИнтерпретация:")
    print("Чем больше оценка чувствительности, тем чаще критерий отвергает гипотезу нормальности.")
    print("Пропущенные прогоны означают, что для некоторых малых выборок критерий chi^2")
    print("оказался неприменим из-за слишком малого числа интервалов после объединения.")


if __name__ == "__main__":
    main()