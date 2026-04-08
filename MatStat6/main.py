import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from openpyxl import Workbook

# -----------------------------
# Параметры задачи
# -----------------------------
np.random.seed(42)

a_true = 2
b_true = 2

x = np.arange(-1.8, 2.0 + 0.0001, 0.2)
eps = np.random.normal(0, 1, len(x))
y = a_true + b_true * x + eps

# Данные с выбросами
y_out = y.copy()
y_out[0] += 10
y_out[-1] -= 10


# -----------------------------
# МНК
# -----------------------------
def lsm(x, y):
    b, a = np.polyfit(x, y, 1)
    return a, b


# -----------------------------
# МНМ
# -----------------------------
def lad(x, y):
    def objective(params):
        a, b = params
        return np.sum(np.abs(y - (a + b * x)))

    a0, b0 = lsm(x, y)
    result = minimize(objective, x0=[a0, b0], method='Nelder-Mead')
    return result.x


# -----------------------------
# Ошибки
# -----------------------------
def calc_errors(a_hat, b_hat):
    delta_a = abs(a_true - a_hat)
    delta_b = abs(b_true - b_hat)

    rel_a = delta_a / abs(a_true) * 100
    rel_b = delta_b / abs(b_true) * 100

    return delta_a, rel_a, delta_b, rel_b


# -----------------------------
# Вычисления
# -----------------------------
a_lsm, b_lsm = lsm(x, y)
a_lad, b_lad = lad(x, y)

a_lsm_out, b_lsm_out = lsm(x, y_out)
a_lad_out, b_lad_out = lad(x, y_out)


# -----------------------------
# Excel
# -----------------------------
wb = Workbook()

# Лист без выбросов
ws1 = wb.active
ws1.title = "Без выбросов"

headers = ["Метод", "a", "Δa", "δa (%)", "b", "Δb", "δb (%)"]
ws1.append(headers)

for name, a_hat, b_hat in [
    ("МНК", a_lsm, b_lsm),
    ("МНМ", a_lad, b_lad)
]:
    da, ra, db, rb = calc_errors(a_hat, b_hat)
    ws1.append([name, a_hat, da, ra, b_hat, db, rb])


# Лист с выбросами
ws2 = wb.create_sheet("С выбросами")
ws2.append(headers)

for name, a_hat, b_hat in [
    ("МНК", a_lsm_out, b_lsm_out),
    ("МНМ", a_lad_out, b_lad_out)
]:
    da, ra, db, rb = calc_errors(a_hat, b_hat)
    ws2.append([name, a_hat, da, ra, b_hat, db, rb])


# Сохранение файла
wb.save("regression_results.xlsx")


# -----------------------------
# Графики
# -----------------------------
x_line = np.linspace(x.min(), x.max(), 300)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Без выбросов
axes[0].scatter(x, y)
axes[0].plot(x_line, a_true + b_true * x_line, label='Истинная')
axes[0].plot(x_line, a_lsm + b_lsm * x_line, label='МНК')
axes[0].plot(x_line, a_lad + b_lad * x_line, label='МНМ')
axes[0].set_title('Без выбросов')
axes[0].legend()
axes[0].grid()

# С выбросами
axes[1].scatter(x, y_out)
axes[1].plot(x_line, a_true + b_true * x_line, label='Истинная')
axes[1].plot(x_line, a_lsm_out + b_lsm_out * x_line, label='МНК')
axes[1].plot(x_line, a_lad_out + b_lad_out * x_line, label='МНМ')
axes[1].set_title('С выбросами')
axes[1].legend()
axes[1].grid()

plt.show()