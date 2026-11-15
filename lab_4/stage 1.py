
from itertools import product
import math

# Исходные параметры
A = (100, 800, 400)
C = 600

q = (25, 200, 100)
c = (0.04, 0.07, 0.05)
L = (30, 150, 100)

# Сценарии первого этапа
scenarios = [
    (0.60, (1.20, 1.10, 1.07)),
    (0.30, (1.05, 1.02, 1.03)),
    (0.10, (0.80, 0.95, 1.00))
]

EPS = 1e-9


# Ожидаемое число пакетов ЦБ1,
# которые можно продать на втором этапе
def b(u1):
    result = 0.0

    for p, multipliers in scenarios:
        a1 = (100 + 25 * u1) * multipliers[0]

        n = max(0, math.floor((a1 - 30) / 25 + EPS))

        result += p * n

    return result


# Целевая функция первого этапа
def F(u1, u2, u3):
    return (
        2015.8882088
        + 0.442225 * u1
        + 5.58456 * u2
        + 2.9452672 * u3
        + 0.285 * b(u1)
    )


# Проверка допустимости управления
def is_feasible(u1, u2, u3):

    u = (u1, u2, u3)

    new_cash = C

    for i in range(3):

        new_asset = A[i] + q[i] * u[i]

        # Ограничение на остаток актива
        if new_asset < L[i] - EPS:
            return False

        # Изменение денежных средств
        if u[i] > 0:
            new_cash -= q[i] * u[i] * (1 + c[i])

        elif u[i] < 0:
            new_cash += q[i] * abs(u[i]) * (1 - c[i])

    return new_cash >= -EPS


# Границы перебора
max_sell = [
    max(0, math.floor((A[i] - L[i]) / q[i] + EPS))
    for i in range(3)
]

max_sale_income = sum(
    max_sell[i] * q[i] * (1 - c[i])
    for i in range(3)
)

max_buy = [
    math.floor(
        (C + max_sale_income) /
        (q[i] * (1 + c[i])) + EPS
    )
    for i in range(3)
]

ranges = [
    range(-max_sell[i], max_buy[i] + 1)
    for i in range(3)
]


# Полный перебор
best_value = -float("inf")
best_action = None
feasible_count = 0

for u in product(*ranges):

    if not is_feasible(*u):
        continue

    feasible_count += 1

    value = F(*u)

    if value > best_value:
        best_value = value
        best_action = u


# Вывод результатов
print("Количество допустимых управлений:", feasible_count)
print("Оптимальное управление:", best_action)
print("Максимальное значение F:", best_value)
print("b(u1):", b(best_action[0]))
