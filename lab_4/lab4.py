import math
import time
import tracemalloc
import numpy as np

# исходные данные модели
DEFAULT_DATA = {
    "stages": 3,  # количество этапов
    # начальное состояние портфеля
    "initial_state": (100.0, 800.0, 400.0, 600.0),  # (ЦБ1, ЦБ2, Деп, Свободные)
    "pack_sizes": (25.0, 200.0, 100.0),  # Размер одного торгового пакета для каждого актива.
    "commissions": (0.04, 0.07, 0.05),  # Комиссионные издержки при покупке и продаже активов.
    "min_limits": (30.0, 150.0, 100.0),  # Минимально допустимые остатки активов после продажи.
    # Рыночные сценарии для каждого этапа.
    # Каждый сценарий задаётся парой:
    # (вероятность реализации, вектор коэффициентов изменения стоимости).
    "scenarios": [
        # Этап 1
        [(0.60, (1.20, 1.10, 1.07)), (0.30, (1.05, 1.02, 1.03)), (0.10, (0.80, 0.95, 1.00))],
        # Этап 2
        [(0.30, (1.40, 1.15, 1.01)), (0.20, (1.05, 1.00, 1.00)), (0.50, (0.60, 0.90, 1.00))],
        # Этап 3
        [(0.40, (1.15, 1.12, 1.05)), (0.40, (1.05, 1.01, 1.01)), (0.20, (0.70, 0.94, 1.00))]
    ]
}

EPS = 1e-7


class PortfolioDP:
    def __init__(self, data):
        # проверяется корректность исходных данных
        self.validate(data)
        # сохранение основных параметров задачи.
        self.N = data["stages"]
        self.s0 = tuple(float(x) for x in data["initial_state"])
        self.q = tuple(float(x) for x in data["pack_sizes"])
        self.c = tuple(float(x) for x in data["commissions"])
        self.L = tuple(float(x) for x in data["min_limits"])
        self.scenarios = data["scenarios"]
        # Словарь мемоизации для хранения уже вычисленных
        # значений функции Беллмана и соответствующих оптимальных действий.
        self.memo = {}
        # Преобразование параметров в массивы NumPy
        # для последующего выполнения векторизированных операций.
        self.q_arr = np.array(self.q)
        self.c_arr = np.array(self.c)
        self.L_arr = np.array(self.L)
        # Предварительное вычисление математических ожиданий
        # коэффициентов изменения стоимости активов на последнем этапе.
        self.Em_last = np.array([sum(p * m[i] for p, m in self.scenarios[-1]) for i in range(3)])
        # Оценка ожидаемого финансового результата покупки и продажи
        # одного пакета каждого актива на последнем этапе.
        gain_buy = self.q_arr * (self.Em_last - (1.0 + self.c_arr))
        gain_sell = self.q_arr * ((1.0 - self.c_arr) - self.Em_last)
        # Проверка возможности аналитического упрощения последнего этапа.
        # Если ожидаемый результат покупки и продажи отрицателен
        # для всех активов, оптимальным является отсутствие сделок.
        self.last_linear = bool(np.all(gain_buy < -EPS) and np.all(gain_sell < -EPS))

    @staticmethod
    def validate(data):
        # Проверка положительности количества этапов.
        if data["stages"] <= 0:
            raise ValueError("Число этапов должно быть положительным.")
        # Количество наборов сценариев должно соответствовать
        # количеству этапов планирования.
        if len(data["scenarios"]) != data["stages"]:
            raise ValueError("Неполная таблица сценариев: не совпадает с числом этапов.")
        # Начальные объёмы активов и денежных средств
        # не должны принимать отрицательные значения.
        if any(x < 0 for x in data["initial_state"]):
            raise ValueError("Начальные суммы активов и средств не могут быть отрицательными.")
        # Размер каждого торгового пакета должен быть положительным.
        if any(q <= 0 for q in data["pack_sizes"]):
            raise ValueError("Размеры пакетов должны быть строго положительными.")
        # Для каждого этапа проверяется нормировка вероятностей:
        # сумма вероятностей всех сценариев должна равняться единице.
        for k, stage_scen in enumerate(data["scenarios"]):
            prob_sum = sum(p for p, _ in stage_scen)
            if not math.isclose(prob_sum, 1.0, rel_tol=1e-5):
                raise ValueError(f"Сумма вероятностей на этапе {k + 1} не равна 1 (получено {prob_sum}).")
            for _, m in stage_scen:
                if len(m) != 3:
                    raise ValueError(f"Неполный набор множителей на этапе {k + 1}: ожидалось 3, получено {len(m)}.")

    def _cash_delta(self, idx, u_val):
        # положительно значение - покупка
        if u_val > 0:
            return -u_val * self.q[idx] * (1.0 + self.c[idx])
        # отрицательное  - продажа
        elif u_val < 0:
            return abs(u_val) * self.q[idx] * (1.0 - self.c[idx])
        # 0 - отсутсвтие сделки
        return 0.0

    def feasible_array(self, state):
        """Формирование множества допустимых управляющих воздействий U(s).
            Результат представляется матрицей размера (n, 3), где каждая строка соответствует отдельной комбинации сделок.
        """
        # Извлечение текущих объёмов активов и денежных средств.
        A = np.array(state[:3])
        cash = state[3]
        # Определение максимально возможного количества пакетов
        # для продажи с сохранением установленных минимальных остатков.
        max_sell = np.where(A >= self.L_arr, np.floor((A - self.L_arr + EPS) / self.q_arr), 0).astype(int)
        # Расчёт потенциальной выручки от продажи максимально допустимого количества пакетов всех активов.
        gain = float(np.sum(max_sell * self.q_arr * (1.0 - self.c_arr)))
        # Определение максимального количества пакетов,
        # которое можно приобрести с учётом имеющихся средств
        # и возможной выручки от продаж.
        max_buy = np.floor((cash + gain + EPS) / (self.q_arr * (1.0 + self.c_arr))).astype(int)
        # Формирование диапазонов управляющих переменных:
        # отрицательные значения — продажа,
        # положительные — покупка,
        # нулевое значение — отсутствие операции.
        axes = [np.arange(-max_sell[i], max_buy[i] + 1) for i in range(3)]
        # комбинаций управляющих воздействий
        U = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
        # Расчёт изменения денежных средств для каждой комбинации.
        # При покупке учитывается полная стоимость с комиссией,
        # при продаже — поступление средств за вычетом комиссии.
        delta = (np.where(U > 0, -U * self.q_arr * (1.0 + self.c_arr), 0.0)
                 + np.where(U < 0, -U * self.q_arr * (1.0 - self.c_arr), 0.0))
        # Исключение комбинаций, при которых денежный баланс
        # становится отрицательным.
        return U[cash + delta.sum(axis=1) >= -EPS]

    def get_feasible_actions(self, state):
        """Множество допустимых управлений U(s) (список кортежей)."""
        return [tuple(int(x) for x in u) for u in self.feasible_array(state)]

    def apply_trade(self, state, u):
        """Промежуточное состояние s' после совершения сделок."""
        a1, a2, d, cash = state
        new_cash = cash + self._cash_delta(0, u[0]) + self._cash_delta(1, u[1]) + self._cash_delta(2, u[2])
        # Формирование нового состояния портфеля.
        return (
            round(a1 + u[0] * self.q[0], 8),
            round(a2 + u[1] * self.q[1], 8),
            round(d + u[2] * self.q[2], 8),
            round(new_cash, 8)
        )

    def next_state(self, state_after_trade, mults):
        """Переход f(s, u, j) после реализации рыночного сценария."""
        a1, a2, d, cash = state_after_trade
        return (
            round(a1 * mults[0], 8),
            round(a2 * mults[1], 8),
            round(d * mults[2], 8),
            round(cash, 8)
        )

    def solve(self):
        """Обратный проход"""
        N = self.N

        def bellman(k, state):
            if k == N + 1:
                return sum(state), [None]
            if k == N and self.last_linear:  # последний этап: замкнутая формула
                return float(np.dot(self.Em_last, state[:3]) + state[3]), [(0, 0, 0)]
            key = (k, tuple(round(x, 7) for x in state))
            if key in self.memo:
                return self.memo[key]

            U = self.feasible_array(state)
            A = np.array(state[:3]) + U * self.q_arr  # сделки (s')
            delta = (np.where(U > 0, -U * self.q_arr * (1.0 + self.c_arr), 0.0)
                     + np.where(U < 0, -U * self.q_arr * (1.0 - self.c_arr), 0.0))
            P = np.column_stack([A, state[3] + delta.sum(axis=1)])
            vals = np.zeros(len(U))
            for prob, mults in self.scenarios[k - 1]:
                S2 = P * np.array(list(mults) + [1.0])  # рыночный сценарий
                if k == N:
                    v = S2.sum(axis=1)
                elif k + 1 == N and self.last_linear:
                    v = S2[:, :3] @ self.Em_last + S2[:, 3]
                else:
                    v = np.array([bellman(k + 1, tuple(r))[0] for r in S2])
                vals += prob * v
            best = vals.max()
            actions = [tuple(int(x) for x in U[i]) for i in np.where(vals >= best - EPS)[0]]
            self.memo[key] = (float(best), actions)
            return self.memo[key]

        opt_val, opt_actions = bellman(1, self.s0)
        return opt_val, opt_actions, bellman


# ПРОВЕРКА ПОЛНЫМ ПЕРЕБОРОМ НА МАЛОМ КОНТРОЛЬНОМ ПРИМЕРЕ (п. 8.2)
def brute_force_check(solver):
    def dfs(k, state):
        if k == solver.N + 1:
            return sum(state), []
        best_v = float("-inf")
        best_u = []
        for u in solver.get_feasible_actions(state):
            s_p = solver.apply_trade(state, u)
            ev = sum(p * dfs(k + 1, solver.next_state(s_p, m))[0] for p, m in solver.scenarios[k - 1])
            if ev > best_v + EPS:
                best_v = ev
                best_u = [u]
            elif abs(ev - best_v) <= EPS:
                best_u.append(u)
        return best_v, best_u

    return dfs(1, solver.s0)


if __name__ == "__main__":
    tracemalloc.start()
    t0 = time.perf_counter()

    dp = PortfolioDP(DEFAULT_DATA)
    opt_capital, opt_u1, bellman_fn = dp.solve()

    elapsed = (time.perf_counter() - t0) * 1000
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    w0 = sum(DEFAULT_DATA["initial_state"])
    print(f"Ожидаемый конечный капитал V1(s1): {opt_capital:.7f} д.е.")
    print(f"Ожидаемый чистый доход E[W4 - W0]: {opt_capital - w0:.7f} д.е.")
    print(f"Оптимальное управление на 1-м этапе u*1: {opt_u1}")
    print(f"Время выполнения ДП: {elapsed:.2f} мс | Пиковая память: {peak_mem / 1024:.2f} КБ\n")

    # Проверка полным перебором на малом контрольном примере (2 этапа, небольшие суммы)
    small_data = {
        **DEFAULT_DATA,
        "stages": 2,
        "initial_state": (50.0, 200.0, 100.0, 100.0),
        "scenarios": DEFAULT_DATA["scenarios"][:2]
    }
    small_dp = PortfolioDP(small_data)
    dp_small_val, dp_small_u1, _ = small_dp.solve()
    bf_small_val, bf_small_u1 = brute_force_check(small_dp)
    assert math.isclose(dp_small_val, bf_small_val, abs_tol=EPS)
    assert dp_small_u1 == bf_small_u1
    print(f"[OK] Полный перебор на контрольном примере совпал с ДП: {bf_small_val:.6f} д.е., u*1 = {bf_small_u1}\n")

    # Вывод ключевых состояний для сравнения с Таблицей 5 ручного расчета
    print("--- Контрольные состояния (сравнение с ручным расчетом) ---")
    s1_prime = dp.apply_trade(dp.s0, opt_u1[0])
    for name, (p1, m1) in zip(["Благопр.", "Нейтр.", "Негатив."], dp.scenarios[0]):
        s2 = dp.next_state(s1_prime, m1)
        v2, u2_list = bellman_fn(2, s2)
        print(f"Этап 2 ({name}, p={p1}): s2={s2} -> u*2={u2_list[0]}, V2(s2)={v2:.6f}")
