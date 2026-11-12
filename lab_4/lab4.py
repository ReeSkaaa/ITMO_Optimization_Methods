import tracemalloc
from functools import lru_cache
from itertools import product
from math import floor
from time import perf_counter

START_ASSETS = (100.0, 800.0, 400.0)
START_CASH = 600.0
MIN_LIMITS = (30.0, 150.0, 100.0)
PACK_SIZES = (25.0, 200.0, 100.0)
BROKER_FEES = (0.04, 0.07, 0.05)

TOLERANCE = 1e-8
ENABLE_PRUNING = True

MARKET_STAGES = (
    ((0.60, (1.20, 1.10, 1.07)),
     (0.30, (1.05, 1.02, 1.03)),
     (0.10, (0.80, 0.95, 1.00))),
    ((0.30, (1.40, 1.15, 1.01)),
     (0.20, (1.05, 1.00, 1.00)),
     (0.50, (0.60, 0.90, 1.00))),
    ((0.40, (1.15, 1.12, 1.05)),
     (0.40, (1.05, 1.01, 1.01)),
     (0.20, (0.70, 0.94, 1.00))),
)


def check_parameters():
    """Проверка корректности входных параметров модели."""
    if not (len(START_ASSETS) == len(MIN_LIMITS) == len(PACK_SIZES) == len(BROKER_FEES) == 3):
        raise ValueError("Параметры должны быть заданы для всех трех активов")

    if START_CASH < 0 or any(val < 0 for val in START_ASSETS):
        raise ValueError("Стартовые объемы не могут быть отрицательными")

    if any(size <= 0 for size in PACK_SIZES):
        raise ValueError("Размер пакета должен быть строго больше нуля")

    if any(not 0 <= rate < 1 for rate in BROKER_FEES):
        raise ValueError("Комиссия должна лежать в полуинтервале [0, 1)")

    if any(lim < 0 for lim in MIN_LIMITS):
        raise ValueError("Минимальные остатки не могут быть отрицательными")

    if len(MARKET_STAGES) != 3:
        raise ValueError("Период планирования должен состоять из трех этапов")

    for stage_data in MARKET_STAGES:
        if abs(sum(prob for prob, _ in stage_data) - 1.0) > TOLERANCE:
            raise ValueError("Сумма вероятностей на этапе должна быть равна 1")
        for prob, coeffs in stage_data:
            if prob < 0 or len(coeffs) != 3 or any(c <= 0 for c in coeffs):
                raise ValueError("Неверно задана вероятность или множитель доходности")


def generate_valid_trades(current_state, expected_growth=None):
    """Формирует список допустимых сделок (состояние_после_сделки, вектор_управления)."""
    holdings, free_cash = current_state[:3], current_state[3]

    sell_caps = [
        max(0, floor((holdings[idx] - MIN_LIMITS[idx] + TOLERANCE) / PACK_SIZES[idx]))
        for idx in range(3)
    ]

    max_sale_revenue = sum(
        sell_caps[idx] * PACK_SIZES[idx] * (1.0 - BROKER_FEES[idx]) for idx in range(3)
    )

    trade_bounds = []
    for idx in range(3):
        buy_cap = max(
            0,
            floor((free_cash + max_sale_revenue + TOLERANCE)
                  / (PACK_SIZES[idx] * (1.0 + BROKER_FEES[idx])))
        )
        lower, upper = -sell_caps[idx], buy_cap

        if expected_growth is not None:
            growth = expected_growth[idx]
            if growth > 1.0 + BROKER_FEES[idx] + TOLERANCE:
                lower = 0
            elif growth < 1.0 - BROKER_FEES[idx] - TOLERANCE:
                upper = 0
            elif (abs(growth - (1.0 + BROKER_FEES[idx])) <= TOLERANCE or
                  abs(growth - (1.0 - BROKER_FEES[idx])) <= TOLERANCE):
                pass
            else:
                lower = upper = 0
        trade_bounds.append(range(lower, upper + 1))

    valid_list = []
    for action in product(*trade_bounds):
        updated_holdings = tuple(holdings[idx] + action[idx] * PACK_SIZES[idx] for idx in range(3))
        updated_cash = free_cash

        for idx, packs in enumerate(action):
            volume = packs * PACK_SIZES[idx]
            if packs > 0:
                updated_cash -= volume * (1.0 + BROKER_FEES[idx])
            elif packs < 0:
                updated_cash += (-volume) * (1.0 - BROKER_FEES[idx])

        if updated_cash < -TOLERANCE:
            continue

        if any(action[idx] < 0 and updated_holdings[idx] < MIN_LIMITS[idx] - TOLERANCE for idx in range(3)):
            continue

        valid_list.append((
            tuple(round(val, 8) for val in (*updated_holdings, updated_cash)),
            tuple(action),
        ))
    return valid_list


def apply_market_step(post_trade_state, coeffs):
    """Пересчитывает стоимость активов после наступления рыночного сценария."""
    return tuple(round(val, 8) for val in (
        post_trade_state[0] * coeffs[0],
        post_trade_state[1] * coeffs[1],
        post_trade_state[2] * coeffs[2],
        post_trade_state[3],
    ))


def calc_total_capital(current_state):
    """Вычисляет общую стоимость портфеля и свободных средств."""
    return sum(current_state)


def calc_mean_coeffs(stage_idx):
    """Находит ожидаемые множители роста для каждого актива на этапе."""
    return tuple(
        sum(prob * coeffs[idx] for prob, coeffs in MARKET_STAGES[stage_idx])
        for idx in range(3)
    )


def within_commission_band(mean_coeffs):
    """Проверяет, попадают ли средние множители внутрь интервала комиссий."""
    return all(
        1.0 - fee + TOLERANCE < mean < 1.0 + fee - TOLERANCE
        for mean, fee in zip(mean_coeffs, BROKER_FEES)
    )


def evaluate_last_stage(current_state):
    """Вычисляет оптимальное значение функции Беллмана для финального этапа (k = 3)."""
    means = calc_mean_coeffs(2)
    if within_commission_band(means):
        exp_cap = sum(current_state[idx] * means[idx] for idx in range(3)) + current_state[3]
        return exp_cap, (((0, 0, 0), tuple(current_state)),)

    max_val = float("-inf")
    optimal_choices = []
    for post_trade, action in generate_valid_trades(current_state):
        exp_val = sum(
            prob * calc_total_capital(apply_market_step(post_trade, coeffs))
            for prob, coeffs in MARKET_STAGES[2]
        )
        if exp_val > max_val + TOLERANCE:
            max_val, optimal_choices = exp_val, [(action, post_trade)]
        elif abs(exp_val - max_val) <= TOLERANCE:
            optimal_choices.append((action, post_trade))
    return max_val, tuple(optimal_choices)


@lru_cache(maxsize=None)
def solve_bellman(stage_idx, current_state):
    """Рекуррентное вычисление функции Беллмана для этапов 1 и 2."""
    if stage_idx == 2:
        return evaluate_last_stage(current_state)

    growth_filter = None
    if stage_idx == 1 and ENABLE_PRUNING:
        last_means = calc_mean_coeffs(2)
        joint_means = tuple(a * b for a, b in zip(calc_mean_coeffs(1), last_means))
        if within_commission_band(last_means) and all(
                factor < 1.0 + fee - TOLERANCE for factor, fee in zip(joint_means, BROKER_FEES)
        ):
            growth_filter = joint_means

    candidates = generate_valid_trades(current_state, expected_growth=growth_filter)
    max_val = float("-inf")
    optimal_choices = []

    for post_trade, action in candidates:
        exp_val = 0.0
        for prob, coeffs in MARKET_STAGES[stage_idx]:
            next_st = apply_market_step(post_trade, coeffs)
            future_val, _ = solve_bellman(stage_idx + 1, next_st)
            exp_val += prob * future_val

        if exp_val > max_val + TOLERANCE:
            max_val, optimal_choices = exp_val, [(action, post_trade)]
        elif abs(exp_val - max_val) <= TOLERANCE:
            optimal_choices.append((action, post_trade))

    if not optimal_choices:
        raise ValueError(f"Не найдены допустимые решения на этапе {stage_idx + 1}: {current_state}")
    return max_val, tuple(optimal_choices)


def display_strategy_tree():
    """Выводит пошаговый план управления портфелем по всем ветвям."""
    print(
        f"{'Этап':<6} | {'Портфель до сделки (ЦБ1, ЦБ2, Деп, Кэш)':<40} | {'Решение u':<14} | {'Портфель после сделки':<30}")
    print("-" * 102)
    start_st = tuple((*START_ASSETS, START_CASH))

    def step_through(stage_idx, st):
        if stage_idx == 3:
            return
        _, choices = solve_bellman(stage_idx, st)
        action, post_trade = choices[0]

        st_fmt = f"({st[0]:.2f}, {st[1]:.2f}, {st[2]:.2f}, {st[3]:.2f})"
        post_fmt = f"({post_trade[0]:.2f}, {post_trade[1]:.2f}, {post_trade[2]:.2f}, {post_trade[3]:.2f})"
        print(f"Этап {stage_idx + 1} | {st_fmt:<40} | {str(action):<14} | {post_fmt:<30}")

        for _, coeffs in MARKET_STAGES[stage_idx]:
            step_through(stage_idx + 1, apply_market_step(post_trade, coeffs))

    step_through(0, start_st)
    print("-" * 102)


def run_benchmark():
    """Замеряет время расчета и пиковое потребление памяти."""
    solve_bellman.cache_clear()
    start_st = tuple((*START_ASSETS, START_CASH))
    tracemalloc.start()
    t0 = perf_counter()
    res_val, _ = solve_bellman(0, start_st)
    duration = perf_counter() - t0
    _, max_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return res_val, duration, max_bytes / (1024 * 1024)


def main():
    check_parameters()
    start_st = tuple((*START_ASSETS, START_CASH))
    start_cap = calc_total_capital(start_st)

    opt_cap, exec_sec, mem_mb = run_benchmark()
    _, best_first_step = solve_bellman(0, start_st)

    print("=" * 65)
    print("Итоги оптимизации инвестиционного портфеля")
    print("=" * 65)
    print(f"Стартовый капитал:                  {start_cap:.2f} д.е.")
    print(f"Лучшая сделка на 1-м этапе:         {best_first_step[0][0]}")
    print(f"Количество оптимальных планов (Э1): {len(best_first_step)}")
    print(f"Ожидаемый конечный капитал:         {opt_cap:.6f} д.е.")
    print(f"Ожидаемый чистый прирост:           {opt_cap - start_cap:.6f} д.е.")
    print("=" * 65)

    print("\nСводная таблица оптимальной стратегии:")
    display_strategy_tree()

    print("\nКонтрольные значения функции Беллмана V_k:")
    for st2 in [(810.0, 880.0, 428.0, 2.0),
                (708.75, 816.0, 412.0, 2.0),
                (540.0, 760.0, 400.0, 2.0)]:
        val2, _ = solve_bellman(1, st2)
        print(f"  V2{st2} = {val2:.6f}")

    for st3 in [(49.0, 1012.0, 432.28, 746.0),
                (36.75, 880.0, 428.0, 746.0),
                (21.0, 792.0, 428.0, 746.0)]:
        val3, _ = evaluate_last_stage(st3)
        print(f"  V3{st3} = {val3:.6f}")
    print("\n" + "=" * 65)
    print(f"Затраченное время:       {exec_sec:.4f} с")
    print(f"Пик памяти:              {mem_mb:.2f} МБ")
    print(f"Записей в кэше Беллмана: {solve_bellman.cache_info().currsize}")
    print("=" * 65)
    print(f"  V1{start_st} = {opt_cap:.6f}")
    s4_blg = apply_market_step((49.0, 1012.0, 432.28, 746.0), MARKET_STAGES[2][0][1])
    print(f"  W(s4) для ветви Блг -> Блг -> Блг: s4 = {s4_blg}, W4 = {calc_total_capital(s4_blg):.6f} д.е.")


if __name__ == "__main__":
    main()
