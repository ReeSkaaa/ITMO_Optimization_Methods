import itertools
from lab4 import (
    START_ASSETS, START_CASH, PACK_SIZES, BROKER_FEES, MARKET_STAGES,
    apply_market_step, calc_total_capital, solve_bellman, check_parameters
)

SCENARIO_LABELS = ("Благоприятный", "Нейтральный", "Негативный")
BASE_CAPITAL = sum(START_ASSETS) + START_CASH


def write_tree_nodes(stage_idx, st, out_file, level=0, branch_prob=1.0):
    """Рекурсивно записывает узлы и переходы дерева решений в текстовый файл."""
    pad = "  " * level

    if stage_idx == 3:
        end_cap = calc_total_capital(st)
        net_gain = end_cap - BASE_CAPITAL
        print(
            f"{pad}[ИТОГ] Капитал = {end_cap:.2f} д.е. | "
            f"Доход = {net_gain:+.2f} д.е. | "
            f"Вероятность ветви = {branch_prob:.3f}",
            file=out_file,
        )
        return

    _, choices = solve_bellman(stage_idx, st)
    action, post_trade = choices[0]

    print(
        f"{pad}Этап {stage_idx + 1}: "
        f"состояние = ({st[0]:.2f}, {st[1]:.2f}, {st[2]:.2f}, {st[3]:.2f}) | "
        f"сделка u = {action} | "
        f"после сделки = ({post_trade[0]:.2f}, {post_trade[1]:.2f}, {post_trade[2]:.2f}, {post_trade[3]:.2f})",
        file=out_file,
    )

    for idx, (prob, coeffs) in enumerate(MARKET_STAGES[stage_idx]):
        label = SCENARIO_LABELS[idx]
        next_st = apply_market_step(post_trade, coeffs)
        print(
            f"{pad}  -> {label} (p = {prob:.2f}, P_пути = {branch_prob * prob:.3f}): "
            f"s{stage_idx + 2} = ({next_st[0]:.2f}, {next_st[1]:.2f}, {next_st[2]:.2f}, {next_st[3]:.2f})",
            file=out_file,
        )
        write_tree_nodes(stage_idx + 1, next_st, out_file, level + 2, branch_prob * prob)


def export_tree_to_file(filepath="decision_tree.txt"):
    """Сохраняет полное дерево исходов в файл."""
    start_st = tuple((*START_ASSETS, START_CASH))
    with open(filepath, "w", encoding="utf-8") as f:
        write_tree_nodes(0, start_st, f)
    print(f"Полное дерево исходов успешно записано в файл: {filepath}")


def evaluate_all_outcomes():
    """Перебирает все 27 сценариев, строит таблицу исходов и считает метрики риска."""
    start_st = tuple((*START_ASSETS, START_CASH))
    start_cap = sum(start_st)

    print("=" * 68)
    print("Анализ всех 27 рыночных исходов для оптимальной стратегии")
    print("=" * 68)

    all_paths = []

    for combo in itertools.product(range(3), repeat=3):
        st = start_st
        path_prob = 1.0
        fees_paid = 0.0
        labels_list = []
        trades_list = []

        for stage_idx in range(3):
            scen_id = combo[stage_idx]
            prob, coeffs = MARKET_STAGES[stage_idx][scen_id]
            path_prob *= prob

            labels_list.append(f"Эт.{stage_idx + 1}: {SCENARIO_LABELS[scen_id]}")

            _, choices = solve_bellman(stage_idx, st)
            action, post_trade = choices[0]

            step_fee = sum(
                abs(packs) * PACK_SIZES[i] * BROKER_FEES[i]
                for i, packs in enumerate(action) if packs != 0
            )
            fees_paid += step_fee
            trades_list.append(f"({action[0]},{action[1]},{action[2]})")

            st = apply_market_step(post_trade, coeffs)

        end_cap = calc_total_capital(st)
        net_gain = end_cap - start_cap
        roi_pct = (net_gain / start_cap) * 100.0

        all_paths.append({
            "path": " -> ".join(labels_list),
            "prob": path_prob,
            "actions": " -> ".join(trades_list),
            "commission": fees_paid,
            "final_capital": end_cap,
            "net_income": net_gain,
            "return_pct": roi_pct,
        })

    exp_cap = sum(item["final_capital"] * item["prob"] for item in all_paths)
    exp_gain = exp_cap - start_cap
    exp_roi = (exp_gain / start_cap) * 100.0

    risk_of_loss = sum(item["prob"] for item in all_paths if item["net_income"] < 0)
    worst_gain = min(item["net_income"] for item in all_paths)
    best_gain = max(item["net_income"] for item in all_paths)

    print(f"Ожидаемый капитал в конце:  {exp_cap:.2f} д.е.")
    print(f"Ожидаемый чистый доход:     {exp_gain:+.2f} д.е.")
    print(f"Ожидаемая доходность:       {exp_roi:+.2f} %")
    print(f"Вероятность ухода в минус:  {risk_of_loss * 100:.2f} %")
    print(f"Худший результат (убыток):  {worst_gain:+.2f} д.е.")
    print(f"Лучший результат (прибыль): {best_gain:+.2f} д.е.")
    print("-" * 102)

    print("\nСводка по всем 27 сценариям (по убыванию конечного капитала):")
    print(
        f"{'Эт1':<3} | {'Эт2':<3} | {'Эт3':<3} | {'Вер.':<5} | "
        f"{'Последовательность сделок':<29} | {'Комис.':>7} | {'Капитал':>9} | "
        f"{'Доход':>9} | {'Дох-ть':>7}"
    )
    print("-" * 102)

    all_paths.sort(key=lambda item: item["final_capital"], reverse=True)
    abbr = {"Благоприятный": "Благоприятный", "Нейтральный": "Нейтральный", "Негативный": "Негативный"}

    for item in all_paths:
        parts = item["path"].split(" -> ")
        s1 = abbr[parts[0].split(": ")[1]]
        s2 = abbr[parts[1].split(": ")[1]]
        s3 = abbr[parts[2].split(": ")[1]]

        print(
            f"{s1:<3} | {s2:<3} | {s3:<3} | {item['prob']:.3f} | "
            f"{item['actions']:<29} | {item['commission']:>7.2f} | "
            f"{item['final_capital']:>9.2f} | {item['net_income']:>+9.2f} | "
            f"{item['return_pct']:>+6.2f}%"
        )

    print("-" * 102)


if __name__ == "__main__":
    check_parameters()
    export_tree_to_file()
    evaluate_all_outcomes()