# -*- coding: utf-8 -*-
"""Бонусное задание: полное дерево исходов найденной стратегии + тесты.
Использует класс PortfolioDP из lab4_fast.py (лежит в той же папке)."""
import csv
import math

from lab4 import PortfolioDP, DEFAULT_DATA, EPS

NAMES = ["Благопр.", "Нейтр.", "Негатив."]
SHORT = "БНО"


def build_tree(dp, bellman):
    """Обход всех 3^N рыночных последовательностей по оптимальной стратегии.
    Возвращает (узлы принятия решения, конечные исходы)."""
    nodes, leaves = [], []

    def rec(k, state, path, prob, ops, fees):
        if k == dp.N + 1:
            w = sum(state)
            w0 = sum(dp.s0)
            leaves.append(dict(path=path, prob=prob, ops=ops, fees=fees, final=state,
                               capital=w, profit=w - w0, ret=(w - w0) / w0))
            return
        value, actions = bellman(k, state)
        u = actions[0]                                   # при равнозначных - первое
        fee = sum(abs(u[i]) * dp.q[i] * dp.c[i] for i in range(3))
        nodes.append(dict(stage=k, path=path, prob=prob, state=state, u=u, fee=fee, V=value))
        s_prime = dp.apply_trade(state, u)
        for j, (p, mults) in enumerate(dp.scenarios[k - 1]):
            rec(k + 1, dp.next_state(s_prime, mults), path + (j,), prob * p, ops + (u,), fees + (fee,))

    rec(1, dp.s0, (), 1.0, (), ())
    return nodes, leaves


def tree_stats(leaves):
    e_cap = sum(l["prob"] * l["capital"] for l in leaves)
    e_profit = sum(l["prob"] * l["profit"] for l in leaves)
    e_ret = sum(l["prob"] * l["ret"] for l in leaves)
    return dict(
        leaves=len(leaves),
        prob_sum=sum(l["prob"] for l in leaves),
        E_capital=e_cap, E_profit=e_profit, E_return=e_ret,
        P_loss=sum(l["prob"] for l in leaves if l["profit"] < -EPS),
        min_profit=min(l["profit"] for l in leaves),
        max_profit=max(l["profit"] for l in leaves),
        std_profit=math.sqrt(sum(l["prob"] * (l["profit"] - e_profit) ** 2 for l in leaves)),
    )


def print_tree(nodes, leaves, stats):
    print("Узлы принятия решения:")
    for n in nodes:
        s = n["state"]
        print(f"  этап {n['stage']}  путь {''.join(SHORT[j] for j in n['path']) or '-':<3} p={n['prob']:.3f}  "
              f"портфель=({s[0]:.2f}; {s[1]:.2f}; {s[2]:.2f})  свободные={s[3]:.2f}  u*={n['u']}  V={n['V']:.4f}")
    print("\nКонечные исходы:")
    for i, l in enumerate(leaves, 1):
        print(f"  {i:>2} {'-'.join(SHORT[j] for j in l['path'])}  p={l['prob']:.3f}  операции={l['ops']}  "
              f"комиссии={sum(l['fees']):.2f}  W4={l['capital']:.3f}  доход={l['profit']:+.3f}  доходность={100*l['ret']:+.2f}%")
    print("\nИтоги: ", {k: (round(v, 6) if isinstance(v, float) else v) for k, v in stats.items()})


def save_csv(leaves, path="tree_leaves.csv"):
    with open(path, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh, delimiter=";")
        w.writerow(["№", "Последовательность", "Вероятность", "Операции по этапам", "Комиссии по этапам",
                    "Комиссии всего", "Конечный капитал", "Чистый доход", "Доходность, %"])
        for i, l in enumerate(leaves, 1):
            w.writerow([i, "-".join(SHORT[j] for j in l["path"]), round(l["prob"], 6),
                        " | ".join(map(str, l["ops"])), " + ".join(f"{x:.2f}" for x in l["fees"]),
                        round(sum(l["fees"]), 2), round(l["capital"], 4), round(l["profit"], 4),
                        round(100 * l["ret"], 4)])


if __name__ == "__main__":
    dp = PortfolioDP(DEFAULT_DATA)
    v1, u1, bellman = dp.solve()
    nodes, leaves = build_tree(dp, bellman)
    st = tree_stats(leaves)
    print_tree(nodes, leaves, st)
    print(f"\nКонтроль: E[W4] по дереву = {st['E_capital']:.7f}, V1(s1) из ДП = {v1:.7f}")
    save_csv(leaves)
    print("Таблица исходов сохранена в tree_leaves.csv")