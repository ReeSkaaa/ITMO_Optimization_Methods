import copy
import itertools
import math

from lab4 import PortfolioDP, DEFAULT_DATA, EPS

results = []


def check(name, inp, expected, actual, ok):
    results.append(ok)
    print(f"[{'OK' if ok else 'FAIL'}] {name}\n    вход: {inp}\n    ожидается: {expected}\n    получено:  {actual}")


def small(**kw):
    d = copy.deepcopy(DEFAULT_DATA)
    d.update(stages=2, initial_state=(100.0, 100.0, 100.0, 100.0), pack_sizes=(50.0, 50.0, 50.0),
             min_limits=(30.0, 30.0, 30.0), scenarios=DEFAULT_DATA["scenarios"][:2])
    d.update(kw)
    return d


def brute(data):
    """Независимая проверка: простая рекурсия без кэша и без numpy, управления - тремя циклами."""
    q, c, L, N = data["pack_sizes"], data["commissions"], data["min_limits"], data["stages"]

    def rec(k, s):
        if k == N:
            return sum(s)
        best = -1e18
        rng = [range(-max(0, int((s[i] - L[i]) / q[i] + 1e-9)), 40) for i in range(3)]
        for u in itertools.product(*rng):
            cash = s[3]
            for i in range(3):
                if u[i] < 0:
                    cash += -u[i] * q[i] * (1 - c[i])
            for i in range(3):
                if u[i] > 0:
                    cash -= u[i] * q[i] * (1 + c[i])
            if cash < -1e-9:
                continue
            A = [s[i] + u[i] * q[i] for i in range(3)]
            v = sum(p * rec(k + 1, (A[0] * m[0], A[1] * m[1], A[2] * m[2], cash)) for p, m in data["scenarios"][k])
            best = max(best, v)
        return best

    return rec(0, tuple(data["initial_state"]))


# ---------------------------------------------------------------- 1. сверка с ручным расчётом
dp = PortfolioDP(DEFAULT_DATA)
V1, U1, bell = dp.solve()
check("Ручной расчёт: V1(s1) и u*1", "вариант из задания", "2034.2388838; [(23, 0, 0)]", f"{V1:.7f}; {U1}",
      abs(V1 - 2034.2388838) < 1e-6 and U1 == [(23, 0, 0)])

hand2 = {"Благопр.": ((810.0, 880.0, 428.0, 2.0), (-31, 0, 0), 2129.411816),
         "Нейтр.": ((708.75, 816.0, 412.0, 2.0), (-27, 0, 0), 1949.565714),
         "Негатив.": ((540.0, 760.0, 400.0, 2.0), (-20, 0, 0), 1717.220800)}
for name, (s, u, v) in hand2.items():
    V, a = bell(2, s)
    check(f"Ручной расчёт: этап 2, {name}", s, f"{u}; {v}", f"{a[0]}; {V:.6f}", a[0] == u and abs(V - v) < 1e-5)

hand3 = {"Блг→Блг": ((49.0, 1012.0, 432.28, 746.0), 2291.114720),
         "Блг→Ней": ((36.75, 880.0, 428.0, 746.0), 2136.957000),
         "Блг→Нег": ((21.0, 792.0, 428.0, 746.0), 2029.372000)}
for name, (s, v) in hand3.items():
    V, a = bell(3, s)
    check(f"Ручной расчёт: этап 3, {name}", s, f"(0, 0, 0); {v}", f"{a[0]}; {V:.6f}", a[0] == (0, 0, 0) and abs(V - v) < 1e-5)

# ---------------------------------------------------------------- 2. независимый перебор 1-го этапа
def hand_V2(s):  # замкнутая формула V2 из ручного решения
    n = max(0, int((s[0] - 30) // 25))
    return 0.9486 * s[0] + 1.0348 * s[1] + 1.027072 * s[2] + s[3] + 0.285 * n

acts = dp.get_feasible_actions(dp.s0)
best, bu = -1, None
for u in acts:
    sp = dp.apply_trade(dp.s0, u)
    val = sum(p * hand_V2(dp.next_state(sp, m)) for p, m in dp.scenarios[0])
    if val > best + 1e-9:
        best, bu = val, u
check("Независимый перебор всех управлений 1-го этапа (V2 по формуле)", f"{len(acts)} управлений",
      f"{V1:.6f}; (23, 0, 0)", f"{best:.6f}; {bu}", abs(best - V1) < 1e-6 and bu == (23, 0, 0))

# допустимость всех управлений: свободные средства >= 0, остатки >= L при продаже
bad = 0
for u in acts:
    sp = dp.apply_trade(dp.s0, u)
    if sp[3] < -1e-6 or any(u[i] < 0 and sp[i] < dp.L[i] - 1e-6 for i in range(3)):
        bad += 1
check("Все допустимые управления 1-го этапа удовлетворяют ограничениям", f"{len(acts)} управлений", "0 нарушений",
      f"{bad} нарушений", bad == 0)
# полнота: управление за границей допустимости не попало в список
over = (24, 0, 0)
check("Недопустимое управление (24, 0, 0) отсутствует в U(s1)", "26*24 = 624 > 600", "нет в списке",
      "нет в списке" if over not in acts else "есть в списке", over not in acts)

# ---------------------------------------------------------------- 3. малые примеры против простой рекурсии
d = small()
bf, v = brute(d), PortfolioDP(d).solve()[0]
check("Малый пример N = 2: ДП против рекурсии без кэша", "старт (100;100;100;100), пакет 50", f"{bf:.6f}", f"{v:.6f}", abs(bf - v) < 1e-6)
d = small(commissions=(0.0, 0.0, 0.0))
d["scenarios"] = [[(0.5, (1.3, 1.0, 1.0)), (0.3, (1.0, 1.2, 1.0)), (0.2, (0.9, 0.9, 1.0))], DEFAULT_DATA["scenarios"][1]]
bf, v = brute(d), PortfolioDP(d).solve()[0]
check("Малый пример без комиссий (выгодные сделки)", "c = 0", f"{bf:.6f}", f"{v:.6f}", abs(bf - v) < 1e-6)
d = small(scenarios=[[(0.5, (1.5, 1.3, 1.0)), (0.5, (0.8, 1.0, 1.0))], [(0.4, (1.2, 1.1, 1.0)), (0.6, (0.9, 1.0, 1.05))]])
bf, v = brute(d), PortfolioDP(d).solve()[0]
check("Малый пример: по 2 сценария на этапе", "другая структура сценариев", f"{bf:.6f}", f"{v:.6f}", abs(bf - v) < 1e-6)

# ---------------------------------------------------------------- 4. дополнительные тесты
d = small(initial_state=(30.0, 30.0, 30.0, 0.0), pack_sizes=(25.0, 25.0, 25.0))
V, a, _ = PortfolioDP(d).solve()
Em = lambda k, i: sum(p * m[i] for p, m in d["scenarios"][k])
exp = sum(30 * Em(0, i) * Em(1, i) for i in range(3))
check("T1. Нулевой ресурс: нет средств, активы на минимуме", "(30;30;30;0)", f"[(0,0,0)]; V={exp:.6f}", f"{a}; V={V:.6f}",
      a == [(0, 0, 0)] and abs(V - exp) < 1e-9)

d = small(stages=1, scenarios=[DEFAULT_DATA["scenarios"][2]])
V, a, _ = PortfolioDP(d).solve()
bf = brute(d)
check("T2. Один этап", "N = 1", f"[(0,0,0)]; V={bf:.6f}", f"{a}; V={V:.6f}", a == [(0, 0, 0)] and abs(V - bf) < 1e-9)

d = small(scenarios=[[(0.5, (1.0, 1.0, 1.0)), (0.5, (1.0, 1.0, 1.0))]] * 2)
V, a, _ = PortfolioDP(d).solve()
check("T3. Нулевая доходность (все множители 1)", "m = 1", "[(0,0,0)]; V=400", f"{a}; V={V:.4f}", a == [(0, 0, 0)] and abs(V - 400) < 1e-9)

d = small(stages=1, commissions=(0.0, 0.0, 0.0), scenarios=[[(0.5, (1.0, 1.0, 1.0)), (0.5, (1.0, 1.0, 1.0))]])
V, a, _ = PortfolioDP(d).solve()
check("T4. Несколько оптимальных планов", "c = 0, m = 1, N = 1", "V=400; управлений > 1", f"V={V:.4f}; управлений: {len(a)}",
      abs(V - 400) < 1e-9 and len(a) > 1)

d = small(commissions=(0.0, 0.0, 0.0), scenarios=[[(1.0, (1.5, 1.0, 1.0))]] * 2)
V, a, _ = PortfolioDP(d).solve()
check("T5. Весь ресурс в один актив", "m(ЦБ1) = 1.5, остальные 1, c = 0", "покупка ЦБ1 на все средства и выручку",
      f"u*={a[0]}; V={V:.2f}", a[0][0] > 0 and a[0][1] <= 0 and a[0][2] <= 0)

bad_data = {
    "отрицательные свободные средства": small(initial_state=(100.0, 100.0, 100.0, -5.0)),
    "вероятности не дают 1": small(scenarios=[[(0.5, (1, 1, 1)), (0.2, (1, 1, 1))]] * 2),
    "неполная таблица сценариев": small(scenarios=[DEFAULT_DATA["scenarios"][0]]),
    "неполный набор множителей": small(scenarios=[[(1.0, (1.1, 1.0))]] * 2),
    "нулевой размер пакета": small(pack_sizes=(0.0, 50.0, 50.0)),
}
for name, dd in bad_data.items():
    try:
        PortfolioDP(dd); res, ok = "ошибка не обнаружена", False
    except ValueError as e:
        res, ok = f"ValueError: {e}", True
    check(f"T6. Некорректные данные: {name}", name, "ValueError", res, ok)

print(f"\nИТОГО: {sum(results)} из {len(results)} проверок пройдено")
assert all(results)