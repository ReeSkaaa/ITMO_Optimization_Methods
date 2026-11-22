"""
Запуск метода на тестовой функции с выводом аргументов, значения функции,
числа вычислений и затраченного времени.

    python run.py --func rastrigin --seed 1
    python run.py --func rastrigin --runs 30          # серия запусков со статистикой
"""
import argparse
import json
import time

import numpy as np

from functions import get_function
from optimizer import optimize


class BudgetExceeded(BaseException):
    pass


class CountedF:
    """Внешний счётчик вычислений (так же устроен счётчик на битве)."""

    def __init__(self, tf, max_nfe):
        self.tf, self.max_nfe, self.nfe = tf, max_nfe, 0

    def __call__(self, x):
        self.nfe += 1
        if self.nfe > self.max_nfe:
            raise BudgetExceeded("Превышен бюджет вычислений")
        return self.tf(x)


def run_once(tf, seed, nfe):
    cf = CountedF(tf, nfe)
    kw = {"constraints": list(tf.constraints)} if tf.constraints else {}
    t0 = time.perf_counter()
    res = optimize(cf, list(tf.bounds), max_nfe=nfe, seed=seed, **kw)
    elapsed = time.perf_counter() - t0
    if cf.nfe != res["nfe"]:
        raise RuntimeError(
            f"Счётчики вычислений не совпадают: "
            f"внешний={cf.nfe}, внутренний={res['nfe']}"
        )
    x = np.clip(np.asarray(res["x"], dtype=float), tf.lower, tf.upper)
    fx = tf(x)                                   # независимый пересчёт значения
    return {"func": tf.name, "seed": seed, "x1": float(x[0]), "x2": float(x[1]), "f": fx,
            "e": tf.error(fx), "success": bool(tf.error(fx) <= tf.eps and tf.is_feasible(x)),
            "nfe": cf.nfe, "time_s": elapsed}, res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--func", default="himmelblau")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--nfe", type=int, default=10_000)
    ap.add_argument("--runs", type=int, default=1, help="число запусков с зёрнами seed..seed+runs-1")
    a = ap.parse_args()
    tf = get_function(a.func)

    rows = []
    for s in range(a.seed, a.seed + a.runs):
        out, _ = run_once(tf, s, a.nfe)
        rows.append(out)
        print(f"seed={s:<3} x* = ({out['x1']:.6f}, {out['x2']:.6f})  f(x*) = {out['f']:.10g}  "
              f"NFE = {out['nfe']}  время = {out['time_s']:.4f} с  {'успех' if out['success'] else ''}")
    if a.runs == 1:
        print("JSON:", json.dumps(rows[0], ensure_ascii=False))
    else:
        f = np.array([r["f"] for r in rows])
        t = np.array([r["time_s"] for r in rows])
        print(f"\n{tf.title}: f* = {tf.f_star:.10g}")
        print(f"лучшее {f.min():.10g} | медиана {np.median(f):.10g} | худшее {f.max():.10g}")
        print(f"доля успехов {np.mean([r['success'] for r in rows]):.2f} | "
              f"время: среднее {t.mean():.4f} с, медиана {np.median(t):.4f} с")
        print("| Лучшее f | Медиана f | Худшее f | Доля успехов | Среднее время, с | Среднее NFE |")
        print("|---|---|---|---|---|---|")
        print(f"| {f.min():.6g} | {np.median(f):.6g} | {f.max():.6g} | "
              f"{np.mean([r['success'] for r in rows]):.2f} | {t.mean():.4f} | "
              f"{np.mean([r['nfe'] for r in rows]):.0f} |")
