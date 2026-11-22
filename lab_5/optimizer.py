"""
Гибридный метод оптимизации: ACO_R (архив решений, гауссовы ядра) + NM, вариант № 8.
Автор: Степанова Е.А., группа K3341.

Обязательный интерфейс (не менять):
    optimize(f, bounds, max_nfe=10_000, seed=None, constraints=None, **params) -> dict

Возвращаемый словарь:
    x         — np.ndarray shape (2,) — найденная точка (обязательно)
    f         — float — значение f(x) (обязательно)
    nfe       — int — число вычислений f (обязательно)
    history   — list[float] — лучшее значение после каждого вычисления f (для графика сходимости)
    switches  — list[(int, str)] — моменты переключения методов: (nfe, "название этапа")
    snapshots — list[(int, str, np.ndarray)] — положения популяции/роя: (nfe, подпись, точки (n, 2))

Правила:
  * все компоненты гибрида — собственная реализация; scipy.optimize и другие готовые оптимизаторы запрещены;
  * не импортируйте functions.py и не используйте известный минимум: метод знает только f и bounds;
  * никакой печати и рисования внутри optimize (это искажает время);
  * все случайные числа — только из rng = np.random.default_rng(seed);
  * не превышать max_nfe: на битве лишнее вычисление прерывает запуск (запуск — неуспешный);
  * параметры метода задаются значениями по умолчанию ниже и на защите не меняются.
"""
from __future__ import annotations

import numpy as np

# ------------------------------------------------------------- параметры метода
DEFAULTS = dict(
    archive_size=30,      # k: размер архива решений ACO_R
    n_ants=10,            # m: число муравьёв, генерируемых на одной итерации
    q=0.25,               # параметр селективности гауссовых весов архива
    xi=0.85,              # скорость испарения (масштаб разброса sigma)
    local_share=0.15,     # доля бюджета NFE на локальное уточнение Нелдером–Мидом
    n_starts=3,           # число стартовых кластеров из архива для Нелдера–Мида
    nm_step=0.03,         # начальный шаг симплекса (доля от ширины области)
)


class Budget:
    """Учёт вычислений, лучшей точки и истории. Используйте self.eval вместо f."""

    def __init__(self, f, bounds, max_nfe, constraints=None, penalty=1e6):
        self.f, self.max_nfe = f, max_nfe
        self.lo = np.array([b[0] for b in bounds], dtype=float)
        self.hi = np.array([b[1] for b in bounds], dtype=float)
        self.constraints = list(constraints or [])
        self.penalty = penalty
        self.nfe = 0
        self.best_x, self.best_f = None, np.inf
        self.history, self.switches, self.snapshots = [], [], []

    @property
    def left(self) -> int:
        return self.max_nfe - self.nfe

    def clip(self, x):
        return np.clip(x, self.lo, self.hi)

    def violation(self, x):
        return sum(max(0.0, g(x)) for g in self.constraints)

    def eval(self, x) -> float:
        """Значение со штрафом за нарушение ограничений; inf, если бюджет исчерпан."""
        if self.nfe >= self.max_nfe:
            return np.inf
        x = self.clip(np.asarray(x, dtype=float))
        self.nfe += 1
        fx = float(self.f(x))
        v = self.violation(x)
        val = fx + self.penalty * v
        if v == 0 and fx < self.best_f:
            self.best_f, self.best_x = fx, x.copy()
        self.history.append(self.best_f)
        return val

    def stage(self, name):
        self.switches.append((self.nfe, name))

    def snap(self, name, points):
        self.snapshots.append((self.nfe, name, np.array(points, dtype=float).copy()))

    def result(self):
        x = self.best_x if self.best_x is not None else (self.lo + self.hi) / 2
        return {"x": x, "f": self.best_f, "nfe": self.nfe, "history": self.history,
                "switches": self.switches, "snapshots": self.snapshots}


# ------------------------------------------------------------- компоненты гибрида
def acor_stage(B: Budget, rng, budget: int, archive_size: int, n_ants: int,
               q: float, xi: float, **_):
    """Глобальный этап: муравьиный алгоритм для непрерывных областей (ACO_R)."""
    if budget <= 0:
        return np.empty((0, 2)), np.empty(0)

    B.stage("ACO_R (глобальный поиск)")
    stop = min(B.max_nfe, B.nfe + budget)

    init_k = min(archive_size, stop - B.nfe)
    archive_x = rng.uniform(B.lo, B.hi, size=(init_k, 2))
    archive_f = np.array([B.eval(x) for x in archive_x], dtype=float)

    order = np.argsort(archive_f)
    archive_x, archive_f = archive_x[order], archive_f[order]
    B.snap("старт ACO_R", archive_x)

    k = len(archive_x)
    ranks = np.arange(k, dtype=float)
    weights = np.exp(-(ranks ** 2) / (2.0 * (q * k) ** 2)) / (q * k * np.sqrt(2.0 * np.pi))
    probs = weights / np.sum(weights)

    mid_nfe = B.nfe + (stop - B.nfe) // 2
    mid_snapped = False

    while B.nfe < stop and k > 1:
        m = min(n_ants, stop - B.nfe)
        new_x = np.empty((m, 2), dtype=float)
        new_f = np.empty(m, dtype=float)

        for ant in range(m):
            idx = int(rng.choice(k, p=probs))
            center = archive_x[idx]
            sigma = xi * np.sum(np.abs(archive_x - center), axis=0) / (k - 1)
            sigma = np.maximum(sigma, 1e-12 * (B.hi - B.lo))
            cand = B.clip(rng.normal(center, sigma))
            new_x[ant] = cand
            new_f[ant] = B.eval(cand)

        comb_x = np.vstack([archive_x, new_x])
        comb_f = np.concatenate([archive_f, new_f])
        order = np.argsort(comb_f)[:k]
        archive_x, archive_f = comb_x[order], comb_f[order]

        if not mid_snapped and B.nfe >= mid_nfe:
            B.snap("середина ACO_R", archive_x)
            mid_snapped = True

    B.snap("конец ACO_R", archive_x)
    return archive_x, archive_f


def _nelder_mead_single(B: Budget, x0: np.ndarray, step: np.ndarray, stop_nfe: int):
    """Классический симплексный метод Нелдера–Мида из одной стартовой точки."""
    if B.nfe >= stop_nfe:
        return x0

    pts = [B.clip(x0.copy())]
    for d in range(2):
        if B.nfe + len(pts) > stop_nfe:
            break
        v = x0.copy()
        v[d] = v[d] + step[d] if v[d] + step[d] <= B.hi[d] else v[d] - step[d]
        pts.append(B.clip(v))

    simplex = np.array(pts, dtype=float)
    vals = np.array([B.eval(p) for p in simplex], dtype=float)
    if len(simplex) < 3:
        return simplex[int(np.argmin(vals))]

    alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5

    while B.nfe < stop_nfe:
        order = np.argsort(vals)
        simplex, vals = simplex[order], vals[order]

        if np.max(np.abs(simplex[1:] - simplex[0])) < 1e-11 and (vals[-1] - vals[0]) < 1e-13:
            break

        centroid = 0.5 * (simplex[0] + simplex[1])

        # 1. Отражение
        xr = B.clip(centroid + alpha * (centroid - simplex[2]))
        fr = B.eval(xr)

        if vals[0] <= fr < vals[1]:
            simplex[2], vals[2] = xr, fr
            continue

        # 2. Растяжение
        if fr < vals[0]:
            if B.nfe < stop_nfe:
                xe = B.clip(centroid + gamma * (xr - centroid))
                fe = B.eval(xe)
                if fe < fr:
                    simplex[2], vals[2] = xe, fe
                else:
                    simplex[2], vals[2] = xr, fr
            else:
                simplex[2], vals[2] = xr, fr
            continue

        # 3. Сжатие (внешнее или внутреннее)
        if B.nfe < stop_nfe:
            if fr < vals[2]:
                xc = B.clip(centroid + rho * (xr - centroid))
            else:
                xc = B.clip(centroid + rho * (simplex[2] - centroid))
            fc = B.eval(xc)
            if fc <= min(fr, vals[2]):
                simplex[2], vals[2] = xc, fc
                continue

        # 4. Редукция симплекса к лучшей вершине
        for i in range(1, 3):
            if B.nfe >= stop_nfe:
                break
            simplex[i] = B.clip(simplex[0] + sigma * (simplex[i] - simplex[0]))
            vals[i] = B.eval(simplex[i])

    return simplex[0]


def nelder_mead_stage(B: Budget, rng, budget: int, archive_x: np.ndarray,
                      n_starts: int, nm_step: float, **_):
    """Локальный этап: многостартовый Нелдер–Мид по лучшим кластерам архива."""
    if budget <= 0:
        return

    B.stage("Нелдер–Мид (уточнение)")
    stop = min(B.max_nfe, B.nfe + budget)
    span = B.hi - B.lo
    min_dist = 0.15 * np.linalg.norm(span)

    starts = []
    if len(archive_x) > 0:
        for cand in archive_x:
            if all(np.linalg.norm(cand - s) >= min_dist for s in starts):
                starts.append(cand.copy())
                if len(starts) >= n_starts:
                    break
    if not starts:
        if B.best_x is not None:
            starts = [B.best_x.copy()]
        for _ in range(n_starts - len(starts)):
            starts.append(rng.uniform(B.lo, B.hi))

    refined = []
    for i, x0 in enumerate(starts):
        rem_starts = len(starts) - i
        rem_budget = stop - B.nfe
        if rem_budget <= 0:
            break
        sub_stop = B.nfe + rem_budget // rem_starts
        step = span * nm_step
        best_pt = _nelder_mead_single(B, x0, step, sub_stop)
        refined.append(best_pt)

    if refined:
        B.snap("финал Нелдер–Мид", np.array(refined))


# ------------------------------------------------------------- точка входа
def optimize(f, bounds, max_nfe=10_000, seed=None, constraints=None, **params):
    p = {**DEFAULTS, **params}
    rng = np.random.default_rng(seed)
    B = Budget(f, bounds, max_nfe, constraints)
    n_local = int(max_nfe * p["local_share"])
    archive_x, _ = acor_stage(B, rng, max_nfe - n_local, **p)
    nelder_mead_stage(B, rng, B.left, archive_x, **p)
    return B.result()
