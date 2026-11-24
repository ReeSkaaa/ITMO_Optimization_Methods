"""
Визуализация для отчёта. Создаёт в папке img/:
  contour.png      — линии уровня, популяция в моменты snapshots и траектория лучшего решения
  convergence.png  — сходимость (lg ошибки от числа вычислений) с моментами переключения методов
  boxplot.png      — гибрид против отдельных компонентов (серия запусков)
  surface.png      — 3D-поверхность функции

    python demo.py --func rastrigin --runs 30
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import optimizer as OPT
from functions import get_function
from run import run_once

os.makedirs("img", exist_ok=True)


def grid(tf, n=300):
    x1 = np.linspace(*tf.bounds[0], n)
    x2 = np.linspace(*tf.bounds[1], n)
    X1, X2 = np.meshgrid(x1, x2)
    Z = np.apply_along_axis(tf, 0, np.stack([X1, X2]))
    return X1, X2, Z


def plot_contour(tf, res, X1, X2, Z):
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    zl = Z - Z.min() + 1e-9
    cs = ax.contourf(X1, X2, np.log10(zl), levels=40, cmap="viridis", alpha=0.85)
    fig.colorbar(cs, ax=ax, label="lg(f − min f + 1e−9)")
    colors = plt.cm.autumn(np.linspace(0, 1, max(1, len(res.get("snapshots", [])))))
    for (nfe, name, pts), c in zip(res.get("snapshots", []), colors):
        ax.scatter(pts[:, 0], pts[:, 1], s=10, color=c, edgecolor="k", lw=0.3, label=f"{name} (NFE={nfe})")
    for p in tf.x_star:
        ax.plot(*p, "w*", ms=14, mec="k", label=None)
    ax.plot(*res["x"], "rx", ms=12, mew=2.5, label="найденное решение")
    ax.set_xlabel("x1"); ax.set_ylabel("x2")
    ax.set_title(f"{tf.title}: популяция и решение (★ — x*)")
    ax.legend(fontsize=7, loc="upper right")
    fig.tight_layout(); fig.savefig("img/contour.png", dpi=140); plt.close(fig)


def plot_convergence(tf, res):
    h = np.array(res["history"], dtype=float)
    e = np.maximum((h - tf.f_star) / max(1, abs(tf.f_star)), 1e-30)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogy(np.arange(1, len(e) + 1), e, lw=1.6)
    for nfe, name in res.get("switches", []):
        ax.axvline(nfe, color="gray", ls="--", lw=1)
        ax.text(nfe, 0.96, f"{name} ", rotation=90, va="top", ha="right", fontsize=8,
                transform=ax.get_xaxis_transform(),
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8))
    ax.axhline(tf.eps, color="tab:red", ls=":", label="порог успеха ε")
    ax.set_xlabel("число вычислений f"); ax.set_ylabel("нормированная ошибка лучшего решения")
    ax.set_title("Сходимость"); ax.legend(); ax.grid(alpha=0.3, which="both")
    fig.tight_layout(); fig.savefig("img/convergence.png", dpi=140); plt.close(fig)



def plot_box(tf, runs, nfe, variants):
    data, labels = [], []

    print("\nСравнение методов:")
    print("| Метод | Доля успехов | Медиана lg e | Среднее время, с |")
    print("|---|---:|---:|---:|")

    for label, params in variants:
        errs = []
        times = []

        for s in range(1, runs + 1):
            old = dict(OPT.DEFAULTS)
            OPT.DEFAULTS.update(params)

            try:
                out, _ = run_once(tf, s, nfe)
            finally:
                OPT.DEFAULTS.clear()
                OPT.DEFAULTS.update(old)

            errs.append(np.log10(max(out["e"], 1e-30)))
            times.append(out["time_s"])

        data.append(errs)
        labels.append(label)

        success_rate = np.mean(np.array(errs) <= np.log10(tf.eps))
        median_lg_e = np.median(errs)
        mean_time = np.mean(times)

        print(
            f"| {label} | {success_rate:.2f} "
            f"| {median_lg_e:.2f} | {mean_time:.4f} |"
        )

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.boxplot(data)
    ax.set_xticks(range(1, len(labels) + 1), labels)
    ax.axhline(
        np.log10(tf.eps),
        color="tab:red",
        ls=":",
        label="порог успеха ε"
    )
    ax.set_ylabel("lg нормированной ошибки")
    ax.set_title(f"Гибрид и компоненты, {runs} запусков")
    ax.legend()
    fig.tight_layout()
    fig.savefig("img/boxplot.png", dpi=140)
    plt.close(fig)



def plot_surface(X1, X2, Z, tf):
    fig = plt.figure(figsize=(6.5, 5))
    ax = fig.add_subplot(projection="3d")
    s = slice(None, None, 3)
    ax.plot_surface(X1[s, s], X2[s, s], Z[s, s], cmap="viridis", lw=0, antialiased=True)
    ax.set_title(tf.title); ax.set_xlabel("x1"); ax.set_ylabel("x2")
    fig.tight_layout(); fig.savefig("img/surface.png", dpi=140); plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--func", default="himmelblau")
    ap.add_argument("--runs", type=int, default=30)
    ap.add_argument("--nfe", type=int, default=10_000)
    a = ap.parse_args()
    tf = get_function(a.func)
    out, res = run_once(tf, 1, a.nfe)
    print(f"x* = ({out['x1']:.6f}, {out['x2']:.6f})  f = {out['f']:.10g}  время = {out['time_s']:.4f} с")
    print("запусков NM:", len(res["snapshots"][-1][2]))
    X1, X2, Z = grid(tf)
    plot_contour(tf, res, X1, X2, Z)
    plot_convergence(tf, res)
    plot_surface(X1, X2, Z, tf)
    plot_box(tf, a.runs, a.nfe, [
        ("Гибрид (ACO_R + NM)", {}),
        ("Только ACO_R", {"local_share": 0.0}),
        ("Только Нелдер–Мид", {"local_share": 1.0, "n_starts": 25, "nm_step": 0.12}),
    ])
    print("Рисунки сохранены в img/")
