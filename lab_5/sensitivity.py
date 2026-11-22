
import csv
import numpy as np
import matplotlib.pyplot as plt

import optimizer as OPT
from functions import get_function
from run import run_once


# Настройки эксперимента
FUNC = "himmelblau"
RUNS = 30
NFE = 10_000

# Проверяемые значения параметров
PARAMETERS = {
    "q": [0.05, 0.1, 0.25, 0.5, 1.0],
    "xi": [0.1, 0.25, 0.5, 0.85, 1.2, 1.5],
}


def test_parameter(tf, name, values):
    results = []

    original_defaults = dict(OPT.DEFAULTS)

    try:
        for value in values:
            OPT.DEFAULTS.clear()
            OPT.DEFAULTS.update(original_defaults)
            OPT.DEFAULTS[name] = value

            errors = []
            successes = 0

            for seed in range(1, RUNS + 1):
                out, _ = run_once(tf, seed, NFE)

                error = max(float(out["e"]), 1e-30)
                errors.append(np.log10(error))

                if out["success"]:
                    successes += 1

            success_rate = successes / RUNS
            median_error = float(np.median(errors))

            results.append({
                "parameter": name,
                "value": value,
                "successes": successes,
                "success_rate": success_rate,
                "median_lg_error": median_error,
            })

            print(
                f"{name} = {value}: "
                f"успехов {successes}/{RUNS}, "
                f"доля = {success_rate:.2f}, "
                f"медиана lg e = {median_error:.2f}"
            )

    finally:
        OPT.DEFAULTS.clear()
        OPT.DEFAULTS.update(original_defaults)

    return results


def main():
    tf = get_function(FUNC)
    all_results = []

    for name, values in PARAMETERS.items():
        print(f"\nПроверка параметра {name}")
        results = test_parameter(tf, name, values)
        all_results.extend(results)

    # Сохранение таблицы
    with open("sensitivity_results.csv", "w",
              newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "parameter",
                "value",
                "successes",
                "success_rate",
                "median_lg_error",
            ],
        )
        writer.writeheader()
        writer.writerows(all_results)

    # Построение графиков
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    for ax, name in zip(axes, PARAMETERS):
        rows = [r for r in all_results if r["parameter"] == name]

        x = [r["value"] for r in rows]
        y = [r["success_rate"] for r in rows]

        ax.plot(x, y, marker="o")
        ax.set_xlabel(f"Значение параметра {name}")
        ax.set_ylabel("Доля успешных запусков")
        ax.set_title(f"Чувствительность к параметру {name}")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True)

    fig.tight_layout()
    fig.savefig("sensitivity.png", dpi=150)
    plt.close(fig)

    # Сравнение чувствительности по диапазону доли успехов
    print("\nСравнение параметров:")
    for name in PARAMETERS:
        rates = [
            r["success_rate"]
            for r in all_results
            if r["parameter"] == name
        ]
        print(
            f"{name}: изменение доли успехов = "
            f"{max(rates) - min(rates):.2f}"
        )

    print("\nСохранены файлы:")
    print("sensitivity_results.csv")
    print("sensitivity.png")


if __name__ == "__main__":
    main()
