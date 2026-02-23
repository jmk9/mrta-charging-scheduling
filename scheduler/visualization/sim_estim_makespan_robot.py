from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

def plot_time_vs_robots() -> None:
    robots: list[int] = [4, 5, 6, 7]


    # data: dict[str, list[float]] = {
    #     "PSO":  [71.673, 74.040, 60.244, 64.720, 62.403],
    #     "WCA":  [118.169, 125.277, 131.171, 141.633, 145.967],
    #     "QSA":  [121.202, 128.388, 135.212, 145.908, 149.561],
    #     "SMA":  [119.784, 125.838, 131.954, 139.688, 145.978],
    #     "FLA":  [32.259, 34.840, 36.27, 38.127, 39.595],
    #     "RIME": [40.974, 42.122, 44.657, 45.372, 47.458],
    # }



    data: dict[str, list[float]] = {
        "Estimator": [1498.6, 1511.2, 1356.9, 1134.6],
        "Simulator": [1498.6 + 94 * 2, 1511.2 + 86 * 2, 1356.9 + 74 * 2, 1134.6 + 78 * 2],
    }

    estim_key = "Estimator"
    sim_key = "Simulator"
    if estim_key in data and sim_key in data:
        estim = data[estim_key]
        sim = data[sim_key]
        print("[오차 로그] (sim - estim) / estim * 100")
        for robot, est_time, sim_time in zip(robots, estim, sim):
            diff_percent = ((sim_time - est_time) / est_time) * 100.0 if est_time else float("nan")
            print(
                f"  로봇 {robot}: 오차={diff_percent:.2f}% "
                f"(estim={est_time:.1f}s, sim={sim_time:.1f}s)"
            )

    plt.figure()
    color_red = "#d62728"
    for algo, times in data.items():
        is_estim = "estim" in algo.lower()
        linestyle = "--" if is_estim else "-"
        plt.plot(
            robots,
            times,
            marker="o",
            color=color_red,
            linestyle=linestyle,
            label=algo,
        )

    plt.xlabel("Number of robots", fontsize=14)
    plt.ylabel("Makespan (s)", fontsize=14)
    plt.xticks(robots, fontsize=12)
    plt.yticks(fontsize=12)

    # 좌/우 여백(padding) 추가: 오른쪽 여백을 더 크게 잡아서 legend 영역 확보
    pad: float = 0.3
    plt.xlim(min(robots) - pad, max(robots) + pad)

    plt.grid(True)
    plt.legend(fontsize=15)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_time_vs_robots()
