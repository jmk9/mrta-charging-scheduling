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
        "GA":  [95.756, 68.512, 44.179, 47.842],
        "PSO":  [74.040, 60.244, 64.720, 62.403],
        "WCA":  [125.277, 131.171, 141.633, 145.967],
        "JA":  [42.681, 43.357, 48.236, 47.956],
        "SRSR":  [79.774, 84.913, 95.388, 93.789],
        "QSA":  [128.388, 135.212, 145.908, 149.561],
        "SMA":  [125.838, 131.954, 139.688, 145.978],
        "WarSO":  [39.246, 41.329, 46.145, 45.547],
        "FLA":  [34.840, 36.27, 38.127, 39.595],
        "RIME": [42.122, 44.657, 45.372, 47.458],
    }


    plt.figure()
    for algo, times in data.items():
        plt.plot(robots, times, marker="o", label=algo)

    plt.xlabel("Number of robots", fontsize=14)
    plt.ylabel("Estimator Runtime (s)", fontsize=14)
    plt.xticks(robots, fontsize=12)
    plt.yticks(fontsize=12)

    # 좌/우 여백(padding) 추가: 오른쪽 여백을 더 크게 잡아서 legend 영역 확보
    pad_left: float = 0.25
    pad_right: float = 1.0
    plt.xlim(min(robots) - pad_left, max(robots) + pad_right)

    plt.grid(True)
    plt.legend(fontsize=10, loc="center right", bbox_to_anchor=(1.01, 0.5))
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_time_vs_robots()
