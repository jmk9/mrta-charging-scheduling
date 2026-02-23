from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

def plot_time_vs_tasks() -> None:
    tasks: list[int] = [40, 45, 50, 55, 60, 65, 70]


    data: dict[str, list[float]] = {
        "GA":  [75.600, 97.818, 95.756, 105.900, 162.012, 145.558, 131.289],
        "PSO":  [47.398, 76.879, 74.040, 94.484, 191.071, 152.569, 139.022],
        "WCA":  [101.701, 116.860, 125.277, 147.268, 160.645, 170.427, 177.314],
        "JA":  [33.776, 40.505, 42.681, 50.501, 54.527, 58.719, 59.737],
        "SRSR":  [65.687, 77.351, 79.774, 92.456, 95.635, 109.260, 112.888],
        "QSA":  [103.259, 120.508, 128.388, 146.638, 152.177, 174.277, 183.146],
        "SMA":  [100.703, 114.012, 125.838, 142.390, 147.346, 171.618, 176.403],
        "WarSO":  [31.769, 37.417, 39.246, 45.730, 47.924, 53.314, 55.656],
        "FLA":  [27.661, 32.691, 34.840, 40.625, 41.834, 49.536, 48.430],
        "RIME": [32.806, 40.055, 42.122, 49.100, 52.531, 59.479, 60.739],
    }

    plt.figure()
    for algo, times in data.items():
        plt.plot(tasks, times, marker="o", label=algo)

    plt.xlabel("Number of tasks", fontsize=14)
    plt.ylabel("Estimator Runtime (s)", fontsize=14)
    plt.xticks(tasks, fontsize=12)
    plt.yticks(fontsize=12)

    # 좌/우 여백(padding) 추가: 오른쪽 여백을 더 크게 잡아서 legend 영역 확보
    pad_left: float = 0.8
    pad_right: float = 7.5
    plt.xlim(min(tasks) - pad_left, max(tasks) + pad_right)

    plt.grid(True)
    plt.legend(fontsize=10, loc="center right", bbox_to_anchor=(1.01, 0.5))
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_time_vs_tasks()
