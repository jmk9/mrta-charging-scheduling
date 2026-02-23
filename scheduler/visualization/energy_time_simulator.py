"""
energy_time_simulator.py

- timeline.py의 구조를 참고하여, 실행 시 태스크별 시간/에너지 변화 로그를 파일로 저장
- test_check.py의 estimate_task_energy() 구조를 참고해 pick/drop/drive 등 실제 동작별 시간, 에너지 변화를 반영
"""
from dataclasses import dataclass
from typing import List, Tuple, Sequence
import math
import os

# 기본 파라미터 (test_check.py와 동일하게 맞춤)
SPEED_EMPTY_MPS = 0.3           # 무적재 주행 속도 (m/s)
SPEED_LOADED_MPS = 0.25         # 적재 주행 속도 (m/s)
LOADED_FACTOR = 1.3             # 적재 주행 에너지 계수
K_DRIVE_WH_PER_M = 0.275        # 무적재 주행 시 1m당 소모 에너지(Wh/m)
E_PICK_WH = 0.5                 # 픽 1회당 에너지(Wh)
E_DROP_WH = 0.5                 # 드랍 1회당 에너지(Wh)
PICK_WAIT_S = 2.0               # 픽 대기 시간(s)
DROP_WAIT_S = 2.0               # 드랍 대기 시간(s)
LOAD_WEIGHT_FACTOR = 0.05       # kg당 추가 에너지 계수

# 관성/가속도 기반 에너지 파라미터 (test_check.py 참고)
BASE_ROBOT_MASS_KG = 4.4
INERTIA_EFFICIENCY = 0.7  # 운동에너지 중 실제 소모되는 비율(예시)
WH_PER_J = 1.0 / 3600.0

@dataclass(frozen=True)
class TaskSpec:
    task_id: int
    kind: str  # "one_way" | "round_trip" | "via_point"
    waypoints: List[Tuple[float, float]]
    pick_weights: List[float] = None  # 각 픽의 무게(kg)

from dataclasses import dataclass
from typing import List, Tuple, Sequence
import math
import os

# 데모 태스크 시퀀스 (timeline.py 참고)
demo_specs: List[Tuple[float, float, float, float]] = [
    (-5.0, -4.0, -5.0, -8.0),     # 1
    ( 3.0,  1.0,  3.0, -4.0),     # 2
    ( 4.8,  1.0,  5.0, -4.0),     # 3
    (-5.0, -3.0,  1.0, -9.0),     # 4
    (-5.0, -1.0,  0.0, -7.0),     # 5
    (-5.0,  5.0, -5.0,  7.0),     # 6
    (-5.0,  4.0,  1.0,  5.0),     # 7
    (-5.0,  0.0,  1.2,  6.0),     # 8
    ( 3.0, -5.0,  3.0, -9.0),     # 9
    ( 5.0, -5.0,  5.0, -8.8),     # 10
    (-3.0, -7.0, -3.0,  1.0),     # 11
    (-3.0,  7.0, -1.5, -0.5),     # 12
    (-5.0,  1.0, -5.0,  5.0),     # 13
    (-3.0,  1.0, -3.0,  4.0),     # 14
    ( 0.0,  1.0,  0.0,  4.0),     # 15
    ( 2.0,  1.0,  1.0,  4.0),     # 16
    ( 1.5,  0.0,  0.0,  5.0),     # 17
    ( 1.5, -1.0,  1.5, -4.0),     # 18
    ( 1.5, -5.0,  1.5, -9.0),     # 19
    ( 0.0, -6.0,  3.0, -6.0),     # 20
]

demo_pairs: List[Tuple[float, float, float, float]] = [
    ( -0.35,  -3.95,   0.45,   5.35),  # task 1
    ( -3.75,   8.55,  -0.30,  -0.85),  # task 2
    ( -3.55,   6.00,  -5.75,   8.60),  # task 3
    (  3.90,  -8.20,  -1.45,  -1.00),  # task 4
    ( -2.95,  -9.75,  -1.90,  -3.70),  # task 5
    ( -3.65,   6.50,  -2.75,  -0.80),  # task 6
    (  0.55,  -7.85,   0.70,  -5.10),  # task 7
    (  1.55,  -6.15,  -3.95,  -5.60),  # task 8
    (  1.30,   7.15,  -1.50,  -1.35),  # task 9
    ( -0.30,  -3.90,   0.05,   2.55),  # task 10
    ( -3.55,  -0.30,   1.25,   2.00),  # task 11
    (  0.05,  -7.45,  -5.60,   5.20),  # task 12
    ( -3.90,  -6.50,  -3.40,   5.85),  # task 13
    ( -3.40,  -8.60,  -4.90,   0.35),  # task 14
    (  3.70,  -4.60,   3.50,   1.05),  # task 15
]

demo_pairs_multi: List[Tuple[float, float, float, float, float, float]] = [
    ( -2.95,  -2.10,  -4.35,   8.85,  -5.40,   6.90),  # group 1
    ( -5.85,  -6.60,  -0.20,  -3.55,   0.95,   2.65),  # group 2
    ( -3.90,   8.50,  -6.20,   1.35,  -4.80,   0.65),  # group 3
    (  5.60,  -6.30,  -2.15,  -0.60,   0.60,   1.55),  # group 4
    (  1.85,  -5.90,   0.75,   3.35,  -4.05,   8.70),  # group 5
    ( -3.45,   3.80,   2.25,   1.50,   1.35,   2.95),  # group 6
    (  1.85,  -3.55,  -1.65,  -0.40,  -2.65,   0.25),  # group 7
    ( -1.80,  -5.10,  -3.95,  -5.45,   1.55,  -5.90),  # group 8
    ( -0.70,  -3.05,  -3.30,  -9.35,  -6.00,  -2.85),  # group 9
    ( -3.85,   6.90,  -3.05,  -2.15,  -1.55,   0.25),  # group 10
    ( -4.60,  -3.75,  -5.60,   8.80,  -4.50,  -0.15),  # group 11
    (  1.55,  -2.45,  -0.05,  -3.80,  -6.05,  -4.35),  # group 12
    ( -3.70,   8.75,   0.10,   5.25,   0.50,  -2.80),  # group 13
    ( -0.40,  -6.60,  -3.75,  -1.70,   1.85,  -7.85),  # group 14
    ( -5.70,  -4.55,  -2.45,  -0.80,  -3.35,  -7.00),  # group 15
]


def euclidean(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])

def simulate_task_energy_time(task: TaskSpec) -> Tuple[float, float, List[str]]:
    """
    실제 픽/드롭/주행 단계를 따라가며 시간, 에너지 변화를 시뮬레이션.
    로그를 리스트로 반환.
    """
    log = []
    E_total = 0.0
    t_total = 0.0
    carried_weight = 0.0
    waypoints = task.waypoints
    pick_weights = task.pick_weights or [0.0] * len(waypoints)
    n = len(waypoints)
    last_speed = 0.0
    for i in range(n - 1):
        src = waypoints[i]
        dst = waypoints[i + 1]
        dist = euclidean(src, dst)
        # 픽/드롭 구분: 첫 이동은 empty, 이후는 loaded
        if i == 0:
            speed = SPEED_EMPTY_MPS
            factor = 1.0
        else:
            speed = SPEED_LOADED_MPS
            factor = LOADED_FACTOR + LOAD_WEIGHT_FACTOR * carried_weight
        t_drive = dist / speed if speed > 1e-6 else 0.0
        E_drive = K_DRIVE_WH_PER_M * factor * dist

        # 관성(가속/감속) 기반 에너지: Δv에 따라 운동에너지 변화
        total_mass = BASE_ROBOT_MASS_KG + carried_weight
        delta_v = speed - last_speed
        inertia_j = 0.5 * total_mass * (delta_v ** 2)
        inertia_wh = inertia_j * INERTIA_EFFICIENCY * WH_PER_J
        last_speed = speed

        t_total += t_drive
        E_total += E_drive + inertia_wh
        log.append(f"drive {i}: {src}->{dst} dist={dist:.2f}m t={t_drive:.2f}s E={E_drive:.3f}Wh inertia={inertia_wh:.5f}Wh (carried={carried_weight:.2f}kg)")
        # pick/drop 이벤트
        if i < len(pick_weights):
            # pick
            E_total += E_PICK_WH
            t_total += PICK_WAIT_S
            carried_weight += pick_weights[i]
            log.append(f"pick {i}: +{pick_weights[i]:.2f}kg, t+{PICK_WAIT_S}s, E+{E_PICK_WH}Wh")
        if i > 0:
            # drop
            E_total += E_DROP_WH
            t_total += DROP_WAIT_S
            carried_weight = max(0.0, carried_weight - pick_weights[i-1])
            log.append(f"drop {i}: -{pick_weights[i-1]:.2f}kg, t+{DROP_WAIT_S}s, E+{E_DROP_WH}Wh")
    return t_total, E_total, log

def main():


    # timeline.py의 build_tasks() 구조 참고
    tasks = []

    # 1..20 one-way: demo_specs
    for tid, (x1, y1, x2, y2) in enumerate(demo_specs, start=1):
        tasks.append(TaskSpec(task_id=tid, kind="one_way", waypoints=[(x1, y1), (x2, y2)], pick_weights=[8.0, 0.0]))

    # 21..35 round-trip: demo_pairs
    for k, (x1, y1, x2, y2) in enumerate(demo_pairs, start=21):
        tasks.append(TaskSpec(task_id=k, kind="round_trip", waypoints=[(x1, y1), (x2, y2), (x1, y1)], pick_weights=[8.0, 8.0, 0.0]))

    # 36..50 via-point: demo_pairs_multi
    for k, (px, py, d1x, d1y, d2x, d2y) in enumerate(demo_pairs_multi, start=36):
        tasks.append(TaskSpec(task_id=k, kind="via_point", waypoints=[(px, py), (px, py), (d1x, d1y), (d2x, d2y)], pick_weights=[8.0, 4.0, 0.0, 0.0]))

    # if len(tasks) != 50:
    #     raise ValueError(f"Expected 50 tasks, got {len(tasks)}. Adjust slices.")

    out_name = "task_time_energy_simulation_log.txt"
    out_path = os.path.join(os.getcwd(), out_name)
    with open(out_path, "w", encoding="utf-8") as f:
        for task in tasks:
            t, e, log = simulate_task_energy_time(task)
            f.write(f"Task {task.task_id} ({task.kind}):\n")
            for line in log:
                f.write(f"  {line}\n")
            f.write(f"  TOTAL: t={t:.2f}s, E={e:.3f}Wh\n\n")
    print(f"Saved: {out_path}")

if __name__ == "__main__":
    main()
