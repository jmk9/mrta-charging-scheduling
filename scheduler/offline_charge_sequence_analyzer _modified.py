"""
offline_charge_sequence_analyzer.py

scheduler_node.py 의 estimate_task_energy / _objective_function 로직을 그대로 사용하여
threshold, feasibility, optimized 3가지 충전 전략을 비교 분석하는 스크립트.

사용법:
1. DECODE_NO_CHARGE_TEXT, DECODE_OPTIMIZED_TEXT 에 sequence_log.txt 의 [decode-RIME] 라인을 복붙
2. Nav2 + Gazebo 실행 후 python3 offline_charge_sequence_analyzer.py 실행
3. 전역 최적화(그리디가 아닌 경우의 수 탐색): --global-phase2 [--phase2-method beam|exhaustive] [--phase2-beam-width N]
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(SCRIPT_DIR, "log_sequence")
os.makedirs(LOG_DIR, exist_ok=True)


def _resolve_output_path(path: str) -> str:
    """Return absolute path within log_sequence unless absolute provided."""
    if os.path.isabs(path):
        return path
    return os.path.join(LOG_DIR, path)

# ROS / Nav2
try:
    import rclpy
    from rclpy.node import Node
    from rclpy.action import ActionClient
    from nav2_msgs.action import ComputePathToPose
    from geometry_msgs.msg import PoseStamped
    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False
    Node = object
    ActionClient = object
    ComputePathToPose = object
    PoseStamped = object


# =============================================================================
# Data Classes (scheduler_node.py 와 동일)
# =============================================================================

@dataclass
class TaskSpec:
    task_id: int
    task_type: str
    picks: List[Tuple[float, float]]
    drops: List[Tuple[float, float]]
    pick_weights: List[float] = field(default_factory=list)
    deliveries: List[List[int]] = field(default_factory=list)
    pick_wait_s: float = 1.5
    drop_wait_s: float = 2.0
    charge_duration_s: float = 0.0

    @property
    def drop_x(self) -> float:
        return float(self.drops[0][0]) if self.drops else 0.0

    @property
    def drop_y(self) -> float:
        return float(self.drops[0][1]) if self.drops else 0.0


@dataclass
class RobotState:
    robot_id: str
    x: float
    y: float
    soc: float
    available: bool = True
    namespace: str = ""


@dataclass
class ChargerSpec:
    charger_id: int
    x: float
    y: float


# =============================================================================
# 상수 (scheduler_node.py 와 동일)
# =============================================================================

K_DRIVE_WH_PER_M = 0.275
E_PICK_WH = 0.5
E_DROP_WH = 0.5
V_EMPTY_MPS = 0.3
V_LOADED_MPS = 0.25
LOADED_FACTOR = 1.3
LOAD_WEIGHT_FACTOR = 0.05
# 60Wh 완충 600초 → 0.1 Wh/s
CHARGE_RATE_WH_PER_S = 0.1

# =============================================================================
# temp.py 스타일 시각화 색상/항목
# =============================================================================

# temp.py와 동일한 색상키
COLOR_MAP = {
    "one_way": "#3399ff",
    "round_trip": "#33cc33",
    "via_point": "#ffb347",
    "move": "#888888",
    "charge": "#ff66cc",  # (실제 bar는 흰색+해치로 그리고, legend용으로만 사용)
}

# offline analyzer의 task_type -> temp.py의 색상키 매핑(일관성만 보장; 룩앤필 동일)
TASKTYPE_TO_COLORKEY = {
    "PICK_DROP": "one_way",
    "DUAL_PICK_MULTI_DROP": "round_trip",
    "MULTI_PICK_DROP": "via_point",
}

TASK_TYPE_WEIGHTS = {
    "PICK_DROP": [8.0],
    "DUAL_PICK_MULTI_DROP": [8.0, 8.0],
    "MULTI_PICK_DROP": [8.0, 4.0],
}

ROBOT_INITIAL_POSITIONS = {
    "tb1": (-3.5, 1.5),
    "tb2": (4.5, 1.5),
    "tb3": (-1.5, -0.5),
    "tb4": (1.0, -8.0),
    'tb5': (-5.0, -3.0),
    'tb6': (5.0, -3.0),
    'tb7': (-3.0, 5.0),  
            # 'tb8': (3.0, 0.0),
            # 'tb9': (0.0, 0.0),
            # 'tb10': (-2.0, -5.0),
            # 'tb11': (0.65, -9.00),
            # 'tb12': (-3.75, 8.90),
            # 'tb13': (-5.25, -2.15),
            # 'tb14': (-4.85, 4.40),
            # 'tb15': (0.20, -7.15),
}

ROBOT_CAP_WH = {
    "tb1": 40.0,
    "tb2": 60.0,
    "tb3": 40.0,
    "tb4": 60.0,
    "tb5": 40.0,
    "tb6": 60.0,
    "tb7": 40.0,
    # "tb8": 60.0,
    # "tb9": 40.0,
    # "tb10": 60.0,
    # "tb11": 40.0,
    # "tb12": 60.0,
    # "tb13": 40.0,
    # "tb14": 60.0,
}

# c1 = (0, -6), c2 = (0, 4) 로 표시 (이름만 반대)
CHARGERS = [
    ChargerSpec(charger_id=2, x=0.0, y=4.0),   # ch_idx 0 → c2
    ChargerSpec(charger_id=1, x=0.0, y=-6.0),  # ch_idx 1 → c1
    # ChargerSpec(charger_id=3, x=-3.0, y=-1.0),  # ch_idx 2 → c3
]


# =============================================================================
# Task 정의 (scheduler_node._init_demo_tasks 와 동일)
# =============================================================================

def _init_demo_tasks() -> Dict[int, TaskSpec]:
    tasks: Dict[int, TaskSpec] = {}
    tid = 1

    # Type 1: PICK_DROP
    demo_specs = [
    # id 1 ~ 4
            (-5.0, -4.0, -5.0, -8.0),     # 1
            ( 3.0,  1.0,  3.0, -4.0),     # 2
            ( 4.8,  1.0,  5.0, -4.0),     # 3
            (-5.0, -3.0,  1.0, -9.0),     # 4
            # 5 ~ 8
            (-5.0, -1.0,  0.0, -7.0),     # 5
            (-5.0,  5.0, -5.0,  7.0),     # 6
            (-5.0,  4.0,  1.0,  5.0),     # 7
            (-5.0,  0.0,  1.2,  6.0),     # 8
            # 9 ~ 12
            ( 3.0, -5.0,  3.0, -9.0),     # 9
            ( 5.0, -5.0,  5.0, -8.8),     # 10
            (-3.0, -7.0, -3.0,  1.0),     # 11
            (-3.0,  7.0, -1.5, -0.5),     # 12
            # 13 ~ 16
            (-5.0,  1.0, -5.0,  5.0),     # 13
            (-3.0,  1.0, -3.0,  4.0),     # 14
            ( 0.0,  1.0,  0.0,  4.0),     # 15
            ( 2.0,  1.0,  1.0,  4.0),     # 16
            # 17 ~ 20
            ( 1.5,  0.0,  0.0,  5.0),     # 17
            ( 1.5, -1.0,  1.5, -4.0),     # 18
            ( 1.5, -5.0,  1.5, -9.0),     # 19
            ( 0.0, -6.0,  3.0, -6.0),     # 20
            (  0.05,   4.55,  -2.15,  -4.30),  # 21
            ( -0.30,  -2.20,  -2.25,  -3.00),  # 22
            (  0.65,  -4.75,   1.35,  -8.55),  # 23
            ( -3.90,   1.40,  -6.05,   0.35),  # 24
            ( -4.10,  -5.35,   0.55,   3.45),  # 25
            ( -3.55,   0.25,  -4.15,  -8.00),  # 26
            ( -5.10,   9.15,  -3.80,   0.20),  # 27
            ( -6.15,   2.40,  -2.55,  -4.00),  # 28
            ( -3.85,  -9.10,  -3.60,  -2.50),  # 29
            (  0.00,   5.10,  -4.35,   4.70),  # 30
    ]
    for (px, py, dx, dy) in demo_specs:
        tasks[tid] = TaskSpec(
            task_id=tid, task_type="PICK_DROP",
            picks=[(px, py)], drops=[(dx, dy)],
            pick_weights=[round(0.9 + 0.15 * (tid % 4), 2)],
            pick_wait_s=2.0, drop_wait_s=2.0,
        )
        tid += 1

    # Type 2: DUAL_PICK_MULTI_DROP
    demo_pairs = [
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
            ( -3.45,   0.30,   0.65,   1.40),  # task 16
            (  3.70,  -4.70,   5.75,  -4.50),  # task 17
            ( -4.85,   5.55,  -2.95,  -0.45),  # task 18
            ( -2.15,  -1.20,   1.25,   6.30),  # task 19
            ( -3.65,  -7.40,   1.65,  -1.90),  # task 20
    ]
    for i, (px, py, dx, dy) in enumerate(demo_pairs):
        pick_loc, drop = (px, py), (dx, dy)
        picks = [pick_loc, drop]
        drops = [drop, pick_loc]
        pick_weights = [round(0.9 + 0.15 * ((i + j) % 4), 2) for j in range(2)]
        deliveries = [[0], [1]]
        tasks[tid] = TaskSpec(
            task_id=tid, task_type="DUAL_PICK_MULTI_DROP",
            picks=picks, drops=drops,
            pick_weights=pick_weights, deliveries=deliveries,
            pick_wait_s=1.5, drop_wait_s=2.0,
        )
        tid += 1

    # Type 3: MULTI_PICK_DROP
    demo_pairs_multi = [
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
            (  0.60,   3.70,  -4.30,  -7.65,  -0.20,  -7.45),  # group 16
            ( -5.60,  -0.40,  -2.90,  -2.35,  -0.70,  -5.10),  # group 17
            ( -2.85,  -4.55,  -0.35,  -5.20,  -1.70,  -9.50),  # group 18
            (  0.05,  -3.55,   5.60,  -3.05,   1.70,  -1.30),  # group 19
            ( -5.20,   5.65,   1.60,  -2.50,  -4.55,  -3.70),  # group 20
    ]
    for i, entry in enumerate(demo_pairs_multi):
        pick_loc = (entry[0], entry[1])
        drop1, drop2 = (entry[2], entry[3]), (entry[4], entry[5])
        picks = [pick_loc, pick_loc]
        drops = [drop1, drop2]
        pick_weights = [round(0.9 + 0.15 * ((i + j) % 4), 2) for j in range(2)]
        deliveries = [[0], [1]]
        tasks[tid] = TaskSpec(
            task_id=tid, task_type="MULTI_PICK_DROP",
            picks=picks, drops=drops,
            pick_weights=pick_weights, deliveries=deliveries,
            pick_wait_s=1.5, drop_wait_s=2.0,
        )
        tid += 1

    return tasks


TASKS = _init_demo_tasks()


# =============================================================================
# Nav2 Path Query Node
# =============================================================================

def _calculate_path_length(path) -> float:
    if not getattr(path, "poses", None) or len(path.poses) < 2:
        return 0.0
    total = 0.0
    for i in range(1, len(path.poses)):
        p1, p2 = path.poses[i - 1].pose.position, path.poses[i].pose.position
        total += math.hypot(p1.x - p2.x, p1.y - p2.y)
    return total


class PathQueryNode(Node):
    def __init__(self, robot_ids: List[str]) -> None:
        super().__init__("offline_analyzer")
        self.robot_ids = list(robot_ids)
        self.path_cache: Dict[Tuple[float, float, float, float], float] = {}
        self._nav2_clients: Dict[str, ActionClient] = {}
        self._working: List[Tuple[str, ActionClient]] = []

        for rid in self.robot_ids:
            action_name = f"/{rid}/compute_path_to_pose"
            client = ActionClient(self, ComputePathToPose, action_name)
            self._nav2_clients[rid] = client
            if client.wait_for_server(timeout_sec=1.0):
                self._working.append((rid, client))

        if not self._working:
            raise RuntimeError(f"Nav2 required: no action server available for {self.robot_ids}")
        self.get_logger().info(f"Nav2: {len(self._working)} server(s) ready ({[r for r, _ in self._working]})")

    def nav2_distance(self, x1: float, y1: float, x2: float, y2: float, robot_id: str = "") -> float:
        if abs(x1 - x2) < 1e-3 and abs(y1 - y2) < 1e-3:
            return 0.0
        key = (x1, y1, x2, y2)
        if key in self.path_cache:
            return self.path_cache[key]

        now = self.get_clock().now().to_msg()
        goal_msg = ComputePathToPose.Goal()
        goal_msg.goal = PoseStamped()
        goal_msg.goal.header.frame_id = "map"
        goal_msg.goal.header.stamp = now
        goal_msg.goal.pose.position.x, goal_msg.goal.pose.position.y = float(x2), float(y2)
        goal_msg.goal.pose.orientation.w = 1.0
        goal_msg.start = PoseStamped()
        goal_msg.start.header.frame_id = "map"
        goal_msg.start.header.stamp = now
        goal_msg.start.pose.position.x, goal_msg.start.pose.position.y = float(x1), float(y1)
        goal_msg.start.pose.orientation.w = 1.0
        goal_msg.use_start = True
        goal_msg.planner_id = ""

        for rid, client in self._working:
            if not client.server_is_ready():
                continue
            try:
                sf = client.send_goal_async(goal_msg)
                rclpy.spin_until_future_complete(self, sf, timeout_sec=2.0)
                if not sf.done():
                    continue
                gh = sf.result()
                if not gh or not gh.accepted:
                    continue
                rf = gh.get_result_async()
                rclpy.spin_until_future_complete(self, rf, timeout_sec=2.0)
                if not rf.done():
                    continue
                res = rf.result()
                if res.status != 4:
                    continue
                L = _calculate_path_length(res.result.path)
                if L > 0.0:
                    self.path_cache[key] = L
                    return L
            except Exception:
                continue

        raise RuntimeError(f"Nav2 path failed ({x1:.2f},{y1:.2f}) -> ({x2:.2f},{y2:.2f})")


# =============================================================================
# estimate_task_energy (scheduler_node.py 와 동일)
# =============================================================================

def get_item_weight(task_type: str, pick_index: int) -> float:
    weights = TASK_TYPE_WEIGHTS.get(task_type, [8.0])
    if pick_index < len(weights):
        return weights[pick_index]
    return weights[-1] if weights else 8.0


def estimate_task_energy(
    robot: RobotState,
    task: TaskSpec,
    path_node: PathQueryNode,
) -> Tuple[float, float]:
    """(E_total_wh, t_total_s)"""
    E_total, t_total = 0.0, 0.0

    if task.task_type == "CHARGE":
        # drive from robot -> charger (use drop as charger location) — scheduler_node와 동일
        if task.drops:
            cx, cy = task.drops[0]
        else:
            cx, cy = task.drop_x, task.drop_y
        d_drive = path_node.nav2_distance(robot.x, robot.y, cx, cy, robot.robot_id)
        E_drive = K_DRIVE_WH_PER_M * d_drive
        t_drive = d_drive / V_EMPTY_MPS if V_EMPTY_MPS > 1e-6 else 0.0
        E_total += E_drive
        t_total += t_drive
        # 완충까지 시간: 현재 배터리(robot.soc)와 용량 기반 (scheduler Phase1/2와 동일)
        cap_wh = ROBOT_CAP_WH.get(robot.robot_id, 60.0)
        charge_duration_s = _charge_duration(cap_wh, robot.soc)
        t_total += charge_duration_s
        return E_total, t_total

    curr_x, curr_y = float(robot.x), float(robot.y)
    picks = list(task.picks) if task.picks else []
    drops = list(task.drops) if task.drops else []

    if task.task_type == "MULTI_PICK_DROP":
        carried_weight, carried_items, picked_count = 0.0, [], 0
        for i, (px, py) in enumerate(picks):
            d = path_node.nav2_distance(curr_x, curr_y, px, py, robot.robot_id)
            if picked_count == 0:
                E_total += K_DRIVE_WH_PER_M * d
                t_total += d / V_EMPTY_MPS if V_EMPTY_MPS > 1e-6 else 0.0
            else:
                wt = LOAD_WEIGHT_FACTOR * carried_weight
                E_total += K_DRIVE_WH_PER_M * (LOADED_FACTOR + wt) * d
                t_total += d / V_LOADED_MPS if V_LOADED_MPS > 1e-6 else 0.0
            E_total += E_PICK_WH
            t_total += float(task.pick_wait_s)
            w = task.pick_weights[i] if i < len(task.pick_weights) else get_item_weight(task.task_type, i)
            carried_items.append((i, w))
            carried_weight += w
            picked_count += 1
            curr_x, curr_y = float(px), float(py)

        if not drops:
            return E_total, t_total

        deliveries = list(task.deliveries) if task.deliveries else []
        if not deliveries or all(not d for d in deliveries):
            remaining = picked_count
            for j, (dx, dy) in enumerate(drops):
                if remaining <= 0:
                    break
                d = path_node.nav2_distance(curr_x, curr_y, dx, dy, robot.robot_id)
                wt = LOAD_WEIGHT_FACTOR * carried_weight
                E_total += K_DRIVE_WH_PER_M * (LOADED_FACTOR + wt) * d
                t_total += d / V_LOADED_MPS if V_LOADED_MPS > 1e-6 else 0.0
                if carried_items:
                    dw = carried_items[0][1]
                    carried_items.pop(0)
                    carried_weight = max(0.0, carried_weight - dw)
                E_total += E_DROP_WH
                t_total += float(task.drop_wait_s)
                remaining -= 1
                curr_x, curr_y = float(dx), float(dy)
            return E_total, t_total

        for j, (dx, dy) in enumerate(drops):
            deliver_idxs = list(deliveries[j]) if j < len(deliveries) else []
            d = path_node.nav2_distance(curr_x, curr_y, dx, dy, robot.robot_id)
            wt = LOAD_WEIGHT_FACTOR * carried_weight
            E_total += K_DRIVE_WH_PER_M * (LOADED_FACTOR + wt) * d
            t_total += d / V_LOADED_MPS if V_LOADED_MPS > 1e-6 else 0.0
            deliver_count, delivered_weight = 0, 0.0
            for idx in deliver_idxs:
                for k, (pi, w) in enumerate(list(carried_items)):
                    if pi == idx:
                        delivered_weight += w
                        deliver_count += 1
                        carried_items = [(a, b) for a, b in carried_items if a != pi]
                        break
            E_total += E_DROP_WH * deliver_count
            t_total += float(task.drop_wait_s) * deliver_count
            carried_weight = max(0.0, carried_weight - delivered_weight)
            curr_x, curr_y = float(dx), float(dy)
        return E_total, t_total

    # Interleaved picks/drops (PICK_DROP, DUAL_PICK_MULTI_DROP)
    n = max(len(picks), len(drops))
    carried_weight = 0.0
    for i in range(n):
        if i < len(picks):
            px, py = picks[i]
            d = path_node.nav2_distance(curr_x, curr_y, px, py, robot.robot_id)
            E_total += K_DRIVE_WH_PER_M * d
            t_total += d / V_EMPTY_MPS if V_EMPTY_MPS > 1e-6 else 0.0
            E_total += E_PICK_WH
            t_total += float(task.pick_wait_s)
            w = task.pick_weights[i] if i < len(task.pick_weights) else get_item_weight(task.task_type, i)
            carried_weight += w
            curr_x, curr_y = float(px), float(py)
        if i < len(drops):
            dx, dy = drops[i]
            d = path_node.nav2_distance(curr_x, curr_y, dx, dy, robot.robot_id)
            wt = LOAD_WEIGHT_FACTOR * carried_weight
            E_total += K_DRIVE_WH_PER_M * (LOADED_FACTOR + wt) * d
            t_total += d / V_LOADED_MPS if V_LOADED_MPS > 1e-6 else 0.0
            E_total += E_DROP_WH
            t_total += float(task.drop_wait_s)
            dropped = get_item_weight(task.task_type, i)
            carried_weight = max(0.0, carried_weight - dropped)
            curr_x, curr_y = float(dx), float(dy)

    return E_total, t_total


# =============================================================================
# 시뮬레이션 (_objective_function Phase 1 + Phase 2 와 동일)
# =============================================================================

@dataclass
class SimResult:
    robot_id: str
    strategy: str
    makespan: float
    energy_used: float
    charge_count: int
    waiting_total: float
    charging_total: float  # 로봇별 충전 시간 합계(초)
    discharge_count: int
    final_sequence: List[int]
    charger_assignments: Optional[List[int]] = None  # i번째 충전 -> ch_idx (0=c1, 1=c2)
    travel_total: float = 0.0  # 로봇별 이동(Travel) 시간 합계(초): task 이동 + 충전소 이동


# Phase1 시뮬레이션에서 task별 "이동(move)" + "작업(pick~종료)" 세그먼트를 기록:
# PHASE1_TASK_TIMELINE[strategy][robot_id] = List[(kind, task_id, start, end)] where kind in ("move", "task")
# task 하나당 ("move", None, s1, e1), ("task", task_id, s2, e2) 두 개가 순서대로 들어감.
PHASE1_TASK_TIMELINE: Dict[str, Dict[str, List[Tuple[str, Optional[int], float, float]]]] = {}


def _charge_duration(cap_wh: float, energy: float) -> float:
    """완충까지 시간(초). 남은 Wh / (Wh/s)."""
    if cap_wh <= 0.0 or CHARGE_RATE_WH_PER_S <= 1e-6:
        return 0.0
    remaining_wh = max(0.0, cap_wh - energy)
    return remaining_wh / CHARGE_RATE_WH_PER_S


def simulate_robot(
    robot_id: str,
    seq: List[TaskSpec],
    cap_wh: float,
    init_pos: Tuple[float, float],
    chargers: List[ChargerSpec],
    path_node: PathQueryNode,
    threshold_frac: float = 0.15,
    strategy: str = "optimized",
    insert_charge: bool = False,
    charger_choices: Optional[List[int]] = None,
) -> Tuple[SimResult, List[Tuple[str, int, List[Tuple[int, float, float]], float, float, float]]]:
    """Phase 1: 로봇별 시뮬레이션 (scheduler_node._objective_function 과 동일)"""
    sim = RobotState(robot_id=robot_id, x=init_pos[0], y=init_pos[1], soc=cap_wh)
    energy = cap_wh
    robot_time, robot_energy = 0.0, 0.0
    discharge_count, charge_count = 0, 0
    charge_events: List[Tuple[str, int, List[Tuple[int, float, float]], float, float, float]] = []
    final_seq: List[int] = []
    seq_charge_idx = 0  # optimized 시퀀스 내 CHARGE task 순번 (c1/c2가 있을 때 매핑용)

    # 이 로봇/전략에 대한 Phase1 task 타임라인 버퍼 준비
    phase1_timeline_for_robot = PHASE1_TASK_TIMELINE.setdefault(strategy, {}).setdefault(robot_id, [])

    for task_idx, task in enumerate(seq):
        sim.soc = energy  # estimate_task_energy(CHARGE)에서 현재 배터리로 완충시간 계산
        # threshold / feasibility 시 동적 충전 삽입
        if insert_charge and task.task_type != "CHARGE":
            try:
                req_e, _ = estimate_task_energy(sim, task, path_node)
            except Exception:
                req_e = 0.0

            need_charge = False
            if strategy == "threshold":
                if energy - req_e < cap_wh * threshold_frac:
                    need_charge = True
            elif strategy == "feasibility":
                if energy < req_e:
                    need_charge = True

            if need_charge:
                # 충전 삽입
                time_before = robot_time
                duration = _charge_duration(cap_wh, energy)
                reachable: List[Tuple[int, float, float]] = []
                for ch_idx, ch in enumerate(chargers):
                    try:
                        e_ch, t_ch = estimate_task_energy(
                            sim,
                            TaskSpec(task_id=-1, task_type="CHARGE", picks=[(ch.x, ch.y)], drops=[(ch.x, ch.y)]),
                            path_node,
                        )
                        if energy >= e_ch:
                            reachable.append((ch_idx, robot_time + t_ch, e_ch))
                    except Exception:
                        pass

                if reachable:
                    # t_ch = 이동+완충시간(robot.soc 반영) → arrive = 충전 종료 시각
                    best = min(reachable, key=lambda x: x[1])
                    ch_idx, arrive, e_cost = best
                    ch = chargers[ch_idx]
                    energy -= e_cost
                    robot_energy += e_cost
                    travel_t = arrive - time_before
                    robot_time += travel_t
                    energy = cap_wh
                    sim.x, sim.y = ch.x, ch.y
                    charge_events.append((robot_id, task_idx, reachable, time_before, arrive, duration))
                    charge_count += 1
                    final_seq.append(-1)
                else:
                    discharge_count += 1

        # CHARGE task (optimized 시퀀스: decode 가 삽입한 충전)
        if task.task_type == "CHARGE":
            time_before = robot_time
            duration = _charge_duration(cap_wh, energy)
            reachable: List[Tuple[int, float, float]] = []

            # optimized + charger_choices 가 있으면, decode 가 선택한 충전소만 사용
            fixed_ch_idx: Optional[int] = None
            if strategy == "optimized" and charger_choices is not None and seq_charge_idx < len(charger_choices):
                fixed_ch_idx = charger_choices[seq_charge_idx]

            if fixed_ch_idx is not None:
                ch = chargers[fixed_ch_idx]
                try:
                    e_ch, t_ch = estimate_task_energy(
                        sim,
                        TaskSpec(task_id=-1, task_type="CHARGE", picks=[(ch.x, ch.y)], drops=[(ch.x, ch.y)]),
                        path_node,
                    )
                    if energy >= e_ch:
                        reachable.append((fixed_ch_idx, robot_time + t_ch, e_ch))
                except Exception:
                    pass
            else:
                # fallback: 기존처럼 모든 충전소 후보 중 best 선택
                for ch_idx, ch in enumerate(chargers):
                    try:
                        e_ch, t_ch = estimate_task_energy(
                            sim,
                            TaskSpec(task_id=-1, task_type="CHARGE", picks=[(ch.x, ch.y)], drops=[(ch.x, ch.y)]),
                            path_node,
                        )
                        if energy >= e_ch:
                            reachable.append((ch_idx, robot_time + t_ch, e_ch))
                    except Exception:
                        pass

            if reachable:
                # t_ch = 이동+완충시간(robot.soc 반영) → arrive = 충전 종료 시각
                best = min(reachable, key=lambda x: x[1])
                ch_idx, arrive, e_cost = best
                ch = chargers[ch_idx]
                energy -= e_cost
                robot_energy += e_cost
                travel_t = arrive - time_before
                robot_time += travel_t
                energy = cap_wh
                sim.x, sim.y = ch.x, ch.y
                charge_events.append((robot_id, task_idx, reachable, time_before, arrive, duration))
                charge_count += 1
                final_seq.append(-1)
            else:
                discharge_count += 1
            # decode 기반 CHARGE 하나 처리 완료
            seq_charge_idx += 1
            continue

        # 일반 task 실행: "이동(로봇→첫 pick)" / "작업(pick~종료)" 구간을 나눠 기록
        start_x, start_y = float(sim.x), float(sim.y)
        if task.picks:
            pick_x, pick_y = float(task.picks[0][0]), float(task.picks[0][1])
        elif task.drops:
            pick_x, pick_y = float(task.drops[0][0]), float(task.drops[0][1])
        else:
            pick_x, pick_y = start_x, start_y

        try:
            d_to_pick = path_node.nav2_distance(start_x, start_y, pick_x, pick_y, robot_id)
        except Exception:
            d_to_pick = math.hypot(pick_x - start_x, pick_y - start_y)
        t_to_start = d_to_pick / V_EMPTY_MPS if V_EMPTY_MPS > 1e-9 else 0.0

        try:
            req_e, t_task = estimate_task_energy(sim, task, path_node)
        except Exception:
            req_e, t_task = 0.0, 0.0

        if energy < req_e:
            discharge_count += 1
            energy = 0.0

        move_start = robot_time
        move_end = robot_time + t_to_start
        task_start = move_end
        task_end = robot_time + t_task

        phase1_timeline_for_robot.append(("move", None, float(move_start), float(move_end)))
        phase1_timeline_for_robot.append(("task", int(task.task_id), float(task_start), float(task_end)))

        energy -= req_e
        robot_energy += req_e
        robot_time = task_end
        if task.drops:
            sim.x, sim.y = task.drops[-1]
        final_seq.append(task.task_id)

    charging_total = sum(d for (_, _, _, _, _, d) in charge_events)
    return (
        SimResult(
            robot_id=robot_id,
            strategy=strategy,
            makespan=robot_time,
            energy_used=robot_energy,
            charge_count=charge_count,
            waiting_total=0.0,
            charging_total=charging_total,
            discharge_count=discharge_count,
            final_sequence=final_seq,
        ),
        charge_events,
    )


def phase2_charger_contention(
    charge_events: List[Tuple[str, int, List[Tuple[int, float, float]], float, float, float]],
    num_chargers: int,
) -> Tuple[Dict[str, float], Dict[str, float], List[Tuple[str, int, int]], List[Tuple[str, int, int, float, float, float]]]:
    """Phase 2: 전 로봇 충전소 경합 처리.
    반환: (delay_map, wait_map, event_charger_list, timeline_events).
    timeline_events: (robot_id, task_idx, ch_idx, arrival_t, start_t, end_t) 리스트."""
    charger_next_free: Dict[int, float] = {i: 0.0 for i in range(num_chargers)}
    robot_total_delay: Dict[str, float] = {}
    robot_wait_total: Dict[str, float] = {}
    event_charger_list: List[Tuple[str, int, int]] = []
    timeline_events: List[Tuple[str, int, int, float, float, float]] = []

    # reachable의 두 번째 값 = 충전 종료 시각(finish). Phase2에서는 도착 시각(arrival = finish - duration)으로 사용
    event_list = []
    for robot_id, task_idx, reachable, time_before, sim_end, duration in charge_events:
        if reachable:
            earliest_arrival = min(finish - duration for _, finish, _ in reachable)
            event_list.append((earliest_arrival, robot_id, task_idx, reachable, time_before, sim_end, duration))
    event_list.sort(key=lambda x: x[0])

    for _earliest_arrival, robot_id, task_idx, reachable, time_before, sim_end, duration in event_list:
        acc = robot_total_delay.get(robot_id, 0.0)
        best_idx, best_end = None, float("inf")
        best_start, best_arrival = None, None
        for ch_idx, finish_base, _ in reachable:
            arrival = finish_base - duration + acc  # 도착 시각
            start = max(arrival, charger_next_free[ch_idx])  # 실제 충전 시작 시각
            end = start + duration
            if end < best_end:
                best_end, best_idx = end, ch_idx
                best_start, best_arrival = start, arrival

        if best_idx is None:
            continue

        charger_next_free[best_idx] = best_end
        event_charger_list.append((robot_id, task_idx, best_idx))
        timeline_events.append((robot_id, task_idx, best_idx, best_arrival, best_start, best_end))
        sim_adj = sim_end + acc
        delay = max(0.0, best_end - sim_adj)
        robot_total_delay[robot_id] = acc + delay
        # waiting = 충전소 도착 후 대기한 시간만 (도착~충전시작). 이동 시간 제외
        wait_at_charger = max(0.0, best_start - best_arrival)
        robot_wait_total[robot_id] = robot_wait_total.get(robot_id, 0.0) + wait_at_charger

    return robot_total_delay, robot_wait_total, event_charger_list, timeline_events


def _build_event_list(
    charge_events: List[Tuple[str, int, List[Tuple[int, float, float]], float, float, float]],
) -> List[Tuple[float, str, int, List[Tuple[int, float, float]], float, float, float]]:
    """charge_events에서 도착 시각 기준 정렬된 event_list 생성 (phase2와 동일 순서)."""
    event_list = []
    for robot_id, task_idx, reachable, time_before, sim_end, duration in charge_events:
        if reachable:
            earliest_arrival = min(finish - duration for _, finish, _ in reachable)
            event_list.append((earliest_arrival, robot_id, task_idx, reachable, time_before, sim_end, duration))
    event_list.sort(key=lambda x: x[0])
    return event_list


def evaluate_assignment(
    charge_events: List[Tuple[str, int, List[Tuple[int, float, float]], float, float, float]],
    num_chargers: int,
    assignment_list: List[int],
    process_all_events: bool = True,
) -> Tuple[Dict[str, float], Dict[str, float], List[Tuple[str, int, int]], List[Tuple[str, int, int, float, float, float]]]:
    """고정 충전소 배정으로 Phase2 타임라인 시뮬레이션.
    assignment_list[i] = event_list[i]에 쓸 ch_idx. (event_list 순서 = _build_event_list)
    해당 ch_idx가 reachable에 없으면 reachable 중 첫 번째 사용.
    process_all_events=False면 assignment_list 길이만큼만 처리(부분 배정, beam용)."""
    event_list = _build_event_list(charge_events)
    n_process = len(event_list) if process_all_events else min(len(assignment_list), len(event_list))
    if n_process <= 0:
        return {}, {}, [], []

    charger_next_free: Dict[int, float] = {i: 0.0 for i in range(num_chargers)}
    robot_total_delay: Dict[str, float] = {}
    robot_wait_total: Dict[str, float] = {}
    event_charger_list: List[Tuple[str, int, int]] = []
    timeline_events: List[Tuple[str, int, int, float, float, float]] = []

    for idx in range(n_process):
        (_earliest_arrival, robot_id, task_idx, reachable, time_before, sim_end, duration) = event_list[idx]
        want_ch = assignment_list[idx] if idx < len(assignment_list) else 0
        ch_idx, finish_base, _ = next((r for r in reachable if r[0] == want_ch), reachable[0])

        acc = robot_total_delay.get(robot_id, 0.0)
        arrival = finish_base - duration + acc
        start = max(arrival, charger_next_free[ch_idx])
        end = start + duration

        charger_next_free[ch_idx] = end
        event_charger_list.append((robot_id, task_idx, ch_idx))
        timeline_events.append((robot_id, task_idx, ch_idx, arrival, start, end))
        sim_adj = sim_end + acc
        delay = max(0.0, end - sim_adj)
        robot_total_delay[robot_id] = acc + delay
        wait_at_charger = max(0.0, start - arrival)
        robot_wait_total[robot_id] = robot_wait_total.get(robot_id, 0.0) + wait_at_charger

    if process_all_events and len(assignment_list) < len(event_list):
        # 나머지 이벤트는 reachable[0]으로 처리
        for idx in range(len(assignment_list), len(event_list)):
            (_ea, robot_id, task_idx, reachable, time_before, sim_end, duration) = event_list[idx]
            ch_idx, finish_base, _ = reachable[0]
            acc = robot_total_delay.get(robot_id, 0.0)
            arrival = finish_base - duration + acc
            start = max(arrival, charger_next_free[ch_idx])
            end = start + duration
            charger_next_free[ch_idx] = end
            event_charger_list.append((robot_id, task_idx, ch_idx))
            timeline_events.append((robot_id, task_idx, ch_idx, arrival, start, end))
            delay = max(0.0, end - (sim_end + acc))
            robot_total_delay[robot_id] = robot_total_delay.get(robot_id, 0.0) + delay
            wait_at_charger = max(0.0, start - arrival)
            robot_wait_total[robot_id] = robot_wait_total.get(robot_id, 0.0) + wait_at_charger

    return robot_total_delay, robot_wait_total, event_charger_list, timeline_events


def _global_makespan(phase1_results: Dict[str, SimResult], delay_map: Dict[str, float]) -> float:
    """전역 메이크스팬 = max(로봇별 phase1_makespan + delay)."""
    if not phase1_results:
        return 0.0
    return max(
        (phase1_results[rid].makespan + delay_map.get(rid, 0.0))
        for rid in phase1_results
    )


def phase2_charger_contention_global(
    charge_events: List[Tuple[str, int, List[Tuple[int, float, float]], float, float, float]],
    num_chargers: int,
    phase1_results: Dict[str, SimResult],
    method: str = "beam",
    beam_width: int = 10,
    max_exhaustive: int = 100000,
) -> Tuple[Dict[str, float], Dict[str, float], List[Tuple[str, int, int]], List[Tuple[str, int, int, float, float, float]]]:
    """Phase2 충전소 배정을 전역 메이크스팬 최소화로 탐색.
    - method 'greedy': 기존 그리디 (로봇 시점 최적).
    - method 'exhaustive': 모든 조합 시도 (조합 수 <= max_exhaustive일 때만).
    - method 'beam': 비임 탐색으로 상위 beam_width개만 유지하며 탐색.
    반환: (delay_map, wait_map, event_charger_list, timeline_events)."""
    event_list = _build_event_list(charge_events)
    if not event_list:
        return phase2_charger_contention(charge_events, num_chargers)

    # 각 이벤트별 가능한 ch_idx 목록 (reachable의 ch_idx)
    options_per_event: List[List[int]] = []
    for _ea, _rid, _ti, reachable, _tb, _se, _du in event_list:
        options_per_event.append([r[0] for r in reachable])

    if method == "greedy":
        return phase2_charger_contention(charge_events, num_chargers)

    if method == "exhaustive":
        import itertools
        total = 1
        for o in options_per_event:
            total *= len(o)
        if total > max_exhaustive:
            # 폭주 방지: exhaustive 대신 greedy 반환
            return phase2_charger_contention(charge_events, num_chargers)
        best_ms, best_out = float("inf"), None
        for assignment in itertools.product(*options_per_event):
            delay_map, wait_map, ev_ch, tl = evaluate_assignment(charge_events, num_chargers, list(assignment))
            ms = _global_makespan(phase1_results, delay_map)
            if ms < best_ms:
                best_ms, best_out = ms, (delay_map, wait_map, ev_ch, tl)
        return best_out if best_out else phase2_charger_contention(charge_events, num_chargers)

    # method == "beam"
    # beam: list of (assignment_prefix, score). score = 해당 접두사까지 시뮬레이션한 전역 메이크스팬 하한
    beam: List[Tuple[List[int], float]] = [([], 0.0)]
    n = len(event_list)

    for i in range(n):
        reachable_ch = options_per_event[i]
        candidates: List[Tuple[List[int], float]] = []
        for prefix, _ in beam:
            for ch in reachable_ch:
                assign = prefix + [ch]
                delay_map, _w, _ec, _tl = evaluate_assignment(
                    charge_events, num_chargers, assign, process_all_events=False
                )
                score = _global_makespan(phase1_results, delay_map)
                candidates.append((assign, score))
        candidates.sort(key=lambda x: (x[1], len(x[0])))
        beam = candidates[:beam_width]

    if not beam:
        return phase2_charger_contention(charge_events, num_chargers)
    best_assign = min(beam, key=lambda x: x[1])[0]
    return evaluate_assignment(charge_events, num_chargers, best_assign, process_all_events=True)


def debug_print_charger_decisions(
    charge_events: List[Tuple[str, int, List[Tuple[int, float, float]], float, float, float]],
    num_chargers: int,
) -> None:
    """Phase 2 의 '충전소 선택' 판단 과정을 이벤트별로 상세 출력.

    각 충전 이벤트에 대해:
      - 두 충전소 후보 각각의 도착 시각, 예상 대기 시간, 충전 시작/종료 시각
      - 그 중 어떤 충전소를 선택했는지 (가장 빨리 끝나는 쪽)
    를 텍스트로 시각화해서 보여준다.
    """
    if not charge_events:
        return

    charger_next_free: Dict[int, float] = {i: 0.0 for i in range(num_chargers)}
    robot_total_delay: Dict[str, float] = {}

    event_list = []
    for robot_id, task_idx, reachable, time_before, sim_end, duration in charge_events:
        if reachable:
            earliest_arrival = min(finish - duration for _, finish, _ in reachable)
            event_list.append((earliest_arrival, robot_id, task_idx, reachable, time_before, sim_end, duration))
    event_list.sort(key=lambda x: x[0])

    print("\n" + "=" * 100)
    print("Charger decision details (Phase 2) — 각 충전 이벤트별 c1/c2 비교")
    print("=" * 100)

    for _earliest_arrival, robot_id, task_idx, reachable, time_before, sim_end, duration in event_list:
        acc = robot_total_delay.get(robot_id, 0.0)
        print(f"\n[Robot {robot_id}] charge_event at task_idx={task_idx}, 누적 delay={acc:.2f}s, duration={duration:.2f}s")

        best_idx, best_end = None, float("inf")
        option_lines = []
        for ch_idx, finish_base, _ in reachable:
            arrival = finish_base - duration + acc
            start = max(arrival, charger_next_free[ch_idx])
            end = start + duration
            wait = max(0.0, start - arrival)
            name = _charger_name(ch_idx)
            option_lines.append(
                (end, f"  {name}: arrival={arrival:8.2f}s, wait={wait:6.2f}s, start={start:8.2f}s, end={end:8.2f}s")
            )
            if end < best_end:
                best_end, best_idx = end, ch_idx

        # 후보들을 end 기준으로 정렬해서 보기 쉽게 출력
        option_lines.sort(key=lambda x: x[0])
        for _e, line in option_lines:
            print(line)

        if best_idx is not None:
            chosen_name = _charger_name(best_idx)
            print(f"  => 선택: {chosen_name} (가장 빨리 끝나는 end={best_end:.2f}s)")
            # Phase 2와 동일하게 상태 갱신 (다음 이벤트 계산을 위해)
            charger_next_free[best_idx] = best_end
            sim_adj = sim_end + acc
            delay = max(0.0, best_end - sim_adj)
            robot_total_delay[robot_id] = acc + delay

    print("=" * 100)


def _sequence_display(seq: List[int], charger_assignments: Optional[List[int]] = None) -> List[str]:
    """시퀀스를 출력용 리스트로. -1은 charger_assignments(ch_idx)에 따라 c1/c2. CHARGERS[0]=c1, [1]=c2."""
    out: List[str] = []
    charge_idx = 0
    for x in seq:
        if x == -1:
            if charger_assignments and charge_idx < len(charger_assignments):
                ch_idx = charger_assignments[charge_idx]
                # ch_idx 0 = CHARGERS[0] = charger_id 1 = c1, ch_idx 1 = c2
                cid = CHARGERS[ch_idx].charger_id if ch_idx < len(CHARGERS) else 0
                out.append(f"c{cid}")
            else:
                out.append("c?")
            charge_idx += 1
        else:
            out.append(str(x))
    return out


# =============================================================================
# 사용자 입력 파싱
# =============================================================================

DECODE_NO_CHARGE_TEXT = """
[decode-RIME] tb1 sequence=[4,27,15,2,25,24,35,6,22,48,5,30,20] (length=13, tasks=[DUAL_PICK_MULTI_DROP:6, MULTI_PICK_DROP:1, PICK_DROP:6], charges=0, makespan=815.12)
[decode-RIME] tb2 sequence=[21,46,14,41,38,50,11,39,45,1,37,16,13] (length=13, tasks=[DUAL_PICK_MULTI_DROP:1, MULTI_PICK_DROP:7, PICK_DROP:5], charges=0, makespan=850.47)
[decode-RIME] tb3 sequence=[28,42,31,9,47,44,29,23,12,18,49,17,26] (length=13, tasks=[DUAL_PICK_MULTI_DROP:5, MULTI_PICK_DROP:4, PICK_DROP:4], charges=0, makespan=823.66)
[decode-RIME] tb4 sequence=[32,19,34,33,43,8,7,3,10,40,36] (length=11, tasks=[DUAL_PICK_MULTI_DROP:3, MULTI_PICK_DROP:3, PICK_DROP:5], charges=0, makespan=811.50)



""".strip()

DECODE_OPTIMIZED_TEXT = """

[decode-RIME] tb1 sequence=[19,6,45,15,12,27,-1,37,40,38,-1,28,-1,24] (length=14, tasks=[DUAL_PICK_MULTI_DROP:3, MULTI_PICK_DROP:4, PICK_DROP:4], charges=3, makespan=1694.15)
[decode-RIME] tb2 sequence=[18,44,-1,10,32,20,50,-1,5,13,7,2,21,11,36] (length=15, tasks=[DUAL_PICK_MULTI_DROP:2, MULTI_PICK_DROP:3, PICK_DROP:8], charges=2, makespan=1366.81)
[decode-RIME] tb3 sequence=[33,4,-1,8,48,35,1,-1,30,16,17,3,-1,46,14,23,41] (length=17, tasks=[DUAL_PICK_MULTI_DROP:4, MULTI_PICK_DROP:3, PICK_DROP:7], charges=3, makespan=1776.70)
[decode-RIME] tb4 sequence=[49,39,34,-1,22,43,9,-1,26,31,42,29,47,25] (length=14, tasks=[DUAL_PICK_MULTI_DROP:6, MULTI_PICK_DROP:5, PICK_DROP:1], charges=2, makespan=1740.89)





""".strip()


# [decode-OPT] tb1 sequence=[35,17,-1,44,36,23,14,-1,22,38,-1,50,43,21,49,20]
# [decode-OPT] tb2 sequence=[10,32,5,4,39,-1,6,46,11,13,33,45,47]
# [decode-OPT] tb3 sequence=[19,3,-1,24,-1,27,25,48,9,-1,28,34,31,7,41]
# [decode-OPT] tb4 sequence=[8,15,42,37,-1,30,2,40,29,26,12,1,18,16]

# [decode-OPT] tb1 sequence=[35,17,c2,44,36,23,14,c2,22,38,c1,50,43,21,49,20] (length=16, charges=3, makespan=0.0)
# [decode-OPT] tb2 sequence=[10,32,5,4,39,c1,6,46,11,13,33,45,47] (length=13, charges=1, makespan=0.0)
# [decode-OPT] tb3 sequence=[19,3,c1,24,c1,27,25,48,9,c1,28,34,31,7,41] (length=15, charges=3, makespan=0.0)
# [decode-OPT] tb4 sequence=[8,15,42,37,c2,30,2,40,29,26,12,1,18,16] (length=14, charges=1, makespan=0.0)

#1821.69


def parse_decode_text(text: str) -> Tuple[Dict[str, List[int]], Dict[str, List[int]]]:
    """decode 텍스트에서 시퀀스와 (있다면) c1/c2 충전소 선택을 함께 파싱.

    반환:
      - seqs: {robot_id: [task_id1, -1, task_id2, ...]}
      - charger_choices: {robot_id: [ch_idx0, ch_idx1, ...]}  # -1(충전) 순서대로 CHARGERS 인덱스

    예:
      sequence=[6,38,1,c2,26,c1,15]
        -> seqs['tb1'] = [6,38,1,-1,26,-1,15]
        -> charger_choices['tb1'] = [idx(c2), idx(c1)]
    """
    seqs: Dict[str, List[int]] = {}
    charger_choices: Dict[str, List[int]] = {}

    # sequence=[...] 부분만 추출. [decode-XXX] 또는 [decode-OPT][RIME] 형식 모두 매칭 (뒤에 (length=...) 등 있어도 첫 ] 까지만)
    pattern = re.compile(r"\[decode-[^\]]+\](?:\[[^\]]+\])?\s+(\w+)\s+sequence=\[([^\]]+)\]")
    for line in text.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        m = pattern.search(line)
        if not m:
            continue
        rid = m.group(1)
        sequence_str = m.group(2).strip()
        seq: List[int] = []
        ch_list: List[int] = []

        for token in sequence_str.split(","):
            s = token.strip()
            if not s:
                continue
            # c1/c2 대소문자 구분 없이 인식
            s_lower = s.lower()
            if s_lower in ("c1", "c2"):
                target_cid = 1 if s_lower == "c1" else 2
                ch_idx = 0
                for i, ch in enumerate(CHARGERS):
                    if ch.charger_id == target_cid:
                        ch_idx = i
                        break
                seq.append(-1)
                ch_list.append(ch_idx)
            else:
                try:
                    seq.append(int(s))
                except ValueError:
                    continue

        if seq:
            seqs[rid] = seq
            charger_choices[rid] = ch_list

    return seqs, charger_choices


def parse_decode_makespan(text: str) -> Dict[str, float]:
    """decode 로그에 찍힌 makespan 값을 그대로 뽑아서 비교용으로 사용."""
    out: Dict[str, float] = {}
    pat = re.compile(r"\[decode-[^\]]+\]\s+(\w+).*?makespan=([0-9]+(?:\.[0-9]+)?)\)")
    for line in text.strip().splitlines():
        m = pat.search(line)
        if m:
            out[m.group(1)] = float(m.group(2))
    return out


def parse_decode_multi_algorithm(
    text: str,
) -> List[Tuple[str, Dict[str, List[int]], Dict[str, List[int]]]]:
    """decode 텍스트에서 알고리즘별 블록을 파싱. [decode-PSO], [decode-WCA], ..., [decode-OPT][RIME] 구분.

    반환: [(algo_name, robot_seqs, charger_choices), ...]
    algo_name: PSO, WCA, QSA, SMA, FLA, RIME (괄호 제목용)
    """
    # 라인 단위로 [decode-XXX] 또는 [decode-OPT][RIME] + rid + sequence=[...] 매칭
    line_pat = re.compile(
        r"^\s*\[decode-([^\]]+)\](?:\[([^\]]+)\])?\s+(\w+)\s+sequence=\[([^\]]+)\]"
    )
    blocks: List[Tuple[str, Dict[str, List[int]], Dict[str, List[int]]]] = []
    current_algo: Optional[str] = None
    current_seqs: Dict[str, List[int]] = {}
    current_ch: Dict[str, List[int]] = {}

    def flush_block() -> None:
        nonlocal current_algo, current_seqs, current_ch
        if current_algo and current_seqs:
            blocks.append((current_algo, dict(current_seqs), dict(current_ch)))
        current_algo = None
        current_seqs = {}
        current_ch = {}

    for line in text.strip().splitlines():
        line = line.strip()
        if not line:
            flush_block()
            continue
        m = line_pat.search(line)
        if not m:
            continue
        tag1, tag2, rid, seq_str = m.group(1), m.group(2), m.group(3), m.group(4)
        # [decode-OPT][RIME] -> RIME, 그 외 -> tag1 (PSO, WCA, ...)
        algo = (tag2.strip() if tag2 else tag1).strip().upper()
        seq: List[int] = []
        ch_list: List[int] = []
        for token in seq_str.split(","):
            s = token.strip()
            if not s:
                continue
            s_lower = s.lower()
            if s_lower in ("c1", "c2"):
                target_cid = 1 if s_lower == "c1" else 2
                ch_idx = next((i for i, ch in enumerate(CHARGERS) if ch.charger_id == target_cid), 0)
                seq.append(-1)
                ch_list.append(ch_idx)
            else:
                try:
                    seq.append(int(s))
                except ValueError:
                    continue
        if not seq:
            continue
        if current_algo is None:
            current_algo = algo
        elif current_algo != algo:
            flush_block()
            current_algo = algo
        current_seqs[rid] = seq
        current_ch[rid] = ch_list
    flush_block()
    return blocks


def build_task_sequence(task_ids: List[int], chargers: List[ChargerSpec]) -> List[TaskSpec]:
    seq: List[TaskSpec] = []
    for tid in task_ids:
        if tid == -1:
            ch = chargers[0]
            seq.append(TaskSpec(task_id=-1, task_type="CHARGE", picks=[(ch.x, ch.y)], drops=[(ch.x, ch.y)]))
        else:
            if tid in TASKS:
                seq.append(TASKS[tid])
    return seq


# =============================================================================
# 6개 알고리즘(PSO, WCA, QSA, SMA, FLA, RIME) 타임라인만 따로 저장
# =============================================================================

def run_single_algorithm(
    path_node: PathQueryNode,
    algo_name: str,
    robot_seqs: Dict[str, List[int]],
    charger_choices: Dict[str, List[int]],
) -> Tuple[List[SimResult], List[Tuple[str, int, int, float, float, float]], Dict[str, List[float]]]:
    """단일 알고리즘에 대해 Phase1 + Phase2 수행 후 (results, timeline_events, phase1_charge_ends) 반환."""
    events_all: List[Tuple[str, int, List[Tuple[int, float, float]], float, float, float]] = []
    phase1_results: Dict[str, SimResult] = {}
    robot_ids = sorted(robot_seqs.keys())

    for rid in robot_ids:
        cap_wh = ROBOT_CAP_WH.get(rid, 60.0)
        init_pos = ROBOT_INITIAL_POSITIONS.get(rid, (0.0, 0.0))
        task_ids = robot_seqs.get(rid, [])
        seq = build_task_sequence(task_ids, CHARGERS)
        ch_choices = charger_choices.get(rid)
        res, evs = simulate_robot(
            robot_id=rid,
            seq=seq,
            cap_wh=cap_wh,
            init_pos=init_pos,
            chargers=CHARGERS,
            path_node=path_node,
            strategy=algo_name,
            insert_charge=False,
            charger_choices=ch_choices,
        )
        phase1_results[rid] = res
        events_all.extend(evs)

    delay_map, wait_map, event_charger_list, timeline_events = phase2_charger_contention(
        events_all, len(CHARGERS)
    )
    phase1_charge_ends: Dict[str, List[float]] = {}
    for rid in robot_ids:
        events_rid = [
            (task_idx, sim_end)
            for (robot_id, task_idx, _, _, sim_end, _) in events_all
            if robot_id == rid
        ]
        events_rid.sort(key=lambda x: x[0])
        phase1_charge_ends[rid] = [sim_end for _, sim_end in events_rid]

    per_robot_charger: Dict[str, List[int]] = {}
    for rid, task_idx, ch_idx in event_charger_list:
        per_robot_charger.setdefault(rid, []).append((task_idx, ch_idx))
    for rid in per_robot_charger:
        per_robot_charger[rid].sort(key=lambda x: x[0])
        per_robot_charger[rid] = [ch_idx for _, ch_idx in per_robot_charger[rid]]

    per_robot_charge_events: Dict[str, List[Tuple[int, float, float, float, int]]] = {}
    for rid, task_idx, ch_idx, arrival, start, end in timeline_events:
        cid = CHARGERS[ch_idx].charger_id if ch_idx < len(CHARGERS) else (ch_idx + 1)
        per_robot_charge_events.setdefault(rid, []).append(
            (int(task_idx), float(arrival), float(start), float(end), int(cid))
        )
    for rid in per_robot_charge_events:
        per_robot_charge_events[rid].sort(key=lambda x: x[0])

    results: List[SimResult] = []
    for rid in phase1_results:
        res = phase1_results[rid]
        d = delay_map.get(rid, 0.0)
        w = wait_map.get(rid, 0.0)
        charger_assignments = per_robot_charger.get(rid, [])
        phase1_tl = PHASE1_TASK_TIMELINE.get(algo_name, {}).get(rid, [])
        charge_ev = per_robot_charge_events.get(rid, [])
        phase1_ends = phase1_charge_ends.get(rid, [])
        travel_total = _compute_travel_total(
            rid, algo_name, res.final_sequence, phase1_tl, charge_ev, phase1_ends,
        )
        results.append(
            SimResult(
                robot_id=rid,
                strategy=algo_name,
                makespan=res.makespan + d,
                energy_used=res.energy_used,
                charge_count=res.charge_count,
                waiting_total=w,
                charging_total=res.charging_total,
                discharge_count=res.discharge_count,
                final_sequence=res.final_sequence,
                charger_assignments=charger_assignments or None,
                travel_total=travel_total,
            )
        )
    return results, timeline_events, phase1_charge_ends


def run_multi_algorithm_timelines(
    path_node: PathQueryNode,
    text: str,
    out_dir: str = ".",
    out_filename: str = "timeline_6algos.png",
) -> None:
    """DECODE_OPTIMIZED_TEXT 등에서 6개 알고리즘(PSO, WCA, QSA, SMA, FLA, RIME)을 파싱해
    하나의 figure에 6개 타임라인을 모아 통합 PNG로 저장. 제목: Task Timeline (PSO), ...
    """
    out_dir = os.path.abspath(_resolve_output_path(out_dir))
    os.makedirs(out_dir, exist_ok=True)
    blocks = parse_decode_multi_algorithm(text)
    if not blocks:
        print("[WARN] multi-algo: no algorithm blocks parsed")
        return
    all_results: List[SimResult] = []
    timeline_events_by_strategy: Dict[str, List[Tuple[str, int, int, float, float, float]]] = {}
    phase1_charge_ends_by_strategy: Dict[str, Dict[str, List[float]]] = {}
    for algo_name, robot_seqs, charger_choices in blocks:
        if not robot_seqs:
            continue
        print(f"[multi-algo] Running {algo_name} ...")
        results, timeline_events, phase1_charge_ends = run_single_algorithm(
            path_node, algo_name, robot_seqs, charger_choices,
        )
        all_results.extend(results)
        timeline_events_by_strategy[algo_name] = timeline_events
        phase1_charge_ends_by_strategy[algo_name] = phase1_charge_ends
    if not all_results:
        print("[WARN] multi-algo: no results to plot")
        return
    # 6개 알고리즘 결과 로그 출력 및 CSV 저장 (일반 실행과 동일)
    print_results(all_results)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_results(all_results, os.path.join(out_dir, f"charging_analysis_6algos_{timestamp}.csv"))
    out_path = os.path.join(out_dir, out_filename)
    generate_internal_timeline_png(
        results=all_results,
        path_node=path_node,
        timeline_events_by_strategy=timeline_events_by_strategy,
        phase1_charge_ends_by_strategy=phase1_charge_ends_by_strategy,
        out_path=out_path,
    )
    print(f"[multi-algo] Done. {len(blocks)} algorithms → 1 file: {out_path}")


# =============================================================================
# 메인: 3가지 전략 비교
# =============================================================================

def run_analysis(
    path_node: PathQueryNode,
    use_global_phase2: bool = False,
    phase2_method: str = "beam",
    phase2_beam_width: int = 10,
) -> Tuple[List[SimResult], Dict[str, List[Tuple[str, int, int, float, float, float]]], Dict[str, Dict[str, List[float]]]]:
    """use_global_phase2=True면 optimized 전략에서 전역 메이크스팬 최소화로 충전소 배정 탐색.
    phase2_method: 'greedy' | 'beam' | 'exhaustive', phase2_beam_width: beam 탐색 시 상위 유지 개수.
    세 번째 반환: phase1_charge_ends_by_strategy[strat][rid] = 해당 로봇의 Phase1 충전 종료 시각(sim_end) 리스트 (충전 순서)."""
    no_charge_seqs, _ = parse_decode_text(DECODE_NO_CHARGE_TEXT)
    optimized_seqs, optimized_charger_choices = parse_decode_text(DECODE_OPTIMIZED_TEXT)
    optimized_logged_ms = parse_decode_makespan(DECODE_OPTIMIZED_TEXT)

    strategies = [ "threshold", "feasibility", "optimized"]
    all_results: List[SimResult] = []
    timeline_events_by_strategy: Dict[str, List[Tuple[str, int, int, float, float, float]]] = {}
    phase1_charge_ends_by_strategy: Dict[str, Dict[str, List[float]]] = {}

    for strat in strategies:
        events_all: List[Tuple[str, int, List[Tuple[int, float, float]], float, float, float]] = []
        phase1_results: Dict[str, SimResult] = {}

        for rid in sorted(no_charge_seqs.keys()):
            cap_wh = ROBOT_CAP_WH.get(rid, 60.0)
            init_pos = ROBOT_INITIAL_POSITIONS.get(rid, (0.0, 0.0))

            if strat == "optimized":
                task_ids = optimized_seqs.get(rid, [])
                seq = build_task_sequence(task_ids, CHARGERS)
                insert = False
                charger_choices = optimized_charger_choices.get(rid)
            else:
                task_ids = [t for t in no_charge_seqs.get(rid, []) if t != -1]
                seq = build_task_sequence(task_ids, CHARGERS)
                insert = True
                charger_choices = None

            res, evs = simulate_robot(
                robot_id=rid, seq=seq, cap_wh=cap_wh, init_pos=init_pos,
                chargers=CHARGERS, path_node=path_node,
                strategy=strat, insert_charge=insert, charger_choices=charger_choices,
            )
            phase1_results[rid] = res
            events_all.extend(evs)

        if strat == "optimized" and use_global_phase2:
            g_delay, g_wait, g_event_charger, _ = phase2_charger_contention(events_all, len(CHARGERS))
            delay_map, wait_map, event_charger_list, timeline_events = phase2_charger_contention_global(
                events_all, len(CHARGERS), phase1_results,
                method=phase2_method, beam_width=phase2_beam_width,
            )
            g_ms = _global_makespan(phase1_results, g_delay)
            opt_ms = _global_makespan(phase1_results, delay_map)
            print(f"\n[Phase2 전역 최적화] method={phase2_method}, beam_width={phase2_beam_width}")
            print(f"  그리디 전역 makespan={g_ms:.2f}s  →  전역 최적={opt_ms:.2f}s  (차이 {opt_ms - g_ms:+.2f}s)")
            print_phase2_greedy_vs_global(
                phase1_results,
                g_delay, g_wait, g_event_charger,
                delay_map, wait_map, event_charger_list,
            )
        else:
            delay_map, wait_map, event_charger_list, timeline_events = phase2_charger_contention(events_all, len(CHARGERS))

        if strat == "optimized":
            timeline_events_by_strategy[strat] = timeline_events
            debug_print_charger_decisions(events_all, len(CHARGERS))
        else:
            timeline_events_by_strategy[strat] = timeline_events

        # Phase1 충전 종료 시각(sim_end) 로봇별·충전 순서대로 저장 (Phase2 delay 반영 타임라인 구성용)
        phase1_charge_ends_by_strategy[strat] = {}
        for rid in phase1_results:
            events_rid = [
                (task_idx, sim_end)
                for (robot_id, task_idx, _, _, sim_end, _) in events_all
                if robot_id == rid
            ]
            events_rid.sort(key=lambda x: x[0])
            phase1_charge_ends_by_strategy[strat][rid] = [sim_end for _, sim_end in events_rid]

        # (robot_id, task_idx, ch_idx) -> 로봇별 충전 이벤트 순서대로 ch_idx 리스트
        per_robot_charger: Dict[str, List[int]] = {}
        for rid, task_idx, ch_idx in event_charger_list:
            per_robot_charger.setdefault(rid, []).append((task_idx, ch_idx))
        for rid in per_robot_charger:
            per_robot_charger[rid].sort(key=lambda x: x[0])
            per_robot_charger[rid] = [ch_idx for _, ch_idx in per_robot_charger[rid]]

        # 로봇별 Phase2 충전 이벤트 (task_idx, arrival, start, end, cid) - travel_total 계산용
        per_robot_charge_events: Dict[str, List[Tuple[int, float, float, float, int]]] = {}
        for rid, task_idx, ch_idx, arrival, start, end in timeline_events:
            cid = CHARGERS[ch_idx].charger_id if ch_idx < len(CHARGERS) else (ch_idx + 1)
            per_robot_charge_events.setdefault(rid, []).append(
                (int(task_idx), float(arrival), float(start), float(end), int(cid))
            )
        for rid in per_robot_charge_events:
            per_robot_charge_events[rid].sort(key=lambda x: x[0])

        for rid, res in phase1_results.items():
            d = delay_map.get(rid, 0.0)
            w = wait_map.get(rid, 0.0)
            charger_assignments = per_robot_charger.get(rid, [])
            phase1_tl = PHASE1_TASK_TIMELINE.get(strat, {}).get(rid, [])
            charge_ev = per_robot_charge_events.get(rid, [])
            phase1_ends = phase1_charge_ends_by_strategy.get(strat, {}).get(rid, [])
            travel_total = _compute_travel_total(
                rid, strat, res.final_sequence, phase1_tl, charge_ev, phase1_ends,
            )
            final = SimResult(
                robot_id=rid, strategy=strat,
                makespan=res.makespan + d,
                energy_used=res.energy_used,
                charge_count=res.charge_count,
                waiting_total=w,
                charging_total=res.charging_total,
                discharge_count=res.discharge_count,
                final_sequence=res.final_sequence,
                charger_assignments=charger_assignments if charger_assignments else None,
                travel_total=travel_total,
            )
            all_results.append(final)

    # 디버그/검증: optimized의 로그 makespan과 재계산 값을 비교 출력
    if optimized_logged_ms:
        by_robot = {r.robot_id: r for r in all_results if r.strategy == "optimized"}
        print("\n[check] optimized makespan (logged vs recomputed):")
        for rid in sorted(optimized_logged_ms.keys()):
            logged = optimized_logged_ms[rid]
            recomputed = by_robot.get(rid).makespan if rid in by_robot else None
            if recomputed is None:
                print(f"  {rid}: logged={logged:.2f}s, recomputed=NA (no result)")
            else:
                diff = recomputed - logged
                wait = by_robot[rid].waiting_total if rid in by_robot else 0.0
                print(f"  {rid}: logged={logged:.2f}s, recomputed={recomputed:.2f}s, diff={diff:+.2f}s  (Phase2 대기={wait:.2f}s)")
        print("  ※ diff 원인 후보: 1) 충전 속도(60Wh=600초 vs decode는 더 빠른 모델) 2) Phase2 충전소 대기")

    return all_results, timeline_events_by_strategy, phase1_charge_ends_by_strategy


def _charger_name(ch_idx: int) -> str:
    """ch_idx -> c1/c2 표시 (charger_id 기준)."""
    if ch_idx < len(CHARGERS):
        return f"c{CHARGERS[ch_idx].charger_id}"
    return f"ch{ch_idx}"


def _event_charger_to_per_robot(
    event_charger_list: List[Tuple[str, int, int]],
) -> Dict[str, List[int]]:
    """(robot_id, task_idx, ch_idx) 리스트 → 로봇별 ch_idx 리스트 (충전 순서)."""
    per: Dict[str, List[Tuple[int, int]]] = {}
    for rid, task_idx, ch_idx in event_charger_list:
        per.setdefault(rid, []).append((task_idx, ch_idx))
    for rid in per:
        per[rid].sort(key=lambda x: x[0])
        per[rid] = [ch_idx for _, ch_idx in per[rid]]
    return per


def _compute_travel_total(
    rid: str,
    strat: str,
    final_sequence: List[int],
    phase1_timeline: List[Tuple[str, Optional[int], float, float]],
    charge_events_rid: List[Tuple[int, float, float, float, int]],
    phase1_ends: List[float],
) -> float:
    """Phase1 이동(move) 합 + Phase2 충전소 이동 합을 반환."""
    travel = 0.0
    for kind, _tid, s, e in phase1_timeline:
        if kind == "move":
            travel += max(0.0, e - s)
    task_iter = iter(phase1_timeline)
    charge_idx = 0
    current_delay = 0.0
    current_end = 0.0
    for token in final_sequence:
        if token == -1:
            if charge_idx >= len(charge_events_rid):
                continue
            _ti, arrival, start, end, _cid = charge_events_rid[charge_idx]
            move_dur = max(0.0, arrival - current_end)
            travel += move_dur
            current_end = end
            if charge_idx < len(phase1_ends):
                current_delay = end - phase1_ends[charge_idx]
            charge_idx += 1
        else:
            try:
                next(task_iter)  # move
                _, _, _s2, e2 = next(task_iter)  # task
            except StopIteration:
                continue
            current_end = e2 + current_delay
    return travel


def print_phase2_greedy_vs_global(
    phase1_results: Dict[str, SimResult],
    greedy_delay: Dict[str, float],
    greedy_wait: Dict[str, float],
    greedy_event_charger: List[Tuple[str, int, int]],
    opt_delay: Dict[str, float],
    opt_wait: Dict[str, float],
    opt_event_charger: List[Tuple[str, int, int]],
) -> None:
    """로봇 시점 그리디 vs 전역 최적화 Phase2 결과를 나란히 비교 출력."""
    greedy_per = _event_charger_to_per_robot(greedy_event_charger)
    opt_per = _event_charger_to_per_robot(opt_event_charger)
    robots = sorted(phase1_results.keys())

    print("\n" + "=" * 100)
    print("Phase2 비교: 로봇 시점 그리디 vs 전역 최적화")
    print("=" * 100)
    print(f"{'로봇':6} | {'makespan(그리디)':>14} {'makespan(전역)':>14} {'차이':>8} | {'대기(그리디)':>10} {'대기(전역)':>10} | 충전소 배정")
    print("-" * 100)

    for rid in robots:
        res = phase1_results[rid]
        ms_g = res.makespan + greedy_delay.get(rid, 0.0)
        ms_o = res.makespan + opt_delay.get(rid, 0.0)
        diff = ms_o - ms_g
        w_g = greedy_wait.get(rid, 0.0)
        w_o = opt_wait.get(rid, 0.0)
        ch_g = greedy_per.get(rid, [])
        ch_o = opt_per.get(rid, [])
        ch_g_str = ",".join(_charger_name(c) for c in ch_g) if ch_g else "-"
        ch_o_str = ",".join(_charger_name(c) for c in ch_o) if ch_o else "-"
        print(f"{rid:6} | {ms_g:14.2f} {ms_o:14.2f} {diff:+8.2f} | {w_g:10.2f} {w_o:10.2f} | 그리디:[{ch_g_str}]  전역:[{ch_o_str}]")

    g_global = _global_makespan(phase1_results, greedy_delay)
    o_global = _global_makespan(phase1_results, opt_delay)
    print("-" * 100)
    print(f"{'전역':6} | {g_global:14.2f} {o_global:14.2f} {o_global - g_global:+8.2f} | (전역 makespan = max(로봇별 makespan))")
    print("=" * 100)


def print_timeline(timeline_events: List[Tuple[str, int, int, float, float, float]]) -> None:
    """optimized 전략의 충전 타임라인: 충전소별·로봇별 도착/시작/종료 시각 출력 (c1 vs c2 비교용)."""
    if not timeline_events:
        return
    print("\n" + "=" * 100)
    print("Timeline (optimized) — 충전소별 / 로봇별 (도착 → 시작 → 종료 [초], c1 vs c2 비교용)")
    print("=" * 100)

    # 충전소별: ch_idx -> [(robot_id, arrival, start, end), ...] (start 순 정렬)
    by_charger: Dict[int, List[Tuple[str, float, float, float]]] = {}
    for robot_id, _task_idx, ch_idx, arrival, start, end in timeline_events:
        by_charger.setdefault(ch_idx, []).append((robot_id, arrival, start, end))
    for ch_idx in sorted(by_charger.keys()):
        by_charger[ch_idx].sort(key=lambda x: x[2])

    print("\n[충전소별]")
    for ch_idx in sorted(by_charger.keys()):
        name = _charger_name(ch_idx)
        print(f"  {name}:")
        for robot_id, arrival, start, end in by_charger[ch_idx]:
            wait = max(0.0, start - arrival)
            print(f"    [{start:7.1f} ~ {end:7.1f}] {robot_id}  (도착 {arrival:7.1f}, 대기 {wait:.1f}s)")

    # 로봇별: robot_id -> [(charger_name, arrival, start, end), ...] (start 순 정렬)
    by_robot: Dict[str, List[Tuple[str, float, float, float]]] = {}
    for robot_id, _task_idx, ch_idx, arrival, start, end in timeline_events:
        by_robot.setdefault(robot_id, []).append((_charger_name(ch_idx), arrival, start, end))
    for rid in by_robot:
        by_robot[rid].sort(key=lambda x: x[2])

    print("\n[로봇별]")
    for rid in sorted(by_robot.keys()):
        print(f"  {rid}:")
        for cname, arrival, start, end in by_robot[rid]:
            wait = max(0.0, start - arrival)
            print(f"    {cname}  도착 {arrival:7.1f} → 시작 {start:7.1f} → 종료 {end:7.1f}  (대기 {wait:.1f}s)")
    print("=" * 100)


def print_results(results: List[SimResult]) -> None:
    print("=" * 100)
    print("Offline Charging Strategy Analysis (Phase 1 + Phase 2 charger contention)")
    print("=" * 100)

    by_robot: Dict[str, Dict[str, SimResult]] = {}
    for r in results:
        by_robot.setdefault(r.robot_id, {})[r.strategy] = r

    for rid in sorted(by_robot.keys()):
        print(f"\n[Robot {rid}]")
        for strat in ["threshold", "feasibility", "optimized"]:
            if strat not in by_robot[rid]:
                continue
            r = by_robot[rid][strat]
            print(f"  {strat:12s}: makespan={r.makespan:8.2f}s, "
                  f"energy={r.energy_used:6.2f}Wh, "
                  f"charges={r.charge_count}, "
                  f"travel={r.travel_total:6.2f}s, "
                  f"waiting={r.waiting_total:6.2f}s, "
                  f"charging={r.charging_total:6.2f}s, "
                  f"discharge={r.discharge_count}")
            disp = _sequence_display(r.final_sequence, r.charger_assignments)
            print(f"                sequence: [{', '.join(disp)}]")

    print("\n" + "=" * 100)
    print("Summary (max makespan per strategy):")
    by_strat: Dict[str, List[SimResult]] = {}
    for r in results:
        by_strat.setdefault(r.strategy, []).append(r)
    for strat in ["threshold", "feasibility", "optimized"]:
        if strat not in by_strat:
            continue
        mx = max(r.makespan for r in by_strat[strat])
        print(f"  {strat:12s}: max_makespan={mx:8.2f}s")


def save_results(results: List[SimResult], filename: str) -> None:
    os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
    with open(filename, "w") as f:
        f.write("robot_id,strategy,makespan,energy_used,charge_count,travel_total,waiting_total,charging_total,discharge_count,sequence\n")
        for r in results:
            disp = _sequence_display(r.final_sequence, r.charger_assignments)
            seq_str = ";".join(disp)
            f.write(f"{r.robot_id},{r.strategy},{r.makespan:.2f},{r.energy_used:.2f},"
                    f"{r.charge_count},{r.travel_total:.2f},{r.waiting_total:.2f},{r.charging_total:.2f},{r.discharge_count},{seq_str}\n")
    print(f"\nResults saved to: {filename}")


def _charge_durations_from_timeline(
    timeline_events: Optional[List[Tuple[str, int, int, float, float, float]]],
) -> Dict[str, List[float]]:
    if not timeline_events:
        return {}
    per_robot: Dict[str, List[Tuple[int, float]]] = {}
    for rid, task_idx, _ch_idx, _arrival, start, end in timeline_events:
        per_robot.setdefault(rid, []).append((int(task_idx), max(0.0, float(end) - float(start))))
    ordered: Dict[str, List[float]] = {}
    for rid, entries in per_robot.items():
        entries.sort(key=lambda x: x[0])
        ordered[rid] = [dur for _, dur in entries]
    return ordered


def _sequence_specs_from_result(res: SimResult, charge_durations: List[float]) -> List[TaskSpec]:
    specs: List[TaskSpec] = []
    if not CHARGERS:
        return specs
    assignments = res.charger_assignments or []
    charge_idx = 0
    for token in res.final_sequence:
        if token == -1:
            ch_idx = assignments[charge_idx] if charge_idx < len(assignments) else 0
            ch_idx = max(0, min(ch_idx, len(CHARGERS) - 1))
            charger = CHARGERS[ch_idx]
            duration = charge_durations[charge_idx] if charge_idx < len(charge_durations) else 0.0
            specs.append(
                TaskSpec(
                    task_id=-1,
                    task_type="CHARGE",
                    picks=[(charger.x, charger.y)],
                    drops=[(charger.x, charger.y)],
                    pick_wait_s=0.0,
                    drop_wait_s=0.0,
                    charge_duration_s=float(duration),
                )
            )
            charge_idx += 1
        else:
            task = TASKS.get(token)
            if task:
                specs.append(task)
    return specs


def _format_task_spec_line(task: TaskSpec) -> str:
    return (
        "TaskSpec("
        f"task_id={task.task_id}, "
        f"task_type=\"{task.task_type}\", "
        f"picks={task.picks}, "
        f"drops={task.drops}, "
        f"pick_wait_s={task.pick_wait_s}, "
        f"drop_wait_s={task.drop_wait_s}, "
        f"charge_duration_s={task.charge_duration_s}"
        ")"
    )


def _print_time_sequence_dict(sequences: Dict[str, List[TaskSpec]]) -> None:
    print("\ntime_sequence = {")
    for rid in sorted(sequences.keys()):
        print(f'  "{rid}": [')
        for task in sequences[rid]:
            print(f"    {_format_task_spec_line(task)},")
        print("  ],")
    print("}")


def _write_time_sequence_csv(
    sequences: Dict[str, List[TaskSpec]],
    filename: str,
    strategy: str,
) -> None:
    os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "robot_id",
            "strategy",
            "order",
            "task_id",
            "task_type",
            "picks",
            "drops",
            "pick_wait_s",
            "drop_wait_s",
            "charge_duration_s",
        ])
        for rid in sorted(sequences.keys()):
            for order_idx, task in enumerate(sequences[rid], start=1):
                writer.writerow([
                    rid,
                    strategy,
                    order_idx,
                    task.task_id,
                    task.task_type,
                    repr(task.picks),
                    repr(task.drops),
                    f"{task.pick_wait_s:.2f}",
                    f"{task.drop_wait_s:.2f}",
                    f"{task.charge_duration_s:.2f}",
                ])
    print(f"[Saved] time sequence csv: {filename}")


def export_time_sequences(
    results: List[SimResult],
    filename: str,
    strategy: str = "optimized",
    timeline_events: Optional[List[Tuple[str, int, int, float, float, float]]] = None,
) -> None:
    sequences: Dict[str, List[TaskSpec]] = {}
    charge_duration_map = _charge_durations_from_timeline(timeline_events)
    for res in results:
        if res.strategy != strategy:
            continue
        durations = charge_duration_map.get(res.robot_id, [])
        specs = _sequence_specs_from_result(res, durations)
        if specs:
            sequences[res.robot_id] = specs
    if not sequences:
        print(f"[WARN] time sequence export skipped: no sequences for strategy '{strategy}'")
        return
    _print_time_sequence_dict(sequences)
    _write_time_sequence_csv(sequences, filename, strategy)


def _ensure_scheduler_importable() -> None:
    """ros2_ws 루트에서 실행 시에도 `from scheduler.temp import ...`가 되도록 sys.path 보정."""
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    cand = os.path.join(here, "src", "scheduler")
    if os.path.isdir(cand) and cand not in sys.path:
        sys.path.insert(0, cand)


def _results_to_temp_cases(results: List[SimResult]) -> List[Dict[str, object]]:
    """SimResult 리스트를 temp.py의 cases 형식으로 변환 (색/항목/스타일 동일 시각화용)."""
    # strategy -> robot_id -> result
    by_strat: Dict[str, Dict[str, SimResult]] = {}
    for r in results:
        by_strat.setdefault(r.strategy, {})[r.robot_id] = r

    cases: List[Dict[str, object]] = []
    for strat in ["threshold", "feasibility", "optimized"]:
        if strat not in by_strat:
            continue
        robot_sequences: Dict[str, List[object]] = {}
        target_makespan: Dict[str, float] = {}
        for rid, r in by_strat[strat].items():
            # final_sequence: -1(충전) -> charger_assignments 기반으로 "c1"/"c2"로 변환
            seq_tokens: List[object] = []
            charge_i = 0
            assigns = r.charger_assignments or []
            for tid in r.final_sequence:
                if tid == -1:
                    ch_idx = assigns[charge_i] if charge_i < len(assigns) else 0
                    cid = CHARGERS[ch_idx].charger_id if ch_idx < len(CHARGERS) else 1
                    seq_tokens.append(f"c{cid}")
                    charge_i += 1
                else:
                    seq_tokens.append(int(tid))
            robot_sequences[rid] = seq_tokens
            target_makespan[rid] = float(r.makespan)

        cases.append({
            "algo_name": strat,
            "robot_sequences": robot_sequences,
            "target_makespan": target_makespan,
        })
    return cases


def generate_timeline_png_from_results(
    results: List[SimResult],
    task_log_path: str,
    out_path: str,
) -> None:
    """temp.py 시각화 엔진을 사용해 threshold/feasibility/optimized 타임라인 PNG 생성."""
    _ensure_scheduler_importable()
    try:
        from scheduler.temp import parse_simulation_log, make_6_timelines_figure  # type: ignore
    except Exception as e:
        print(f"[WARN] timeline skipped: cannot import temp.py plotter: {e}")
        return

    try:
        tasks = parse_simulation_log(task_log_path)
    except Exception as e:
        print(f"[WARN] timeline skipped: cannot read task log '{task_log_path}': {e}")
        return

    cases = _results_to_temp_cases(results)
    if not cases:
        print("[WARN] timeline skipped: no cases to render")
        return

    # temp.py의 함수 이름은 6이지만 n개 케이스도 처리 가능 (n=len(cases))
    make_6_timelines_figure(tasks, cases, save_path=out_path)
    print(f"[Saved] timeline: {out_path}")


def generate_internal_timeline_png(
    results: List[SimResult],
    path_node: PathQueryNode,
    timeline_events_by_strategy: Dict[str, List[Tuple[str, int, int, float, float, float]]],
    phase1_charge_ends_by_strategy: Optional[Dict[str, Dict[str, List[float]]]] = None,
    out_path: str = "",
) -> None:
    """외부 task log 없이, Phase1 task 타임라인 + Phase2 충전 이벤트(delay 반영)로 temp.py 스타일 타임라인 PNG 생성."""
    if not out_path:
        out_path = "timeline_offline.png"
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[WARN] internal timeline skipped: import failed: {e}")
        return

    # strategy -> {robot_id: SimResult}
    by_strat: Dict[str, Dict[str, SimResult]] = {}
    for r in results:
        by_strat.setdefault(r.strategy, {})[r.robot_id] = r

    # 시각화 순서: threshold/feasibility/optimized, 그 다음 6개 알고리즘은 PSO→WCA→QSA→SMA→FLA→RIME
    fixed_order = ["threshold", "feasibility", "optimized", "PSO", "WCA", "QSA", "SMA", "FLA", "RIME"]
    strategies = [s for s in fixed_order if s in by_strat]
    for s in sorted(by_strat.keys()):
        if s not in strategies:
            strategies.append(s)
    if not strategies:
        print("[WARN] internal timeline skipped: no strategies in results")
        return

    # temp.py 스타일 legend proxy
    def _proxy_patches():
        return {
            "Travel": plt.Rectangle((0, 0), 1, 1, facecolor=COLOR_MAP["move"], edgecolor="k", linewidth=1),
            "One-way": plt.Rectangle((0, 0), 1, 1, facecolor=COLOR_MAP["one_way"], edgecolor="k", linewidth=1),
            "Round-trip": plt.Rectangle((0, 0), 1, 1, facecolor=COLOR_MAP["round_trip"], edgecolor="k", linewidth=1),
            "Via-point": plt.Rectangle((0, 0), 1, 1, facecolor=COLOR_MAP["via_point"], edgecolor="k", linewidth=1),
            "Charge": plt.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor="k", hatch="//", linewidth=1),
            "Charge wait": plt.Rectangle((0, 0), 1, 1, facecolor="black", edgecolor="k", linewidth=1),
        }

    # (charger_id -> ChargerSpec)
    chargers_by_id: Dict[int, ChargerSpec] = {c.charger_id: c for c in CHARGERS}

    def _segments_for_strategy(strat: str) -> Tuple[Dict[str, List[Dict[str, object]]], Dict[str, float]]:
        """
        Phase1 task 타임라인(PHASE1_TASK_TIMELINE) + Phase2 충전 이벤트(timeline_events)를 사용해
        delay 반영 최종 세그먼트를 구성한다. task는 기록된 시작/종료 시각에 cumulative_delay를 더해 그린다.
        """
        robots = sorted(by_strat[strat].keys())
        segments_by_robot: Dict[str, List[Dict[str, object]]] = {rid: [] for rid in robots}
        makespan_by_robot: Dict[str, float] = {}

        tl_events = timeline_events_by_strategy.get(strat, [])
        # 로봇별 Phase2 충전 이벤트 (task_idx 순서 = final_sequence 내 충전 순서)
        per_robot_charge_events: Dict[str, List[Tuple[int, float, float, float, int]]] = {}
        for rid, task_idx, ch_idx, arrival, start, end in tl_events:
            cid = CHARGERS[ch_idx].charger_id if ch_idx < len(CHARGERS) else (ch_idx + 1)
            per_robot_charge_events.setdefault(rid, []).append(
                (int(task_idx), float(arrival), float(start), float(end), int(cid))
            )
        for rid in per_robot_charge_events:
            per_robot_charge_events[rid].sort(key=lambda x: x[0])

        phase1_ends_map = (phase1_charge_ends_by_strategy or {}).get(strat, {})
        phase1_task_timeline = PHASE1_TASK_TIMELINE.get(strat, {})

        for rid in robots:
            r = by_strat[strat][rid]
            task_list = phase1_task_timeline.get(rid, [])  # ("move"/"task", task_id, start, end) 시퀀스 순
            charge_events_rid = per_robot_charge_events.get(rid, [])
            phase1_ends = phase1_ends_map.get(rid, [])

            segs: List[Dict[str, object]] = []
            task_iter = iter(task_list)
            charge_idx = 0
            current_delay = 0.0
            last_end = 0.0

            for token in r.final_sequence:
                if token == -1:
                    # 충전: Phase2 (arrival, start, end) 사용
                    if charge_idx >= len(charge_events_rid):
                        continue
                    _ti, arrival, start, end, cid = charge_events_rid[charge_idx]
                    if last_end < arrival - 1e-9:
                        segs.append({
                            "start": last_end,
                            "dur": arrival - last_end,
                            "kind": "move",
                        })
                    wait_dur = max(0.0, start - arrival)
                    if wait_dur > 1e-9:
                        segs.append({
                            "start": arrival,
                            "dur": wait_dur,
                            "kind": "charge_wait",
                            "charger_id": cid,
                        })
                    charge_dur = max(0.0, end - start)
                    if charge_dur > 1e-9:
                        segs.append({
                            "start": start,
                            "dur": charge_dur,
                            "kind": "charge",
                            "charger_id": cid,
                        })
                    if charge_idx < len(phase1_ends):
                        current_delay = end - phase1_ends[charge_idx]
                    last_end = end
                    charge_idx += 1
                else:
                    # task: Phase1에서 "move" + "task" 두 세그먼트를 순서대로 꺼내 delay 반영
                    try:
                        kind1, _tid1, s1, e1 = next(task_iter)
                        kind2, task_id, s2, e2 = next(task_iter)
                    except StopIteration:
                        continue
                    if kind1 == "move" and e1 - s1 > 1e-9:
                        segs.append({
                            "start": s1 + current_delay,
                            "dur": e1 - s1,
                            "kind": "move",
                        })
                    if kind2 == "task" and task_id is not None:
                        start_p2 = s2 + current_delay
                        end_p2 = e2 + current_delay
                        task_spec = TASKS.get(int(task_id))
                        color_key = TASKTYPE_TO_COLORKEY.get(task_spec.task_type, "one_way") if task_spec else "one_way"
                        segs.append({
                            "start": start_p2,
                            "dur": end_p2 - start_p2,
                            "kind": "task",
                            "type": color_key,
                            "task_id": int(task_id),
                        })
                        last_end = end_p2

            segments_by_robot[rid] = segs
            makespan_by_robot[rid] = float(r.makespan)

        return segments_by_robot, makespan_by_robot

    # ---- figure ----
    n = len(strategies)
    fig, axes = plt.subplots(n, 1, figsize=(20, 3.2 * n), sharex=False)
    if n == 1:
        axes = [axes]  # type: ignore

    # temp.py에서 algo_name 위치에 들어갈 표시 이름 매핑
    STRAT_DISPLAY_NAME = {
        "threshold": "Threshold",
        "feasibility": "Feasibility",
        "optimized": "RIME",  # 요청: optimized 대신 RIME 표기
    }

    max_spans: List[float] = []
    per_case_data: List[Tuple[str, Dict[str, List[Dict[str, object]]], Dict[str, float]]] = []
    for strat, ax in zip(strategies, axes):
        segs_by_robot, ms_by_robot = _segments_for_strategy(strat)
        per_case_data.append((strat, segs_by_robot, ms_by_robot))
        max_spans.append(max(ms_by_robot.values()) if ms_by_robot else 0.0)

        robots = sorted(segs_by_robot.keys())
        max_makespan = max(ms_by_robot.values()) if ms_by_robot else None
        max_robots = [r for r in robots if max_makespan is not None and abs(ms_by_robot.get(r, 0.0) - max_makespan) < 1e-6]
        for ridx, rid in enumerate(robots):
            for seg in segs_by_robot[rid]:
                kind = str(seg.get("kind", ""))
                left = float(seg.get("start", 0.0))
                width = float(seg.get("dur", 0.0))
                if width <= 0.0:
                    continue
                if kind == "move":
                    ax.barh(ridx, width, left=left, color=COLOR_MAP["move"], edgecolor="k", height=0.6, linewidth=1)
                elif kind == "charge_wait":
                    ax.barh(ridx, width, left=left, color="black", edgecolor="k", height=0.6, linewidth=1)
                elif kind == "charge":
                    ax.barh(ridx, width, left=left, color="white", edgecolor="k", height=0.6, linewidth=1, hatch="//")
                    cid = seg.get("charger_id", None)
                    if cid is not None and width > 2.0:
                        ax.text(left + width / 2.0, ridx, f"c{int(cid)}", va="center", ha="center", fontsize=12, color="black")
                elif kind == "task":
                    ttype = str(seg.get("type", "one_way"))
                    ax.barh(ridx, width, left=left, color=COLOR_MAP.get(ttype, "#cccccc"), edgecolor="k", height=0.6, linewidth=1)
                    tid = seg.get("task_id", None)
                    if tid is not None and width > 2.0:
                        ax.text(left + width / 2.0, ridx, f"{int(tid)}", va="center", ha="center", fontsize=9, color="black")

            # temp.py처럼 로봇별 makespan 라벨을 오른쪽에 표시
            makespan = float(ms_by_robot.get(rid, 0.0))
            fontweight = "bold" if rid in max_robots else "normal"
            ax.text(
                makespan + 5.0,
                ridx,
                f"{makespan:.1f}s",
                va="center",
                ha="left",
                fontsize=13,
                color="black",
                fontweight=fontweight,
            )

        ax.set_yticks(range(len(robots)))
        ax.set_yticklabels(robots, fontsize=14)
        ax.tick_params(axis="both", labelsize=13)
        ax.invert_yaxis()
        ax.set_xlabel("Time (s)", fontsize=14)
        disp_name = STRAT_DISPLAY_NAME.get(strat, strat.upper())
        ax.set_title(f"Task Timeline ({disp_name})", fontsize=18)

    global_max = max(max_spans) if max_spans else 0.0
    # 오른쪽 여백: makespan 라벨(숫자+s)이 안 잘리도록 xlim을 여유 있게
    margin_right = max(80.0, global_max * 0.06)
    if global_max > 0.0:
        for ax in axes:
            ax.set_xlim(0.0, global_max + margin_right)

    proxy = _proxy_patches()
    fig.legend(
        [proxy[k] for k in ["Travel", "One-way", "Round-trip", "Via-point", "Charge", "Charge wait"]],
        ["Travel", "One-way", "Round-trip", "Via-point", "Charge", "Charge wait"],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=6,
        frameon=False,
        fontsize=22,
        handlelength=2.2,
        columnspacing=1.6,
        handletextpad=0.6,
    )

    # 아래 여백 확대: 범례가 플롯/글씨와 겹치지 않도록
    plt.tight_layout(rect=[0, 0.10, 1, 1])
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"[Saved] internal timeline: {out_path}")


def generate_charge_timeline_png(
    timeline_events: List[Tuple[str, int, int, float, float, float]],
    out_path: str,
) -> None:
    """Phase2에서 얻은 충전 대기/충전 구간만 간단히 시각화 (작업 구간은 미포함)."""
    if not timeline_events:
        print("[WARN] charge timeline skipped: no timeline_events")
        return

    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[WARN] charge timeline skipped: matplotlib import failed: {e}")
        return

    # 로봇별 정렬
    by_robot: Dict[str, List[Tuple[str, float, float, float]]] = {}
    for robot_id, _task_idx, ch_idx, arrival, start, end in timeline_events:
        by_robot.setdefault(robot_id, []).append((_charger_name(ch_idx), arrival, start, end))
    for rid in by_robot:
        by_robot[rid].sort(key=lambda x: x[2])  # start 기준

    robots = sorted(by_robot.keys())
    fig, ax = plt.subplots(figsize=(16, 1.2 * len(robots) + 1.0))

    for idx, rid in enumerate(robots):
        events = by_robot[rid]
        for cname, arrival, start, end in events:
            wait = max(0.0, start - arrival)
            charge = max(0.0, end - start)
            if wait > 1e-6:
                ax.barh(idx, wait, left=arrival, color="black", edgecolor="k", height=0.6, linewidth=1)
            if charge > 1e-6:
                ax.barh(idx, charge, left=start, color="white", edgecolor="k", height=0.6, linewidth=1, hatch="//")
                if charge > 2.0:
                    ax.text(start + charge / 2.0, idx, cname, ha="center", va="center", fontsize=10, color="black")

    ax.set_yticks(range(len(robots)))
    ax.set_yticklabels(robots, fontsize=12)
    ax.invert_yaxis()
    ax.set_xlabel("Time (s)", fontsize=12)
    ax.set_title("Charge Timeline (wait: black, charge: white hatched)", fontsize=14)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[Saved] charge timeline: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline charge sequence analyzer (threshold/feasibility/optimized)")
    parser.add_argument("--multi-algo", action="store_true", help="DECODE_OPTIMIZED_TEXT 내 6개 알고리즘(PSO,WCA,QSA,SMA,FLA,RIME) 타임라인만 따로 저장")
    parser.add_argument("--multi-algo-out-dir", default=".", help="--multi-algo 시 PNG 저장 디렉터리 (상대 경로는 log_sequence 기준)")
    parser.add_argument("--save-both", action="store_true", help="한 번 실행으로 timeline_offline.png(3전략) + timeline_6algos.png(6알고리즘) 동시 저장")
    parser.add_argument("--global-phase2", action="store_true", help="optimized 전략에서 전역 메이크스팬 최소화로 충전소 배정 탐색")
    parser.add_argument("--phase2-method", choices=("greedy", "beam", "exhaustive"), default="beam", help="Phase2 탐색 방식 (default: beam)")
    parser.add_argument("--phase2-beam-width", type=int, default=10, help="beam 탐색 시 상위 유지 개수 (default: 10)")
    parser.add_argument("--timeline-out", default="timeline_offline.png", help="생성할 타임라인 PNG 파일명 (상대 경로는 log_sequence 기준)")
    parser.add_argument("--timeline-mode", choices=("internal", "temp"), default="internal", help="타임라인 생성 방식 (default: internal)")
    parser.add_argument("--task-log", default="task_time_energy_simulation_log.txt", help="(timeline-mode=temp) temp.py용 task 로그 파일")
    parser.add_argument("--charge-timeline-out", default="timeline_charge.png", help="충전 구간만 표시한 타임라인 PNG 파일명 (상대 경로는 log_sequence 기준)")
    parser.add_argument("--no-timeline", action="store_true", help="타임라인 PNG 생성 비활성화")
    parser.add_argument("--no-charge-timeline", action="store_true", help="충전 전용 타임라인 PNG 생성 비활성화")
    args = parser.parse_args()
    args.timeline_out = _resolve_output_path(args.timeline_out)
    args.charge_timeline_out = _resolve_output_path(args.charge_timeline_out)

    if not ROS_AVAILABLE:
        print("[ERROR] ROS2 + nav2_msgs required.")
        sys.exit(1)

    rclpy.init()
    try:
        path_node = PathQueryNode(list(ROBOT_CAP_WH.keys()))
    except RuntimeError as e:
        print(f"[ERROR] {e}")
        rclpy.shutdown()
        sys.exit(1)

    if args.multi_algo:
        try:
            run_multi_algorithm_timelines(
                path_node,
                DECODE_OPTIMIZED_TEXT,
                out_dir=args.multi_algo_out_dir,
            )
        finally:
            path_node.destroy_node()
            rclpy.shutdown()
        return

    try:
        results, timeline_events_by_strategy, phase1_charge_ends_by_strategy = run_analysis(
            path_node,
            use_global_phase2=args.global_phase2,
            phase2_method=args.phase2_method,
            phase2_beam_width=args.phase2_beam_width,
        )
        print_results(results)
        timeline_events = timeline_events_by_strategy.get("optimized", [])
        if timeline_events:
            print_timeline(timeline_events)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = _resolve_output_path(f"charging_analysis_{timestamp}.csv")
        save_results(results, csv_path)
        time_seq_path = os.path.join(SCRIPT_DIR, "time_sequence.csv")
        export_time_sequences(
            results=results,
            filename=time_seq_path,
            strategy="optimized",
            timeline_events=timeline_events_by_strategy.get("optimized"),
        )

        if not args.no_timeline:
            if args.timeline_mode == "internal":
                generate_internal_timeline_png(
                    results=results,
                    path_node=path_node,
                    timeline_events_by_strategy=timeline_events_by_strategy,
                    phase1_charge_ends_by_strategy=phase1_charge_ends_by_strategy,
                    out_path=str(args.timeline_out),
                )
            else:
                generate_timeline_png_from_results(
                    results=results,
                    task_log_path=str(args.task_log),
                    out_path=str(args.timeline_out),
                )
        if not args.no_charge_timeline and timeline_events:
            generate_charge_timeline_png(
                timeline_events=timeline_events,
                out_path=str(args.charge_timeline_out),
            )
        if args.save_both:
            out_dir = os.path.dirname(os.path.abspath(args.timeline_out)) or "."
            run_multi_algorithm_timelines(
                path_node,
                DECODE_OPTIMIZED_TEXT,
                out_dir=out_dir,
                out_filename="timeline_6algos.png",
            )
    finally:
        path_node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
