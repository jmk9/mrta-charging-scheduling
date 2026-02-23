from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import math
import random
import time

import rclpy
from rclpy.node import Node

from std_msgs.msg import String, Float32
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped
from nav_msgs.msg import Odometry
from multi_robot_msgs.msg import TaskCommand
from visualization_msgs.msg import Marker, MarkerArray

import numpy as np
from mealpy import FloatVar, SMA


# -----------------------------------------------------------------------------
# Embedded optimization data structures and logic (no ROS dependencies)
# -----------------------------------------------------------------------------

@dataclass
class RobotState:
    robot_id: str      # e.g. "tb1"
    namespace: str     # e.g. "/tb1"
    x: float
    y: float
    soc: float         # interpreted as remaining energy in Wh
    available: bool


@dataclass
class TaskSpec:
    task_id: int
    task_type: str   # "PICK_DROP", "CHARGE", "MULTI_PICK_DROP", "DUAL_PICK_MULTI_DROP"

    picks: List[Tuple[float, float]]   # [(x1,y1), (x2,y2), ...]
    drops: List[Tuple[float, float]]   # [(x1,y1), (x2,y2), ...]
    # Optional per-pick item weights (kg). If provided its length should
    # correspond to `picks`. If omitted, defaults will be used.
    pick_weights: Optional[List[float]] = None
    # Optional per-drop delivery specification: each element is a list of
    # pick indices (0-based) indicating which picked items to unload at
    # the corresponding drop location. Length will be normalized to
    # `drops` with empty lists by default.
    deliveries: Optional[List[List[int]]] = None
    pick_wait_s: float = 1.5
    drop_wait_s: float = 2.0
    charge_duration_s: float = 0.0

    # Backwards-compatible convenience properties returning the first
    # pick/drop coordinates for code paths that expect single-point tasks.
    @property
    def pick_x(self) -> float:
        return float(self.picks[0][0]) if self.picks else 0.0

    @property
    def pick_y(self) -> float:
        return float(self.picks[0][1]) if self.picks else 0.0

    @property
    def drop_x(self) -> float:
        return float(self.drops[0][0]) if self.drops else 0.0

    @property
    def drop_y(self) -> float:
        return float(self.drops[0][1]) if self.drops else 0.0

    def __post_init__(self):
        # Normalize pick_weights to a list matching picks (possibly empty).
        if self.pick_weights is None:
            self.pick_weights = []
        # Pad pick_weights to match picks with zeros
        if len(self.pick_weights) < len(self.picks):
            pad = [0.0] * (len(self.picks) - len(self.pick_weights))
            self.pick_weights = list(self.pick_weights) + pad

        if self.deliveries is None:
            self.deliveries = []
        # Normalize deliveries to have one list per drop
        if len(self.deliveries) < len(self.drops):
            pad2 = [[] for _ in range(len(self.drops) - len(self.deliveries))]
            self.deliveries = list(self.deliveries) + pad2


@dataclass
class ChargerSpec:
    charger_id: int
    x: float
    y: float


@dataclass
class ScheduledAction:
    task_id: int
    task_type: str
    robot_id: str


def _euclidean_distance(x1: float, y1: float, x2: float, y2: float) -> float:
    dx = x1 - x2
    dy = y1 - y2
    return math.hypot(dx, dy)


# -----------------------------------------------------------------------------
# ROS 2 Scheduler Node implementation
# -----------------------------------------------------------------------------


class SchedulerNode(Node):
    """Multi-robot task scheduler using embedded compute_schedule.

    Tracks robot pose/SOC, maintains a task queue, and periodically assigns
    tasks to robots via TaskCommand messages.
    """

    def __init__(self) -> None:
        super().__init__('scheduler')

        self.robot_ids: List[str] = ['tb1', 'tb2', 'tb3', 'tb4']
        self.robot_states: Dict[str, RobotState] = {}
        # Last odom positions per robot for SOC computation
        self._last_odom: Dict[str, tuple] = {}
        # Tasks that have not yet been scheduled/assigned
        self.pending_tasks: List[TaskSpec] = []
        # Backwards-compat alias used by the scheduler loop
        self.task_queue: List[TaskSpec] = self.pending_tasks
        self.chargers: List[ChargerSpec] = []
        # Per-charger simple occupancy/queue: charger_id -> list of robot_ids
        # A robot is considered occupying a charger while executing a CHARGE
        # task to that charger.
        self.charger_queues: Dict[int, List[str]] = {}
        self.task_pubs: Dict[str, rclpy.publisher.Publisher] = {}
        # Unified RViz debug marker publisher for all robots' current targets
        self.goal_debug_pub = self.create_publisher(MarkerArray, '/goal_pose_debug', 10)
        # Per-robot current goal (x, y) for visualization; None if no goal
        self.current_goals: Dict[str, tuple] = {rid: None for rid in self.robot_ids}
        # Separate smoothed pose used only for RViz visualization to avoid
        # showing large instantaneous jumps when AMCL/odometry resets.
        self.viz_pose: Dict[str, tuple] = {rid: (0.0, 0.0) for rid in self.robot_ids}
        # Threshold (m) to consider a pose update a jump (for logging)
        self.viz_jump_threshold = 1.0
        # Exponential smoothing factor for visualization (0..1), higher=less smoothing
        self.viz_alpha = 0.4
        # Track whether a robot is currently carrying an object (True/False)
        # This is maintained by scheduler using odometry proximity and task
        # events so we can deterministically decide retry behavior on
        # failures.
        # Track the task id currently executing on each robot (None if idle)
        self.current_task_id: Dict[str, int] = {rid: None for rid in self.robot_ids}
        # Distance threshold (meters) used to detect that a robot reached
        # Per-robot retry counters for failed tasks (avoid infinite retries)
        self.retry_counts: Dict[str, int] = {rid: 0 for rid in self.robot_ids}
        # Max number of automatic retries before giving up / marking dead
        self.retry_limit: int = 3
        # Per-robot current task sequence (queue of TaskSpec) and index into that sequence.
        # A robot must finish all tasks in its current sequence before receiving a new one.
        self.current_sequences: Dict[str, List[TaskSpec]] = {rid: [] for rid in self.robot_ids}
        self.current_seq_index: Dict[str, int] = {rid: 0 for rid in self.robot_ids}
        # Global jobs: list of jobs, each job is a list of TaskSpec
        # We will build exactly 4 jobs with 5 tasks each from the demo pool.
        self.jobs: List[List[TaskSpec]] = []
        # Real-time measurement of when each robot actually starts and
        # finishes executing its offline sequence (wall-clock).
        self.robot_start_time = {rid: None for rid in self.robot_ids}
        self.robot_finish_time = {rid: None for rid in self.robot_ids}
        # Robots that we consider "dead" for this run: spawned with a
        # non-empty sequence but never started any task within a grace
        # period. These will be excluded from makespan statistics.
        self.dead_robots: Dict[str, bool] = {rid: False for rid in self.robot_ids}

        # ANSI color codes for per-robot log coloring (for terminals that
        # support ANSI escape sequences). This makes it easier to visually
        # distinguish tb1~tb4 in logs.
        self.robot_color_codes: Dict[str, str] = {
            'tb1': "\033[31m",  # red
            'tb2': "\033[33m",  # yellow / orange
            'tb3': "\033[93m",  # bright yellow
            'tb4': "\033[32m",  # green
        }
        self.color_reset: str = "\033[0m"

        # Energy model parameters (simple, shared across robots for now)
        # These mirror the odom-based SOC decrease in _odom_cb so that
        # estimate_task_energy() predictions roughly match actual SOC drops.
        self.k_drive_wh_per_m = 0.0275 * 10.0  # same as drive_wh_per_m in _odom_cb
        # Fixed overheads for pick/drop actions (in Wh); can be tuned later.
        self.E_pick_wh = 0.5
        self.E_drop_wh = 0.5
        # Weight-related parameters: default item weight (kg) and how
        # much extra load factor per kg to apply to drive energy.
        self.default_item_weight_kg = 1.0
        # Multiplier per kg applied on top of loaded_factor when computing energy
        # e.g., energy_multiplier = loaded_factor + load_weight_factor * carried_kg
        self.load_weight_factor = 0.05

        # Per-robot battery capacities in Wh (used in _odom_cb and when
        # applying pick/drop overhead energy to SOC).
        self.robot_cap_wh: Dict[str, float] = {
            'tb1': 120.0,
            'tb2': 120.0,
            'tb3': 60.0,
            'tb4': 60.0,
        }

        # Initial SOC scaling factor so you can start tests with partially
        # charged batteries. For example, 1.0 = 100%, 0.5 = 50%.
        # Later this could be turned into a ROS parameter.
        self.initial_soc_factor: float = 1

        # Define two chargers in the map frame for visualization
        self.chargers = [
            ChargerSpec(charger_id=1, x=-3.0, y=-3.0),
            ChargerSpec(charger_id=2, x=0.0, y=-3.0),
        ]

        # Initialize charger queues
        for ch in self.chargers:
            self.charger_queues[ch.charger_id] = []

        for rid in self.robot_ids:
            self._init_robot_interfaces(rid)

        # Simple global task request interface
        self.create_subscription(
            String,
            '/global_task_request',
            self._on_global_task_request,
            10,
        )

        # Seed a pool of demo tasks for offline scheduling
        self._init_demo_tasks()

        # Charging strategy: choose how to handle CHARGE tasks relative to the
        # stage-1 per-robot task sequence optimization.
        #  - "optimized": use the joint multi-robot charging optimizer (stage 2)
        #    to insert CHARGE offline (both tasks and charging are offline).
        #  - "feasibility": allocate tasks offline but rely on online SOC-based
        #    feasibility logic to decide when to charge.
        #  - "threshold": allocate tasks offline and trigger charging online
        #    whenever SOC falls below a fixed fraction of capacity (e.g. 15%).
        charge_param = self.declare_parameter("charge", "optimized").get_parameter_value().string_value
        if charge_param not in ("optimized", "feasibility", "threshold"):
            self.get_logger().warn(
                f"[scheduler] Unknown charge strategy '{charge_param}', falling back to 'optimized'"
            )
            charge_param = "optimized"
        self.charging_strategy = charge_param

        # Main scheduling timer
        self.timer = self.create_timer(2.0, self._timer_cb)
        # Periodic SOC logging timer (every 10 seconds)
        self.soc_log_timer = self.create_timer(10.0, self._soc_log_timer_cb)
        # Periodic goal visualization timer (publish /goal_pose_debug continuously)
        self.goal_viz_timer = self.create_timer(0.5, self._goal_viz_timer_cb)
        # Log optimizer configuration (mealpy) once at startup for visibility,
        # based on the actually imported SMA class and our parameters.
        optimizer_name = getattr(SMA, 'OriginalSMA', SMA).__name__
        epoch = 50
        pop_size = 30
        pr = 0.03
        self.get_logger().info(
            f"SchedulerNode started for robots={self.robot_ids}; "
            f"task allocation optimizer={optimizer_name}(epoch={epoch}, pop_size={pop_size}, pr={pr})"
        )

        # ------------------------------------------------------------------
        # Health check timer: after a grace period (e.g., 10 seconds) from
        # node startup, detect robots that have never started any task even
        # though they have a non-empty offline sequence. We mark them as
        # "dead" and log once; they will be excluded from makespan stats.
        # ------------------------------------------------------------------
        self.health_check_dead_logged = False
        self.health_check_timer = self.create_timer(300.0, self._health_check_timer_cb)

        # Once the real makespan is reported, suppress periodic SOC logs to
        # keep output concise.
        self.makespan_reported: bool = False

    # Pending start tracker: when we publish a TaskCommand we expect a
        # corresponding `task_started` event from the robot. If no such
        # event arrives within `start_timeout_s`, we consider the robot's
        # start as failed and revert its availability so the scheduler can
        # retry the task. This prevents a global hang when all robots never
        # acknowledge starts.
        self.pending_task_start: Dict[str, float] = {}
        self.start_timeout_s = 8.0
        self.start_timeout_timer = self.create_timer(2.0, self._pending_start_check)

        # ------------------------------------------------------------------
        # Offline scheduling: pre-compute a fixed full sequence per robot.
        # We optimize the order of all tasks assigned to each robot once
        # at initialization time and then execute that sequence without
        # re-optimizing during runtime (rolling horizon is disabled).
        # ------------------------------------------------------------------
        self._offline_initialize_sequences()

    # ------------------------------------------------------------------
    # Energy estimation helper
    # ------------------------------------------------------------------

    def estimate_task_energy(self, robot: RobotState, task: TaskSpec):
        """Estimate energy (Wh) and time (s) to complete a task.

        Returns a tuple (E_total_wh, t_total_s).
        """
        # Generalized multi-stage support: treat picks/drops as an ordered
        # sequence of pick->drop pairs. We walk the robot's current pose
        # through each pick and drop, summing drive energy/time and adding
        # per-pick/drop overheads (waits and fixed energy costs).

        E_total = 0.0
        t_total = 0.0
        
        

        # If this is a CHARGE task, model the drive to charger and the
        # charge duration as time (energy consumed is only the drive).
        if task.task_type == "CHARGE":
            # drive from robot -> charger (use drop as charger location)
            if task.drops:
                cx, cy = task.drops[0]
            else:
                cx, cy = task.drop_x, task.drop_y

            d_drive = _euclidean_distance(robot.x, robot.y, cx, cy)
            E_drive = self.k_drive_wh_per_m * d_drive
            v_empty_mps = 0.3
            t_drive = d_drive / v_empty_mps if v_empty_mps > 1e-6 else 0.0

            E_total += E_drive
            t_total += t_drive
            # charging dwell time (executor-level) contributes to total time
            t_total += float(task.charge_duration_s)
            return E_total, t_total

        # Default speeds and loaded multiplier
        v_empty_mps = 0.3
        v_loaded_mps = 0.25
        loaded_factor = 1.3

        curr_x = float(robot.x)
        curr_y = float(robot.y)

        picks = list(task.picks) if task.picks else []
        drops = list(task.drops) if task.drops else []

        # Special handling for tasks that pick multiple items then deliver
        # them across several drop locations in one carried run.
        if task.task_type == "MULTI_PICK_DROP":
            # We will perform picks in order; after each pick the robot
            # carries the picked items and subsequent travel is loaded.
            total_picks = len(picks)
            carried_weight = 0.0
            carried_items = []
            picked_count = 0

            # perform picks sequentially, accounting for carried weight when
            # moving between picks (loaded after first pick)
            for i, (px, py) in enumerate(picks):
                # travel from current pose to next pick using current carried_weight
                d_to_pick = _euclidean_distance(curr_x, curr_y, px, py)
                # use empty speed if not carrying anything, otherwise loaded speed
                if picked_count == 0:
                    # empty drive
                    E_drive = self.k_drive_wh_per_m * d_to_pick
                    t_drive = d_to_pick / v_empty_mps if v_empty_mps > 1e-6 else 0.0
                else:
                    # loaded drive: scale by weight
                    weight_term = float(getattr(self, 'load_weight_factor', 0.05)) * carried_weight
                    E_drive = self.k_drive_wh_per_m * (loaded_factor + weight_term) * d_to_pick
                    t_drive = d_to_pick / v_loaded_mps if v_loaded_mps > 1e-6 else 0.0

                E_total += E_drive
                t_total += t_drive

                # perform pick overhead and increase carried weight
                E_total += self.E_pick_wh
                t_total += float(task.pick_wait_s)

                # determine picked item weight from pick_weights
                w = 0.0
                try:
                    w = float(task.pick_weights[i]) if (hasattr(task, 'pick_weights') and i < len(task.pick_weights)) else 0.0
                except Exception:
                    w = 0.0
                if w <= 0.0:
                    w = float(getattr(self, 'default_item_weight_kg', 1.0))

                # keep a list of carried items (index, weight)
                if 'carried_items' not in locals():
                    carried_items = []
                carried_items.append((i, w))
                carried_weight += w
                picked_count += 1
                curr_x, curr_y = float(px), float(py)

            # If no drops, we're done after picks
            if not drops:
                return E_total, t_total

            # Deliver according to explicit deliveries list if provided; each
            # deliveries[j] is a list of pick indices to unload at drop j.
            deliveries = list(task.deliveries) if getattr(task, 'deliveries', None) else []
            # If no explicit deliveries, default to delivering FIFO 1-per-drop
            if not deliveries or all((not d) for d in deliveries):
                remaining = picked_count
                for j, (dx, dy) in enumerate(drops):
                    if remaining <= 0:
                        break
                    deliver_idxs = [carried_items[0][0]] if carried_items else []
                    if not deliver_idxs:
                        # still move to drop but nothing to unload
                        d_loaded = _euclidean_distance(curr_x, curr_y, dx, dy)
                        weight_term = float(getattr(self, 'load_weight_factor', 0.05)) * carried_weight
                        E_drive_loaded = self.k_drive_wh_per_m * (loaded_factor + weight_term) * d_loaded
                        t_drive_loaded = d_loaded / v_loaded_mps if v_loaded_mps > 1e-6 else 0.0
                        E_total += E_drive_loaded
                        t_total += t_drive_loaded
                        curr_x, curr_y = float(dx), float(dy)
                        continue

                    # drive while carrying current weight
                    d_loaded = _euclidean_distance(curr_x, curr_y, dx, dy)
                    weight_term = float(getattr(self, 'load_weight_factor', 0.05)) * carried_weight
                    E_drive_loaded = self.k_drive_wh_per_m * (loaded_factor + weight_term) * d_loaded
                    t_drive_loaded = d_loaded / v_loaded_mps if v_loaded_mps > 1e-6 else 0.0
                    E_total += E_drive_loaded
                    t_total += t_drive_loaded

                    # unload picked item(s)
                    deliver_count = len(deliver_idxs)
                    delivered_weight = 0.0
                    # remove items from carried_items by index
                    for idx in deliver_idxs:
                        for k, (pi_idx, w) in enumerate(carried_items):
                            if pi_idx == idx:
                                delivered_weight += w
                                carried_items.pop(k)
                                break

                    E_total += self.E_drop_wh * deliver_count
                    t_total += float(task.drop_wait_s) * deliver_count

                    carried_weight = max(0.0, carried_weight - delivered_weight)
                    remaining -= deliver_count
                    curr_x, curr_y = float(dx), float(dy)

                return E_total, t_total

            # explicit deliveries provided: follow the lists of pick indices
            remaining = picked_count
            for j, (dx, dy) in enumerate(drops):
                deliver_idxs = list(deliveries[j]) if j < len(deliveries) else []

                # move to this drop while carrying current weight
                d_loaded = _euclidean_distance(curr_x, curr_y, dx, dy)
                weight_term = float(getattr(self, 'load_weight_factor', 0.05)) * carried_weight
                E_drive_loaded = self.k_drive_wh_per_m * (loaded_factor + weight_term) * d_loaded
                t_drive_loaded = d_loaded / v_loaded_mps if v_loaded_mps > 1e-6 else 0.0
                E_total += E_drive_loaded
                t_total += t_drive_loaded

                # unload specified picks at this drop
                deliver_count = 0
                delivered_weight = 0.0
                for idx in deliver_idxs:
                    for k, (pi_idx, w) in enumerate(list(carried_items)):
                        if pi_idx == idx:
                            delivered_weight += w
                            deliver_count += 1
                            # remove that item
                            for kk, it in enumerate(carried_items):
                                if it[0] == pi_idx:
                                    carried_items.pop(kk)
                                    break
                            break

                E_total += self.E_drop_wh * deliver_count
                t_total += float(task.drop_wait_s) * deliver_count

                carried_weight = max(0.0, carried_weight - delivered_weight)
                remaining = max(0, remaining - deliver_count)
                curr_x, curr_y = float(dx), float(dy)

            return E_total, t_total

        # Walk through interleaved picks and drops: for i in range(max(len(picks), len(drops)))
        n = max(len(picks), len(drops))
        # Track currently carried weight (kg)
        carried_weight = 0.0
        for i in range(n):
            # Move to next pick (empty drive)
            if i < len(picks):
                px, py = picks[i]
                d_to_pick = _euclidean_distance(curr_x, curr_y, px, py)
                E_drive = self.k_drive_wh_per_m * d_to_pick
                t_drive = d_to_pick / v_empty_mps if v_empty_mps > 1e-6 else 0.0

                E_total += E_drive
                t_total += t_drive

                # Pick overheads
                E_total += self.E_pick_wh
                t_total += float(task.pick_wait_s)

                # Add item weight if provided, otherwise use default
                w = 0.0
                try:
                    w = float(task.pick_weights[i]) if (hasattr(task, 'pick_weights') and i < len(task.pick_weights)) else 0.0
                except Exception:
                    w = 0.0

                if w <= 0.0:
                    w = float(getattr(self, 'default_item_weight_kg', 1.0))

                carried_weight += w

                curr_x, curr_y = float(px), float(py)

            # Move to corresponding drop (loaded drive)
            if i < len(drops):
                dx, dy = drops[i]
                d_loaded = _euclidean_distance(curr_x, curr_y, dx, dy)
                # scale loaded energy by both the static loaded_factor and
                # an additional term proportional to carried weight
                weight_term = float(getattr(self, 'load_weight_factor', 0.05)) * carried_weight
                E_drive_loaded = self.k_drive_wh_per_m * (loaded_factor + weight_term) * d_loaded
                t_drive_loaded = d_loaded / v_loaded_mps if v_loaded_mps > 1e-6 else 0.0

                E_total += E_drive_loaded
                t_total += t_drive_loaded

                # Drop overheads (per-drop). If multiple items are being
                # dropped at once this will be accounted for by MULTI_PICK_DROP
                E_total += self.E_drop_wh
                t_total += float(task.drop_wait_s)

                # After a drop, assume delivered items reduce carried_weight by
                # one default item unless more advanced semantics apply.
                # If pick_weights were used for specific delivery counts, the
                # MULTI_PICK_DROP branch handles that case explicitly.
                carried_weight = max(0.0, carried_weight - float(getattr(self, 'default_item_weight_kg', 1.0)))

                curr_x, curr_y = float(dx), float(dy)

        return E_total, t_total

    # ------------------------------------------------------------------
    # Charge helpers
    # ------------------------------------------------------------------

    def _nearest_charger(self, x: float, y: float) -> ChargerSpec:
        """Return the nearest charger to a given (x, y) position."""
        best = None
        best_dist = float("inf")
        for ch in self.chargers:
            d = _euclidean_distance(x, y, ch.x, ch.y)
            if d < best_dist:
                best_dist = d
                best = ch
        return best

    def _nearest_available_charger(self, x: float, y: float) -> ChargerSpec:
        """Return the nearest charger that is not currently occupied.
        """
        best = None
        best_dist = float("inf")
        for ch in self.chargers:
            queue = self.charger_queues.get(ch.charger_id, [])
            # Charger is considered occupied if its queue is non-empty.
            if queue:
                continue
            d = _euclidean_distance(x, y, ch.x, ch.y)
            if d < best_dist:
                best_dist = d
                best = ch

        # If no charger has an empty queue, return None to signal that
        # no charger is currently available.
        return best

    def _feasibility_charging_for_all_robots(
        self,
        base_sequences: Dict[str, List[TaskSpec]],
    ) -> Dict[str, List[TaskSpec]]:
        """Greedy feasibility-based CHARGE insertion (no joint optimization).

        For each robot independently, walk its base sequence and, whenever
        the estimated energy is insufficient to execute the next PICK_DROP
        task from the current pose, insert a CHARGE task that drives to the
        nearest charger and lets the runtime charging model replenish SOC.

        This is effectively the earlier feasibility-oriented charging logic
        promoted to a sequence-construction step, so that we can easily
        compare this strategy with the full optimization-based stage-2.
        """

        augmented: Dict[str, List[TaskSpec]] = {rid: [] for rid in self.robot_ids}

        for rid in self.robot_ids:
            # Skip robots that are already marked as dead or have no
            # base sequence; they will also be excluded from makespan
            # statistics elsewhere.
            if self.dead_robots.get(rid, False):
                continue
            seq = base_sequences.get(rid, [])
            if not seq:
                continue

            state0 = self.robot_states.get(rid)
            if state0 is None:
                augmented[rid] = list(seq)
                continue

            cap_wh = self.robot_cap_wh.get(rid, 120.0)
            soc = state0.soc
            x = state0.x
            y = state0.y

            for task in seq:
                if task.task_type not in ("PICK_DROP", "MULTI_PICK_DROP", "DUAL_PICK_MULTI_DROP"):
                    augmented[rid].append(task)
                    continue

                # Estimate energy from current pose to pick and then task itself.
                try:
                    required, _ = self.estimate_task_energy(
                        RobotState(
                            robot_id=rid,
                            namespace=state0.namespace,
                            x=x,
                            y=y,
                            soc=soc,
                            available=state0.available,
                        ),
                        task,
                    )
                except Exception:
                    required = 0.0

                # If SOC is not enough, insert a CHARGE task before this task.
                if soc < required:
                    try:
                        charger = self._nearest_available_charger(x, y)
                    except Exception:
                        charger = None

                    if charger is None:
                        # If we cannot find any charger, just proceed without
                        # inserting a charge; runtime may still handle it, but
                        # from the offline planner's perspective this is a
                        # hard-to-fix situation.
                        pass
                    else:
                        charge_task = TaskSpec(
                            task_id=-1,
                            task_type="CHARGE",
                            picks=[(x, y)],
                            drops=[(charger.x, charger.y)],
                            pick_wait_s=0.0,
                            drop_wait_s=0.0,
                            # One CHARGE action corresponds to up to 20s of
                            # dwell time at the charger in the executor.
                            charge_duration_s=20.0,
                        )
                        augmented[rid].append(charge_task)

                        # Offline model: assume we can recharge up to full
                        # capacity with this action so that subsequent tasks
                        # start from a safe SOC. This slightly over-approximates
                        # the runtime behavior but is conservative for
                        # feasibility.
                        soc = cap_wh
                        x, y = charger.x, charger.y

                # Now execute the task in the offline model.
                if soc >= required:
                    soc -= required
                augmented[rid].append(task)
                
                if task.drops:
                    x, y = task.drops[-1]
                else:
                    x, y = task.drop_x, task.drop_y

            augmented[rid] = augmented[rid]

        return augmented

    # ------------------------------------------------------------------
    # Stage-2: multi-robot charging optimization (offline, joint)
    # ------------------------------------------------------------------

    def _optimize_charging_for_all_robots(
        self,
        base_sequences: Dict[str, List[TaskSpec]],
    ) -> Dict[str, List[TaskSpec]]:
        """Second-stage optimizer: insert CHARGE events jointly for all robots.

        - base_sequences: per-robot PICK_DROP-only sequences (output of stage 1)
        - Returns: per-robot augmented sequences [TASK, CHARGE, TASK, ...]

        Hard constraints:
          * No depletion (SOC never negative, must have energy to reach tasks/chargers)
        Soft objectives:
          * Minimize makespan
          * Minimize waiting time at chargers when contention occurs
        """

        # Flatten tasks and build indexing for decision variables
        robot_ids = [rid for rid in self.robot_ids if base_sequences.get(rid)]
        if not robot_ids:
            return {rid: base_sequences.get(rid, []) for rid in self.robot_ids}

        index_map = []  # list of (rid, task_index)
        for rid in robot_ids:
            for i, _ in enumerate(base_sequences[rid]):
                index_map.append((rid, i))

        K = len(index_map)
        if K == 0:
            return {rid: base_sequences.get(rid, []) for rid in self.robot_ids}

        num_chargers = max(1, len(self.chargers))

        # Weights for stage-2 objective. The primary goal is to minimize the
        # overall makespan; waiting at chargers and extra driving time are
        # secondary, and we apply a modest penalty to the number of charges
        # to avoid excessive ping-ponging without preventing necessary
        # charging. An additional small penalty on overlapping reservations
        # at the same charger helps spread robots across chargers in time.
        w_ms = 1.0
        w_wait = 0.25
        w_path = 0.25
        w_charge = 1.0
        w_overlap = 0.1

        HUGE_PENALTY = 1e9

        # Pre-compute which robots cannot complete their base sequence with
        # their current SOC if they never charge. For those robots we will
        # treat "zero charge decisions" as infeasible inside the objective.
        robots_need_charge: Dict[str, bool] = {}
        for rid in robot_ids:
            state0 = self.robot_states.get(rid)
            seq = base_sequences.get(rid, [])
            if state0 is None or not seq:
                robots_need_charge[rid] = False
                continue

            cap_wh = self.robot_cap_wh.get(rid, 120.0)
            soc = state0.soc
            x = state0.x
            y = state0.y
            needs = False

            for task in seq:
                if task.task_type not in ("PICK_DROP", "MULTI_PICK_DROP", "DUAL_PICK_MULTI_DROP"):
                    continue


                try:
                    required, _ = self.estimate_task_energy(
                        RobotState(
                            robot_id=rid,
                            namespace=state0.namespace,
                            x=x,
                            y=y,
                            soc=soc,
                            available=state0.available,
                        ),
                        task,
                    )
                except Exception:
                    required = 0.0

                if soc < required:
                    needs = True
                    break

                soc -= required
                if task.drops:
                    x, y = task.drops[-1]
                else:
                    x, y = task.drop_x, task.drop_y

            robots_need_charge[rid] = needs

        def _decode_solution_to_plan(solution: np.ndarray) -> Dict[str, Dict[int, int]]:
            """Map continuous solution -> per-robot per-task charge decision.

            Returns: charge_plan[rid][i] = -1 (no charge) or charger_index (0..num_chargers-1).
            """
            plan: Dict[str, Dict[int, int]] = {rid: {} for rid in robot_ids}
            for var_idx, (rid, task_idx) in enumerate(index_map):
                z = float(solution[var_idx])
                if z < 0.0:
                    plan[rid][task_idx] = -1
                else:
                    # Map z in [0,1] to charger index 0..num_chargers-1
                    frac = max(0.0, min(1.0, z))
                    cid = int(frac * num_chargers)
                    if cid >= num_chargers:
                        cid = num_chargers - 1
                    plan[rid][task_idx] = cid
            return plan

        # Keep track of the best objective we've seen so far so that we can
        # log only occasional improvements instead of every evaluation.
        best_obj_seen = {
            "value": float("inf"),
        }

        def _objective(solution: np.ndarray) -> float:
            # Decode decision variables
            charge_plan = _decode_solution_to_plan(solution)

            # Per-charger reservations: charger_index -> list of (start, end)
            charger_reservations: Dict[int, List[tuple]] = {i: [] for i in range(num_chargers)}

            finish_times: Dict[str, float] = {}
            total_drive_time = 0.0
            total_wait = 0.0
            overlap_penalty = 0.0
            infeasible_penalty = 0.0
            num_charges_total = 0

            # Pre-cache charger list as array for index access
            chargers_list = list(self.chargers)

            for rid in robot_ids:
                state0 = self.robot_states.get(rid)
                if state0 is None:
                    continue
                seq = base_sequences[rid]
                if not seq:
                    finish_times[rid] = 0.0
                    continue

                cap_wh = self.robot_cap_wh.get(rid, 120.0)

                t = 0.0
                soc = state0.soc
                x = state0.x
                y = state0.y

                for i, task in enumerate(seq):
                    if task.task_type not in ("PICK_DROP", "MULTI_PICK_DROP", "DUAL_PICK_MULTI_DROP"):
                        continue

                    decision = charge_plan[rid].get(i, -1)
                    # Optional pre-task charge
                    if decision >= 0 and chargers_list:
                        num_charges_total += 1
                        ch = chargers_list[decision]
                        # drive to charger
                        dx = x - ch.x
                        dy = y - ch.y
                        dist_to_ch = math.hypot(dx, dy)
                        drive_wh_per_m = self.k_drive_wh_per_m
                        e_to_ch = drive_wh_per_m * dist_to_ch
                        v_empty_mps = 0.3
                        t_to_ch = dist_to_ch / v_empty_mps if v_empty_mps > 1e-6 else 0.0
                        if soc < e_to_ch:
                            infeasible_penalty += HUGE_PENALTY
                            break

                        soc -= e_to_ch
                        t += t_to_ch
                        total_drive_time += t_to_ch

                        # compute waiting time at charger due to existing reservations
                        arrive = t
                        reservations = charger_reservations[decision]
                        wait = 0.0
                        if reservations:
                            latest_free = arrive
                            for (s0, e0) in reservations:
                                if e0 > latest_free and s0 < arrive:
                                    latest_free = e0
                            if latest_free > arrive:
                                wait = latest_free - arrive

                        total_wait += wait
                        t_start = arrive + wait

                        # Charging model: at most 20 seconds, SOC +0.05 per
                        # second in normalized units, capped at full. This
                        # mirrors the runtime behavior applied when a CHARGE
                        # task completes, so that the optimizer reasons about
                        # approximately the same energy dynamics.
                        frac_soc = max(0.0, min(1.0, soc / cap_wh))
                        remaining_frac = 1.0 - frac_soc
                        max_dur = 20.0
                        # Time to full at +0.05 per second in SOC units.
                        time_to_full = remaining_frac / 0.05 if remaining_frac > 0.0 else 0.0
                        t_charge = min(max_dur, max(0.0, time_to_full))

                        # SOC gain during this charging period.
                        delta_frac = min(remaining_frac, 0.05 * t_charge)
                        soc += delta_frac * cap_wh
                        soc = min(soc, cap_wh)

                        t_end = t_start + t_charge
                        # register reservation
                        reservations.append((t_start, t_end))
                        charger_reservations[decision] = reservations

                        # update robot state after charging
                        t = t_end
                        x, y = ch.x, ch.y

                    # Now execute the original task
                    # drive current pose -> pick
                    dxp = x - task.pick_x
                    dyp = y - task.pick_y
                    dist_to_pick = math.hypot(dxp, dyp)
                    v_empty_mps = 0.3
                    t_to_pick = dist_to_pick / v_empty_mps if v_empty_mps > 1e-6 else 0.0

                    # task itself from pick pose
                    try:
                        required, t_task = self.estimate_task_energy(
                            RobotState(
                                robot_id=rid,
                                namespace=state0.namespace,
                                x=x,
                                y=y,
                                soc=soc,
                                available=state0.available,
                            ),
                            task,
                        )
                    except Exception:
                        required, t_task = 0.0, 0.0

                    if soc < required:
                        infeasible_penalty += HUGE_PENALTY
                        break

                    soc -= required
                    t += t_task
                    total_drive_time += t_task

                    # update pose to drop
                    x, y = task.drop_x, task.drop_y

                # If this robot needs charging to finish its sequence but
                # never actually charged in this solution, treat as infeasible.
                if robots_need_charge.get(rid, False):
                    has_charge = any(
                        (charge_plan[rid].get(i, -1) >= 0)
                        for i, task in enumerate(seq)
                        if task.task_type in ("PICK_DROP", "MULTI_PICK_DROP", "DUAL_PICK_MULTI_DROP")
                    )
                    if not has_charge:
                        infeasible_penalty += HUGE_PENALTY

                finish_times[rid] = t

            if len(finish_times) == 0:
                return HUGE_PENALTY

            makespan = max(finish_times.values())

            # Explicitly penalize temporal overlap of reservations on the
            # same physical charger. This encourages the optimizer to avoid
            # stacking multiple robots on one charger at the same time when
            # an alternative schedule (possibly using another charger) can
            # reduce contention and ultimately the makespan.
            for cid, reservations in charger_reservations.items():
                if len(reservations) <= 1:
                    continue
                rs = sorted(reservations, key=lambda r: r[0])
                for i in range(len(rs)):
                    s1, e1 = rs[i]
                    for j in range(i + 1, len(rs)):
                        s2, e2 = rs[j]
                        if s2 >= e1:
                            break
                        overlap = min(e1, e2) - max(s1, s2)
                        if overlap > 0.0:
                            overlap_penalty += overlap

            obj = (
                w_ms * makespan
                + w_wait * total_wait
                + w_path * total_drive_time 
                + w_charge * num_charges_total
                + w_overlap * overlap_penalty
                + infeasible_penalty
            )

            # Only log when we find a strictly better objective than any
            # seen so far in this optimization run. This keeps logs sparse
            # while still giving insight into progress.
            if obj < best_obj_seen["value"]:
                best_obj_seen["value"] = obj
                self.get_logger().info(
                    f"[stage2] obj best-so-far: J={obj:.1f}, "
                    f"makespan={makespan:.1f}s, wait={total_wait:.1f}s, "
                    f"drive={total_drive_time:.1f}s, "
                    f"overlap={overlap_penalty:.1f}s, "
                    f"infeasible_penalty={infeasible_penalty:.1e}"
                )

            return obj

        # Mealpy problem: one variable per (robot, task_index)
        problem = {
            "obj_func": _objective,
            "bounds": FloatVar(lb=(-1.0,) * K, ub=(1.0,) * K),
            "minmax": "min",
            "log_to": None,
        }

        model = SMA.OriginalSMA(epoch=300, pop_size=60, pr=0.03)
        best = model.solve(problem)

        # Decode best solution to final augmented sequences
        best_plan = _decode_solution_to_plan(best.solution)
        chargers_list = list(self.chargers)

        augmented: Dict[str, List[TaskSpec]] = {rid: [] for rid in self.robot_ids}
        for rid in self.robot_ids:
            seq = base_sequences.get(rid, [])
            if not seq:
                continue
            charge_decisions = best_plan.get(rid, {})
            state0 = self.robot_states.get(rid)
            if state0 is None:
                augmented[rid] = list(seq)
                continue

            # Rebuild sequence with CHARGE tasks inserted according to best_plan
            x, y = state0.x, state0.y
            for i, task in enumerate(seq):
                decision = charge_decisions.get(i, -1)
                if decision >= 0 and chargers_list:
                    ch = chargers_list[decision]
                    charge_task = TaskSpec(
                        task_id=-1,
                        task_type="CHARGE",
                        picks=[(x, y)],
                        drops=[(ch.x, ch.y)],
                        pick_wait_s=0.0,
                        drop_wait_s=0.0,
                        # One CHARGE action corresponds to 20 seconds of
                        # dwell time at the charger in the executor.
                        charge_duration_s=20.0,
                    )
                    augmented[rid].append(charge_task)
                    x, y = ch.x, ch.y

                augmented[rid].append(task)
                if task.task_type == "PICK_DROP":
                    x, y = task.drop_x, task.drop_y

        return augmented

    # ------------------------------------------------------------------
    # Demo / initial tasks
    # ------------------------------------------------------------------

    def _init_demo_tasks(self) -> None:
        """Create a pool of demo tasks (>= 10) for later random job building.

        Coordinates are in the 'map' frame and loosely correspond to a
        warehouse-like layout for the AWS small warehouse world.
        """
        # Use the provided 30 pick/drop pairs and generate 20 tasks by
        # grouping every 3 consecutive pairs (wrap-around). Each generated
        # task will be a MULTI_PICK_DROP with 3 picks and 3 drops and a
        # per-pick weights list.
        # demo_triplets: each entry is (pick_x,pick_y, drop1_x,drop1_y, drop2_x,drop2_y)
        demo_pairs = [
            (-3.80, 5.75, -4.10, -9.95),  # task 1
            (1.10, -2.75, 3.15, -9.60),   # task 2
            (-2.95, 7.15, -5.40, 4.15),   # task 3
            (0.25, 0.80, 4.65, -9.55),    # task 4
            (-1.30, -10.10, -3.10, 4.20), # task 5
            (-2.40, 0.25, -3.15, -2.20),  # task 6
            (0.05, 7.25, -2.40, -2.90),   # task 7
            (6.15, -3.30, -0.50, -1.40),  # task 8
            (-3.85, -0.90, 3.25, -7.00),  # task 9
            (-3.65, 4.80, 2.80, -9.70),   # task 10
            (-0.55, -8.70, 4.20, -3.80),  # task 11
            (3.55, 0.60, 5.40, -2.80),    # task 12
            (5.50, 0.45, -5.95, -8.50),   # task 13
            (-4.15, -9.35, 2.80, -9.70),  # task 14
            (-6.05, -3.45, -6.00, 4.50),  # task 15
            (-3.75, -9.65, 0.80, 2.40),   # task 16
            (4.05, -9.40, -5.80, -0.70),  # task 17
            (-2.00, -2.85, 3.25, -0.20),  # task 18
            (-6.40, 5.35, -6.20, 6.50),   # task 19
            (0.40, 5.60, 1.50, -7.00),    # task 20
            (-0.60, -5.00, -1.75, -2.80), # task 21
            (-5.35, -0.45, 5.80, -3.30),  # task 22
            (0.50, 5.25, 1.50, -1.50),    # task 23
            (-3.95, -8.35, -4.70, -9.60), # task 24
            (-6.30, -0.40, 1.00, 6.00),   # task 25
            (3.20, -1.00, 1.20, 4.90),    # task 26
            (4.45, -5.85, 5.10, -9.35),   # task 27
            (1.90, 1.45, -6.20, 4.55),    # task 28
            (-3.10, 6.05, 1.70, -1.25),   # task 29
            (-5.70, 5.45, -0.45, -7.10),  # task 30
            (-6.55, -8.80, -5.80, -9.55), # task 31
            (-4.35, 7.90, -5.90, -8.25),  # task 32
            (-4.55, -8.40, 4.55, -9.90),  # task 33
            (-3.25, 1.95, -1.30, -9.25),  # task 34
            (0.40, 6.00, -3.35, 4.65),    # task 35
            (5.10, -1.75, -1.90, -2.40),  # task 36
            (-2.90, 8.15, 0.65, -2.10),   # task 37
            (0.20, -1.20, 1.85, -3.30),   # task 38
            (-0.40, -7.75, 4.25, -2.90),  # task 39
            (1.50, 5.25, -0.35, -6.05),   # task 40
            (0.45, -8.30, -4.95, 7.20),   # task 41
            (0.65, -0.65, 0.65, 5.45),    # task 42
            (-4.05, -2.90, -4.80, -8.30), # task 43
            (-3.15, 2.25, -2.15, -2.15),  # task 44
            (-3.80, -4.80, -5.00, -4.20), # task 45
            (-6.60, 4.95, 1.40, 3.55),    # task 46
            (-3.10, -7.40, 4.85, -6.05),  # task 47
            (-4.20, -7.45, -6.00, -9.00), # task 48
            (0.00, 4.40, -0.75, 0.30),    # task 49
            (1.75, -8.85, 0.75, 5.10),    # task 50
            (-2.90, -4.85, 0.75, -6.45),  # task 51
            (-6.20, 6.35, -3.80, 1.65),   # task 52
            (-3.00, 1.90, 2.15, -9.50),   # task 53
            (4.65, 9.85, 3.55, -8.10),    # task 54
            (1.05, 6.35, 4.90, -0.75),    # task 55
            (-0.40, -8.25, -2.65, -3.80), # task 56
            (0.80, -7.90, -6.00, 0.10),   # task 57
            (-4.60, 1.30, -3.10, 8.95),   # task 58
            (-5.85, 3.75, -5.55, -6.75),  # task 59
            (-5.05, 6.40, 0.40, -6.20),   # task 60
            (-3.25, 4.85, -2.05, -0.30),  # task 61
            (-1.75, -1.85, -4.10, -3.45), # task 62
            (1.60, 1.20, -2.40, 6.55),    # task 63
            (-3.30, 8.45, -3.90, -1.15),  # task 64
            (3.20, 2.15, 1.05, 5.15),     # task 65
            (0.75, 3.50, -3.95, -9.70),   # task 66
            (-3.65, -3.35, 3.10, -3.55),  # task 67
            (0.15, 6.20, -2.95, 5.10),    # task 68
            (5.05, -0.65, 1.90, -9.80),   # task 69
            (-0.55, -4.45, -3.55, -5.95), # task 70
            (5.05, -0.25, -6.20, 1.50),   # task 71
            (-4.10, -8.85, 4.90, -7.60),  # task 72
            (1.60, -5.80, -3.15, -8.45),  # task 73
            (-6.25, -9.45, 0.15, 6.30),   # task 74
            (1.70, 7.15, 0.10, -5.40),    # task 75
            (0.20, -9.60, -2.35, -9.55),  # task 76
            (5.70, -7.60, 4.20, -4.35),   # task 77
            (3.35, -6.30, -5.65, 4.60),   # task 78
            (-2.25, -0.50, -4.75, -6.00), # task 79
            (-2.70, -0.70, -0.20, -2.70), # task 80
            (-1.55, -3.95, 3.75, 1.15),   # task 81
            (-2.85, 8.20, 4.80, -2.45),   # task 82
            (-2.75, 8.75, -0.50, 5.65),   # task 83
            (-2.55, -2.05, -0.35, 1.55),  # task 84
            (-1.05, -1.45, -6.50, 6.05),  # task 85
            (-1.50, -0.45, -0.45, -0.95), # task 86
            (-3.95, -1.70, 6.00, -8.25),  # task 87
            (-5.85, -4.25, -4.15, 5.30),  # task 88
            (-5.05, -4.30, 0.95, -2.65),  # task 89
            (4.90, -7.70, -3.10, -9.05),  # task 90
            (-1.65, -4.30, -2.65, 7.00),  # task 91
            (-5.45, -3.75, -2.50, 3.50),  # task 92
            (-6.15, -4.35, 5.25, -4.35),  # task 93
            (-5.25, 8.90, 2.85, -2.20),   # task 94
            (-3.45, 1.50, 3.70, -8.35),   # task 95
            (-5.00, -7.30, 0.40, 6.65),   # task 96
            (-6.35, -4.30, -1.05, -5.70), # task 97
            (0.10, 6.00, 5.80, -6.35),    # task 98
            (-4.95, 6.75, -6.55, -3.15),  # task 99
            (-6.45, -0.20, 0.90, -5.40)   # task 100
        ]

        # Each demo_pairs entry already encodes a triplet: (pick_x,pick_y, drop1_x,drop1_y, drop2_x,drop2_y)
        triplets = []
        for entry in demo_pairs:
            pick_loc = (entry[0], entry[1])
            drop = (entry[2], entry[3])
            triplets.append((pick_loc, drop))

        self.pending_tasks = []
        tid = 1
        for i in range(len(triplets)):
            pick_loc, drop = triplets[i]

            # two identical picks at the same location (two items to pick)
            picks = [pick_loc, drop]
            drops = [drop, pick_loc]

            # per-pick weights for the two items
            pick_weights = [round(0.9 + 0.15 * ((i + j) % 4), 2) for j in range(2)]
            # deliveries: first drop receives pick index 0, second drop receives pick index 1

            self.pending_tasks.append(
                TaskSpec(
                    task_id=tid,
                    task_type="DUAL_PICK_MULTI_DROP",
                    picks=picks,
                    drops=drops,
                    pick_weights=pick_weights,
                    pick_wait_s=1.5,
                    drop_wait_s=2.0,
                    charge_duration_s=0.0,
                )
            )
            tid += 1

        # Keep task_queue pointing at the same list (pool of tasks)
        self.task_queue = self.pending_tasks
        self.get_logger().info(f'Initialized demo task pool with {len(self.pending_tasks)} MULTI_PICK_DROP tasks (20 tasks from 30 pairs)')

    def _build_jobs_from_pool(self) -> None:
        """Build a fixed number of global jobs from the demo task pool.

        For now we:
          - shuffle the 20 demo tasks
          - split them into exactly 4 jobs with 5 tasks each

        This gives us a clear job structure that can later be optimized,
        while we still assign tasks to robots randomly at runtime.
        """
        if not self.pending_tasks:
            return

        all_tasks = list(self.pending_tasks)
        random.shuffle(all_tasks)

        num_jobs = 4
        job_size = max(1, len(all_tasks) // num_jobs)
        self.jobs = []

        start = 0
        for j in range(num_jobs):
            end = min(start + job_size, len(all_tasks))
            if start >= len(all_tasks):
                break
            self.jobs.append(all_tasks[start:end])
            start = end

        self.get_logger().info(
            'Built global jobs: ' +
            ', '.join(f"job{idx+1}({len(job)} tasks)" for idx, job in enumerate(self.jobs))
        )

    def _offline_initialize_sequences(self) -> None:
        """Offline scheduling step: build one fixed optimized sequence per robot.
        
        We treat the entire pending task pool as a global set and greedily
        allocate tasks to robots based on estimated energy/time cost. The
        resulting per-robot base sequences are then passed to the joint
        multi-robot charging optimizer (stage 2) when enabled.
        """

        # If there are no pending tasks, nothing to schedule.
        if not self.pending_tasks:
            return

        # Stage 1: global multi-robot allocation using a simple greedy
        # heuristic based on estimate_task_energy. We consider only the
        # supported task types for offline optimization.
        candidate_tasks = [
            t for t in self.pending_tasks
            if t.task_type in ("PICK_DROP", "MULTI_PICK_DROP", "DUAL_PICK_MULTI_DROP")
        ]

        if not candidate_tasks:
            return

        base_sequences: Dict[str, List[TaskSpec]] = {rid: [] for rid in self.robot_ids}

        # Build simulated robot states used only for offline allocation.
        sim_states: Dict[str, RobotState] = {}
        for rid in self.robot_ids:
            state = self.robot_states.get(rid)
            if state is None or self.dead_robots.get(rid, False):
                continue
            sim_states[rid] = RobotState(
                robot_id=state.robot_id,
                namespace=state.namespace,
                x=state.x,
                y=state.y,
                soc=state.soc,
                available=state.available,
            )

        if not sim_states:
            self.get_logger().warn("[offline] No available robots for offline scheduling")
            return

        remaining: List[TaskSpec] = list(candidate_tasks)

        # Greedy global allocation: at each step, choose the (robot, task)
        # pair with minimum estimated cost and assign that task to the
        # corresponding robot's sequence, updating its simulated state.
        while remaining:
            best_rid: Optional[str] = None
            best_idx: Optional[int] = None
            best_cost: float = float("inf")
            best_new_state: Optional[RobotState] = None

            for idx, task in enumerate(remaining):
                for rid, sim in sim_states.items():
                    try:
                        required_energy, t_task = self.estimate_task_energy(sim, task)
                    except Exception:
                        required_energy, t_task = 0.0, 0.0

                    # Same weighting style as _optimize_task_sequence.
                    w_t = 1.0
                    w_e = 0.1
                    cost = w_t * t_task + w_e * required_energy

                    # Softly penalize assignments that are clearly
                    # infeasible from the current SOC so that other
                    # robots (or later tasks) are preferred when
                    # possible. Stage-2 or online charging may still
                    # recover feasibility later.
                    if required_energy > sim.soc:
                        cost += 1e3

                    if cost < best_cost:
                        best_cost = cost
                        best_rid = rid
                        best_idx = idx

                        # Compute the simulated state after executing
                        # this task from the current pose.
                        if task.drops:
                            last_drop = task.drops[-1]
                            new_x, new_y = float(last_drop[0]), float(last_drop[1])
                        else:
                            new_x, new_y = float(task.drop_x), float(task.drop_y)

                        new_soc = max(0.0, sim.soc - required_energy)
                        best_new_state = RobotState(
                            robot_id=sim.robot_id,
                            namespace=sim.namespace,
                            x=new_x,
                            y=new_y,
                            soc=new_soc,
                            available=sim.available,
                        )

            if best_rid is None or best_idx is None or best_new_state is None:
                # Should not normally happen, but break to avoid an
                # infinite loop if we cannot find any viable assignment.
                self.get_logger().warn("[offline] Greedy allocation terminated early: no viable (robot, task) pair found")
                break

            task = remaining.pop(best_idx)
            base_sequences[best_rid].append(task)
            sim_states[best_rid] = best_new_state

        # At this point, every task in candidate_tasks has been assigned to
        # exactly one robot in base_sequences. Next, for each robot, run the
        # SMA-based per-robot optimizer so that the full task set is
        # actually optimized (not just greedily assigned) and mealpy's
        # built-in progress bar is visible during task optimization.

        for rid, seq in base_sequences.items():
            state = self.robot_states.get(rid)
            if state is None or not seq:
                continue
            try:
                self.get_logger().info(
                    f"[stage1] Optimizing task sequence for {rid} with {len(seq)} tasks using SMA.OriginalSMA"
                )
                best_seq = self._optimize_task_sequence(
                    robot_state=state,
                    candidate_tasks=seq,
                )
                base_sequences[rid] = best_seq
            except Exception as e:
                self.get_logger().error(
                    f"[stage1] SMA sequence optimization for {rid} failed: {type(e).__name__}: {e}. Using greedy order."
                )

        # Stage 2: insert CHARGE events according to the selected strategy.
        if self.charging_strategy == "optimized":
            # Fully offline: use the joint multi-robot charging optimizer.
            augmented_sequences = self._optimize_charging_for_all_robots(base_sequences)
        elif self.charging_strategy == "feasibility":
            # Tasks are allocated offline, but charging decisions are made
            # online based on real-time SOC, so we keep only the base
            # sequences here and let the runtime logic decide when to insert
            # CHARGE.
            augmented_sequences = base_sequences
        elif self.charging_strategy == "threshold":
            # Threshold-based online charging: tasks are allocated offline,
            # and we will trigger CHARGE online when SOC falls below a
            # configured fraction of capacity. No offline CHARGE insertion.
            augmented_sequences = base_sequences
        else:
            # Fallback: no extra charging optimization, just use base sequences.
            augmented_sequences = base_sequences

        for rid in self.robot_ids:
            final_seq = augmented_sequences.get(rid, [])
            self.current_sequences[rid] = final_seq
            self.current_seq_index[rid] = 0
            if not final_seq:
                continue
            seq_ids = [t.task_id for t in final_seq]
            self.get_logger().info(
                f"[offline] initialized sequence for {rid}: "
                f"tasks={seq_ids} (len={len(seq_ids)})"
            )

    def _all_jobs_empty(self) -> bool:
        """Return True if there are no remaining tasks in any global job."""
        if not self.jobs:
            return True
        return all((not job) for job in self.jobs)

    def _init_robot_interfaces(self, robot_id: str) -> None:
        pose_topic = f'/{robot_id}/amcl_pose'
        self.create_subscription(
            PoseWithCovarianceStamped,
            pose_topic,
            lambda msg, rid=robot_id: self._pose_cb(rid, msg),
            10,
        )

        # Odom subscriber for SOC decrease inside scheduler
        odom_topic = f'/{robot_id}/odom'
        self.create_subscription(
            Odometry,
            odom_topic,
            lambda msg, rid=robot_id: self._odom_cb(rid, msg),
            10,
        )

        soc_topic = f'/robot/{robot_id}/status'
        self.create_subscription(
            Float32,
            soc_topic,
            lambda msg, rid=robot_id: self._soc_cb(rid, msg),
            10,
        )

        event_topic = f'/{robot_id}/task_event'
        self.create_subscription(
            String,
            event_topic,
            lambda msg, rid=robot_id: self._task_event_cb(rid, msg),
            10,
        )

        cmd_topic = f'/{robot_id}/task_cmd'
        pub = self.create_publisher(TaskCommand, cmd_topic, 10)
        self.task_pubs[robot_id] = pub

        cap_wh = self.robot_cap_wh.get(robot_id, 120.0)
        init_soc = cap_wh * self.initial_soc_factor

        self.robot_states[robot_id] = RobotState(
            robot_id=robot_id,
            namespace=f'/{robot_id}',
            x=0.0,
            y=0.0,
            soc=init_soc,
            available=True,
        )

        rid_str = self._color_robot(robot_id) if hasattr(self, "_color_robot") else robot_id
        self.get_logger().info(
            f"[scheduler] init {rid_str}: cap={cap_wh:.1f}Wh, initial_soc={init_soc:.1f}Wh (factor={self.initial_soc_factor:.2f})"
        )

    def _pose_cb(self, robot_id: str, msg: PoseWithCovarianceStamped) -> None:
        state = self.robot_states.get(robot_id)
        if state is None:
            return
        px = msg.pose.pose.position.x
        py = msg.pose.pose.position.y

        # Always update true state used for scheduling
        state.x = px
        state.y = py

        # Update smoothed viz pose separately to avoid sudden jumps in RViz.
        try:
            vx, vy = self.viz_pose.get(robot_id, (px, py))
            # Detect large jump for logging
            jump = math.hypot(px - vx, py - vy)
            if jump > self.viz_jump_threshold:
                rid_str = self._color_robot(robot_id)
                # self.get_logger().warn(
                #     f"[viz] {rid_str} large pose jump detected: {jump:.2f}m (using smoothing for RViz)"
                # )
                # Immediately set viz pose to current to avoid long interpolation when jump is expected
                vx, vy = px, py
            else:
                # exponential smoothing
                alpha = float(self.viz_alpha)
                vx = alpha * px + (1.0 - alpha) * vx
                vy = alpha * py + (1.0 - alpha) * vy
            self.viz_pose[robot_id] = (vx, vy)
        except Exception:
            self.viz_pose[robot_id] = (px, py)

    def _soc_cb(self, robot_id: str, msg: Float32) -> None:
        state = self.robot_states.get(robot_id)
        if state is None:
            return
        # Treat incoming SOC as an absolute energy in Wh, if the external
        # source publishes it that way. Otherwise you may keep using the
        # internal odom-based model only.
        state.soc = float(msg.data)

    def _odom_cb(self, robot_id: str, msg: Odometry) -> None:
        """Decrease SOC based on distance travelled from odom.

        This keeps the energy model local to the scheduler package, without
        depending on external multi_robot_energy logic.
        """
        state = self.robot_states.get(robot_id)
        if state is None:
            return

        try:
            x = float(msg.pose.pose.position.x)
            y = float(msg.pose.pose.position.y)
        except Exception:
            return

        prev = self._last_odom.get(robot_id)
        self._last_odom[robot_id] = (x, y)

        if prev is None:
            # first odom sample, just seed
            return

        px, py = prev
        dist = math.hypot(x - px, y - py)
        if dist <= 1e-6:
            return

        # Simple per-robot energy model: Wh/m and capacity
        scaling_factor = 10.0  # could be adjusted per robot type
        drive_wh_per_m = 0.0275 * scaling_factor
        cap_wh = self.robot_cap_wh.get(robot_id, 120.0)
        dwh = drive_wh_per_m * dist

        # Apply decrease locally in Wh, but never below 0.0
        state.soc = max(0.0, state.soc - dwh)


    def _health_check_timer_cb(self) -> None:
        """Periodic health check to detect robots that never started.

        After a grace period from startup, mark robots as "dead" if they
        have a non-empty offline sequence but have never emitted a
        task_started event (robot_start_time is still None). This runs once
        effectively when the first non-empty current_sequences are ready.
        """
        # If we've already logged dead robots once, no need to repeat.
        if getattr(self, "health_check_dead_logged", False):
            return

        dead = []
        alive = []

        for rid in self.robot_ids:
            seq = self.current_sequences.get(rid, [])
            start = self.robot_start_time.get(rid)
            if not seq:
                continue
            if start is None:
                self.dead_robots[rid] = True
                dead.append(rid)
            else:
                alive.append(rid)

        if not dead:
            return

        self.health_check_dead_logged = True
        dead_str = ", ".join(dead)
        alive_str = ", ".join(alive) if alive else "none"
        self.get_logger().warn(
            f"[health] Robots that never started any task within 10s and will "
            f"be excluded from makespan stats: [{dead_str}]; alive robots: [{alive_str}]"
        )

    def _color_robot(self, robot_id: str) -> str:
        """Return robot_id wrapped in an ANSI color code for logs.

        If the terminal does not support ANSI colors, the escape codes will
        simply be shown as-is or ignored by the viewer. This is purely a
        cosmetic helper for log readability.
        """
        color = self.robot_color_codes.get(robot_id, "")
        reset = self.color_reset if color else ""
        return f"{color}{robot_id}{reset}" if color else robot_id

    def _task_event_cb(self, robot_id: str, msg: String) -> None:
        state = self.robot_states.get(robot_id)
        if state is None:
            return

        raw = msg.data.strip()
        parts = raw.split(":")

        base = parts[0].lower()   # "task_started"

        task_id_val = None
        wp_idx = None

        if len(parts) >= 2:
            try:
                task_id_val = int(parts[1])
            except Exception:
                task_id_val = None

        if len(parts) >= 3:
            try:
                wp_idx = int(parts[2])
            except Exception:
                wp_idx = None


        if base == 'task_started':
            state.available = False
            # Record the first time this robot starts any task in its
            # offline sequence as the sequence start time (wall-clock).
            if self.robot_start_time.get(robot_id) is None:
                self.robot_start_time[robot_id] = self.get_clock().now()
            # If this is a CHARGE task, mark the corresponding charger as occupied
            # by this robot (add to its queue).

            if task_id_val == -1:
                # Find the active CHARGE task for this robot (if any)
                active_seq = self.current_sequences.get(robot_id, [])
                for t in active_seq:
                    if t.task_type == "CHARGE":
                        # Determine which charger this task is targeting
                        target = None
                        for ch in self.chargers:
                            if abs(ch.x - t.drop_x) < 1e-3 and abs(ch.y - t.drop_y) < 1e-3:
                                target = ch
                                break
                        if target is not None:
                            q = self.charger_queues.setdefault(target.charger_id, [])
                            if robot_id not in q:
                                q.append(robot_id)
                            rid_str = self._color_robot(robot_id)
                            self.get_logger().info(
                                f"[charge] {rid_str} started CHARGE at charger {target.charger_id}, queue={q}"
                            )
                        break
            rid_str = self._color_robot(robot_id)
            self.get_logger().info(
                f"[scheduler] {rid_str} event='{raw}' -> set available=False"
            )
            # If a PICK_DROP task started, track its id and clear carrying
            if task_id_val is not None and task_id_val != -1:
                self.current_task_id[robot_id] = task_id_val
                # reset retry counter when a task actually starts
                self.retry_counts[robot_id] = 0
            # Cancel pending start timeout for this robot (we received start)
            try:
                if robot_id in self.pending_task_start:
                    del self.pending_task_start[robot_id]
            except Exception:
                pass
        elif base == 'task_done':
            state.available = True
            # If this was a CHARGE task, free the charger (remove from queue)
            # and apply the SOC increase corresponding to charging.
            if ':' in raw:
                try:
                    _, tid_str = raw.split(':', 1)
                    task_id_val = int(tid_str)
                except Exception:
                    task_id_val = None
            else:
                task_id_val = None

            if task_id_val == -1:
                # Remove this robot from all charger queues where it appears.
                for cid, q in self.charger_queues.items():
                    if robot_id in q:
                        q.remove(robot_id)
                        # Apply charging effect: up to 20 seconds at +5% of
                        # full capacity per second, capped at full capacity.
                        # This mirrors the stage-2 optimizer's assumption
                        # that a single CHARGE action can at most bring the
                        # robot from its current SOC to full within 20s.
                        cap_wh = self.robot_cap_wh.get(robot_id, 120.0)
                        gain_per_sec = 0.05 * cap_wh
                        max_dur = 20.0
                        # Compute remaining fraction of capacity until full.
                        remaining_frac = max(0.0, (cap_wh - state.soc) / cap_wh)
                        time_to_full = remaining_frac / 0.05 if remaining_frac > 0.0 else 0.0
                        t_charge = min(max_dur, max(0.0, time_to_full))
                        total_gain = gain_per_sec * t_charge
                        before = state.soc
                        state.soc = min(cap_wh, state.soc + total_gain)
                        rid_str = self._color_robot(robot_id)
                        self.get_logger().info(
                            f"[charge] {rid_str} finished CHARGE at charger {cid}, "
                            f"+{total_gain:.2f}Wh over {t_charge:.1f}s ({before:.2f}Wh -> {state.soc:.2f}Wh), queue={q}"
                        )
            # If this was the last task in the robot's offline sequence,
            # record its finish time and, once all *alive* robots have
            # finished, log the real makespan. Robots that never started
            # any task within the health check window are treated as dead
            # and excluded from these stats.
            seq = self.current_sequences.get(robot_id, [])
            idx = self.current_seq_index.get(robot_id, 0)
            # Debug: report sequence/index/start/finish to help diagnose why
            # the per-robot finish logging below may not run.
            try:
                seq_ids_dbg = [t.task_id for t in seq]
            except Exception:
                seq_ids_dbg = []
            self.get_logger().info(
                f"[finish-check] {robot_id}: seq_len={len(seq_ids_dbg)}, idx={idx}, start={self.robot_start_time.get(robot_id)}, finish={self.robot_finish_time.get(robot_id)}, dead={self.dead_robots.get(robot_id)}"
            )
            if seq and idx >= len(seq):
                # Stamp finish time only once.
                if self.robot_finish_time.get(robot_id) is None:
                    finish_time = self.get_clock().now()
                    self.robot_finish_time[robot_id] = finish_time

                    # As soon as this robot finishes its own sequence,
                    # log its individual makespan (start->finish).
                    start_time = self.robot_start_time.get(robot_id)
                    if start_time is not None:
                        elapsed = (finish_time - start_time).nanoseconds / 1e9
                        rid_str = self._color_robot(robot_id)
                        self.get_logger().info(
                            f"[makespan] {rid_str} completed its sequence in {elapsed:.1f} s"
                        )

                # Check if all robots with non-empty sequences have finished.
                all_done = True
                for rid in self.robot_ids:
                    seq_r = self.current_sequences.get(rid, [])
                    # Skip robots with no sequence or marked as dead.
                    if not seq_r or self.dead_robots.get(rid, False):
                        continue
                    if self.robot_finish_time.get(rid) is None:
                        all_done = False
                        break

                if all_done:
                    t0 = None
                    max_t = None
                    # Use the earliest start and latest finish across
                    # robots that actually participated (non-dead with
                    # non-empty sequences).
                    for rid in self.robot_ids:
                        start = self.robot_start_time.get(rid)
                        finish = self.robot_finish_time.get(rid)
                        seq_r = self.current_sequences.get(rid, [])
                        if (
                            not seq_r
                            or self.dead_robots.get(rid, False)
                            or start is None
                            or finish is None
                        ):
                            continue
                        if t0 is None or start < t0:
                            t0 = start
                        if max_t is None or finish > max_t:
                            max_t = finish

                    if t0 is not None and max_t is not None:
                        # Debug: log the earliest start and latest finish timestamps
                        self.get_logger().debug(f"[makespan] t0={t0}, max_t={max_t}")
                        makespan_sec = (max_t - t0).nanoseconds / 1e9
                        # Per-robot finish times (excluding dead robots)
                        for rid in self.robot_ids:
                            start = self.robot_start_time.get(rid)
                            finish = self.robot_finish_time.get(rid)
                            seq_r = self.current_sequences.get(rid, [])
                            if (
                                not seq_r
                                or self.dead_robots.get(rid, False)
                                or start is None
                                or finish is None
                            ):
                                continue
                            elapsed = (finish - start).nanoseconds / 1e9
                            self.get_logger().info(
                                f"Robot {rid} finished at t = {elapsed:.1f} s (real)"
                            )

                        self.get_logger().info(
                            f"All robots completed tasks at t = {makespan_sec:.1f} s (makespan, real)"
                        )
                        # Mark that makespan has been reported so we can stop
                        # noisy periodic SOC logs.
                        self.makespan_reported = True
            rid_str = self._color_robot(robot_id)
            self.get_logger().info(
                f"[scheduler] {rid_str} event='{raw}' -> set available=True"
            )
            # Clear carrying and active task when a PICK_DROP completes
            if task_id_val is not None and task_id_val != -1:
                self.current_task_id[robot_id] = None
                self.retry_counts[robot_id] = 0
            # Clear any pending start marker in case task_done arrived
            try:
                if robot_id in self.pending_task_start:
                    del self.pending_task_start[robot_id]
            except Exception:
                pass
        elif base == 'task_failed':
            # When a task fails, attempt to retry it from the robot's
            # current position instead of marking the robot dead. We:
            #  - parse the failed task_id (if present),
            #  - find the TaskSpec in the robot's current sequence,
            #  - create a TaskCommand with the pick_pose set to the
            #    robot's current pose (so the robot continues from the
            #    failure point), and publish it immediately.
            #  - adjust the sequence index so the retry is reflected
            #    in the offline-sequence bookkeeping.

                
            failed_tid = task_id_val

            rid_str = self._color_robot(robot_id)

            seq = self.current_sequences.get(robot_id, [])
            failed_task = None
            if failed_tid is not None and seq:
                for t in seq:
                    if t.task_id == failed_tid:
                        failed_task = t
                        break
            
            # If event didn't include an explicit task id, try falling
            # back to the task currently recorded as active for this robot.
            if failed_task is None:
                cur_tid = self.current_task_id.get(robot_id)
                if cur_tid is not None and seq:
                    for t in seq:
                        if t.task_id == cur_tid:
                            failed_task = t
                            failed_tid = cur_tid
                            self.get_logger().info(
                                f"[scheduler] {self._color_robot(robot_id)} task_failed: no explicit id, using current_task_id={cur_tid} for retry"
                            )
                            break

            # If we found the failed TaskSpec in the current sequence,
            # craft a retry command. Use the deterministic internal
            # waypoint with the robot's current pose. Additionally handle
            # CHARGE-task failures specially using retry counters.
            pub = self.task_pubs.get(robot_id)
            if pub is None:
                state.available = True
                self.get_logger().warn(
                    f"[scheduler] {rid_str} task_failed event '{raw}': no publisher for robot"
                )
            else:
                # CHARGE failure handling (task_id == -1 or TaskSpec.task_type == 'CHARGE')
                is_charge_fail = (failed_tid == -1) or (failed_task is not None and failed_task.task_type == "CHARGE")

                if is_charge_fail:
                    # Remove robot from any charger queues it may appear in
                    for cid, q in self.charger_queues.items():
                        if robot_id in q:
                            try:
                                q.remove(robot_id)
                            except ValueError:
                                pass

                    # Enforce retry limit
                    rc = self.retry_counts.get(robot_id, 0)
                    if rc >= self.retry_limit:
                        self.dead_robots[robot_id] = True
                        state.available = False
                        self.get_logger().warn(
                            f"[scheduler] {rid_str} CHARGE failed repeatedly (retries={rc}) -> marking dead"
                        )
                    else:
                        # Build and send a fresh CHARGE TaskCommand to nearest available charger
                        try:
                            ch = self._nearest_available_charger(state.x, state.y)
                        except Exception:
                            ch = None

                        if ch is None:
                            ch = self._nearest_charger(state.x, state.y)

                        if ch is not None:
                            charge_task = TaskSpec(
                                task_id=-1,
                                task_type="CHARGE",
                                picks=[(state.x, state.y)],
                                drops=[(ch.x, ch.y)],
                                pick_wait_s=0.0,
                                drop_wait_s=0.0,
                                charge_duration_s=20.0,
                            )
                            cmd = self._make_task_command_from_spec(charge_task)
                            pub.publish(cmd)
                            # Reserve the charger in our local queue so other
                            # robots won't be sent to the same charger while
                            # this retry is pending.
                            q = self.charger_queues.setdefault(ch.charger_id, [])
                            if robot_id not in q:
                                q.append(robot_id)
                            state.available = False
                            self.current_task_id[robot_id] = -1
                            self.retry_counts[robot_id] = rc + 1
                            self.get_logger().warn(
                                f"[scheduler] {rid_str} CHARGE failed for id={failed_tid}: retrying at charger {ch.charger_id} (retry {rc+1}/{self.retry_limit}), queue={q}"
                            )
                        else:
                            state.available = True
                            self.get_logger().warn(
                                f"[scheduler] {rid_str} CHARGE failed for id={failed_tid}: no charger available to retry"
                            )
                else:
                    # PICK_DROP (or other) failure: resend starting from the
                    # point nearest the robot so we resume rather than restart.
                    if failed_task is not None:
                        cmd = self._make_task_command_from_spec(failed_task)

                        waypoints = getattr(cmd, 'waypoints', []) or []
                        waits = getattr(cmd, 'waits', []) or []
                        start_idx = wp_idx if wp_idx is not None else 0

                        if not waypoints:
                            state.available = True
                            self.get_logger().warn(
                                f"[scheduler] {rid_str} task_failed for id={failed_tid}: no waypoints to retry"
                            )
                        else:
                            if start_idx >= len(waypoints):
                                state.available = True
                                self.get_logger().warn(
                                    f"[scheduler] {rid_str} task_failed for id={failed_tid}: nothing remaining to retry (start_idx={start_idx})"
                                )
                            else:
                                # build resumed command starting from ordered stage
                                resumed = TaskCommand()
                                resumed.task_id = cmd.task_id
                                resumed.task_type = cmd.task_type

                                new_waypoints = waypoints[start_idx:]
                                new_waits = [float(w) for w in waits[start_idx:]]

                                resumed.waypoints = new_waypoints
                                resumed.waits = new_waits
                                resumed.charge_duration_s = float(getattr(cmd, 'charge_duration_s', 0.0))

                                pub.publish(resumed)
                                state.available = False
                                self.get_logger().warn(
                                    f"[scheduler] {rid_str} task_failed for id={failed_tid}: resumed at ordered stage {start_idx})"
                                )
                    else:
                        state.available = True
                        self.get_logger().warn(
                            f"[scheduler] {rid_str} task_failed event '{raw}': could not find matching task to retry"
                        )
            # end of task_failed handling
        else:
            rid_str = self._color_robot(robot_id)
            self.get_logger().warn(
                f"[scheduler] {rid_str} received unknown task_event '{raw}', keeping available={state.available}"
            )

    def add_task(self, task: TaskSpec) -> None:
        self.pending_tasks.append(task)
        # Keep alias in sync
        self.task_queue = self.pending_tasks

    # ------------------------------------------------------------------
    # Global task request interface
    # ------------------------------------------------------------------

    def _on_global_task_request(self, msg: String) -> None:
        """Parse a simple CSV string and enqueue a new TaskSpec.

        Expected format:
            "task_id,task_type,pick_x,pick_y,drop_x,drop_y"

        Example:
            "10,PICK_DROP,-1.0,0.0,2.0,3.0"
        """
        text = msg.data.strip()
        if not text:
            self.get_logger().warning('Received empty global_task_request string')
            return

        parts = [p.strip() for p in text.split(',')]
        if len(parts) != 6:
            self.get_logger().warning(
                f"Invalid global_task_request format (expected 6 fields): '{text}'"
            )
            return

        try:
            task_id = int(parts[0])
            task_type = parts[1]
            pick_x = float(parts[2])
            pick_y = float(parts[3])
            drop_x = float(parts[4])
            drop_y = float(parts[5])
        except (ValueError, IndexError) as exc:
            self.get_logger().warning(
                f"Failed to parse global_task_request '{text}': {exc}"
            )
            return

            task = TaskSpec(
                task_id=task_id,
                task_type=task_type,
                picks=[(pick_x, pick_y)],
                drops=[(drop_x, drop_y)],
                pick_wait_s=2.0,
                drop_wait_s=2.0,
                charge_duration_s=0.0,
            )

        self.add_task(task)
        self.get_logger().info(
            f"Queued new task from /global_task_request: id={task_id}, "
            f"type={task_type}, pick=({pick_x:.2f},{pick_y:.2f}), "
            f"drop=({drop_x:.2f},{drop_y:.2f})"
        )

    def _soc_log_timer_cb(self) -> None:
        """Log each robot's SOC and availability every 10 seconds."""
        # After makespan is reported, stop emitting SOC logs to keep output clean.
        if getattr(self, "makespan_reported", False):
            return
        parts = []
        for rid in self.robot_ids:
            state = self.robot_states.get(rid)
            if state is None:
                continue
            cap_wh = self.robot_cap_wh.get(rid, 120.0)
            pct = (state.soc / cap_wh * 100.0) if cap_wh > 0.0 else 0.0
            rid_str = self._color_robot(rid)
            parts.append(
                f"{rid_str}={state.soc:.2f}Wh ({pct:.1f}%) avail={state.available}"
            )
        if parts:
            self.get_logger().info('[scheduler] Robot SOCs: ' + ', '.join(parts))

    def _goal_viz_timer_cb(self) -> None:
        """Continuously publish robot poses, current goals, and chargers on /goal_pose_debug.

        - tb1~tb4 are shown with distinct colors:
            tb1=red, tb2=orange, tb3=yellow, tb4=green
        - Current robot pose: solid sphere
        - Current goal (if any) - larger, semi-transparent sphere
        - Chargers: black squares at fixed positions
        """
        if self.goal_debug_pub is None:
            return

        now = self.get_clock().now().to_msg()
        markers: List[Marker] = []

        # Per-robot color map (RGB)
        color_map = {
            'tb1': (1.0, 0.0, 0.0),   # red
            'tb2': (1.0, 0.5, 0.0),   # orange
            'tb3': (1.0, 1.0, 0.0),   # yellow
            'tb4': (0.0, 1.0, 0.0),   # green
        }

        # Robot pose & goal markers
        for rid in self.robot_ids:
            state = self.robot_states.get(rid)
            if state is None:
                continue

            r = color_map.get(rid, (1.0, 1.0, 1.0))

            # Current robot pose marker (small solid sphere)
            pose_marker = Marker()
            pose_marker.header.frame_id = 'map'
            pose_marker.header.stamp = now
            pose_marker.ns = f'robot_pos_{rid}'
            pose_marker.id = 0
            pose_marker.type = Marker.SPHERE
            pose_marker.action = Marker.ADD
            # Use smoothed viz pose to avoid teleporting in RViz
            vx, vy = self.viz_pose.get(rid, (state.x, state.y))
            pose_marker.pose.position.x = vx
            pose_marker.pose.position.y = vy
            pose_marker.pose.position.z = 0.05
            pose_marker.pose.orientation.w = 1.0
            pose_marker.scale.x = 0.2
            pose_marker.scale.y = 0.2
            pose_marker.scale.z = 0.2
            pose_marker.color.r = float(r[0])
            pose_marker.color.g = float(r[1])
            pose_marker.color.b = float(r[2])
            pose_marker.color.a = 1.0
            pose_marker.lifetime.sec = 1
            markers.append(pose_marker)

            # Current goal marker (if any) - larger, semi-transparent sphere
            goal = self.current_goals.get(rid)
            if goal is not None:
                gx, gy = goal
                goal_marker = Marker()
                goal_marker.header.frame_id = 'map'
                goal_marker.header.stamp = now
                goal_marker.ns = f'goal_{rid}'
                goal_marker.id = 0
                goal_marker.type = Marker.SPHERE
                goal_marker.action = Marker.ADD
                goal_marker.pose.position.x = gx
                goal_marker.pose.position.y = gy
                goal_marker.pose.position.z = 0.0
                goal_marker.pose.orientation.w = 1.0
                goal_marker.scale.x = 0.3
                goal_marker.scale.y = 0.3
                goal_marker.scale.z = 0.3
                goal_marker.color.r = float(r[0])
                goal_marker.color.g = float(r[1])
                goal_marker.color.b = float(r[2])
                goal_marker.color.a = 0.5
                goal_marker.lifetime.sec = 1
                markers.append(goal_marker)

        # Charger markers (black squares)
        for charger in self.chargers:
            m = Marker()
            m.header.frame_id = 'map'
            m.header.stamp = now
            m.ns = 'chargers'
            m.id = charger.charger_id
            m.type = Marker.CUBE
            m.action = Marker.ADD
            m.pose.position.x = charger.x
            m.pose.position.y = charger.y
            m.pose.position.z = 0.0
            m.pose.orientation.w = 1.0
            m.scale.x = 0.5
            m.scale.y = 0.5
            m.scale.z = 0.1
            m.color.r = 0.0
            m.color.g = 0.0
            m.color.b = 0.0
            m.color.a = 1.0
            m.lifetime.sec = 0  # persistent while republished
            markers.append(m)

        arr = MarkerArray()
        arr.markers = markers
        self.goal_debug_pub.publish(arr)

    # ------------------------------------------------------------------
    # Task allocation helper (random for now, optimizer-ready later)
    # ------------------------------------------------------------------

    def _optimize_task_sequence(self, robot_state: RobotState, candidate_tasks: List[TaskSpec]) -> List[TaskSpec]:
        """Use mealpy to find a good ordering of candidate_tasks for one robot.

        This uses a random-key encoding: each task gets a real-valued key,
        and sorting by that key defines the sequence. The objective follows
        J = w_t * T_total + w_e * E_total + penalty with a 20%% SOC
        feasibility constraint, using estimate_task_energy (no harvesting).
        """

        K = len(candidate_tasks)
        if K == 0:
            return []

        cap_wh = self.robot_cap_wh.get(robot_state.robot_id, 120.0)

        # makespan-dominant 가중치 (w_t > w_e)
        w_t = 1.0
        w_e = 0.1

        # ---------- objective ----------
        def _objective_function(solution: np.ndarray) -> float:
            # 1) random-key 디코딩: 실수 key 로 정렬해서 순서를 만든다.
            keys = solution
            ordered = sorted(zip(keys, candidate_tasks), key=lambda x: x[0])
            seq = [t for _, t in ordered]

            # 에너지 상태는 Wh 로 해석
            energy = robot_state.soc

            # 로봇 현재 pose 기준으로 첫 task 로 간다고 가정
            sim_robot = RobotState(
                robot_id=robot_state.robot_id,
                namespace=robot_state.namespace,
                x=robot_state.x,
                y=robot_state.y,
                soc=robot_state.soc,
                available=robot_state.available,
            )

            total_energy = 0.0  # Σ 에너지 소비
            total_time = 0.0    # Σ 시간 (충전 포함)
            penalty = 0.0

            for t in seq:
                # ----- 1) 현재 prev_pose 에서 이 task 를 바로 수행 가능 여부 확인 (충전 결정) -----
                # prev_pose -> pick 으로 가는 segment 를 별도 TaskSpec 으로 근사해 에너지/시간 계산
                try:
                    required_energy, t_to_pick = self.estimate_task_energy(
                        RobotState(
                            robot_id=sim_robot.robot_id,
                            namespace=sim_robot.namespace,
                            x=sim_robot.x,
                            y=sim_robot.y,
                            soc=energy,
                            available=sim_robot.available,
                        ),
                        t,
                    )
                except Exception as e:
                    required_energy, t_to_pick = 0.0, 0.0
                    self.get_logger().warning(f"Failed to estimate energy/time for task {t.task_id} in objective function {e}")

                if energy < required_energy:
                    # ----- 2) task 수행 전에 충전 필요 → 충전 이벤트 삽입 -----
                    try:
                        charger = self._nearest_available_charger(sim_robot.x, sim_robot.y)
                    except Exception:
                        charger = None

                    if charger is None:
                        # 어떤 charger 도 선택할 수 없으면 infeasible 로 큰 penalty
                        penalty += 1e6
                        break

                    # prev_pose -> charger 이동 에너지/시간
                    try:
                        e_to_charger, t_to_charger = self.estimate_task_energy(
                            RobotState(
                                robot_id=sim_robot.robot_id,
                                namespace=sim_robot.namespace,
                                x=sim_robot.x,
                                y=sim_robot.y,
                                soc=energy,
                                available=sim_robot.available,
                            ),
                            TaskSpec(
                                task_id=-1,
                                task_type="CHARGE",
                                picks=[(charger.x, charger.y)],
                                drops=[(charger.x, charger.y)],
                                pick_wait_s=0.0,
                                drop_wait_s=0.0,
                                charge_duration_s=0.0,
                            ),
                        )
                    except Exception:
                        e_to_charger, t_to_charger = 0.0, 0.0

                    if energy < e_to_charger:
                        # charger 에 도달할 에너지가 없으면 infeasible
                        penalty += 1e6
                        break

                    # charger 로 이동
                    energy -= e_to_charger
                    total_energy += e_to_charger
                    total_time += t_to_charger

                    # charger 에서 충전 시간 (에너지 수확은 capacity 로 reset)
                    charge_time = 10.0  # 고정 충전 시간 (필요시 파라미터화 가능)
                    total_time += charge_time
                    energy = cap_wh  # full capacity 로 리셋

                    # pose 를 charger 위치로 업데이트
                    sim_robot.x = charger.x
                    sim_robot.y = charger.y

                    # 충전 후, charger_pose 기준으로 다시 prev_pose->pick, task 실행 에너지/시간 계산
                    try:
                        required_energy, t_to_pick = self.estimate_task_energy(
                            RobotState(
                                robot_id=sim_robot.robot_id,
                                namespace=sim_robot.namespace,
                                x=sim_robot.x,
                                y=sim_robot.y,
                                soc=energy,
                                available=sim_robot.available,
                            ),
                            t,
                        )
                    except Exception:
                        required_energy, t_to_pick = 0.0, 0.0

                    if energy < required_energy:
                        # 충전 후에도 task 를 수행할 수 없으면 infeasible
                        penalty += 1e6
                        break

                # ----- 3) (필요시 충전 후) task 실제 실행 -----
                energy -= required_energy          # 에너지 감소
                total_energy += required_energy    # 누적 에너지 소비
                total_time += t_to_pick            # 누적 시간 (drive+task)

                # drop pose 로 로봇 위치 업데이트
                sim_robot.x = t.drop_x
                sim_robot.y = t.drop_y

            # J = w_t * T_total + w_e * E_total + penalty 최소화
            return w_t * total_time + w_e * total_energy + penalty

        # ---------- mealpy problem ----------
        problem = {
            "obj_func": _objective_function,
            "bounds": FloatVar(lb=(-1.0,) * K, ub=(1.0,) * K),
            "minmax": "min",
            "log_to": None,
        }

        model = SMA.OriginalSMA(epoch=300, pop_size=60, pr=0.03)
        best = model.solve(problem)

        # ---------- decode best ----------
        keys = best.solution
        ordered = sorted(zip(keys, candidate_tasks), key=lambda x: x[0])
        best_seq = [t for _, t in ordered]

        return best_seq


    def _maybe_issue_online_charge(self, rid: str, state: RobotState) -> bool:
        """Decide whether to send the robot to charge now (online decision).

        Returns True if a CHARGE task was issued and the caller should skip
        assigning a normal task this cycle. This is only active for
        "feasibility" and "threshold" charging strategies; in "optimized"
        mode, all CHARGE events are decided offline.
        """

        # Optimized mode: do not perform online charging overrides.
        if self.charging_strategy == "optimized":
            return False

        cap_wh = self.robot_cap_wh.get(rid, 120.0)
        if cap_wh <= 0.0:
            return False

        # Threshold-based policy: if SOC is below 15% of capacity, trigger
        # an immediate charge regardless of the next task.
        if self.charging_strategy == "threshold":
            soc_ratio = state.soc / cap_wh
            if soc_ratio >= 0.15:
                return False

        # For both "feasibility" and "threshold" (when SOC < threshold),
        # we attempt to send the robot to the nearest available charger.
        # Availability here is defined by having an empty queue so that
        # two robots are never assigned to the same physical charger at
        # the same time.
        try:
            charger = self._nearest_available_charger(state.x, state.y)
        except Exception:
            charger = None

        if charger is None:
            # No charger available; skip online charging this tick.
            return False

        # Build a one-off CHARGE TaskSpec to drive to this charger and
        # charge for a fixed duration (the actual SOC gain is applied in
        # the task_done handler for CHARGE tasks).
        charge_task = TaskSpec(
            task_id=-1,
            task_type="CHARGE",
            picks=[(charger.x, charger.y)],
            drops=[(charger.x, charger.y)],
            pick_wait_s=0.0,
            drop_wait_s=0.0,
            charge_duration_s=20.0,
        )

        # Update current goal for visualization.
        self.current_goals[rid] = (charger.x, charger.y)

        rid_str = self._color_robot(rid)
        self.get_logger().info(
            f"[online-charge] {rid_str} soc={state.soc:.3f}Wh cap={cap_wh:.1f}Wh "
            f"strategy={self.charging_strategy} -> send to charger {charger.charger_id}"
        )

        cmd = self._make_task_command_from_spec(charge_task)
        pub = self.task_pubs.get(rid)
        if pub is None:
            return False

        pub.publish(cmd)
        state.available = False

        # Reserve this charger immediately in the local queue so that
        # other robots searching for an available charger will avoid it
        # while this CHARGE task is in flight.
        q = self.charger_queues.setdefault(charger.charger_id, [])
        if rid not in q:
            q.append(rid)

        try:
            self.pending_task_start[rid] = self.get_clock().now().nanoseconds / 1e9
        except Exception:
            self.pending_task_start[rid] = time.time()

        self.current_task_id[rid] = charge_task.task_id
        self.retry_counts[rid] = 0

        return True


    def _timer_cb(self) -> None:
        # Offline execution: walk the pre-computed fixed sequence per robot.
        # No re-optimization or rolling-horizon allocation is performed here.

        any_assigned = False

        for rid in self.robot_ids:
            state = self.robot_states.get(rid)
            if state is None:
                continue
            if not state.available:
                # Helpful debug logging when robots are skipped due to
                # availability; this makes it easier to spot when all
                # robots become unavailable.
                self.get_logger().debug(f"[timer] skipping {rid}: available={state.available}, soc={state.soc:.2f}Wh")
                continue

            seq = self.current_sequences.get(rid, [])
            idx = self.current_seq_index.get(rid, 0)

            if not seq or idx >= len(seq):
                # No more tasks left in this robot's offline sequence.
                continue

            task_spec = seq[idx]

            # --- Online charging decisions ---
            # threshold: purely SOC-based (handled inside _maybe_issue_online_charge)
            # feasibility: check whether the next task is energetically
            # feasible from the current SOC; if not, attempt to charge.
            if self.charging_strategy == "feasibility":
                try:
                    required_energy, _ = self.estimate_task_energy(state, task_spec)
                except Exception:
                    required_energy = 0.0

                # Simple feasibility rule: if current SOC is insufficient
                # to execute the next task (no margin), try to charge.
                if state.soc < required_energy:
                    if self._maybe_issue_online_charge(rid, state):
                        any_assigned = True
                        continue
            else:
                # threshold (and any other online policy) is handled here.
                if self._maybe_issue_online_charge(rid, state):
                    any_assigned = True
                    continue

            self.current_seq_index[rid] = idx + 1
            source = 'offline-sequence'

            any_assigned = True

            # Update current goal for visualization.
            # For PICK_DROP we use the pick position; for CHARGE we
            # want the robot to actually move to the charger (drop pose).
            if task_spec.task_type == "CHARGE":
                self.current_goals[rid] = (task_spec.drop_x, task_spec.drop_y)
            else:
                self.current_goals[rid] = (task_spec.pick_x, task_spec.pick_y)

            # Log the sequence and current progress for this robot first,
            # including per-task estimated energy (Wh and %) and time (s).
            seq_specs = self.current_sequences.get(rid, [])
            seq_ids = [t.task_id for t in seq_specs]
            seq_str = ','.join(str(tid) for tid in seq_ids)
            current_pos = self.current_seq_index.get(rid, 0)

            # Build per-task energy/time estimates for the whole sequence using
            # the same accounting style as _maybe_insert_charge_into_sequence.
            cap_wh = self.robot_cap_wh.get(rid, 120.0)
            energy_vals: List[float] = []
            energy_pct: List[float] = []
            time_vals: List[float] = []

            # Start from the robot's current simulated state.
            sim_energy = state.soc
            sim_x, sim_y = state.x, state.y

            for t in seq_specs:
                if t.task_type in ("PICK_DROP", "MULTI_PICK_DROP", "DUAL_PICK_MULTI_DROP"):
                    # prev_pose -> pick segment energy/time
                    try:
                        e_total, t_total = self.estimate_task_energy(
                            RobotState(
                                robot_id=state.robot_id,
                                namespace=state.namespace,
                                x=sim_x,
                                y=sim_y,
                                soc=sim_energy,
                                available=state.available,
                            ),
                            t,
                        )
                    except Exception:
                        e_total, t_total = 0.0, 0.0

                    energy_vals.append(e_total)
                    pct = (e_total / cap_wh * 100.0) if cap_wh > 0.0 else 0.0
                    energy_pct.append(pct)
                    time_vals.append(t_total)

                    # advance simulated pose and energy as in the charge helper
                    sim_x, sim_y = t.drop_x, t.drop_y
                    sim_energy = max(0.0, sim_energy - e_total)
                else:
                    # CHARGE or other unknown types: treat as pure wait (no drive energy here),
                    # and keep energy constant. Duration is charge_duration_s.
                    energy_vals.append(0.0)
                    energy_pct.append(0.0)
                    time_vals.append(float(getattr(t, 'charge_duration_s', 0.0)))

            energy_wh_str = ','.join(f"{e:.3f}" for e in energy_vals)
            energy_pct_str = ','.join(f"{p:.1f}" for p in energy_pct)
            time_str = ','.join(f"{ts:.1f}" for ts in time_vals)

            rid_str = self._color_robot(rid)
            self.get_logger().info(
                f"[jobs] {rid_str} soc={state.soc:.3f}Wh {source}: "
                f"sequence=[{seq_str}] "
                f"energy(Wh)=[{energy_wh_str}] "
                f"time(s)=[{time_str}] "
                f"progress={current_pos}/{len(seq_ids)} -> start task {task_spec.task_id}"
            )

            cmd = self._make_task_command_from_spec(task_spec)
            pub = self.task_pubs.get(rid)
            if pub is None:
                continue

            pub.publish(cmd)
            state.available = False
            # record that we are waiting for a task_started event
            try:
                self.pending_task_start[rid] = self.get_clock().now().nanoseconds / 1e9
            except Exception:
                self.pending_task_start[rid] = time.time()
            # Record that this task is now executing on the robot and the
            # robot is not carrying yet (will be updated from odom/events).
            self.current_task_id[rid] = task_spec.task_id
            # reset retry counter for fresh task
            self.retry_counts[rid] = 0

        # Optionally, we could log here when all robots have finished their
        # offline sequences, but no additional scheduling is performed.

    def _pending_start_check(self) -> None:
        """Periodically check for published tasks that never started.

        If a task was published but no `task_started` event arrived within
        `start_timeout_s`, revert the robot to available and move the
        sequence index back so the task can be retried.
        """
        now = None
        try:
            now = self.get_clock().now().nanoseconds / 1e9
        except Exception:
            now = time.time()

        to_clear = []
        for rid, ts in list(self.pending_task_start.items()):
            age = now - ts
            if age >= self.start_timeout_s:
                state = self.robot_states.get(rid)
                if state is None:
                    to_clear.append(rid)
                    continue
                # If robot is marked dead, skip recovery here.
                if self.dead_robots.get(rid, False):
                    to_clear.append(rid)
                    continue

                # Move seq index back so the same task will be retried.
                idx = self.current_seq_index.get(rid, 0)
                if idx > 0:
                    self.current_seq_index[rid] = max(0, idx - 1)

                state.available = True
                rid_str = self._color_robot(rid)
                self.get_logger().warn(
                    f"[scheduler] {rid_str} did not emit task_started (age={age:.1f}s) -> reverting available=True and will retry"
                )
                to_clear.append(rid)

        for rid in to_clear:
            try:
                del self.pending_task_start[rid]
            except Exception:
                pass

    def _make_task_command_from_spec(self, spec: TaskSpec) -> TaskCommand:
        msg = TaskCommand()
        msg.task_id = spec.task_id
        msg.task_type = spec.task_type

        waypoints: List[PoseStamped] = []
        waits: List[float] = []

        if spec.task_type == "CHARGE":
            # single waypoint: charger
            ps = PoseStamped()
            ps.header.frame_id = 'map'
            ps.pose.position.x = float(spec.drop_x)
            ps.pose.position.y = float(spec.drop_y)
            waypoints.append(ps)
            waits.append(float(spec.charge_duration_s))
        elif spec.task_type == "PICK_DROP":
            p = PoseStamped()
            p.header.frame_id = 'map'
            p.pose.position.x = float(spec.pick_x)
            p.pose.position.y = float(spec.pick_y)
            waypoints.append(p)
            waits.append(float(spec.pick_wait_s))

            d = PoseStamped()
            d.header.frame_id = 'map'
            d.pose.position.x = float(spec.drop_x)
            d.pose.position.y = float(spec.drop_y)
            waypoints.append(d)
            waits.append(float(spec.drop_wait_s))
        elif spec.task_type == "MULTI_PICK_DROP":
            # MULTI_PICK_DROP or other multi-stage: concatenate picks then drops
            # Collapse multiple picks at the same coordinates into a single
            # waypoint with aggregated wait (pick_wait_s * count). Preserve
            # the original order of distinct locations.
            last = None
            count = 0
            for (px, py) in spec.picks:
                if last is None:
                    last = (float(px), float(py))
                    count = 1
                elif (float(px), float(py)) == last:
                    count += 1
                else:
                    ps = PoseStamped()
                    ps.header.frame_id = 'map'
                    ps.pose.position.x = last[0]
                    ps.pose.position.y = last[1]
                    waypoints.append(ps)
                    waits.append(float(spec.pick_wait_s) * count)
                    last = (float(px), float(py))
                    count = 1

            # Same collapsing for drops
            last = None
            count = 0
            for (dx, dy) in spec.drops:
                if last is None:
                    last = (float(dx), float(dy))
                    count = 1
                elif (float(dx), float(dy)) == last:
                    count += 1
                else:
                    ds = PoseStamped()
                    ds.header.frame_id = 'map'
                    ds.pose.position.x = last[0]
                    ds.pose.position.y = last[1]
                    waypoints.append(ds)
                    waits.append(float(spec.drop_wait_s) * count)
                    last = (float(dx), float(dy))
                    count = 1
        
        elif spec.task_type == "DUAL_PICK_MULTI_DROP":
            for i in range(len(spec.picks)):
                pick = spec.picks[i]
                drop = spec.drops[i]

                ps = PoseStamped()
                ps.header.frame_id = 'map'
                ps.pose.position.x = float(pick[0])
                ps.pose.position.y = float(pick[1])
                waypoints.append(ps)
                waits.append(float(spec.pick_wait_s))

                ds = PoseStamped()
                ds.header.frame_id = 'map'
                ds.pose.position.x = float(drop[0])
                ds.pose.position.y = float(drop[1])
                waypoints.append(ds)
                waits.append(float(spec.drop_wait_s))
        else:
            # Unknown task type: treat as simple pick-drop with single pick/drop
            p = PoseStamped()
            p.header.frame_id = 'map'
            p.pose.position.x = float(spec.pick_x)
            p.pose.position.y = float(spec.pick_y)
            waypoints.append(p)
            waits.append(float(spec.pick_wait_s))

            d = PoseStamped()
            d.header.frame_id = 'map'
            d.pose.position.x = float(spec.drop_x)
            d.pose.position.y = float(spec.drop_y)
            waypoints.append(d)
            waits.append(float(spec.drop_wait_s))
        # assign to message
        msg.waypoints = waypoints
        msg.waits = [float(w) for w in waits]
        # keep legacy field for compatibility
        msg.charge_duration_s = float(spec.charge_duration_s)
        return msg


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SchedulerNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
