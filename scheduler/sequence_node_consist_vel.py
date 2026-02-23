from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass
import copy
import math
import random
import time
import yaml
import os
from datetime import datetime

import rclpy
from rclpy.node import Node
from rclpy import time as rclpy_time
from rclpy import duration as rclpy_duration
from rcl_interfaces.msg import ParameterDescriptor, ParameterType

from std_msgs.msg import String, Float32, Empty
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped, TransformStamped
from nav_msgs.msg import Odometry, Path
from multi_robot_msgs.msg import TaskCommand, TaskExecution
from visualization_msgs.msg import Marker, MarkerArray
from rclpy.action import ActionClient, GoalResponse
from nav2_msgs.action import ComputePathToPose, NavigateToPose
from nav2_msgs.srv import ClearEntireCostmap
from rclpy.qos import QoSPresetProfiles
import tf2_ros
from tf2_ros import TransformException


import numpy as np
from mealpy import FloatVar
from mealpy.bio_based import SMA, VCS, TSA
from mealpy.evolutionary_based import GA, CRO
from mealpy.swarm_based import PSO, JA, SRSR, BES
from mealpy.human_based import WarSO, QSA
from mealpy.music_based import HS
from mealpy.physics_based import FLA, RIME, ESO
from mealpy.system_based import WCA
from mealpy.math_based import CGO


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
    carried_weight: float = 0.0  # weight of items being carried (kg)


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


def _calculate_path_length(path: Path) -> float:
    """Calculate total length of a nav_msgs/Path by summing distances between consecutive poses.
    
    Note: Nav2 paths may appear as curves when visualized/interpolated, but the Path message
    contains discrete poses. This function sums the Euclidean distances between consecutive
    poses, which is a good approximation of the actual path length when poses are sufficiently
    dense (typically 10-50cm spacing). This is the standard method used in Nav2 benchmarking.
    
    For more accuracy with sparse paths, spline interpolation could be used, but for typical
    Nav2 paths with dense pose spacing, this method is sufficient and computationally efficient.
    """
    if not path.poses or len(path.poses) < 2:
        return 0.0
    
    total_length = 0.0
    for i in range(1, len(path.poses)):
        p1 = path.poses[i-1].pose.position
        p2 = path.poses[i].pose.position
        dx = p1.x - p2.x
        dy = p1.y - p2.y
        total_length += math.hypot(dx, dy)
    
    return total_length


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

        # Get robot_ids from parameter
        # ROS2 has issues with empty list [] as default for string arrays (infers BYTE_ARRAY).
        # Use ParameterDescriptor to explicitly specify STRING_ARRAY type.
        param_descriptor = ParameterDescriptor(type=ParameterType.PARAMETER_STRING_ARRAY)
        # Provide a non-empty default to force STRING_ARRAY typing; parameter files override it.
        param_value = self.declare_parameter(
            'robot_ids',
            ['tb1'],
            param_descriptor,
        )
        robot_ids_param = param_value.get_parameter_value().string_array_value
        self.robot_ids: List[str] = [str(rid) for rid in robot_ids_param] if robot_ids_param else []

        if not self.robot_ids:
            self.get_logger().warn("No robot_ids provided! Please set robot_ids parameter.")

        self.get_logger().info(f"Initialized scheduler with {len(self.robot_ids)} robots: {self.robot_ids}")
        self.robot_states: Dict[str, RobotState] = {}
        self._last_odom: Dict[str, tuple] = {}
        self.pending_tasks: List[TaskSpec] = []
        self.task_queue: List[TaskSpec] = self.pending_tasks
        self.chargers: List[ChargerSpec] = []
        self.charger_queues: Dict[int, List[str]] = {}
        self.task_pubs: Dict[str, rclpy.publisher.Publisher] = {}
        # Task execution timeline publisher for visualization
        self.task_execution_pub = self.create_publisher(TaskExecution, '/task_execution_log', 10)
        # Optimization time publisher for metrics
        self.optimization_time_pub = self.create_publisher(Float32, '/optimization_time', 10)
        # Unified RViz debug marker publisher for all robots' current targets
        self.goal_debug_pub = self.create_publisher(MarkerArray, '/goal_pose_debug', 10)
        # Per-robot current goal (x, y) for visualization; None if no goal
        self.current_goals: Dict[str, tuple] = {rid: None for rid in self.robot_ids}
        # Separate smoothed pose used only for RViz visualization to avoid showing large jumps
        self.viz_pose: Dict[str, tuple] = {rid: (0.0, 0.0) for rid in self.robot_ids}
        self.viz_jump_threshold = 1.0
        self.viz_alpha = 0.4
        # Track whether a robot is currently carrying an object (True/False)
        self.robot_carrying: Dict[str, bool] = {rid: False for rid in self.robot_ids}
        # Task retry tracking
        self.retry_counts: Dict[str, int] = {rid: 0 for rid in self.robot_ids}
        self.retry_limit: int = 3
        # Per-robot current task sequence (queue of TaskSpec) and index into that sequence.
        # A robot must finish all tasks in its current sequence before receiving a new one.
        self.current_sequences: Dict[str, List[TaskSpec]] = {rid: [] for rid in self.robot_ids}
        self.current_seq_index: Dict[str, int] = {rid: 0 for rid in self.robot_ids}
        self.current_task_id: Dict[str, Optional[int]] = {rid: None for rid in self.robot_ids}
        # Tasks are now optimally assigned to robots (no job structure)
        # Real-time measurement of when each robot actually starts and
        # finishes executing its offline sequence (wall-clock).
        self.robot_start_time = {rid: None for rid in self.robot_ids}
        self.robot_finish_time = {rid: None for rid in self.robot_ids}
        # Robots that we consider "dead" for this run: spawned with a
        # non-empty sequence but never started any task within a grace
        # period. These will be excluded from makespan statistics.
        self.dead_robots: Dict[str, bool] = {rid: False for rid in self.robot_ids}

        # ANSI color codes for per-robot log coloring (for terminals that
        # support ANSI escape sequences). Supports up to 10 robots (tb1~tb10).
        self.robot_color_codes: Dict[str, str] = {
            'tb1': "\033[31m",   # red
            'tb2': "\033[33m",   # yellow / orange
            'tb3': "\033[93m",   # bright yellow
            'tb4': "\033[32m",   # green
            'tb5': "\033[34m",   # blue
            'tb6': "\033[35m",   # magenta
            'tb7': "\033[36m",   # cyan
            'tb8': "\033[91m",   # bright red
            'tb9': "\033[92m",   # bright green
            'tb10': "\033[94m",  # bright blue
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
        self.default_item_weight_kg = 1.0  # Fallback default
        # Task type specific weights (kg)
        # PICK_DROP: single item weight
        # DUAL_PICK_MULTI_DROP: (first_pick, second_pick) weights - round-trip task
        # MULTI_PICK_DROP: (first_pick, second_pick) weights
        self.task_type_weights = {
            "PICK_DROP": [8.0],
            "DUAL_PICK_MULTI_DROP": [8.0, 8.0],
            "MULTI_PICK_DROP": [8.0, 4.0],
        }
        # Multiplier per kg applied on top of loaded_factor when computing energy
        # e.g., energy_multiplier = loaded_factor + load_weight_factor * carried_kg
        self.load_weight_factor = 0.05

        # Acceleration/inertia-based energy consumption parameters
        # Base robot mass (kg) - affects inertia-based energy consumption
        # Higher mass = more energy needed to accelerate/decelerate
        self.base_robot_mass_kg = 4.4  # Fixed robot mass (4.4kg)
        # Store last velocity per robot for calculating acceleration energy
        self._last_velocity: Dict[str, Tuple[float, float]] = {}  # (vx, vy) in m/s
        
        # Charging rate (Wh/s) - absolute charging speed regardless of battery capacity
        # 0.1 Wh/s means: 40Wh battery takes 400s to fully charge, 60Wh takes 600s
        self.charge_rate_wh_per_s = 0.1  # 0.1 Wh/s charging speed
        
        # Static energy consumption (time-based)
        # Robots consume energy even when stationary (computers, sensors, electronics, idle motors)
        # This is independent of movement and proportional to time elapsed
        self.E_static_wh_per_s = 0.001  # Static power consumption in Wh/s (0.001 Wh/s = 3.6W)
        # Store last odom timestamp per robot for calculating static energy
        self._last_odom_time: Dict[str, float] = {}  # Last odom message timestamp (seconds)
        
        # Real-world energy consumption noise/uncertainty (sigma)
        # Real robots have manufacturing variations, temperature effects, battery degradation, etc.
        # Add Gaussian noise to energy calculations to reflect real-world uncertainty
        # sigma determines the standard deviation of energy consumption noise
        # Formula: E_noise ~ N(0, sigma^2), added to total energy consumption
        self.energy_noise_sigma = 0.01  # Standard deviation of energy noise (Wh)
        # Higher values = more uncertainty/variation in energy consumption
        
        # Deceleration energy handling policy
        # Options for handling deceleration energy (kinetic energy decrease):
        # 1. "none": Deceleration consumes no energy (physical: KE decreases naturally)
        #    - Most physically accurate for coasting/inertial deceleration
        # 2. "regenerative": Deceleration recovers energy (negative energy = energy gained)
        #    - Models regenerative braking systems (e.g., electric vehicles)
        #    - Efficiency determines how much kinetic energy can be recovered
        # 3. "braking_loss": Deceleration loses energy as heat from friction braking
        #    - Models conventional friction brakes (mechanical energy → heat)
        #    - Typically a small fraction of kinetic energy lost
        self.deceleration_policy = "none"  # Options: "none", "regenerative", "braking_loss"
        
        # Regenerative braking efficiency (0.0 to 1.0)
        # Represents the fraction of kinetic energy that can be recovered during deceleration
        # 0.0 = no recovery (same as "none"), 1.0 = perfect recovery (physically impossible)
        # Typical values: 0.3-0.7 for electric vehicles, 0.0 for TurtleBot3 (no regenerative braking)
        self.regenerative_efficiency = 0.0  # Default: no regenerative braking (TurtleBot3)
        
        # Braking energy loss coefficient (for "braking_loss" policy)
        # Represents the fraction of kinetic energy lost as heat during friction braking
        # Only used when deceleration_policy = "braking_loss"
        # Typical values: 0.1-0.3 (small fraction, as most KE is just converted to heat, not "consumed")
        self.braking_loss_coefficient = 0.1  # 10% of kinetic energy lost as heat during braking

        # Per-robot battery capacities in Wh (used in _odom_cb and when
        # applying pick/drop overhead energy to SOC).
        # ROS2 parameters don't support dict type, so we pass as YAML string
        self.declare_parameter('robot_cap_wh', '')
        cap_yaml = self.get_parameter('robot_cap_wh').value
        
        # Parse YAML string to dict, use 120.0 as fallback default
        self.robot_cap_wh: Dict[str, float] = {}
        if cap_yaml:
            try:
                parsed_dict = yaml.safe_load(cap_yaml)
                if isinstance(parsed_dict, dict):
                    for robot_id in self.robot_ids:
                        if robot_id in parsed_dict:
                            self.robot_cap_wh[robot_id] = float(parsed_dict[robot_id])
                        else:
                            self.robot_cap_wh[robot_id] = 120.0
                else:
                    # Fallback: use 120.0 Wh for all robots if parsing failed
                    self.robot_cap_wh = {rid: 120.0 for rid in self.robot_ids}
            except Exception as e:
                self.get_logger().warn(f"Failed to parse robot_cap_wh: {e}, using defaults")
                self.robot_cap_wh = {rid: 120.0 for rid in self.robot_ids}
        else:
            # Fallback: use 120.0 Wh for all robots if not specified
            self.robot_cap_wh = {rid: 120.0 for rid in self.robot_ids}
        
        # Log the battery capacities
        self.get_logger().info(
            f"[scheduler] Robot battery capacities: {self.robot_cap_wh}"
        )

        # Initial SOC scaling factor so you can start tests with partially
        # charged batteries. For example, 1.0 = 100%, 0.5 = 50%.
        # Later this could be turned into a ROS parameter.
        self.initial_soc_factor: float = 1

        # Define two chargers in the map frame for visualization
        self.chargers = [
            ChargerSpec(charger_id=1, x=0.0, y=-4.0),
            ChargerSpec(charger_id=2, x=0.0, y=6.0),
            # ChargerSpec(charger_id=3, x=-3.0, y=-1.0)
        ]

        # Initialize charger queues
        for ch in self.chargers:
            self.charger_queues[ch.charger_id] = []

        # ------------------------------------------------------------------
        # Nav2 path planning clients for accurate distance estimation
        # One client per robot, using each robot's own planner
        # ------------------------------------------------------------------
        self.path_planning_clients: Dict[str, ActionClient] = {}
        self.path_cache: Dict[Tuple[float, float, float, float], float] = {}
        self.cache_enabled = True
        self.use_nav2_path_planning = True  # Set to False to use euclidean distance
        
        # For scheduling-only experiments: use specific robot(s) Nav2 for path planning
        # Option 1: Single robot - set path_planning_robot_id to a robot_id (e.g., 'tb1')
        # Option 2: Multiple robots - set path_planning_robot_ids to a list (e.g., ['tb1', 'tb2', 'tb3'])
        #           Requests will be distributed round-robin across available servers
        # Option 3: Set both to None to use each robot's own action server (normal operation)
        self.path_planning_robot_id: Optional[str] = None  # Single robot mode (disabled)
        self.path_planning_robot_ids: List[str] = ['tb1', 'tb2', 'tb3', 'tb4', 'tb5', 'tb6','tb7']  # Multi-robot mode: round-robin
        self._path_planning_round_robin_idx: int = 0  # Counter for round-robin distribution
        
        # Robot initial positions (for scheduling-only experiments without simulation)
        # These positions match the Gazebo spawn positions in launch file
        self.robot_initial_positions: Dict[str, Tuple[float, float]] = {
            'tb1': (-3.5, 1.5),
            'tb2': (4.5, 1.5),
            'tb3': (-1.5, -0.5),
            'tb4': (1.0, -8.0),
            'tb5': (-5.0, -3.0),
            'tb6': (5.0, -3.0),
            'tb7': (-3.0, 5.0),
            'tb8': (3.0, 0.0),
            'tb9': (0.0, 0.0),
            'tb10': (-2.0, -5.0),
            'tb11': (0.65, -9.00),
            'tb12': (-3.75, 8.90),
            'tb13': (-5.25, -2.15),
            'tb14': (-4.85, 4.40),
            'tb15': (0.20, -7.15),
            'tb16': (-5.40, 2.20),
            'tb17': (-0.50, -3.15),
            'tb18': (0.40, 6.25),
            'tb19': (-3.75, -7.85),
            'tb20': (-3.60, 3.40),
            'tb21': (-1.15, -4.85),
            'tb22': (-3.70, -2.05),
            'tb23': (0.85, -4.80),
            'tb24': (-5.95, 8.95),
            'tb25': (-4.70, -6.55),
            'tb26': (0.50, -1.10),
            'tb27': (5.50, -1.85),
            'tb28': (-6.00, -8.70),
            'tb29': (1.25, -2.40),
            'tb30': (5.45, -7.70),
            'tb31': (3.45, -4.20),
            'tb32': (3.60, -1.95),
            'tb33': (1.75, -6.85),
            'tb34': (-3.55, 0.75),
            'tb35': (-4.45, -3.95),
            'tb36': (5.55, 0.40),
            'tb37': (-6.05, 5.35),
            'tb38': (3.80, -6.10),
            'tb39': (5.45, -3.75),
            'tb40': (-3.40, -9.60),
            'tb41': (3.80, -7.65),
            'tb42': (0.10, 3.40),
            'tb43': (-1.80, -9.55),
            'tb44': (5.40, -6.00),
            'tb45': (3.70, -9.15),
            'tb46': (-2.20, -1.85),
            'tb47': (-6.05, -4.20),
            'tb48': (3.45, -0.40),
            'tb49': (-5.20, 6.75),
            'tb50': (4.85, 1.75),
        }

        
        
        # Nav2 recovery clients for task failure recovery
        # NavigateToPose action clients (to cancel goals)
        self.nav_action_clients: Dict[str, ActionClient] = {}
        # Active goal handles for proper cancellation (robot_id -> goal_handle)
        self.active_nav_goals: Dict[str, any] = {}
        # ClearCostmap service clients (global and local costmaps)
        from rclpy.node import Client
        self.clear_costmap_global_clients: Dict[str, Client] = {}
        self.clear_costmap_local_clients: Dict[str, Client] = {}
        # InitialPose publishers (to reset robot localization)
        self.initial_pose_pubs: Dict[str, rclpy.node.Publisher] = {}
        # TF2 buffer for coordinate transformation (odom -> map)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        # TF broadcaster for publishing map->odom transform (Gazebo ground truth recovery)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        # Gazebo ModelStates subscriber for ground truth pose
        self.gazebo_model_states_sub = None
        self.gazebo_ground_truth: Dict[str, Tuple[float, float, float]] = {}  # robot_id -> (x, y, yaw)
        try:
            from gazebo_msgs.msg import ModelStates
            self.gazebo_model_states_sub = self.create_subscription(
                ModelStates,
                '/gazebo/model_states',
                self._gazebo_model_states_cb,
                10
            )
        except ImportError:
            self.get_logger().warn(
                "[nav2-recovery] gazebo_msgs not available - will use TF fallback for ground truth pose"
            )
        

        for rid in self.robot_ids:
            self._init_robot_interfaces(rid)

        # Simple global task request interface
        self.create_subscription(
            String,
            '/global_task_request',
            self._on_global_task_request,
            10,
        )

        # Seed a pool of demo tasks (no job structure - tasks will be optimally assigned)
        self._init_demo_tasks()

        # Optimization is disabled; keep placeholders for compatibility with legacy helpers.
        self.optimization_epoch = 0
        self.optimization_pop_size = 0

        # Hard-code manual sequences so we can bypass the optimizer entirely.
        # Negative entries act as CHARGE placeholders handled later.
        self.manual_sequence_source: str = "<embedded>"
        self.manual_sequence_map: Dict[str, List[Any]] = {
            "tb1": [35, 17, -1, 44, 36, 23, 14, -1, 22, 38, -1, 50, 43, 21, 49, 20],
            "tb2": [10, 32, 5, 4, 39, -1, 6, 46, 11, 13, 33, 45, 47],
            "tb3": [19, 3, -1, 24, -1, 27, 25, 48, 9, -1, 28, 34, 31, 7, 41],
            "tb4": [8, 15, 42, 37, -1, 30, 2, 40, 29, 26, 12, 1, 18, 16],
        }
        robots_list = ", ".join(sorted(self.manual_sequence_map.keys())) or "<none>"
        self.get_logger().info(
            f"[manual] Using embedded manual sequences for robots: {robots_list}"
        )

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
        
        # Whether to include charging in objective function optimization
        # - True: objective function considers charging (charge_key optimization)
        # - False: objective function ignores charging, only optimizes task allocation/sequencing
        #          Charging is handled later by feasibility/threshold strategy
        # This allows comparison between:
        #   1. Optimized charging in objective (include_charge_in_objective=True)
        #   2. Post-hoc charging via feasibility/threshold (include_charge_in_objective=False)
        include_charge_param = self.declare_parameter("include_charge_in_objective", True).get_parameter_value().bool_value
        self.include_charge_in_objective = include_charge_param

        default_variant = "with_charge" if self.include_charge_in_objective else "task_only"
        seq_variant_param = self.declare_parameter(
            "sequence_execution_variant",
            default_variant,
        ).get_parameter_value().string_value
        seq_variant = (seq_variant_param or default_variant).strip().lower()
        if seq_variant not in ("with_charge", "task_only"):
            self.get_logger().warn(
                f"[scheduler] Unknown sequence_execution_variant='{seq_variant_param}', falling back to '{default_variant}'"
            )
            seq_variant = default_variant
        self.sequence_execution_variant = seq_variant
        self.sequence_variants: Dict[str, Dict[str, Any]] = {}
        self.active_sequence_variant: Optional[str] = None

        self.get_logger().info(
            f"[scheduler] Charging config: strategy={self.charging_strategy}, "
            f"include_charge_in_objective={self.include_charge_in_objective}, "
            f"sequence_execution_variant={self.sequence_execution_variant}"
        )

        # Main scheduling timer
        self.timer = self.create_timer(2.0, self._timer_cb)
        # Periodic SOC logging timer (every 10 seconds)
        self.soc_log_timer = self.create_timer(10.0, self._soc_log_timer_cb)
        # Periodic goal visualization timer (publish /goal_pose_debug continuously)
        self.goal_viz_timer = self.create_timer(0.5, self._goal_viz_timer_cb)
        # Log manual sequence mode once at startup for visibility
        self.get_logger().info(
            f"SchedulerNode started for robots={self.robot_ids}; task allocation uses embedded manual sequences (optimizer disabled)"
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
        
        # Flag to prevent energy consumption during precompute/optimization
        # Energy should only decrease after tasks are assigned and execution begins
        self.optimization_in_progress: bool = False
        
        # Log file output directory
        base_output_dir = self.declare_parameter("log_output_dir", "metrics_output").get_parameter_value().string_value
        os.makedirs(base_output_dir, exist_ok=True)
        
        # Create a subdirectory with timestamp for this run's logs
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_output_dir = os.path.join(base_output_dir, f"logs_{timestamp}")
        os.makedirs(self.log_output_dir, exist_ok=True)
        
        # Initialize log files in the timestamped directory
        self.sequence_log_file = os.path.join(self.log_output_dir, "sequence_log.txt")
        self.makespan_log_file = os.path.join(self.log_output_dir, "makespan_log.txt")
        self.execution_time_log_file = os.path.join(self.log_output_dir, "execution_time_log.txt")

        # Track path planning (Nav2 precompute) time for logging alongside
        # per-algorithm optimization time in execution_time_log.txt
        self.path_planning_time_s: float = 0.0

    # Pending start tracker: when we publish a TaskCommand we expect a
        # corresponding `task_started` event from the robot. If no such
        # event arrives within `start_timeout_s`, we consider the robot's
        # start as failed and revert its availability so the scheduler can
        # retry the task. This prevents a global hang when all robots never
        # acknowledge starts.
        self.pending_task_start: Dict[str, float] = {}
        # Track actual start times and per-waypoint SOC/timing to evaluate execution progress
        self.task_start_times: Dict[Tuple[str, int], float] = {}
        self.task_progress: Dict[Tuple[str, int], Dict[str, set]] = {}
        self.task_start_soc: Dict[Tuple[str, int], float] = {}
        self.task_waypoint_times: Dict[Tuple[str, int], List[float]] = {}
        self.task_waypoint_end_soc: Dict[Tuple[str, int], List[float]] = {}
        self.task_waypoint_start: Dict[Tuple[str, int], int] = {}
        self.start_timeout_s = 8.0
        self.start_timeout_timer = self.create_timer(2.0, self._pending_start_check)

        # ------------------------------------------------------------------
        # Offline scheduling: bootstrap the embedded manual sequences once.
        # No rolling horizon or optimizer is active; Gazebo follows the
        # authored order verbatim for each robot.
        # ------------------------------------------------------------------
        self._offline_initialize_sequences()

    # ------------------------------------------------------------------
    # Weight helpers
    # ------------------------------------------------------------------
    
    def get_item_weight(self, task_type: str, pick_index: int) -> float:
        """Get item weight for a specific task type and pick index.
        
        Args:
            task_type: Task type (PICK_DROP, DUAL_PICK_MULTI_DROP, MULTI_PICK_DROP)
            pick_index: Index of the pick (0-based)
        
        Returns:
            Weight in kg for the specified pick
        """
        weights = self.task_type_weights.get(task_type, [self.default_item_weight_kg])
        if pick_index < len(weights):
            return weights[pick_index]
        elif weights:
            return weights[-1]  # Use last weight if index exceeds list
        return self.default_item_weight_kg

    def _resolve_pick_weight(self, task: TaskSpec, pick_index: int) -> float:
        """Return the effective pick weight honoring task-type defaults."""
        if task.task_type in self.task_type_weights:
            return self.get_item_weight(task.task_type, pick_index)

        if getattr(task, 'pick_weights', None) and pick_index < len(task.pick_weights):
            value = task.pick_weights[pick_index]
            if value is not None:
                try:
                    weight = float(value)
                except (TypeError, ValueError):
                    weight = 0.0
                else:
                    if weight > 0.0:
                        return weight

        return self.default_item_weight_kg

    def _compute_charge_duration(self, robot_id: str, current_soc: Optional[float] = None) -> float:
        """Compute SOC-based charge duration (seconds) for a robot."""
        cap_wh = self.robot_cap_wh.get(robot_id, 120.0)
        rate = getattr(self, 'charge_rate_wh_per_s', 0.0)
        if cap_wh <= 0.0 or rate <= 0.0:
            return 0.0

        if current_soc is None:
            # Fallback to initial SOC assumption if we have no live reading
            current_soc = cap_wh * float(getattr(self, 'initial_soc_factor', 1.0))

        current_soc = max(0.0, min(cap_wh, float(current_soc)))
        remaining_wh = max(0.0, cap_wh - current_soc)
        if remaining_wh <= 0.0:
            return 0.0
        return remaining_wh / rate

    # ------------------------------------------------------------------
    # Log file helpers
    # ------------------------------------------------------------------
    
    def _save_sequence_log(self, algorithm_name: str, robot_id: str, sequence_str: str, 
                          seq_length: int, task_type_counts: Dict[str, int], 
                          charge_count: int, makespan: Optional[float]):
        """Save sequence log to file with additional information."""
        try:
            # Format task type counts
            task_type_str = ", ".join([f"{task_type}:{count}" for task_type, count in sorted(task_type_counts.items())]) if task_type_counts else "none"
            makespan_str = f", makespan={makespan:.2f}" if makespan is not None else ", makespan=N/A"
            
            log_line = (f"[decode-{algorithm_name}] {robot_id} sequence=[{sequence_str}] "
                       f"(length={seq_length}, tasks=[{task_type_str}], charges={charge_count}{makespan_str})\n")
            
            with open(self.sequence_log_file, 'a') as f:
                f.write(log_line)
        except Exception as e:
            self.get_logger().warn(f"Failed to save sequence log: {e}")
    
    def _save_makespan_log(self, algorithm_name: str, best_fitness: float, makespan: Optional[float], energy: Optional[float] = None):
        """Save makespan and energy log to file."""
        try:
            makespan_info = f", makespan={makespan:.2f}" if makespan is not None else ""
            energy_info = f", energy={energy:.2f}Wh" if energy is not None else ""
            # Ensure directory exists in case it was removed after initialization
            os.makedirs(os.path.dirname(self.makespan_log_file), exist_ok=True)
            with open(self.makespan_log_file, 'a') as f:
                f.write(f"[optimization-{algorithm_name}] Best fitness = {best_fitness:.2f}{makespan_info}{energy_info}\n")
        except Exception as e:
            self.get_logger().warn(f"Failed to save makespan log: {e}")
    
    def _save_execution_time_log(self, algorithm_name: str, optimization_time: float,
                                 path_planning_time: Optional[float] = None) -> None:
        """Save algorithm execution time log to file.

        When ``path_planning_time`` is provided, the log line includes a
        breakdown of total time into ``path planning + optimization`` as:

            Execution time(path planning + optimization) = TOTALs (PPs + OPTs)

        where TOTAL = path_planning_time + optimization_time.
        """
        try:
            with open(self.execution_time_log_file, 'a') as f:
                if path_planning_time is not None:
                    pp = max(0.0, float(path_planning_time))
                    opt = max(0.0, float(optimization_time))
                    total = pp + opt
                    f.write(
                        f"[execution-time-{algorithm_name}] "
                        f"Execution time(path planning + optimization) = "
                        f"{total:.3f}s ({pp:.3f}s + {opt:.3f}s)\n"
                    )
                else:
                    # Backwards-compatible simple format (optimization only)
                    f.write(
                        f"[execution-time-{algorithm_name}] "
                        f"Execution time = {optimization_time:.3f}s\n"
                    )
        except Exception as e:
            self.get_logger().warn(f"Failed to save execution time log: {e}")
    
    def _save_robot_completion_times_log(self, algorithm_name: str, robot_completion_times: Dict[str, float]):
        """Save robot completion times log to file."""
        try:
            with open(self.execution_time_log_file, 'a') as f:
                f.write(f"[robot-completion-times-{algorithm_name}]\n")
                for robot_id in sorted(robot_completion_times.keys()):
                    completion_time = robot_completion_times[robot_id]
                    f.write(f"  {robot_id}: {completion_time:.2f}s\n")
        except Exception as e:
            self.get_logger().warn(f"Failed to save robot completion times log: {e}")
    
    def _save_round_result_log(self, round_num: int, algo_name: str, fitness: float, makespan: Optional[float], run_time: float, energy: Optional[float] = None):
        """Save individual round result to log file."""
        try:
            makespan_str = f", makespan={makespan:.2f}" if makespan is not None else ", makespan=N/A"
            energy_str = f", energy={energy:.2f}Wh" if energy is not None else ", energy=N/A"
            # Ensure directory exists in case it was removed after initialization
            os.makedirs(os.path.dirname(self.makespan_log_file), exist_ok=True)
            with open(self.makespan_log_file, 'a') as f:
                f.write(f"[round-{round_num:02d}] {algo_name}: fitness={fitness:.2f}{makespan_str}{energy_str}, time={run_time:.3f}s\n")
        except Exception as e:
            self.get_logger().warn(f"Failed to save round result log: {e}")
    
    def _save_multi_round_statistics_log(self, total_rounds: int, algorithms: List[str], 
                                          all_fitnesses_per_algo: Dict[str, List[float]],
                                          all_makespans_per_algo: Dict[str, List[Optional[float]]],
                                          all_energies_per_algo: Dict[str, List[Optional[float]]],
                                          all_waiting_times_per_algo: Dict[str, List[Optional[float]]],
                                          best_fitness_per_algo: Dict[str, float],
                                          best_makespan_per_algo: Dict[str, Optional[float]],
                                          best_energy_per_algo: Dict[str, Optional[float]],
                                          best_waiting_time_per_algo: Dict[str, Optional[float]],
                                          total_time_per_algo: Dict[str, float],
                                          global_best_algo: str, global_best_round: int, 
                                          global_best_fitness: float, global_best_makespan: Optional[float],
                                          global_best_energy: Optional[float]):
        """Save final statistics for multi-round optimization to log file."""
        try:
            # Ensure directory exists in case it was removed after initialization
            os.makedirs(os.path.dirname(self.makespan_log_file), exist_ok=True)
            with open(self.makespan_log_file, 'a') as f:
                f.write(f"\n{'='*80}\n")
                f.write(f"[FINAL STATISTICS] {total_rounds} rounds completed\n")
                f.write(f"{'='*80}\n\n")
                
                # Per-algorithm statistics (fitness)
                f.write(f"[Per-Algorithm Statistics - Fitness]\n")
                for algo_name in algorithms:
                    valid_fitnesses = [fit for fit in all_fitnesses_per_algo[algo_name] if math.isfinite(fit)]
                    if valid_fitnesses:
                        avg_fitness = sum(valid_fitnesses) / len(valid_fitnesses)
                        min_fitness = min(valid_fitnesses)
                        max_fitness = max(valid_fitnesses)
                        std_fitness = (sum((f - avg_fitness) ** 2 for f in valid_fitnesses) / len(valid_fitnesses)) ** 0.5
                        f.write(f"  {algo_name}: best={min_fitness:.2f}, avg={avg_fitness:.2f}, worst={max_fitness:.2f}, "
                               f"std={std_fitness:.2f}, total_time={total_time_per_algo[algo_name]:.3f}s\n")
                    else:
                        f.write(f"  {algo_name}: all {total_rounds} runs failed, "
                               f"total_time={total_time_per_algo[algo_name]:.3f}s\n")
                
                # Per-algorithm statistics (makespan)
                f.write(f"\n[Per-Algorithm Statistics - Makespan]\n")
                for algo_name in algorithms:
                    valid_makespans = [m for m in all_makespans_per_algo[algo_name] if m is not None]
                    if valid_makespans:
                        avg_makespan = sum(valid_makespans) / len(valid_makespans)
                        min_makespan = min(valid_makespans)
                        max_makespan = max(valid_makespans)
                        std_makespan = (sum((m - avg_makespan) ** 2 for m in valid_makespans) / len(valid_makespans)) ** 0.5
                        best_makespan = best_makespan_per_algo[algo_name]
                        best_makespan_str = f"{best_makespan:.2f}" if best_makespan is not None else "N/A"
                        f.write(f"  {algo_name}: best_at_best_fitness={best_makespan_str}, "
                               f"min={min_makespan:.2f}, avg={avg_makespan:.2f}, max={max_makespan:.2f}, std={std_makespan:.2f}\n")
                    else:
                        f.write(f"  {algo_name}: no valid makespan data\n")
                
                # Per-algorithm statistics (energy)
                f.write(f"\n[Per-Algorithm Statistics - Energy]\n")
                for algo_name in algorithms:
                    valid_energies = [e for e in all_energies_per_algo[algo_name] if e is not None]
                    if valid_energies:
                        avg_energy = sum(valid_energies) / len(valid_energies)
                        min_energy = min(valid_energies)
                        max_energy = max(valid_energies)
                        std_energy = (sum((e - avg_energy) ** 2 for e in valid_energies) / len(valid_energies)) ** 0.5
                        best_energy = best_energy_per_algo[algo_name]
                        best_energy_str = f"{best_energy:.2f}" if best_energy is not None else "N/A"
                        f.write(f"  {algo_name}: best_at_best_fitness={best_energy_str}, "
                               f"min={min_energy:.2f}, avg={avg_energy:.2f}, max={max_energy:.2f}, std={std_energy:.2f}\n")
                    else:
                        f.write(f"  {algo_name}: no valid energy data\n")

                # Per-algorithm statistics (waiting time)
                f.write(f"\n[Per-Algorithm Statistics - Waiting Time]\n")
                for algo_name in algorithms:
                    raw_waits = all_waiting_times_per_algo.get(algo_name, [])
                    valid_waits = [w for w in raw_waits if (w is not None and math.isfinite(w))]
                    if valid_waits:
                        avg_wait = sum(valid_waits) / len(valid_waits)
                        min_wait = min(valid_waits)
                        max_wait = max(valid_waits)
                        std_wait = (sum((w - avg_wait) ** 2 for w in valid_waits) / len(valid_waits)) ** 0.5
                        best_wait = best_waiting_time_per_algo.get(algo_name)
                        best_wait_str = f"{best_wait:.2f}" if (best_wait is not None and math.isfinite(best_wait)) else "N/A"
                        f.write(
                            f"  {algo_name}: best_at_best_fitness={best_wait_str}, "
                            f"min={min_wait:.2f}, avg={avg_wait:.2f}, max={max_wait:.2f}, std={std_wait:.2f}\n"
                        )
                    else:
                        f.write(f"  {algo_name}: no valid waiting time data\n")
                
                # Best fitness comparison
                f.write(f"\n[Best Fitness Comparison]\n")
                sorted_algos = sorted(algorithms, key=lambda a: best_fitness_per_algo[a])
                for rank, algo_name in enumerate(sorted_algos, 1):
                    makespan = best_makespan_per_algo[algo_name]
                    energy = best_energy_per_algo[algo_name]
                    makespan_str = f", makespan={makespan:.2f}" if makespan is not None else ""
                    energy_str = f", energy={energy:.2f}Wh" if energy is not None else ""
                    f.write(f"  #{rank}: {algo_name} = fitness {best_fitness_per_algo[algo_name]:.2f}{makespan_str}{energy_str}\n")
                
                # Global best
                f.write(f"\n[GLOBAL BEST]\n")
                f.write(f"  Algorithm: {global_best_algo}\n")
                f.write(f"  Round: {global_best_round}\n")
                f.write(f"  Fitness: {global_best_fitness:.2f}\n")
                makespan_str = f"{global_best_makespan:.2f}" if global_best_makespan is not None else "N/A"
                energy_str = f"{global_best_energy:.2f}" if global_best_energy is not None else "N/A"
                f.write(f"  Makespan: {makespan_str}\n")
                f.write(f"  Energy: {energy_str}Wh\n")
                f.write(f"{'='*80}\n\n")
        except Exception as e:
            self.get_logger().warn(f"Failed to save multi-round statistics log: {e}")
    
    # ------------------------------------------------------------------
    # Manual sequence helpers
    # ------------------------------------------------------------------

    def _load_manual_sequence_map(self, source: str) -> Optional[Dict[str, List[Any]]]:
        """Load manual sequences from inline YAML/JSON text or a file path."""
        if not source:
            return None

        text = source.strip()
        if not text:
            return None

        candidate_path = text
        if candidate_path.startswith("file://"):
            candidate_path = candidate_path[7:]
        candidate_path = os.path.expanduser(candidate_path)

        if os.path.isfile(candidate_path):
            try:
                with open(candidate_path, "r", encoding="utf-8") as f:
                    text = f.read()
            except Exception as exc:
                self.get_logger().error(
                    f"[manual] Failed to read manual sequence file '{candidate_path}': {exc}"
                )
                return None

        try:
            data = yaml.safe_load(text)
        except Exception as exc:
            self.get_logger().error(f"[manual] Failed to parse manual sequence content: {exc}")
            return None

        if not isinstance(data, dict):
            self.get_logger().warn(
                "[manual] Manual sequence content must be a mapping of robot_id -> sequence list"
            )
            return None

        manual_map: Dict[str, List[Any]] = {}
        for robot_id, sequence in data.items():
            rid = str(robot_id)
            if sequence is None:
                manual_map[rid] = []
            elif isinstance(sequence, (list, tuple)):
                manual_map[rid] = list(sequence)
            else:
                manual_map[rid] = [sequence]

        return manual_map

    def _create_manual_charge_task(self, robot_id: str, entry: Any) -> Optional[TaskSpec]:
        """Create a CHARGE TaskSpec from a manual entry when requested."""
        if not self.chargers:
            return None

        charger_id = None
        duration_s: Optional[float] = None

        if isinstance(entry, (int, float)) and not isinstance(entry, bool):
            try:
                sentinel = int(entry)
            except (TypeError, ValueError):
                return None
            if sentinel >= 0:
                return None
            # Negative integers (e.g., -1) are treated as default CHARGE placeholders
        elif isinstance(entry, str):
            token = entry.strip()
            if not token:
                return None
            duration_part = None
            charger_part = None
            if ':' in token:
                token, duration_part = token.split(':', 1)
                duration_part = duration_part.strip()
            if '@' in token:
                token, charger_part = token.split('@', 1)
                charger_part = charger_part.strip()
            if token.strip().upper() != "CHARGE":
                return None
            if charger_part:
                try:
                    charger_id = int(charger_part)
                except ValueError:
                    charger_id = None
            if duration_part:
                try:
                    duration_s = float(duration_part)
                except ValueError:
                    duration_s = None
        elif isinstance(entry, dict):
            task_type = str(entry.get("task_type", "")).upper()
            if task_type != "CHARGE":
                return None
            if "charger_id" in entry:
                try:
                    charger_id = int(entry.get("charger_id"))
                except (TypeError, ValueError):
                    charger_id = None
            elif "charger" in entry:
                try:
                    charger_id = int(entry.get("charger"))
                except (TypeError, ValueError):
                    charger_id = None
            duration_key = (
                entry.get("charge_duration_s")
                or entry.get("duration_s")
                or entry.get("duration")
            )
            if duration_key is not None:
                try:
                    duration_s = float(duration_key)
                except (TypeError, ValueError):
                    duration_s = None
        else:
            return None

        charger = None
        if charger_id is not None:
            for spec in self.chargers:
                if spec.charger_id == charger_id:
                    charger = spec
                    break
        if charger is None:
            charger = self.chargers[0]

        if duration_s is None:
            state = self.robot_states.get(robot_id)
            current_soc = state.soc if state is not None else None
            duration_s = self._compute_charge_duration(robot_id, current_soc)
        try:
            duration_value = max(0.0, float(duration_s))
        except (TypeError, ValueError):
            duration_value = 0.0

        return TaskSpec(
            task_id=-1,
            task_type="CHARGE",
            picks=[(charger.x, charger.y)],
            drops=[(charger.x, charger.y)],
            pick_wait_s=0.0,
            drop_wait_s=0.0,
            charge_duration_s=duration_value,
        )

    def _apply_manual_sequences(self) -> bool:
        """Apply user-provided manual sequences instead of running optimization."""
        if not getattr(self, 'manual_sequence_map', None):
            return False

        task_lookup: Dict[int, TaskSpec] = {t.task_id: t for t in self.pending_tasks}
        assigned_ids: Set[int] = set()
        missing_ids: Set[int] = set()
        duplicate_ids: Set[int] = set()
        invalid_entries: List[str] = []
        charge_inserts = 0

        sequences: Dict[str, List[TaskSpec]] = {rid: [] for rid in self.robot_ids}

        for robot_id, entries in self.manual_sequence_map.items():
            rid = str(robot_id)
            if rid not in self.robot_ids:
                self.get_logger().warn(
                    f"[manual] Robot '{rid}' not in configured robot_ids; skipping its manual sequence"
                )
                continue

            normalized_entries = entries if isinstance(entries, list) else [entries]
            for entry in normalized_entries:
                charge_task = self._create_manual_charge_task(rid, entry)
                if charge_task is not None:
                    sequences[rid].append(charge_task)
                    charge_inserts += 1
                    continue

                value = entry
                if isinstance(entry, dict) and "task_id" in entry:
                    value = entry.get("task_id")

                try:
                    task_id = int(value)
                except (TypeError, ValueError):
                    invalid_entries.append(f"{rid}:{entry}")
                    continue

                if task_id in assigned_ids:
                    duplicate_ids.add(task_id)
                    continue

                task = task_lookup.get(task_id)
                if task is None:
                    missing_ids.add(task_id)
                    continue

                sequences[rid].append(task)
                assigned_ids.add(task_id)

        if not any(sequences.values()):
            self.get_logger().warn(
                "[manual] Manual sequence override provided but no valid tasks were matched"
            )
            return False

        if missing_ids:
            missing_list = ", ".join(str(tid) for tid in sorted(missing_ids))
            self.get_logger().warn(
                f"[manual] Unknown task IDs in manual sequence: {missing_list}"
            )
        if duplicate_ids:
            dup_list = ", ".join(str(tid) for tid in sorted(duplicate_ids))
            self.get_logger().warn(
                f"[manual] Duplicate task IDs detected in manual sequence (kept first occurrence): {dup_list}"
            )
        if invalid_entries:
            self.get_logger().warn(
                f"[manual] Ignored non-numeric manual entries: {invalid_entries}"
            )

        for rid in self.robot_ids:
            seq = sequences.get(rid, [])
            self.current_sequences[rid] = seq
            self.current_seq_index[rid] = 0
            if seq:
                seq_ids = [t.task_id for t in seq]
                self.get_logger().info(
                    f"[manual] Sequence for {rid}: {seq_ids} (charges={sum(1 for t in seq if t.task_type == 'CHARGE')})"
                )

        self.sequence_variants.clear()
        self.sequence_variants["manual"] = {
            "include_charge": self.include_charge_in_objective,
            "sequences": sequences,
            "stats": {
                "assigned_tasks": len(assigned_ids),
                "charge_events": charge_inserts,
            },
        }
        self.active_sequence_variant = "manual"

        return True

    # ------------------------------------------------------------------
    # Path planning helper for accurate distance calculation
    # ------------------------------------------------------------------

    def _get_path_planning_client(self, robot_id: Optional[str]) -> Optional[ActionClient]:
        """Get or create action client for path planning.
        
        IMPORTANT: For scheduling-only experiments (without full simulation),
        path planning requests can be distributed across multiple Nav2 servers.
        - path_planning_robot_id: Use a single robot's Nav2 (simple mode)
        - path_planning_robot_ids: Use multiple robots' Nav2 in round-robin (parallel mode)
        """
        # Option 1: Multi-robot round-robin mode (fastest for scheduling-only)
        path_planning_robots = getattr(self, 'path_planning_robot_ids', None)
        if path_planning_robots and len(path_planning_robots) > 0:
            # Round-robin: select next robot in list
            idx = self._path_planning_round_robin_idx % len(path_planning_robots)
            self._path_planning_round_robin_idx += 1
            robot_id = path_planning_robots[idx]
        # Option 2: Single robot mode
        elif getattr(self, 'path_planning_robot_id', None) is not None:
            robot_id = self.path_planning_robot_id
        # Option 3: Use robot's own server (default)
        elif robot_id is None:
            robot_id = self.robot_ids[0] if self.robot_ids else None
            if robot_id is None:
                return None
        
        if robot_id not in self.path_planning_clients:
            action_name = f'/{robot_id}/compute_path_to_pose'
            client = ActionClient(self, ComputePathToPose, action_name)
            self.path_planning_clients[robot_id] = client
            self.get_logger().info(f"Initialized Nav2 path planning client for {robot_id}: {action_name}")
        
        client = self.path_planning_clients[robot_id]
        # Minimal timeout for scheduling-only experiments (single robot Nav2)
        if not client.wait_for_server(timeout_sec=0.1):
            # Only log warning once per robot to avoid spam
            if robot_id not in getattr(self, '_logged_missing_action_server', set()):
                if not hasattr(self, '_logged_missing_action_server'):
                    self._logged_missing_action_server = set()
                self._logged_missing_action_server.add(robot_id)
                self.get_logger().warn(
                    f"Nav2 action server for {robot_id} not available: "
                    f"/{robot_id}/compute_path_to_pose action server not found. "
                    f"Will use euclidean distance for this robot."
                )
            # Return None to fallback to euclidean distance
            return None
        return client

    def _compute_nav2_path_distance(self, start_x: float, start_y: float, 
                                     goal_x: float, goal_y: float,
                                     robot_id: Optional[str] = None) -> Optional[float]:
        """Compute path distance using Nav2 ComputePathToPose action.
        
        Returns path length in meters, or None if planning fails.
            Uses cache to avoid redundant planning requests.
        """
        if not self.use_nav2_path_planning:
            return None
        
        # Early return if start and goal are the same (or very close)
        if abs(start_x - goal_x) < 1e-3 and abs(start_y - goal_y) < 1e-3:
            return 0.0
        
        # Check cache first
        cache_key = (start_x, start_y, goal_x, goal_y)
        if self.cache_enabled and cache_key in self.path_cache:
            return self.path_cache[cache_key]
        
        # Get action client
        client = self._get_path_planning_client(robot_id)
        if client is None:
            return None
        
        # Check if server is still ready (may have gone down since last check)
        if not client.server_is_ready():
            self.get_logger().debug(
                f"Nav2 action server for {robot_id} not ready at request time, "
                f"falling back to euclidean distance"
            )
            return None
        
        # Create goal message
        now = self.get_clock().now().to_msg()
        goal_msg = ComputePathToPose.Goal()
        goal_msg.goal = PoseStamped()
        goal_msg.goal.header.frame_id = 'map'
        goal_msg.goal.header.stamp = now
        goal_msg.goal.pose.position.x = float(goal_x)
        goal_msg.goal.pose.position.y = float(goal_y)
        goal_msg.goal.pose.orientation.w = 1.0
        goal_msg.start = PoseStamped()
        goal_msg.start.header.frame_id = 'map'
        goal_msg.start.header.stamp = now
        goal_msg.start.pose.position.x = float(start_x)
        goal_msg.start.pose.position.y = float(start_y)
        goal_msg.start.pose.orientation.w = 1.0
        goal_msg.use_start = True
        goal_msg.planner_id = ''
        
        # Execute action and get result
        # Minimal timeouts for fast scheduling-only experiments
        try:
            send_goal_future = client.send_goal_async(goal_msg)
            rclpy.spin_until_future_complete(self, send_goal_future, timeout_sec=0.1)
            if not send_goal_future.done():
                return None
            goal_handle = send_goal_future.result()
            if not goal_handle or not goal_handle.accepted:
                return None
            
            result_future = goal_handle.get_result_async()
            rclpy.spin_until_future_complete(self, result_future, timeout_sec=0.1)
            if not result_future.done():
                return None
            
            result = result_future.result()
            if result.status != 4:  # SUCCEEDED = 4
                # Log path planning failure but don't spam - only log if significant
                # Path planning failures are already logged by planner_server
                # We just return None and fallback to euclidean distance
                return None
            
            path_length = _calculate_path_length(result.result.path)
            if path_length <= 0.0:
                # self.get_logger().warn(
                #     f"Nav2 returned empty or invalid path for {robot_id}: "
                #     f"({start_x:.2f}, {start_y:.2f}) -> ({goal_x:.2f}, {goal_y:.2f})"
                # )
                return None
            
            if self.cache_enabled:
                self.path_cache[cache_key] = path_length
            return path_length
        except Exception as e:
            self.get_logger().debug(
                f"Nav2 path planning exception for {robot_id}: {e}: "
                f"({start_x:.2f}, {start_y:.2f}) -> ({goal_x:.2f}, {goal_y:.2f})"
            )
            return None

    def _gazebo_model_states_cb(self, msg) -> None:
        """Callback for Gazebo ModelStates - stores ground truth poses for all robots.
        
        This provides the true robot position from Gazebo simulation, which is used
        for Nav2 recovery to reset map->odom transform.
        """
        try:
            from gazebo_msgs.msg import ModelStates
            if not isinstance(msg, ModelStates):
                return
            
            # Log available model names on first callback (for debugging)
            if not hasattr(self, '_gazebo_models_logged'):
                self._gazebo_models_logged = True
                self.get_logger().info(
                    f"[nav2-recovery] Available Gazebo models: {msg.name}"
                )
            
            # Update ground truth poses for all known robots
            matched_count = 0
            for robot_id in self.robot_ids:
                model_name = robot_id  # e.g., "tb1"
                if model_name in msg.name:
                    idx = msg.name.index(model_name)
                    if idx < len(msg.pose):
                        pose = msg.pose[idx]
                        x = pose.position.x
                        y = pose.position.y
                        
                        # Extract yaw from quaternion
                        qx = pose.orientation.x
                        qy = pose.orientation.y
                        qz = pose.orientation.z
                        qw = pose.orientation.w
                        yaw = math.atan2(
                            2.0 * (qw * qz + qx * qy),
                            1.0 - 2.0 * (qy * qy + qz * qz)
                        )
                        
                        self.gazebo_ground_truth[robot_id] = (x, y, yaw)
                        matched_count += 1
                    else:
                        self.get_logger().debug(
                            f"[nav2-recovery] Model {model_name} found but pose index out of range"
                        )
                else:
                    self.get_logger().debug(
                        f"[nav2-recovery] Robot {robot_id} (model {model_name}) not found in Gazebo models"
                    )
            
            if matched_count == 0 and len(self.robot_ids) > 0:
                self.get_logger().warn(
                    f"[nav2-recovery] No robots matched in Gazebo ModelStates. "
                    f"Looking for: {self.robot_ids}, Available: {msg.name}"
                )
        except Exception as e:
            self.get_logger().warn(
                f"[nav2-recovery] Exception in _gazebo_model_states_cb: {type(e).__name__}: {e}"
            )

    def _recover_nav2_for_robot(self, robot_id: str, state: RobotState) -> None:
        """Recover Nav2 state when a task fails (paper-grade + real-world stable implementation).

        Guiding principle: mirror the Gazebo pose into TF so Nav2 naturally realigns.

        Preferred recovery structure:
        Step 1. Obtain the ground-truth pose from Gazebo (/gazebo/model_states).
        Step 2. Keep odom → base_link untouched (Gazebo updates it automatically).
        Step 3. Reset the map → odom TF by broadcasting it directly.

        Following this keeps the costmap, planner, and controller aligned automatically.
        
        Args:
            robot_id: Robot ID
            state: Current robot state (contains current position)
        """
        rid_str = self._color_robot(robot_id)
        
        self.get_logger().info(
            f"[nav2-recovery] {rid_str}: Starting Nav2 recovery sequence (current state: "
            f"pos=({state.x:.2f}, {state.y:.2f}), soc={state.soc:.2f}Wh)"
        )
        
        try:
            # Step 1: Cancel current navigation action using active goal handle
            # This ensures proper BT state cleanup in Nav2
            self.get_logger().info(f"[nav2-recovery] {rid_str}: [Step 1/4] Cancelling active navigation goal...")
            goal_handle = self.active_nav_goals.get(robot_id)
            cancel_success = False
            
            if goal_handle is not None:
                self.get_logger().info(
                    f"[nav2-recovery] {rid_str}: Found active goal handle, cancelling via handle..."
                )
                try:
                    # Cancel the active goal using its handle (proper way)
                    cancel_future = goal_handle.cancel_goal_async()
                    if cancel_future:
                        rclpy.spin_until_future_complete(self, cancel_future, timeout_sec=1.0)
                        if cancel_future.done():
                            cancel_success = True
                            self.active_nav_goals[robot_id] = None  # Clear active goal
                            self.get_logger().info(
                                f"[nav2-recovery] {rid_str}: ✓ Successfully cancelled navigation goal via goal handle"
                            )
                        else:
                            self.get_logger().warn(
                                f"[nav2-recovery] {rid_str}: ⚠ Goal cancellation timeout (1.0s)"
                            )
                    else:
                        self.get_logger().warn(
                            f"[nav2-recovery] {rid_str}: ⚠ cancel_goal_async() returned None"
                        )
                except Exception as e:
                    self.get_logger().warn(
                        f"[nav2-recovery] {rid_str}: ✗ Failed to cancel goal via handle: {type(e).__name__}: {e}"
                    )
            else:
                self.get_logger().info(
                    f"[nav2-recovery] {rid_str}: No active goal handle found (may not have active navigation)"
                )
            
            # Fallback: Try cancel via action client if no goal handle available
            if not cancel_success:
                nav_client = self.nav_action_clients.get(robot_id)
                if nav_client is not None:
                    if nav_client.server_is_ready():
                        self.get_logger().info(
                            f"[nav2-recovery] {rid_str}: Attempting fallback cancel via action client..."
                        )
                        try:
                            # This is a less reliable method but may work if goal handle is missing
                            nav_client.cancel_all_goals_async()
                            self.get_logger().info(
                                f"[nav2-recovery] {rid_str}: ✓ Cancelled all goals via action client (fallback)"
                            )
                            cancel_success = True
                        except Exception as e:
                            self.get_logger().warn(
                                f"[nav2-recovery] {rid_str}: ✗ Failed to cancel via action client: {type(e).__name__}: {e}"
                            )
                    else:
                        self.get_logger().info(
                            f"[nav2-recovery] {rid_str}: Nav2 action server not ready, skipping cancel"
                        )
                else:
                    self.get_logger().warn(
                        f"[nav2-recovery] {rid_str}: No Nav2 action client available for {robot_id}"
                    )
            
            # Step 2: Reset internal odom/energy state to prevent incorrect calculations
            # This is critical for accurate energy model after recovery
            self.get_logger().info(f"[nav2-recovery] {rid_str}: [Step 2/4] Resetting internal odom/energy state...")
            
            prev_odom = self._last_odom.get(robot_id)
            prev_velocity = self._last_velocity.get(robot_id)
            prev_odom_time = self._last_odom_time.get(robot_id)
            
            if robot_id in self._last_odom:
                self._last_odom[robot_id] = None
            if robot_id in self._last_velocity:
                self._last_velocity[robot_id] = (0.0, 0.0)
            if robot_id in self._last_odom_time:
                now_sec = self.get_clock().now().seconds_nanoseconds()[0]
                self._last_odom_time[robot_id] = now_sec
            
            self.get_logger().info(
                f"[nav2-recovery] {rid_str}: ✓ Reset internal state - "
                f"prev_odom={prev_odom}, prev_velocity={prev_velocity}, "
                f"prev_time={prev_odom_time if prev_odom_time else 'None'}s -> now={self._last_odom_time.get(robot_id, 'None')}s"
                    )
            
            # Step 3: Clear costmaps (global and local) to remove stale obstacle data
            # This helps if the robot is stuck due to incorrect costmap data
            self.get_logger().info(f"[nav2-recovery] {rid_str}: [Step 3/4] Clearing costmaps...")
            
            clear_global_client = self.clear_costmap_global_clients.get(robot_id)
            global_cleared = False
            if clear_global_client is not None:
                try:
                    service_name = f"/{robot_id}/global_costmap/clear_entirely_global_costmap"
                    if clear_global_client.wait_for_service(timeout_sec=0.5):
                        request = Empty()
                        future = clear_global_client.call_async(request)
                        if rclpy.spin_until_future_complete(self, future, timeout_sec=0.5):
                            if future.done():
                                response = future.result()
                                global_cleared = True
                                self.get_logger().info(
                                    f"[nav2-recovery] {rid_str}: ✓ Cleared global costmap (service: {service_name})"
                                )
                            else:
                                self.get_logger().warn(
                                    f"[nav2-recovery] {rid_str}: ⚠ Global costmap clear timeout"
                                )
                    else:
                        self.get_logger().warn(
                            f"[nav2-recovery] {rid_str}: ⚠ Global costmap service unavailable: {service_name}"
                        )
                except Exception as e:
                    self.get_logger().warn(
                        f"[nav2-recovery] {rid_str}: ✗ Failed to clear global costmap: {type(e).__name__}: {e}"
                    )
            else:
                self.get_logger().warn(
                    f"[nav2-recovery] {rid_str}: ⚠ No global costmap client available for {robot_id}"
                )
            
            clear_local_client = self.clear_costmap_local_clients.get(robot_id)
            local_cleared = False
            if clear_local_client is not None:
                try:
                    service_name = f"/{robot_id}/local_costmap/clear_entirely_local_costmap"
                    if clear_local_client.wait_for_service(timeout_sec=0.5):
                        request = Empty()
                        future = clear_local_client.call_async(request)
                        if rclpy.spin_until_future_complete(self, future, timeout_sec=0.5):
                            if future.done():
                                response = future.result()
                                local_cleared = True
                                self.get_logger().info(
                                    f"[nav2-recovery] {rid_str}: ✓ Cleared local costmap (service: {service_name})"
                                )
                            else:
                                self.get_logger().warn(
                                    f"[nav2-recovery] {rid_str}: ⚠ Local costmap clear timeout"
                                )
                    else:
                        self.get_logger().warn(
                            f"[nav2-recovery] {rid_str}: ⚠ Local costmap service unavailable: {service_name}"
                        )
                except Exception as e:
                    self.get_logger().warn(
                        f"[nav2-recovery] {rid_str}: ✗ Failed to clear local costmap: {type(e).__name__}: {e}"
                    )
            else:
                self.get_logger().warn(
                    f"[nav2-recovery] {rid_str}: ⚠ No local costmap client available for {robot_id}"
                )
            
            self.get_logger().info(
                f"[nav2-recovery] {rid_str}: Costmap clear results - global={'✓' if global_cleared else '✗'}, "
                    f"local={'✓' if local_cleared else '✗'}"
                    )
            
            # Step 4: Reset map->odom TF using Gazebo ground truth pose
            # Key idea: mirror the Gazebo pose into TF so Nav2 converges without hacks.
            # Keep odom->base_footprint untouched and only reset map->odom.
            self.get_logger().info(f"[nav2-recovery] {rid_str}: [Step 4/4] Resetting map->odom TF using Gazebo ground truth...")
            
            # Get Gazebo ground truth pose
            gazebo_gt = self.gazebo_ground_truth.get(robot_id)
            if gazebo_gt is None:
                self.get_logger().warn(
                    f"[nav2-recovery] {rid_str}: ✗ No Gazebo ground truth available - waiting for ModelStates..."
                )
                # Wait briefly for ModelStates callback to update
                for i in range(10):
                    rclpy.spin_once(self, timeout_sec=0.1)
                    gazebo_gt = self.gazebo_ground_truth.get(robot_id)
                    if gazebo_gt is not None:
                        self.get_logger().info(
                            f"[nav2-recovery] {rid_str}: ✓ Got Gazebo ground truth after {i+1} attempts"
                        )
                        break
                
                if gazebo_gt is None:
                    self.get_logger().warn(
                        f"[nav2-recovery] {rid_str}: ✗ Failed to get Gazebo ground truth - using TF fallback. "
                        f"Available robots in cache: {list(self.gazebo_ground_truth.keys())}"
                    )
                    # Fallback: Use odom->base_footprint TF to estimate map pose
                    # This is less accurate but should work if Gazebo ModelStates is unavailable
                    try:
                        odom_frame = f"{robot_id}/odom"
                        base_frame = f"{robot_id}/base_footprint"
                        odom_to_base = self.tf_buffer.lookup_transform(
                            odom_frame, base_frame, rclpy_time.Time(), timeout=rclpy_duration.Duration(seconds=0.5)
                        )
                        
                        # Use state.x, y as map pose (from AMCL, may be inaccurate)
                        # This is a fallback when Gazebo ground truth is unavailable
                        map_x = state.x
                        map_y = state.y
                        
                        # Extract yaw from state or odom
                        odom_yaw = 0.0
                        try:
                            odom_qx = odom_to_base.transform.rotation.x
                            odom_qy = odom_to_base.transform.rotation.y
                            odom_qz = odom_to_base.transform.rotation.z
                            odom_qw = odom_to_base.transform.rotation.w
                            odom_yaw = math.atan2(
                                2.0 * (odom_qw * odom_qz + odom_qx * odom_qy),
                                1.0 - 2.0 * (odom_qy * odom_qy + odom_qz * odom_qz)
                            )
                        except:
                            pass
                        
                        # For fallback, assume map_yaw = odom_yaw (no rotation offset)
                        map_yaw = odom_yaw
                        gazebo_gt = (map_x, map_y, map_yaw)
                        
                        self.get_logger().warn(
                            f"[nav2-recovery] {rid_str}: Using TF fallback - map pose from state: "
                            f"({map_x:.3f}, {map_y:.3f}), yaw={math.degrees(map_yaw):.1f}° "
                            f"[may be inaccurate if AMCL failed]"
                                )
                    except Exception as e:
                        self.get_logger().warn(
                            f"[nav2-recovery] {rid_str}: ✗ TF fallback failed: {e}. "
                            f"Using local calculation from state.x,y"
                        )
                        # Final fallback: Use state.x, y directly (AMCL-based, may be inaccurate)
                        # Calculate map->odom assuming odom at origin (0,0,0)
                        map_x = state.x
                        map_y = state.y
                        map_yaw = 0.0  # Assume no rotation
                        gazebo_gt = (map_x, map_y, map_yaw)
                        self.get_logger().warn(
                            f"[nav2-recovery] {rid_str}: Using local fallback - map pose from state: "
                            f"({map_x:.3f}, {map_y:.3f}), yaw=0° [assumes odom at origin]"
                        )
            
            map_x, map_y, map_yaw = gazebo_gt
            self.get_logger().info(
                f"[nav2-recovery] {rid_str}: ✓ Got Gazebo ground truth: pos=({map_x:.3f}, {map_y:.3f}), "
                f"yaw={math.degrees(map_yaw):.1f}°"
            )
            
            # Get odom->base_footprint transform (this is maintained by Gazebo, we don't modify it)
            odom_frame = f"{robot_id}/odom"
            base_frame = f"{robot_id}/base_footprint"
            odom_to_base = None
            try:
                odom_to_base = self.tf_buffer.lookup_transform(
                    odom_frame, base_frame, rclpy_time.Time(), timeout=rclpy_duration.Duration(seconds=0.5)
                )
                odom_x = odom_to_base.transform.translation.x
                odom_y = odom_to_base.transform.translation.y
                
                # Extract yaw from odom->base_footprint quaternion
                odom_qx = odom_to_base.transform.rotation.x
                odom_qy = odom_to_base.transform.rotation.y
                odom_qz = odom_to_base.transform.rotation.z
                odom_qw = odom_to_base.transform.rotation.w
                odom_yaw = math.atan2(
                    2.0 * (odom_qw * odom_qz + odom_qx * odom_qy),
                    1.0 - 2.0 * (odom_qy * odom_qy + odom_qz * odom_qz)
                )
                
                self.get_logger().info(
                    f"[nav2-recovery] {rid_str}: ✓ Got odom->base_footprint: pos=({odom_x:.3f}, {odom_y:.3f}), "
                    f"yaw={math.degrees(odom_yaw):.1f}°"
                )
            except TransformException as e:
                self.get_logger().warn(
                    f"[nav2-recovery] {rid_str}: ✗ Failed to get odom->base_footprint: {e}. "
                    f"Using local calculation - assuming robot at odom origin (0,0,0)."
                )
                # Final fallback: Assume odom frame at origin
                # This means map->odom translation = map pose directly
                odom_x, odom_y, odom_yaw = 0.0, 0.0, 0.0
            
            # Calculate map->odom transform
            # If odom is at origin (fallback case), map->odom = map pose directly
            if odom_x == 0.0 and odom_y == 0.0 and odom_yaw == 0.0:
                # Local calculation: assume odom at origin
                # map->odom translation = map pose, rotation = map_yaw
                map_to_odom_x = map_x
                map_to_odom_y = map_y
                map_to_odom_yaw = map_yaw
                self.get_logger().info(
                    f"[nav2-recovery] {rid_str}: Using local calculation - map->odom = map pose directly: "
                    f"translation=({map_to_odom_x:.3f}, {map_to_odom_y:.3f}), "
                    f"yaw={math.degrees(map_to_odom_yaw):.1f}°"
                )
            else:
                # Normal case: Calculate map->odom from map pose and odom pose
                # map_pose = map->odom * odom_pose
                # Therefore: map->odom = map_pose * inverse(odom_pose)
                # For 2D: map->odom_translation = map_pos - rotate(map->odom_rotation, odom_pos)
                #         map->odom_yaw = map_yaw - odom_yaw
                
                map_to_odom_yaw = map_yaw - odom_yaw
                
                # Rotate odom position by map->odom rotation (inverse: -map_to_odom_yaw)
                cos_yaw = math.cos(-map_to_odom_yaw)
                sin_yaw = math.sin(-map_to_odom_yaw)
                rotated_x = cos_yaw * odom_x - sin_yaw * odom_y
                rotated_y = sin_yaw * odom_x + cos_yaw * odom_y
                
                # Calculate translation: map_pos - rotated_odom_pos
                map_to_odom_x = map_x - rotated_x
                map_to_odom_y = map_y - rotated_y
            
            self.get_logger().info(
                f"[nav2-recovery] {rid_str}: ✓ Calculated map->odom transform: "
                f"translation=({map_to_odom_x:.3f}, {map_to_odom_y:.3f}), "
                f"yaw={math.degrees(map_to_odom_yaw):.1f}°"
            )
            
            # Publish map->odom transform using TF broadcaster
            # This directly resets the localization without needing AMCL initial pose
            try:
                map_frame = "map"
                transform_stamped = TransformStamped()
                transform_stamped.header.stamp = self.get_clock().now().to_msg()
                transform_stamped.header.frame_id = map_frame
                transform_stamped.child_frame_id = odom_frame
                
                transform_stamped.transform.translation.x = map_to_odom_x
                transform_stamped.transform.translation.y = map_to_odom_y
                transform_stamped.transform.translation.z = 0.0
                
                # Convert yaw to quaternion
                cy = math.cos(map_to_odom_yaw * 0.5)
                sy = math.sin(map_to_odom_yaw * 0.5)
                transform_stamped.transform.rotation.x = 0.0
                transform_stamped.transform.rotation.y = 0.0
                transform_stamped.transform.rotation.z = sy
                transform_stamped.transform.rotation.w = cy
                
                # Publish the transform
                self.tf_broadcaster.sendTransform(transform_stamped)
                self.get_logger().info(
                    f"[nav2-recovery] {rid_str}: ✓ Published map->odom TF transform. "
                    f"Nav2 components (costmap, planner, controller) will automatically align."
                )
                
                # Publish multiple times to ensure it's received (TF can be rate-limited)
                for _ in range(5):
                    rclpy.spin_once(self, timeout_sec=0.05)
                    transform_stamped.header.stamp = self.get_clock().now().to_msg()
                    self.tf_broadcaster.sendTransform(transform_stamped)
                
            except Exception as e:
                self.get_logger().error(
                    f"[nav2-recovery] {rid_str}: ✗ Failed to publish map->odom TF: {type(e).__name__}: {e}"
                )
                import traceback
                self.get_logger().error(f"Traceback:\n{traceback.format_exc()}")
            
            # Final recovery summary
            recovery_summary = (
                f"[nav2-recovery] {rid_str}: Nav2 recovery sequence completed - "
                f"cancel={'✓' if cancel_success else '✗'}, "
                    f"costmaps=[global={'✓' if global_cleared else '✗'}, local={'✓' if local_cleared else '✗'}], "
                    f"map->odom_TF={'✓' if gazebo_gt is not None else '✗'}. "
                    f"Nav2 components (costmap, planner, controller) automatically aligned."
            )
            self.get_logger().info(recovery_summary)
            
        except Exception as e:
            import traceback
            self.get_logger().error(
                f"[nav2-recovery] {rid_str}: ✗ CRITICAL ERROR during Nav2 recovery: {type(e).__name__}: {e}\n"
                f"Traceback:\n{traceback.format_exc()}"
            )
    
    def _nav2_distance(self, x1: float, y1: float, x2: float, y2: float, 
                       robot_id: Optional[str] = None) -> float:
        """Calculate distance using Nav2 path planning if available, otherwise euclidean.
        
        This is a drop-in replacement for _euclidean_distance that uses actual
        path planning when Nav2 is available.
        
        Args:
            x1, y1: Start position
            x2, y2: Goal position
            robot_id: Robot ID to use its planner. If None, uses first available robot.
        """
        # Calculate euclidean distance for comparison
        euclidean_dist = _euclidean_distance(x1, y1, x2, y2)
        
        # Try Nav2 path planning
        nav2_dist = self._compute_nav2_path_distance(x1, y1, x2, y2, robot_id)
        if nav2_dist is not None:
            # Always log comparison (even from cache) at debug level
            # Log comparison between euclidean and Nav2 path distance
            diff = nav2_dist - euclidean_dist
            diff_pct = (diff / euclidean_dist * 100.0) if euclidean_dist > 0.0 else 0.0
            return nav2_dist
        
        # Fallback to euclidean distance
        # Log which robot failed (for debugging intermittent failures)
        robot_str = f"robot={robot_id}" if robot_id else "robot=unknown"
        # self.get_logger().debug(
        #     f"[path_dist] {robot_str} ({x1:.2f}, {y1:.2f}) -> ({x2:.2f}, {y2:.2f}): "
        #     f"euclidean={euclidean_dist:.3f}m (Nav2 unavailable, using euclidean)"
        # )
        return euclidean_dist

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

            d_drive = self._nav2_distance(robot.x, robot.y, cx, cy, robot.robot_id)
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
                d_to_pick = self._nav2_distance(curr_x, curr_y, px, py, robot.robot_id)
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

                # determine picked item weight from task-type defaults or explicit overrides
                w = self._resolve_pick_weight(task, i)

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
                        d_loaded = self._nav2_distance(curr_x, curr_y, dx, dy, robot.robot_id)
                        weight_term = float(getattr(self, 'load_weight_factor', 0.05)) * carried_weight
                        E_drive_loaded = self.k_drive_wh_per_m * (loaded_factor + weight_term) * d_loaded
                        t_drive_loaded = d_loaded / v_loaded_mps if v_loaded_mps > 1e-6 else 0.0
                        E_total += E_drive_loaded
                        t_total += t_drive_loaded
                        curr_x, curr_y = float(dx), float(dy)
                        continue

                    # drive while carrying current weight
                    d_loaded = self._nav2_distance(curr_x, curr_y, dx, dy, robot.robot_id)
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
                d_loaded = self._nav2_distance(curr_x, curr_y, dx, dy, robot.robot_id)
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
                d_to_pick = self._nav2_distance(curr_x, curr_y, px, py, robot.robot_id)
                E_drive = self.k_drive_wh_per_m * d_to_pick
                t_drive = d_to_pick / v_empty_mps if v_empty_mps > 1e-6 else 0.0

                E_total += E_drive
                t_total += t_drive

                # Pick overheads
                E_total += self.E_pick_wh
                t_total += float(task.pick_wait_s)

                # Add item weight honoring task-type defaults
                w = self._resolve_pick_weight(task, i)

                carried_weight += w

                curr_x, curr_y = float(px), float(py)

            # Move to corresponding drop (loaded drive)
            if i < len(drops):
                dx, dy = drops[i]
                d_loaded = self._nav2_distance(curr_x, curr_y, dx, dy, robot.robot_id)
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
                # the weight resolved for this pick/drop pair
                dropped_weight = self._resolve_pick_weight(task, i)
                carried_weight = max(0.0, carried_weight - dropped_weight)

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
            d = self._nav2_distance(x, y, ch.x, ch.y, None)  # No specific robot for charger selection
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
            d = self._nav2_distance(x, y, ch.x, ch.y, None)  # No specific robot for charger selection
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
                        # Charge duration based on remaining energy and absolute Wh/s charge rate.
                        # Example: charge_rate_wh_per_s=0.1 → 40Wh needs 400s, 60Wh needs 600s.
                        remaining_wh = max(0.0, cap_wh - soc)
                        if self.charge_rate_wh_per_s <= 0.0 or remaining_wh <= 0.0:
                            charge_duration_s = 0.0
                        else:
                            charge_duration_s = remaining_wh / self.charge_rate_wh_per_s

                        charge_task = TaskSpec(
                            task_id=-1,
                            task_type="CHARGE",
                            picks=[(x, y)],
                            drops=[(charger.x, charger.y)],
                            pick_wait_s=0.0,
                            drop_wait_s=0.0,
                            charge_duration_s=charge_duration_s,
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
    # Demo / initial tasks
    # ------------------------------------------------------------------

    def _init_demo_tasks(self) -> None:
        """Create a pool of demo tasks with 3 different task types randomly combined.

        Coordinates are in the 'map' frame and loosely correspond to a
        warehouse-like layout for the AWS small warehouse world.
        """
        # Type 1: Simple PICK_DROP tasks (1 pick, 1 drop)
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
            # ( -1.90,  -4.75,  -3.45,   1.10),  # 31
            # (  0.20,  -2.70,   0.25,   5.30),  # 32
            # (  0.50,   5.10,  -4.25,   0.60),  # 33
            # (  0.95,  -6.40,  -3.75,   3.95),  # 34
            # ( -3.85,   3.10,   0.75,   1.05),  # 35
            # ( -0.55,  -4.15,  -4.75,   0.75),  # 36
            # ( -4.20,  -9.45,   1.85,  -4.80),  # 37
            # ( -3.30,  -0.25,  -3.90,   6.30),  # 38
            # (  5.10,   1.30,  -3.95,  -2.95),  # 39
            # ( -3.10,  -8.35,  -2.30,  -4.45),  # 40
        ]

        # Type 2: MULTI_PICK_DROP tasks (2 picks -> drop -> drop, three-stop route)
        # picks = [pick_loc, pick_loc] (two picks at the same location)
        # drops = [drop1, drop2]
        # Waypoint order: pick_loc -> pick_loc -> drop1 -> drop2
        # Each entry is (pick_x, pick_y, drop1_x, drop1_y, drop2_x, drop2_y)
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
            # ( -5.50,   6.75,  -5.55,  -0.20,  -6.10,  -8.85),  # group 21
            # ( -3.65,  -5.20,  -0.35,  -6.55,   3.85,  -5.35),  # group 22
            # ( -3.75,   8.75,  -5.85,   2.25,   1.25,   5.65),  # group 23
            # ( -3.40,  -7.35,  -0.35,  -4.25,  -3.20,  -9.55),  # group 24
            # ( -3.85,  -1.00,   0.20,  -2.25,  -4.75,  -4.35),  # group 25
            # (  1.20,  -8.60,  -0.70,  -1.20,  -3.70,  -1.00),  # group 26
            # (  0.60,   5.10,  -5.75,   6.55,  -0.20,  -0.75),  # group 27
            # ( -5.85,  -4.25,   0.40,  -5.80,   1.00,   2.00),  # group 28
            # ( -0.05,  -3.65,   1.00,  -8.20,   3.45,  -8.85),  # group 29
            # (  5.75,  -5.75,  -0.05,   0.45,  -1.50,  -0.85),  # group 30
        ]

        # Type 3: DUAL_PICK_MULTI_DROP tasks
        # picks = [pick_loc, drop], drops = [drop, pick_loc]
        # Waypoint order: pick_loc -> drop -> drop -> pick_loc
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
            # ( -5.80,  -4.10,   0.85,   3.65),  # task 21
            # ( -4.90,   5.75,  -0.05,  -8.10),  # task 22
            # ( -3.20,   8.80,  -1.75,  -5.15),  # task 23
            # (  3.25,   1.70,   5.40,  -7.55),  # task 24
            # ( -4.00,  -1.80,   0.85,   2.50),  # task 25
            # ( -2.75,  -4.15,  -3.10,   7.70),  # task 26
            # (  5.45,  -5.50,  -4.40,  -8.25),  # task 27
            # ( -6.00,   6.60,  -2.95,  -3.30),  # task 28
            # (  1.40,  -9.55,  -4.60,  -3.70),  # task 29
            # ( -4.30,  -4.60,  -1.15,  -1.00),  # task 30
        ]

        # Combine all task data and randomly select task types
        self.pending_tasks = []
        tid = 1
        
        # Create tasks from demo_specs (PICK_DROP)
        pick_drop_tasks = []
        for (px, py, dx, dy) in demo_specs:
            pick_drop_tasks.append(('PICK_DROP', (px, py), (dx, dy), None, None))
        
        # Create tasks from demo_pairs_multi (MULTI_PICK_DROP: 2 picks -> drop -> drop)
        # picks = [pick_loc, pick_loc] (two picks at the same location)
        # drops = [drop1, drop2]
        triplets = []
        for entry in demo_pairs_multi:
            pick_loc = (entry[0], entry[1])
            drop1 = (entry[2], entry[3])
            drop2 = (entry[4], entry[5])
            triplets.append((pick_loc, drop1, drop2))


        multi_tasks = []
        for i in range(len(triplets)):
            pick_loc, drop1, drop2 = triplets[i]

            # two identical picks at the same location (two items to pick)
            picks = [pick_loc, pick_loc]
            drops = [drop1, drop2]

            # per-pick weights for the two items
            pick_weights = [round(0.9 + 0.15 * ((i + j) % 4), 2) for j in range(2)]
            # deliveries: first drop receives pick index 0, second drop receives pick index 1
            deliveries = [[0], [1]]
            multi_tasks.append(('MULTI_PICK_DROP', picks, drops, pick_weights, deliveries))
        
        # Create tasks from demo_pairs (DUAL_PICK_MULTI_DROP)
        # picks = [pick_loc, drop], drops = [drop, pick_loc]
        # Waypoint order: pick_loc -> drop -> drop -> pick_loc
        triplets = []
        for entry in demo_pairs:
            pick_loc = (entry[0], entry[1])
            drop = (entry[2], entry[3])
            triplets.append((pick_loc, drop))

        dual_tasks = []
        for i in range(len(triplets)):
            pick_loc, drop = triplets[i]

            # two identical picks at the same location (two items to pick)
            picks = [pick_loc, drop]
            drops = [drop, pick_loc]

            # per-pick weights for the two items
            pick_weights = [round(0.9 + 0.15 * ((i + j) % 4), 2) for j in range(2)]
            # deliveries: first drop receives pick index 0, second drop receives pick index 1
            deliveries = [[0], [1]]
            dual_tasks.append(('DUAL_PICK_MULTI_DROP', picks, drops, pick_weights, deliveries))
        
        # Randomly combine all task types
        all_tasks = pick_drop_tasks + dual_tasks + multi_tasks
        # random.shuffle(all_tasks)
        
        for task_data in all_tasks:
            task_type, picks_or_pick, drops_or_drop, pick_weights, deliveries = task_data
            
            if task_type == "PICK_DROP":
                px, py = picks_or_pick
                dx, dy = drops_or_drop
                # Add default weight for PICK_DROP tasks
                default_weight = round(0.9 + 0.15 * (tid % 4), 2)
                self.pending_tasks.append(
                    TaskSpec(
                        task_id=tid,
                        task_type="PICK_DROP",
                        picks=[(px, py)],
                        drops=[(dx, dy)],
                        pick_weights=[default_weight],
                        pick_wait_s=2.0,
                        drop_wait_s=2.0,
                        charge_duration_s=0.0,
                    )
                )
            elif task_type == "DUAL_PICK_MULTI_DROP":
                self.pending_tasks.append(
                    TaskSpec(
                            task_id=tid,
                            task_type="DUAL_PICK_MULTI_DROP",
                            picks=picks_or_pick,
                            drops=drops_or_drop,
                            pick_weights=pick_weights,
                            deliveries=deliveries,
                            pick_wait_s=1.5,
                            drop_wait_s=2.0,
                            charge_duration_s=0.0,
                        )
                    )
            elif task_type == "MULTI_PICK_DROP":
                self.pending_tasks.append(
                    TaskSpec(
                        task_id=tid,
                        task_type="MULTI_PICK_DROP",
                        picks=picks_or_pick,
                        drops=drops_or_drop,
                        pick_weights=pick_weights,
                        deliveries=deliveries,
                        pick_wait_s=1.5,
                        drop_wait_s=2.0,
                        charge_duration_s=0.0,
                )
            )
            tid += 1

        # Keep task_queue pointing at the same list (pool of tasks)
        self.task_queue = self.pending_tasks
        type_counts = {
            'PICK_DROP': sum(1 for t in self.pending_tasks if t.task_type == 'PICK_DROP'),
                'DUAL_PICK_MULTI_DROP': sum(1 for t in self.pending_tasks if t.task_type == 'DUAL_PICK_MULTI_DROP'),
                'MULTI_PICK_DROP': sum(1 for t in self.pending_tasks if t.task_type == 'MULTI_PICK_DROP'),
                }
        self.get_logger().info(
            f'Initialized demo task pool with {len(self.pending_tasks)} tasks '
            f'(PICK_DROP: {type_counts["PICK_DROP"]}, '
            f'DUAL_PICK_MULTI_DROP: {type_counts["DUAL_PICK_MULTI_DROP"]}, '
            f'MULTI_PICK_DROP: {type_counts["MULTI_PICK_DROP"]})'
        )

    def _optimize_with_algorithm(self, candidate_tasks: List[TaskSpec], algorithm_name: str) -> Tuple[Dict[str, List[TaskSpec]], float, Optional[float], Optional[float]]:
        """Use mealpy to optimally assign all tasks to robots and optimize task sequences.
        
        This function optimizes both:
        1. Which robot gets which task (allocation)
        2. The order of tasks for each robot (sequencing)
        
        Encoding: For each task, we use two values:
        - robot_key: [0, num_robots) → determines which robot
        - seq_key: random key → determines order within that robot
        
        Returns:
            Tuple of (Dict mapping robot_id to list of assigned tasks in optimal order, best_fitness, max_makespan)
        """
        if not candidate_tasks:
            return {rid: [] for rid in self.robot_ids}, 0.0, 0.0
        
        if not self.robot_ids:
            self.get_logger().warn("Cannot allocate tasks: no robots available")
            return {}, 0.0, None
        
        K = len(candidate_tasks)
        num_robots = len(self.robot_ids)
        robot_list = list(self.robot_ids)  # Fixed order for indexing
        
        # Makespan-dominant weighting (w_t > w_e)
        w_t = 1.0
        w_e = 0.1
        
        # ---------- objective ----------
        # Storage for max_makespan, total energy, and total waiting time from best solution evaluation
        best_makespan_storage = [None]
        best_energy_storage = [None]  # Store total energy consumption
        best_waiting_time_storage = [None]  # Store total charging waiting time (queue delay)
        robot_times_storage = [None]  # Store robot_times list for per-robot makespan
        store_makespan_flag = [False]  # Flag to control when to store makespan-related metrics
        
        # Capture include_charge_in_objective flag for use in objective function
        include_charge_flag = self.include_charge_in_objective
        
        def _objective_function(solution: np.ndarray) -> float:
            # Decode solution: [robot_key_1, seq_key_1, charge_key_1, robot_key_2, seq_key_2, charge_key_2, ...]
            # Each task has 3 values: robot assignment, sequence key, and charge decision
            assignments: Dict[str, List[Tuple[float, float, TaskSpec]]] = {rid: [] for rid in robot_list}
            
            for i, task in enumerate(candidate_tasks):
                robot_key = solution[3 * i]  # First value: robot assignment [-1.0, 1.0]
                seq_key = solution[3 * i + 1]  # Second value: sequence key
                charge_key = solution[3 * i + 2]  # Third value: charge decision [-1.0, 1.0]
                
                # If charging is not included in objective, ignore charge_key (set to -1.0 = no charge)
                if not include_charge_flag:
                    charge_key = -1.0
                
                # Map robot_key from [-1.0, 1.0] to [0, num_robots) for robot_id
                # Step 1: Convert [-1.0, 1.0] to [0.0, 1.0]
                normalized_key = (robot_key + 1.0) / 2.0
                # Step 2: Scale to [0.0, num_robots) and convert to integer index
                robot_idx = int(np.clip(normalized_key * num_robots, 0, num_robots - 1))
                robot_id = robot_list[robot_idx]
                
                assignments[robot_id].append((seq_key, charge_key, task))
            
            # Calculate total cost across all robots
            num_chargers = max(1, len(self.chargers))
            chargers_list = list(self.chargers)
            
            total_cost = 0.0
            max_makespan = 0.0  # Track maximum makespan across robots
            total_penalty = 0.0
            early_charge_count = 0.0
            
            # Track per-robot costs for load balancing
            robot_times: List[float] = []
            robot_energies: List[float] = []
            robot_task_counts: List[int] = []
            
            # Global charger occupancy tracker (shared across all robots for realistic simulation)
            charger_next_free: Dict[int, float] = {i: 0.0 for i in range(num_chargers)}
            charge_duration = 1.0
            
            # ChargeEvent list for time-based event-driven charging simulation
            # Format: (robot_id, task_idx_in_robot, reachable_chargers, time_before_charge, simulated_charge_end_time)
            # reachable_chargers: List[Tuple[int, float, float]] = (charger_idx, arrive_time, energy_cost)
            # arrive_time: actual arrival time at that charger (based on the Phase 1 simulation)
            # energy_cost: energy needed to travel to the charger
            # time_before_charge: accumulated time prior to initiating the charge
            # simulated_charge_end_time: Phase 1 charge completion time (used to compute post-charge timing)
            charge_events: List[Tuple[str, int, List[Tuple[int, float, float]], float, float]] = []
            
            for rid in robot_list:
                assigned_entries = assignments[rid]
                robot_task_counts.append(len(assigned_entries))

                if not assigned_entries:
                    robot_times.append(0.0)
                    robot_energies.append(0.0)
                    continue
                
                state = self.robot_states.get(rid)
                if state is None:
                    # Infeasibility penalty: robot not available
                    total_penalty += 1e6
                    robot_times.append(0.0)
                    robot_energies.append(0.0)
                    continue

                # Sort tasks by sequence key
                sorted_tasks = sorted(assigned_entries, key=lambda x: x[0])
                seq_with_charge = [(charge_key, t) for _, charge_key, t in sorted_tasks]
                seq = [t for _, t in seq_with_charge]
                charge_keys = [charge_key for charge_key, _ in seq_with_charge]
                
                cap_wh = self.robot_cap_wh.get(rid, 120.0)
                
                # Simulate robot execution
                # Use full charge capacity for optimization (fair comparison across all robots)
                sim_robot = RobotState(
                    robot_id=state.robot_id,
                    namespace=state.namespace,
                    x=state.x,
                    y=state.y,
                    soc=cap_wh,  # Start from 100% SOC for fair optimization
                    available=state.available,
                )

                # Use full charge capacity for optimization (fair comparison)
                energy = cap_wh  # Start from 100% SOC instead of current state.soc
                robot_energy = 0.0  # Energy consumed so far
                robot_time = 0.0
                penalty = 0.0  # Infeasibility penalty: accumulates for infeasible solutions
                
                for task_idx, t in enumerate(seq):
                    charge_key = charge_keys[task_idx]  # Get charge decision for this task
                    
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
                        # Infeasibility penalty 1: task energy estimation failed
                        required_energy, t_to_pick = 0.0, 0.0
                        penalty += 1e6
                        break

                    # ================================================================
                    # If charging is NOT included in objective:
                    # - Skip all charging logic (charge_key is already set to -1.0)
                    # - Assume infinite energy (no energy constraints)
                    # - Only optimize task allocation and sequencing (makespan)
                    # - Charging will be handled later by feasibility/threshold
                    # ================================================================
                    if not include_charge_flag:
                        # No energy constraint: just accumulate time and update position
                        robot_time += t_to_pick
                        
                        # Update robot position
                        if t.drops:
                            sim_robot.x, sim_robot.y = t.drops[-1]
                        else:
                            sim_robot.x = t.drop_x
                            sim_robot.y = t.drop_y
                        continue
                    
                    # ================================================================
                    # Below: charging included in objective (include_charge_flag=True)
                    # ================================================================
                    
                    # Check if charging is needed based on charge_key
                    # charge_key > 0: charge before task, charge_key ≤ 0: do not charge
                    should_charge = (charge_key > 0.0)
                    
                    # Penalty: charge_key ≤ 0 but insufficient energy
                    if not should_charge and energy < required_energy:
                        # Infeasibility penalty: charge_key ≤ 0 but insufficient energy to perform task
                        penalty += 1e6
                        break

                    # Early charge count
                    if should_charge:
                        energy_ratio = energy / cap_wh
                        if energy_ratio > 0.8:
                            early_charge_count += 1

                    # If should_charge, process charging (1st phase: simplified simulation)
                    if should_charge:
                        # Collect all charger candidates
                        if not chargers_list:
                            # Infeasibility penalty: no charger available
                            penalty += 1e6
                            break
                        
                        # Store time before charge (for Phase 2 time propagation)
                        time_before_charge = robot_time
                        
                        # ------------------------------------------------------------------
                        # Dynamic charge duration based on remaining energy and Wh/s rate
                        # remaining_wh: Wh needed to reach full charge
                        # charge_rate_wh_per_s: Wh charged per second
                        # Example: charge_rate_wh_per_s=0.1 → 40Wh needs 400s, 60Wh needs 600s.
                        # ------------------------------------------------------------------
                        remaining_wh = max(0.0, cap_wh - energy)
                        if self.charge_rate_wh_per_s <= 0.0 or remaining_wh <= 0.0:
                            charge_duration = 0.0
                        else:
                            charge_duration = remaining_wh / self.charge_rate_wh_per_s
                        
                        # Calculate travel time and energy to each charger candidate
                        reachable_chargers_list: List[Tuple[int, float, float]] = []  # (charger_idx, arrive_time, energy_cost)
                        for ch_idx, ch in enumerate(chargers_list):
                            try:
                                e_to_ch, t_to_ch = self.estimate_task_energy(
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
                                        picks=[(ch.x, ch.y)],
                                        drops=[(ch.x, ch.y)],
                                        pick_wait_s=0.0,
                                        drop_wait_s=0.0,
                                        charge_duration_s=0.0,
                                    ),
                                )
                                # Only include reachable chargers
                                if energy >= e_to_ch:
                                    arrive_time_at_ch = robot_time + t_to_ch
                                    reachable_chargers_list.append((ch_idx, arrive_time_at_ch, e_to_ch))
                            except Exception:
                                pass
                        
                        if not reachable_chargers_list:
                            # Infeasibility penalty: charge_key > 0 but cannot reach any charger
                            penalty += 1e6
                            break
                        
                        # Select best charger considering occupancy (same logic as Phase 2)
                        # This ensures charging time is properly reflected in makespan
                        best_ch_idx = None
                        best_charge_end = float('inf')
                        best_arrive_time = 0.0
                        best_e_cost = 0.0
                        
                        for ch_idx, arrive_time, e_cost in reachable_chargers_list:
                            charge_end_time = arrive_time + charge_duration
                            
                            if charge_end_time < best_charge_end:
                                best_charge_end = charge_end_time
                                best_ch_idx = ch_idx
                                best_arrive_time = arrive_time
                                best_e_cost = e_cost
                        
                        if best_ch_idx is None:
                            penalty += 1e6
                            break
                        
                        first_ch = chargers_list[best_ch_idx]
                        
                        # Update energy and time
                        energy -= best_e_cost
                        robot_energy += best_e_cost
                        
                        # Move to charger: accumulate travel time (DO NOT overwrite robot_time)
                        # best_arrive_time = time_before_charge + t_to_ch (calculated at line 2020)
                        travel_time_to_charger = best_arrive_time - time_before_charge
                        robot_time += travel_time_to_charger  # Accumulate travel time
                        robot_time += charge_duration        # Accumulate charge duration

                        energy = cap_wh  # Instant full charge
                        
                        # Record charge event for Phase 2 processing (including time_before_charge, simulated_charge_end_time, and charge_duration)
                        charge_events.append((rid, task_idx, reachable_chargers_list, time_before_charge, best_charge_end, charge_duration))
                        
                        # Update robot position (using first reachable charger for continuity)
                        sim_robot.x = first_ch.x
                        sim_robot.y = first_ch.y
                    
                    # Re-estimate after charging / task execution
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
                        # Infeasibility penalty 4: insufficient energy after charging (task will fail)
                        penalty += 1e6
                        break

                    # Execute task (after charging, energy is sufficient)
                    energy -= required_energy
                    robot_energy += required_energy
                    robot_time += t_to_pick
                    
                    # Update robot position
                    if t.drops:
                        sim_robot.x, sim_robot.y = t.drops[-1]
                    else:
                        sim_robot.x = t.drop_x
                        sim_robot.y = t.drop_y
                
                total_penalty += penalty
                # Store robot time (will be updated in Phase 2 based on actual charge end times)
                robot_times.append(robot_time)
                robot_energies.append(robot_energy)
                
                # Add infeasibility penalty: penalize solutions where tasks will fail
                # This includes:
                # - Energy estimation failures
                # - No charger available when needed
                # - Insufficient energy to reach charger
                # - Insufficient energy after charging to complete task
            
            # ========================================================================
            # Phase 2: Event-driven time-based charger selection and time update
            # ========================================================================
            
            # Track charger availability: charger_idx -> next_free_time
            charger_next_free: Dict[int, float] = {i: 0.0 for i in range(num_chargers)}
            
            # Track total delay per robot: accumulates delays from all charge events
            # Delay = actual_charge_end_time - simulated_charge_end_time (from Phase 1)
            robot_total_delay: Dict[str, float] = {rid: 0.0 for rid in robot_list}
            
            # Track charge events per robot: robot_id -> list of (task_idx, charge_end_time)
            # Used to update robot completion time
            robot_charge_end_times: Dict[str, List[Tuple[int, float]]] = {rid: [] for rid in robot_list}
            
            # Sort charge events by earliest arrive_time (across all reachable chargers)
            # This ensures we process events in chronological order
            if charge_events:
                # Create event list with arrive_time for sorting
                # Each reachable_chargers_list entry is a 3-tuple (charger_idx, arrive_time, energy_cost).
                event_list: List[Tuple[float, str, int, List[Tuple[int, float, float]], float, float, float]] = []
                for robot_id, task_idx, reachable_chargers_list, time_before_charge, simulated_charge_end_time, charge_duration in charge_events:
                    # Use earliest arrive_time for sorting
                    if reachable_chargers_list:
                        # reachable_chargers_list uses the (ch_idx, arrive_time, energy_cost) structure,
                        # so unpacking needs to follow the same three-field layout.
                        earliest_arrive = min(arrive_time for _, arrive_time, _ in reachable_chargers_list)
                        event_list.append(
                            (earliest_arrive, robot_id, task_idx, reachable_chargers_list, time_before_charge, simulated_charge_end_time, charge_duration)
                        )
                
                # Sort by earliest arrive_time
                event_list.sort(key=lambda x: x[0])
                
                # Process each event: select best charger and update robot time immediately
                for earliest_arrive, robot_id, task_idx, reachable_chargers_list, time_before_charge, simulated_charge_end_time, charge_duration in event_list:
                    # Get accumulated delay from previous charge events for this robot
                    # Previous delays affect when this robot actually arrives at chargers
                    accumulated_delay = robot_total_delay.get(robot_id, 0.0)
                    
                    # Select best charger: earliest charge_end = max(arrive_time, charger_next_free) + charge_duration (per-event)
                    # IMPORTANT: Add accumulated delay to arrive_time to reflect delays from previous charges
                    best_charger_idx = None
                    best_charge_end = float('inf')
                    best_arrive_time = 0.0
                    
                    for ch_idx, arrive_time_base, _ in reachable_chargers_list:
                        # Adjust arrive_time with accumulated delay from previous charge events
                        # If robot was delayed by previous charges, it arrives later at all chargers
                        arrive_time = arrive_time_base + accumulated_delay
                        
                        charger_free_time = charger_next_free[ch_idx]
                        charge_start_time = max(arrive_time, charger_free_time)
                        charge_end_time = charge_start_time + charge_duration
                        
                        if charge_end_time < best_charge_end:
                            best_charge_end = charge_end_time
                            best_charger_idx = ch_idx
                            best_arrive_time = arrive_time
                    
                    if best_charger_idx is None:
                        continue
                    
                    # Update charger availability
                    charger_next_free[best_charger_idx] = best_charge_end
                    
                    # Calculate delay: difference between actual charge_end and Phase 1 simulated charge_end
                    # IMPORTANT: Adjust simulated_charge_end_time with previous delays
                    # If robot was delayed by previous charges, simulated_charge_end_time should also be adjusted
                    simulated_charge_end_time_adjusted = simulated_charge_end_time + accumulated_delay
                    
                    # Delay = actual charge_end - adjusted simulated charge_end
                    # This represents the NEW delay caused by charger occupancy (wait time) for this charge event
                    delay = best_charge_end - simulated_charge_end_time_adjusted

                    EPS = 1e-6

                    if delay < -EPS:
                        self.get_logger().warn(
                            f"[objective] Negative delay: delay={delay:.2f}, "
                            f"simulated_charge_end_time_adjusted={simulated_charge_end_time_adjusted:.2f}, "
                            f"best_charge_end={best_charge_end:.2f}"
                        )
                        delay = 0.0
                    
                    # Accumulate delay for this robot (multiple charges can add up)
                    robot_total_delay[robot_id] += delay
            
            # Update robot_times based on event-driven time updates from Phase 2
            # For robots with charges, use robot_current_time (which was updated to charge_end)
            # For robots without charges, use the original robot_time (robot_current_time == robot_times[rid_idx])
            for rid_idx, rid in enumerate(robot_list):
                if rid_idx < len(robot_times):
                    # robot_current_time was initialized for all robots in Phase 2
                    # For robots with charges: updated to charge_end in Phase 2
                    # For robots without charges: remains as Phase 1 completion time
                    robot_times[rid_idx] += robot_total_delay[rid]
                    max_makespan = max(max_makespan, robot_times[rid_idx])
            
            # Store max_makespan, total energy, robot_times, and total waiting time only when flag is set
            if store_makespan_flag[0]:
                best_makespan_storage[0] = max_makespan
                total_energy = sum(robot_energies) if robot_energies else 0.0
                best_energy_storage[0] = total_energy
                robot_times_storage[0] = robot_times.copy()  # Store per-robot times
                # Total waiting time = sum of charger-queue delays over all robots
                try:
                    total_waiting_time = float(sum(robot_total_delay.values())) if robot_total_delay else 0.0
                except Exception:
                    total_waiting_time = 0.0
                best_waiting_time_storage[0] = total_waiting_time
            
            # ========================================================================
            # Stage 1 Objective Function: Multi-Robot Task Allocation & Sequencing
            # ========================================================================
            # 
            # Objective: Minimize makespan (max completion time)
            # 
            # 📌 NORMALIZED OBJECTIVE: Makespan is normalized to unitless ratio
            #    This ensures the objective is scale-invariant and weights remain meaningful
            #    regardless of experimental settings (task count, map size, robot count, etc.)
            # 
            # Components:
            # 1. Normalized makespan: max_makespan / estimated_max_makespan
            # 2. Infeasibility penalty: fixed large value (1e6) for invalid solutions
            # ========================================================================
            
            # Normalization factors (estimated from current problem instance)
            # These are computed dynamically to adapt to problem scale
            num_tasks = len(candidate_tasks)
            
            # Estimate maximum possible makespan for normalization
            # Strategy: Use a fixed multiplier based on current max_makespan to ensure
            # that the objective function properly penalizes unbalanced solutions
            if max_makespan > 0:
                # Use a fixed multiplier (2.0x) to normalize makespan
                estimated_max_makespan = max_makespan * 2.0
            else:
                # No makespan yet: use heuristic based on task count and number of robots
                # Estimate: if tasks are evenly distributed, each robot gets ~num_tasks/num_robots tasks
                # Rough estimate: ~50 seconds per task (conservative)
                if num_robots > 0:
                    tasks_per_robot = max(1, num_tasks / num_robots)
                    estimated_max_makespan = max(100.0, tasks_per_robot * 50.0)
                else:
                    estimated_max_makespan = max(100.0, num_tasks * 50.0)
            
            # Prevent division by zero
            if estimated_max_makespan < 1e-6:
                estimated_max_makespan = 1.0
            if max_makespan < 1e-6:
                max_makespan = 0.0
            
            # ========================================================================
            # NORMALIZED OBJECTIVE COMPONENT
            # ========================================================================
            
            # Normalized makespan: max_makespan / estimated_max_makespan
            # Range: 0.0 ~ 1.0+ (typically close to 1.0 for good solutions)
            normalized_makespan = max_makespan / estimated_max_makespan
            
            # ========================================================================
            # OBJECTIVE FUNCTION
            # ========================================================================
            
            # Calculate additional cost components
            sum_robot_times = sum(robot_times) if robot_times else 0.0
            total_charge_count = len(charge_events) if charge_events else 0

            # Load-balance penalties (soft constraints)
            total_assigned_tasks = sum(robot_task_counts)
            idle_robot_count = sum(1 for count in robot_task_counts if count == 0)
            if total_assigned_tasks == 0:
                idle_robot_count = 0

            avg_time = (sum_robot_times / num_robots) if (robot_times and num_robots > 0) else 0.0
            if avg_time > 1e-6:
                time_variance = sum((t - avg_time) ** 2 for t in robot_times) / max(1, num_robots)
                normalized_time_std = math.sqrt(time_variance) / max(1e-6, avg_time)
            else:
                normalized_time_std = 0.0

            if num_robots > 0 and total_assigned_tasks > 0:
                avg_tasks_per_robot = total_assigned_tasks / num_robots
                task_variance = sum((c - avg_tasks_per_robot) ** 2 for c in robot_task_counts) / num_robots
                normalized_task_std = math.sqrt(task_variance) / max(1.0, avg_tasks_per_robot)
            else:
                normalized_task_std = 0.0

            time_balance_weight = 0.5
            task_balance_weight = 0.3
            idle_robot_weight = 0.6
            imbalance_scale = max(max_makespan, 1.0)
            fairness_penalty = (
                time_balance_weight * normalized_time_std * imbalance_scale
                + task_balance_weight * normalized_task_std * imbalance_scale
                + idle_robot_weight * idle_robot_count * imbalance_scale
            )
            
            if include_charge_flag:
                # Include charging in objective: penalize charge count
                total_cost = (
                    max_makespan           # Previously normalized_makespan; now use the raw value
                    + 0.3 * sum_robot_times  # Covers travel, detours, and idle waiting
                    + 100 * total_charge_count  # Discourage excessive charging
                    + 1e5 * early_charge_count  # Discourage charging too early
                    + total_penalty
                    + 10 * fairness_penalty
                )
            else:
                # Exclude charging from objective: only optimize makespan and task distribution
                # No penalty for energy constraints, no charge count penalty
                total_cost = (
                    max_makespan           # Makespan minimization
                    + 0.02 * sum_robot_times  # Load balancing
                    + total_penalty
                )
            
            # Final objective: minimize total_cost (all components are unitless)
            return total_cost
        
        # ---------- mealpy problem ----------
        # Solution dimension: 3 * K (robot_key + seq_key + charge_key for each task)
        problem = {
            "obj_func": _objective_function,
            "bounds": FloatVar(lb=(-1.0,) * (3 * K), ub=(1.0,) * (3 * K)),
            "minmax": "min",
            # Use console logging to disable tqdm progress bar and enable epoch logs
            "log_to": "console",
        }

        # Create optimizer based on algorithm_name
        epoch = self.optimization_epoch
        pop_size = self.optimization_pop_size
        
        if algorithm_name == 'BaseGA':
            # Try BaseGA first, fallback to OriginalGA if BaseGA doesn't exist
            try:
                model = GA.BaseGA(epoch=epoch, pop_size=pop_size, pc=0.9, pm=0.05)
            except AttributeError:
                model = GA.OriginalGA(epoch=epoch, pop_size=pop_size, pc=0.9, pm=0.05)
        elif algorithm_name == 'GA':
            model = GA.OriginalGA(epoch=epoch, pop_size=pop_size, pc=0.9, pm=0.05)
        elif algorithm_name == 'PSO' or algorithm_name == 'OriginalPSO':
            model = PSO.OriginalPSO(epoch=epoch, pop_size=pop_size, c1=2.05, c2=2.05, w=0.4)
        elif algorithm_name == 'OriginalHS':
            model = HS.OriginalHS(epoch=epoch, pop_size=pop_size)
        elif algorithm_name == 'OriginalWCA':
            model = WCA.OriginalWCA(epoch=epoch, pop_size=pop_size)
        elif algorithm_name == 'CRO' or algorithm_name == 'OriginalCRO':
            model = CRO.OriginalCRO(epoch=epoch, pop_size=pop_size, po=0.4, Fb=0.9, Fa=0.1, Fd=0.1, Pd=0.1, GCR=0.1, G=0.02)
        elif algorithm_name == 'JA' or algorithm_name == 'OriginalJA':
            model = JA.OriginalJA(epoch=epoch, pop_size=pop_size)
        elif algorithm_name == 'VCS' or algorithm_name == 'OriginalVCS':
            model = VCS.OriginalVCS(epoch=epoch, pop_size=pop_size)
        elif algorithm_name == 'SRSR' or algorithm_name == 'OriginalSRSR':
            model = SRSR.OriginalSRSR(epoch=epoch, pop_size=pop_size)
        elif algorithm_name == 'QSA' or algorithm_name == 'OriginalQSA':
            model = QSA.OriginalQSA(epoch=epoch, pop_size=pop_size)
        elif algorithm_name == 'BES' or algorithm_name == 'OriginalBES':
            model = BES.OriginalBES(epoch=epoch, pop_size=pop_size)
        elif algorithm_name == 'SMA' or algorithm_name == 'OriginalSMA':
            model = SMA.OriginalSMA(epoch=epoch, pop_size=pop_size, pr=0.03)
        elif algorithm_name == 'TSA' or algorithm_name == 'OriginalTSA':
            model = TSA.OriginalTSA(epoch=epoch, pop_size=pop_size)
        elif algorithm_name == 'CGO' or algorithm_name == 'OriginalCGO':
            model = CGO.OriginalCGO(epoch=epoch, pop_size=pop_size)
        elif algorithm_name == 'WarSO' or algorithm_name == 'OriginalWarSO':
            model = WarSO.OriginalWarSO(epoch=epoch, pop_size=pop_size)
        elif algorithm_name == 'FLA' or algorithm_name == 'OriginalFLA':
            model = FLA.OriginalFLA(epoch=epoch, pop_size=pop_size)
        elif algorithm_name == 'RIME' or algorithm_name == 'OriginalRIME':
            model = RIME.OriginalRIME(epoch=epoch, pop_size=pop_size)
        elif algorithm_name == 'ESO' or algorithm_name == 'OriginalESO':
            # ESO (Egret Swarm Optimization) - use larger population to avoid percentile issues
            eso_pop_size = max(pop_size, 50)
            model = ESO.OriginalESO(epoch=epoch, pop_size=eso_pop_size)
        elif algorithm_name == 'BBOA':
            model = BBOA.OriginalBBOA(epoch=epoch, pop_size=pop_size)
        elif algorithm_name == 'FOX':
            model = FOX.OriginalFOX(epoch=epoch, pop_size=pop_size)
        elif algorithm_name == 'CDO':
            model = CDO.OriginalCDO(epoch=epoch, pop_size=pop_size)
        elif algorithm_name == 'EVO':
            model = EVO.OriginalEVO(epoch=epoch, pop_size=pop_size)
        else:  # Default to SMA
            model = SMA.OriginalSMA(epoch=epoch, pop_size=pop_size, pr=0.03)
            self.get_logger().info(f"[optimization] The algorithm {algorithm_name} is not supported. Using SMA as default algorithm")
        
        self.get_logger().info(
            f"[optimization] Starting mealpy optimization with {algorithm_name}: epoch={epoch}, pop_size={pop_size}, tasks={K}"
        )
        
        try:
            best = model.solve(problem)
        except Exception as e:
            self.get_logger().error(
                f"[optimization] {algorithm_name} optimization failed: {type(e).__name__}: {e}"
            )
            # Return empty result with infinite makespan to indicate failure
            return {rid: [] for rid in robot_list}, float('inf'), None
        
        # Check if best solution is valid
        if best is None or not hasattr(best, 'solution'):
            self.get_logger().error(
                f"[optimization] {algorithm_name} returned invalid solution"
            )
            return {rid: [] for rid in robot_list}, float('inf'), None
        
        # Re-evaluate best solution to extract max_makespan, total energy, waiting time, and per-robot makespans
        best_solution_makespan = None
        best_solution_energy = None
        best_solution_waiting_time = None
        robot_makespans: Dict[str, float] = {}  # robot_id -> makespan
        try:
            # Set flag to store makespan and energy during this evaluation
            store_makespan_flag[0] = True
            best_makespan_storage[0] = None  # Reset storage
            best_energy_storage[0] = None  # Reset storage
            best_waiting_time_storage[0] = None  # Reset storage
            robot_times_storage[0] = None  # Reset storage
            _ = _objective_function(best.solution)
            best_solution_makespan = best_makespan_storage[0]
            best_solution_energy = best_energy_storage[0]
            best_solution_waiting_time = best_waiting_time_storage[0]
            robot_times_list = robot_times_storage[0]
            if robot_times_list and len(robot_times_list) == len(robot_list):
                # Map robot_times to robot_ids (robot_times is in same order as robot_list)
                for idx, rid in enumerate(robot_list):
                    if idx < len(robot_times_list):
                        robot_makespans[rid] = robot_times_list[idx]
                
                # Log and save robot completion times
                completion_times_msg = f"[robot-completion-times-{algorithm_name}]"
                self.get_logger().info(completion_times_msg)
                for rid in sorted(robot_list):
                    if rid in robot_makespans:
                        completion_time = robot_makespans[rid]
                        self.get_logger().info(f"  {rid}: {completion_time:.2f}s")
                
                # Save to log file
                self._save_robot_completion_times_log(algorithm_name, robot_makespans)
            
            store_makespan_flag[0] = False  # Reset flag
        except Exception as e:
            self.get_logger().debug(f"[optimization] Could not extract makespan from best solution: {e}")
            best_solution_makespan = None
            robot_makespans = {}
            store_makespan_flag[0] = False  # Reset flag on error
        
        # ---------- decode best solution ----------
        # NOTE: According to design principles:
        #  - objective_function is the ONLY place that computes makespan/penalty/fitness.
        #  - This decode step is responsible ONLY for building the structural assignment:
        #    (robot_id -> ordered list of TaskSpec, including CHARGE where charge_key > 0).
        #  - No time/energy/penalty simulation is performed here.
        solution = best.solution
        
        # Extract best fitness from the optimizer result (single source of truth)
        best_fitness = float("inf")
        try:
            target = getattr(best, "target", best)
            if hasattr(target, "fitness"):
                best_fitness = float(target.fitness)
            elif hasattr(target, "objectives") and len(target.objectives) > 0:
                best_fitness = float(target.objectives[0])
            elif isinstance(target, (int, float)):
                best_fitness = float(target)
        except Exception:
            # If anything goes wrong, keep best_fitness as inf and log later
            pass
        
        # Log best fitness, makespan, and energy for this algorithm
        if math.isfinite(best_fitness):
            makespan_info = f", makespan={best_solution_makespan:.2f}" if best_solution_makespan is not None else ""
            energy_info = f", energy={best_solution_energy:.2f}Wh" if best_solution_energy is not None else ""
            log_msg = f"[optimization-{algorithm_name}] Best fitness = {best_fitness:.2f}{makespan_info}{energy_info}"
            self.get_logger().info(log_msg)
            # Save to log file
            self._save_makespan_log(algorithm_name, best_fitness, best_solution_makespan, best_solution_energy)
        else:
            self.get_logger().warn(
                f"[optimization-{algorithm_name}] Could not determine best fitness from optimizer result"
            )
        
        # Verify solution size matches expected dimension
        expected_size = 3 * K  # [robot_key, seq_key, charge_key] per task
        actual_size = len(solution) if hasattr(solution, "__len__") else 0
        if actual_size != expected_size:
            self.get_logger().warn(
                f"[optimization] Solution size mismatch: expected {expected_size}, got {actual_size}"
            )
        
        # Decode random-key representation into per-robot ordered task lists
        assignments: Dict[str, List[Tuple[float, float, TaskSpec]]] = {rid: [] for rid in robot_list}
        
        for i, task in enumerate(candidate_tasks):
            if 3 * i + 2 >= len(solution):
                self.get_logger().warn(
                    f"[optimization] Solution too short for task {i}: need index {3 * i + 2}, have {len(solution)}"
                )
                break
            robot_key = solution[3 * i]  # [-1.0, 1.0]
            seq_key = solution[3 * i + 1]
            charge_key = solution[3 * i + 2]  # [-1.0, 1.0]
            
            # If charging is not included in objective, ignore charge_key
            if not self.include_charge_in_objective:
                charge_key = -1.0
            
            # Map robot_key from [-1.0, 1.0] to [0, num_robots) for robot_id
            normalized_key = (robot_key + 1.0) / 2.0
            robot_idx = int(np.clip(normalized_key * num_robots, 0, num_robots - 1))
            robot_id = robot_list[robot_idx]
            
            assignments[robot_id].append((seq_key, charge_key, task))
        
        # Log assignments for debugging
        self.get_logger().info(
            f"[decode-{algorithm_name}] Assignments:"
        )
        for rid in robot_list:
            robot_assignments = assignments.get(rid, [])
            if robot_assignments:
                assignment_str = ", ".join([
                    f"task_{t.task_id}(seq={sk:.3f},charge={ck:.3f})"
                    for sk, ck, t in robot_assignments
                ])
                self.get_logger().info(
                    f"[decode-{algorithm_name}]   {rid}: {assignment_str}"
                )
        
        # Verify all tasks were assigned
        total_assigned_in_decode = sum(len(tasks) for tasks in assignments.values())
        if total_assigned_in_decode != K:
            self.get_logger().warn(
                f"[optimization] Task count mismatch after decode: expected {K}, got {total_assigned_in_decode}"
            )
        
        # Build final per-robot sequences:
        #  - Sort by seq_key
        #  - For each task, if charge_key > 0, insert a CHARGE TaskSpec before it
        #  - Do NOT compute times/energies here (pure structure generation)
        #  - Charger selection will be handled during actual execution/simulation
        result: Dict[str, List[TaskSpec]] = {}
        chargers_list = list(self.chargers)
        
        for rid in robot_list:
            robot_tasks = assignments.get(rid, [])
            if not robot_tasks:
                result[rid] = []
                continue
            
            state = self.robot_states.get(rid)
            # Fallback position if state is missing
            sim_x = float(state.x) if state is not None else 0.0
            sim_y = float(state.y) if state is not None else 0.0
            
            # Sort by sequence key
            sorted_tasks = sorted(robot_tasks, key=lambda x: x[0])
            
            # Build sequence structure only: task order and charge positions
            # Actual charger selection will be done at execution time based on current position
            seq: List[TaskSpec] = []
            for seq_key, charge_key, t in sorted_tasks:
                # Insert CHARGE placeholder if charge_key > 0
                # Charger location will be determined at execution time
                if charge_key > 0.0 and chargers_list:
                    # Create placeholder CHARGE task with dummy location
                    # Actual charger will be selected based on robot's current position at execution
                    if chargers_list:
                        # Use first charger as placeholder (will be replaced at execution)
                        placeholder_charger = chargers_list[0]
                        charge_task = TaskSpec(
                            task_id=-1,
                            task_type="CHARGE",
                            picks=[(placeholder_charger.x, placeholder_charger.y)],
                            drops=[(placeholder_charger.x, placeholder_charger.y)],
                            pick_wait_s=0.0,
                            drop_wait_s=0.0,
                            charge_duration_s=1.0,
                        )
                        seq.append(charge_task)
                # Append the actual task
                seq.append(t)
            result[rid] = seq
            
            # Log sequence for this robot: task_id for tasks, -1 for CHARGE
            if seq:
                sequence_str = ",".join([
                    str(t.task_id) for t in seq
                ])
                # Calculate additional information
                seq_length = len(seq)
                # Count task types for this robot's sequence
                robot_task_type_counts: Dict[str, int] = {}
                charge_count = 0
                for t in seq:
                    if t.task_type == "CHARGE":
                        charge_count += 1
                    else:
                        robot_task_type_counts[t.task_type] = robot_task_type_counts.get(t.task_type, 0) + 1
                
                # Format task type counts for this robot
                task_type_str = ", ".join([f"{task_type}:{count}" for task_type, count in sorted(robot_task_type_counts.items())]) if robot_task_type_counts else "none"
                
                # Get robot-specific makespan from objective function calculation
                robot_makespan = robot_makespans.get(rid)
                makespan_str = f", makespan={robot_makespan:.2f}" if robot_makespan is not None else ", makespan=N/A"
                
                log_msg = (f"[decode-{algorithm_name}] {rid} sequence=[{sequence_str}] "
                          f"(length={seq_length}, tasks=[{task_type_str}], charges={charge_count}{makespan_str})")
                self.get_logger().info(log_msg)
                # Save to log file
                self._save_sequence_log(algorithm_name, rid, sequence_str, seq_length, 
                                      robot_task_type_counts, charge_count, robot_makespan)
        
        # Optionally log simple assignment summary (counts only, no times/fitness)
        task_type_counts: Dict[str, int] = {}
        total_charge_count = 0
        for rid in robot_list:
            tasks = result.get(rid, [])
            non_charge_tasks = [t for t in tasks if t.task_type != "CHARGE"]
            charge_count = sum(1 for t in tasks if t.task_type == "CHARGE")
            total_charge_count += charge_count
            
            for t in non_charge_tasks:
                task_type_counts[t.task_type] = task_type_counts.get(t.task_type, 0) + 1
        
        if task_type_counts:
            total_tasks = sum(task_type_counts.values())
            overall_summary_parts = [f"{task_type}: {count}" for task_type, count in sorted(task_type_counts.items())]
            overall_summary = ", ".join(overall_summary_parts)
            charge_summary = f", {total_charge_count} CHARGE events" if total_charge_count > 0 else ""
            self.get_logger().info(
                f"[assignment-{algorithm_name}] Overall: {total_tasks} tasks ({overall_summary}{charge_summary})"
            )
        
        # Return decoded assignment, best fitness, max makespan, total energy, and total waiting time
        return result, best_fitness, best_solution_makespan, best_solution_energy, best_solution_waiting_time

    def _optimize_multi_robot_allocation(self, candidate_tasks: List[TaskSpec]) -> Dict[str, List[TaskSpec]]:
        """Run optimization with all algorithms 30 times and select the best result.
        
        The entire set of algorithms is run 30 times (total_rounds), and the overall
        best result across all algorithms and rounds is selected.
        
        Returns the result with the minimum objective fitness.
        """
        if not candidate_tasks:
            return {rid: [] for rid in self.robot_ids}
        
        if not self.robot_ids:
            self.get_logger().warn("Cannot allocate tasks: no robots available")
            return {}
        
        # Number of times to repeat the entire algorithm set
        total_rounds = 10
        
        # Test all available algorithms
        # algorithms = ['RIME']

        algorithms = ['RIME']
        
        # Track best results per algorithm across all rounds
        best_results_per_algo: Dict[str, Dict[str, List[TaskSpec]]] = {}
        best_fitness_per_algo: Dict[str, float] = {algo: float('inf') for algo in algorithms}
        best_makespan_per_algo: Dict[str, Optional[float]] = {algo: None for algo in algorithms}
        best_energy_per_algo: Dict[str, Optional[float]] = {algo: None for algo in algorithms}
        best_waiting_time_per_algo: Dict[str, Optional[float]] = {algo: None for algo in algorithms}
        all_fitnesses_per_algo: Dict[str, List[float]] = {algo: [] for algo in algorithms}
        all_makespans_per_algo: Dict[str, List[Optional[float]]] = {algo: [] for algo in algorithms}
        all_energies_per_algo: Dict[str, List[Optional[float]]] = {algo: [] for algo in algorithms}
        all_waiting_times_per_algo: Dict[str, List[Optional[float]]] = {algo: [] for algo in algorithms}
        total_time_per_algo: Dict[str, float] = {algo: 0.0 for algo in algorithms}
        
        # Global best tracking
        global_best_result = None
        global_best_fitness = float('inf')
        global_best_makespan: Optional[float] = None
        global_best_energy: Optional[float] = None
        global_best_algo = None
        global_best_round = -1
        
        total_runs = total_rounds * len(algorithms)
        self.get_logger().info(
            f"[multi-algo] Running {total_rounds} rounds x {len(algorithms)} algorithms = {total_runs} total runs"
        )
        self.get_logger().info(
            f"[multi-algo] Algorithms: {', '.join(algorithms)}"
        )
        
        # Run all algorithms for each round
        for round_num in range(total_rounds):
            self.get_logger().info(f"[multi-algo] === Round {round_num + 1}/{total_rounds} ===")
            
            for algo_name in algorithms:
                try:
                    # Measure execution time for this run
                    run_t0 = time.time()
                    result, fitness, makespan, energy, waiting_time = self._optimize_with_algorithm(candidate_tasks, algo_name)
                    run_dt = time.time() - run_t0
                    
                    total_time_per_algo[algo_name] += run_dt
                    all_fitnesses_per_algo[algo_name].append(fitness)
                    all_makespans_per_algo[algo_name].append(makespan)
                    all_energies_per_algo[algo_name].append(energy)
                    all_waiting_times_per_algo[algo_name].append(waiting_time)
                    
                    # Save this round's result to log file
                    self._save_round_result_log(round_num + 1, algo_name, fitness, makespan, run_dt, energy)
                    
                    # Update best result for this algorithm
                    if math.isfinite(fitness) and fitness < best_fitness_per_algo[algo_name]:
                        best_fitness_per_algo[algo_name] = fitness
                        best_makespan_per_algo[algo_name] = makespan
                        best_energy_per_algo[algo_name] = energy
                        best_waiting_time_per_algo[algo_name] = waiting_time
                        best_results_per_algo[algo_name] = result
                    
                    # Update global best
                    if math.isfinite(fitness) and fitness < global_best_fitness:
                        global_best_fitness = fitness
                        global_best_makespan = makespan
                        global_best_energy = energy
                        global_best_result = result
                        global_best_algo = algo_name
                        global_best_round = round_num + 1
                        makespan_str = f", makespan={makespan:.2f}" if makespan is not None else ""
                        self.get_logger().info(
                            f"[multi-algo] New global best: {algo_name} round {round_num + 1}, fitness={fitness:.2f}{makespan_str}"
                        )
                    
                    makespan_str = f", makespan={makespan:.2f}" if makespan is not None else ""
                    self.get_logger().debug(
                        f"[multi-algo] Round {round_num + 1} {algo_name}: fitness={fitness:.2f}{makespan_str} (time={run_dt:.3f}s)"
                    )
                    
                except Exception as e:
                    self.get_logger().error(f"[multi-algo] Round {round_num + 1} {algo_name} failed: {e}")
                    all_fitnesses_per_algo[algo_name].append(float('inf'))
                    all_makespans_per_algo[algo_name].append(None)
                    all_energies_per_algo[algo_name].append(None)
                    all_waiting_times_per_algo[algo_name].append(None)
                    # Save failed result to log
                    self._save_round_result_log(round_num + 1, algo_name, float('inf'), None, 0.0, None)
        
        # Log statistics for each algorithm
        self.get_logger().info(f"[multi-algo] === Final Statistics ({total_rounds} rounds) ===")
        for algo_name in algorithms:
            valid_fitnesses = [f for f in all_fitnesses_per_algo[algo_name] if math.isfinite(f)]
            if valid_fitnesses:
                avg_fitness = sum(valid_fitnesses) / len(valid_fitnesses)
                min_fitness = min(valid_fitnesses)
                max_fitness = max(valid_fitnesses)
                self.get_logger().info(
                    f"[multi-algo] {algo_name}: best={min_fitness:.2f}, avg={avg_fitness:.2f}, worst={max_fitness:.2f}, "
                    f"total_time={total_time_per_algo[algo_name]:.3f}s"
                )
            else:
                self.get_logger().info(
                    f"[multi-algo] {algo_name}: all {total_rounds} runs failed, "
                    f"total_time={total_time_per_algo[algo_name]:.3f}s"
                )
            # Save execution time to log file with breakdown into
            # path planning (Nav2 precompute) and optimization time.
            # Path planning time is global for this offline run and is
            # stored in self.path_planning_time_s by _offline_initialize_sequences.
            path_time = getattr(self, 'path_planning_time_s', 0.0)
            self._save_execution_time_log(algo_name, total_time_per_algo[algo_name], path_time)
        
        # Check if any valid result was found
        if global_best_result is None or not math.isfinite(global_best_fitness):
            self.get_logger().error("[multi-algo] All algorithms failed in all rounds, returning empty result")
            return {rid: [] for rid in self.robot_ids}
        
        # Log final comparison
        self.get_logger().info(
            f"[multi-algo] Algorithm comparison (best fitness from {total_rounds} rounds): " +
            ", ".join([f"{algo}={best_fitness_per_algo[algo]:.2f}" for algo in algorithms])
        )
        makespan_str = f", makespan={global_best_makespan:.2f}" if global_best_makespan is not None else ""
        energy_str = f", energy={global_best_energy:.2f}Wh" if global_best_energy is not None else ""
        self.get_logger().info(
            f"[multi-algo] === FINAL SELECTION: {global_best_algo} (round {global_best_round}, fitness={global_best_fitness:.2f}{makespan_str}{energy_str}) ==="
        )
        
        # Save final statistics to log file
        self._save_multi_round_statistics_log(
            total_rounds,
            algorithms,
            all_fitnesses_per_algo,
            all_makespans_per_algo,
            all_energies_per_algo,
            all_waiting_times_per_algo,
            best_fitness_per_algo,
            best_makespan_per_algo,
            best_energy_per_algo,
            best_waiting_time_per_algo,
            total_time_per_algo,
            global_best_algo,
            global_best_round,
            global_best_fitness,
            global_best_makespan,
            global_best_energy,
        )
        
        return global_best_result

    def _offline_initialize_sequences(self) -> None:
        """Offline scheduling now simply applies the embedded manual sequences."""

        if not self.pending_tasks:
            return

        if self.manual_sequence_map:
            if self._apply_manual_sequences():
                self.optimization_in_progress = False
                self.optimization_time = 0.0
                opt_time_msg = Float32()
                opt_time_msg.data = 0.0
                self.optimization_time_pub.publish(opt_time_msg)
                self.get_logger().info(
                    "[manual] Manual sequences applied; optimization pipeline permanently disabled"
                )
                return

            self.get_logger().error(
                "[manual] Manual sequence override failed and optimizer fallback has been removed; no sequences initialized"
            )
            return

        self.get_logger().error(
            "[manual] No manual sequence map configured; optimization pipeline removed so there is nothing to schedule"
        )
        return

    def _compute_sequences_for_current_flag(
        self,
        candidate_tasks: List[TaskSpec],
    ) -> Tuple[Dict[str, List[TaskSpec]], Dict[str, float]]:
        """Run allocation pipeline for the currently configured include_charge flag."""

        if not candidate_tasks:
            sequences: Dict[str, List[TaskSpec]] = {rid: [] for rid in self.robot_ids}
            stage1_dt = 0.0
        else:
            t0 = time.time()
            sequences = self._optimize_multi_robot_allocation(candidate_tasks)
            stage1_dt = time.time() - t0

        total_tasks = 0
        energy_t0 = time.time()
        for rid, seq in sequences.items():
            state = self.robot_states.get(rid)
            if state is None:
                continue
            for spec in seq:
                if spec.task_type in ("PICK_DROP", "MULTI_PICK_DROP", "DUAL_PICK_MULTI_DROP"):
                    _ = self.estimate_task_energy(state, spec)
                    total_tasks += 1
        energy_dt = time.time() - energy_t0

        assigned = sum(len([t for t in seq if t.task_type != "CHARGE"]) for seq in sequences.values())
        charge_events = sum(sum(1 for t in seq if t.task_type == "CHARGE") for seq in sequences.values())
        robots_used = sum(1 for seq in sequences.values() if seq)

        stats = {
            "stage1_time": stage1_dt,
            "energy_time": energy_dt,
            "energy_tasks": total_tasks,
            "assigned_tasks": assigned,
            "charge_events": charge_events,
            "robots_utilized": robots_used,
        }

        return sequences, stats

    def _all_jobs_empty(self) -> bool:
        """Return True if there are no remaining tasks in pending_tasks."""
        return not self.pending_tasks or len(self.pending_tasks) == 0

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

        # SOC topic subscription is commented out to use internal odom-based SOC tracking
        # If external SOC updates are needed, uncomment and ensure the topic format matches
        # soc_topic = f'/robot/{robot_id}/status'
        # self.create_subscription(
        #     Float32,
        #     soc_topic,
        #     lambda msg, rid=robot_id: self._soc_cb(rid, msg),
        #     10,
        # )

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
        
        # Nav2 recovery interfaces for task failure recovery
        # NavigateToPose action client (to cancel current navigation goal)
        nav_action_name = f'/{robot_id}/navigate_to_pose'
        self.nav_action_clients[robot_id] = ActionClient(self, NavigateToPose, nav_action_name)
        
        # ClearCostmap service clients (global and local)
        clear_global_name = f'/{robot_id}/global_costmap/clear_entirely_global_costmap'
        self.clear_costmap_global_clients[robot_id] = self.create_client(
            ClearEntireCostmap, clear_global_name
        )
        clear_local_name = f'/{robot_id}/local_costmap/clear_entirely_local_costmap'
        self.clear_costmap_local_clients[robot_id] = self.create_client(
            ClearEntireCostmap, clear_local_name
        )
        
        # InitialPose publisher (to reset robot localization to current Gazebo pose)
        initial_pose_topic = f'/{robot_id}/initialpose'
        self.initial_pose_pubs[robot_id] = self.create_publisher(
            PoseWithCovarianceStamped, initial_pose_topic, QoSPresetProfiles.SYSTEM_DEFAULT.value
        )

        cap_wh = self.robot_cap_wh.get(robot_id, 120.0)
        init_soc = cap_wh * self.initial_soc_factor
        
        # Get initial position from hardcoded positions (for scheduling-only experiments)
        init_x, init_y = self.robot_initial_positions.get(robot_id, (0.0, 0.0))

        self.robot_states[robot_id] = RobotState(
            robot_id=robot_id,
            namespace=f'/{robot_id}',
            x=init_x,
            y=init_y,
            soc=init_soc,
            available=True,
            carried_weight=0.0,
        )

        rid_str = self._color_robot(robot_id) if hasattr(self, "_color_robot") else robot_id
        self.get_logger().info(
            f"[scheduler] init {rid_str}: pos=({init_x:.1f}, {init_y:.1f}), cap={cap_wh:.1f}Wh, initial_soc={init_soc:.1f}Wh (factor={self.initial_soc_factor:.2f})"
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
        """Decrease SOC based on distance travelled and velocity (inertia) from odom.

        This keeps the energy model local to the scheduler package, without
        depending on external multi_robot_energy logic.
        
        Energy consumption considers:
        - E_roll: Rolling resistance/friction (C_r * m * g * d) - distance and mass based
        - E_acceleration: Acceleration/deceleration energy (ΔE_k = 0.5 * m * (v² - v₀²))
        - E_angular: Rotational energy consumption (time-based, proportional to ω²)
        - E_static: Static consumption (electronics, sensors - time-based)
        - E_noise: Real-world uncertainty (Gaussian noise for manufacturing variations, etc.)
        
        NOTE: E_kinetic is NOT included - kinetic energy is a state variable, not consumed energy.
        Only changes in kinetic energy (acceleration) consume energy.
        
        NOTE: During precompute/optimization phase, energy consumption is paused to ensure
        fair allocation starting from initial 100% SOC. Energy consumption resumes after
        allocation is complete.
        """
        state = self.robot_states.get(robot_id)
        if state is None:
            return

        try:
            x = float(msg.pose.pose.position.x)
            y = float(msg.pose.pose.position.y)
            # Extract velocity from odometry twist message
            vx = float(msg.twist.twist.linear.x)
            vy = float(msg.twist.twist.linear.y)
            # Angular velocity can also contribute to energy consumption
            v_angular = float(msg.twist.twist.angular.z)
        except Exception:
            return

        prev = self._last_odom.get(robot_id)
        self._last_odom[robot_id] = (x, y)

        # Get timestamp from odom message (for static energy calculation)
        try:
            # Convert ROS2 timestamp to seconds
            stamp_sec = float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9
        except Exception:
            # Fallback to current time if timestamp unavailable
                stamp_sec = time.time()

        # Update tracking variables (position, velocity, timestamp) even during optimization
        # This ensures accurate distance calculations after optimization completes
        self._last_velocity[robot_id] = (vx, vy)
        self._last_odom_time[robot_id] = stamp_sec

        # Skip energy consumption during optimization phase
        # Position and velocity tracking continue for accurate post-optimization calculations
        if self.optimization_in_progress:
            return

        if prev is None:
            # first odom sample, just seed position, velocity, and timestamp
            # Already done above, so just return
            return

        px, py = prev
        dist = math.hypot(x - px, y - py)
        
        # Calculate time elapsed since last odom message
        last_time = self._last_odom_time.get(robot_id, stamp_sec)
        dt = max(0.0, stamp_sec - last_time)  # Time delta in seconds
        self._last_odom_time[robot_id] = stamp_sec

        # Battery capacity for clamping SOC
        cap_wh = self.robot_cap_wh.get(robot_id, 120.0)
        
        if dist <= 1e-6:
            # No movement, but still update velocity and calculate static energy
            self._last_velocity[robot_id] = (vx, vy)
            # Even when stationary, robots consume static energy
            if dt > 0.0:
                E_static = getattr(self, 'E_static_wh_per_s', 0.001) * dt
                # Add noise to static energy (non-negative: energy can only be consumed)
                noise_sigma = getattr(self, 'energy_noise_sigma', 0.01)
                E_noise = max(0.0, random.gauss(0.0, noise_sigma)) if noise_sigma > 0.0 else 0.0
                dwh = E_static + E_noise
                # Apply static consumption and clamp to [0, cap_wh]
                new_soc = state.soc - dwh
                state.soc = max(0.0, min(cap_wh, new_soc))
            return

        # Energy model: PHYSICALLY CORRECT SUM MODEL (not multiplicative)
        # E_total = E_roll + E_acceleration + E_angular + E_static + E_noise
        # Each component is calculated independently and summed in Wh units
        # NOTE: E_kinetic is NOT included - kinetic energy is a state, not consumed.
        # Only changes in kinetic energy (acceleration) consume energy.
        
        # Get robot mass and carried weight
        carried_weight = state.carried_weight
        total_mass = self.base_robot_mass_kg + carried_weight  # Total mass (kg)
        
        # Calculate linear velocity magnitude (m/s)
        linear_speed = math.hypot(vx, vy)
        
        # ========================================================================
        # 1. E_roll: Rolling resistance / friction energy (distance-based)
        # ========================================================================
        # Physical model: E_roll = C_r * m * g * d
        # Where:
        #   C_r = rolling resistance coefficient (dimensionless)
        #   m = total mass (kg)
        #   g = gravitational acceleration (9.81 m/s²)
        #   d = distance traveled (m)
        # Convert Joules to Wh: 1 Wh = 3600 J
        C_r = 0.015  # Rolling resistance coefficient for rubber wheels on indoor floor (typical: 0.01-0.02)
        g = 9.81  # Gravitational acceleration (m/s²)
        E_roll = (C_r * total_mass * g * dist) / 3600.0  # Convert J to Wh
        
        # ========================================================================
        # 2. E_acceleration: Acceleration/deceleration energy (velocity change)
        # ========================================================================
        # Physical model: ΔE_k = 0.5 * m * (v² - v₀²)
        # This represents the energy needed to change kinetic energy state
        # NOTE: This replaces E_kinetic - we only account for energy needed to change velocity
        # Kinetic energy itself is not consumed; only changes in kinetic energy require work
        last_v = self._last_velocity.get(robot_id, (0.0, 0.0))
        last_vx, last_vy = last_v
        last_speed = math.hypot(last_vx, last_vy)
        
        # Calculate kinetic energy change
        delta_v_squared = (linear_speed ** 2) - (last_speed ** 2)
        delta_E_k = 0.5 * total_mass * delta_v_squared / 3600.0  # Kinetic energy change (J → Wh)
        
        decel_policy = getattr(self, 'deceleration_policy', 'none')
        
        if delta_v_squared > 1e-6:
            # ========================================================================
            # ACCELERATION: Energy consumed to increase kinetic energy
            # ========================================================================
            # Always consumes energy (motor work to increase velocity)
            E_acceleration = delta_E_k  # Positive: energy consumed
        
        elif delta_v_squared < -1e-6:
            # ========================================================================
            # DECELERATION: Handle according to deceleration policy
            # ========================================================================
            abs_delta_E_k = abs(delta_E_k)  # Magnitude of kinetic energy decrease
            
            if decel_policy == "none":
                # Inertial/coasting deceleration: no energy consumed
                # Physical: KE decreases naturally, no energy input/output
                # Most physically accurate for most mobile robots without regenerative braking
                E_acceleration = 0.0
            
            elif decel_policy == "regenerative":
                # Regenerative braking: recover energy (negative = energy gained)
                # Models electric vehicles with regenerative braking
                regen_eff = getattr(self, 'regenerative_efficiency', 0.0)
                regen_eff = max(0.0, min(1.0, regen_eff))  # Clamp to [0, 1]
                recovered_energy = abs_delta_E_k * regen_eff
                E_acceleration = -recovered_energy  # Negative: energy recovered (added to battery)
            
            elif decel_policy == "braking_loss":
                # Friction braking: lose energy as heat
                # Models conventional friction brakes where kinetic energy is converted to heat
                # Only a small fraction is "consumed" (rest is just converted to heat)
                brake_loss_coef = getattr(self, 'braking_loss_coefficient', 0.1)
                brake_loss_coef = max(0.0, min(1.0, brake_loss_coef))  # Clamp to [0, 1]
                lost_energy = abs_delta_E_k * brake_loss_coef
                E_acceleration = lost_energy  # Positive: small energy loss from friction
            else:
                # Unknown policy, default to "none"
                E_acceleration = 0.0
        else:
            # Constant velocity: no kinetic energy change
            E_acceleration = 0.0
        
        # ========================================================================
        # 3. E_angular: Rotational energy (angular velocity)
        # ========================================================================
        # Physical model: E_rot = 0.5 * I * ω²
        # Where I = moment of inertia (kg·m²), ω = angular velocity (rad/s)
        # For circular robot: I ≈ 0.5 * m * r²
        # NOTE: Rotational energy is consumed when angular velocity changes, not per distance
        # We model it as instantaneous rotational kinetic energy state change
        robot_radius = 0.17  # Approximate TurtleBot3 radius (m)
        I = 0.5 * total_mass * (robot_radius ** 2)  # Moment of inertia (kg·m²)
        
        # Calculate angular kinetic energy (current state)
        # This represents the energy stored in rotation (consumed when rotation changes)
        # For simplicity, we approximate as: energy proportional to ω² and time
        # More accurate: track angular velocity changes similar to linear acceleration
        # For now, use time-based consumption for rotational motion
        if abs(v_angular) > 0.01:  # Only if rotating significantly
            # Rotational power consumption: P_rot ≈ k_rot * I * ω²
            # Energy = Power * time = k_rot * I * ω² * dt
            k_rot = 0.5  # Rotational energy coefficient (dimensionless)
            E_angular = k_rot * I * (v_angular ** 2) * dt / 3600.0  # Convert J to Wh
        else:
            E_angular = 0.0
        
        # Update last velocity for next calculation
        self._last_velocity[robot_id] = (vx, vy)
        
        # ========================================================================
        # 4. E_static: Static energy consumption (time-based)
        # ========================================================================
        # Robots consume energy even when moving (computers, sensors, electronics)
        # This is independent of movement and proportional to time elapsed
        # Examples: CPU processing, sensor operation, communication, idle motor currents
        E_static = getattr(self, 'E_static_wh_per_s', 0.001) * dt  # Static power * time
        
        # ========================================================================
        # 5. E_noise: Real-world energy consumption uncertainty/noise
        # ========================================================================
        # Real robots have manufacturing variations, temperature effects, 
        # battery degradation, motor efficiency variations, etc.
        # Add Gaussian noise to reflect real-world uncertainty
        # E_noise ~ N(0, sigma^2) - zero mean, configurable variance
        # NOTE: Noise should be non-negative (energy can only be consumed, not created)
        noise_sigma = getattr(self, 'energy_noise_sigma', 0.01)
        E_noise = max(0.0, random.gauss(0.0, noise_sigma)) if noise_sigma > 0.0 else 0.0
        
        # ========================================================================
        # Total energy consumption: SUM of all components (physically correct)
        # E_total = E_roll + E_acceleration + E_angular + E_static + E_noise
        # NOTE: E_kinetic removed - kinetic energy is a state, not consumed energy
        # NOTE: E_acceleration can be negative (energy recovered) for regenerative braking
        # ========================================================================
        dwh = E_roll + E_acceleration + E_angular + E_static + E_noise

        # Apply energy change (can be positive or negative)
        # Positive dwh  = energy consumed  → SOC decreases
        # Negative dwh  = energy recovered → SOC increases (regenerative braking)
        new_soc = state.soc - dwh

        # Clamp SOC to valid range [0.0, cap_wh]
        state.soc = max(0.0, min(cap_wh, new_soc))


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

    def _publish_task_execution(self, robot_id: str, task_id: int, task_type: str, 
                                waypoint_index: int, start_time: float, end_time: float,
                                start_soc: float = 0.0, end_soc: float = 0.0) -> None:
        """Publish TaskExecution message for visualization."""
        msg = TaskExecution()
        msg.robot_id = robot_id
        msg.task_id = task_id
        msg.task_type = task_type
        msg.waypoint_index = waypoint_index
        msg.start_time = start_time
        msg.end_time = end_time
        msg.start_soc = start_soc
        msg.end_soc = end_soc
        self.task_execution_pub.publish(msg)

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
            # Note: robot_start_time is now recorded when task is assigned (pub.publish),
            # not when task actually starts. This gives the true makespan from assignment to completion.
            
            # Record task start time for visualization
            current_time = self.get_clock().now().nanoseconds / 1e9
            if task_id_val is not None:
                progress_key = (robot_id, task_id_val)
                self.task_start_times[progress_key] = current_time
                self.task_start_soc[progress_key] = state.soc
                self.task_waypoint_times[progress_key] = []
                self.task_waypoint_end_soc[progress_key] = []
            
            # If this is a CHARGE task, mark the corresponding charger as occupied
            # by this robot (add to its queue).

            if task_id_val == -1:
                # CHARGE task: publish task execution message
                current_time = self.get_clock().now().nanoseconds / 1e9
                progress_key = (robot_id, task_id_val)
                self.task_start_times[progress_key] = current_time
                self.task_start_soc[progress_key] = state.soc
                self.task_waypoint_times[progress_key] = []
                self.task_waypoint_end_soc[progress_key] = []
                
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
            # If a PICK_DROP task started, track its id
            # Note: carried_weight reset logic:
            # - New task (current_task_id was None or different): reset to 0.0
            # - Retry (current_task_id is same): keep current weight (robot already has picked items)
            if task_id_val is not None and task_id_val != -1:
                prev_task_id = self.current_task_id.get(robot_id)
                is_retry = (prev_task_id is not None and prev_task_id == task_id_val)
                
                # Find task type for logging
                task_type = "UNKNOWN"
                active_seq = self.current_sequences.get(robot_id, [])
                for t in active_seq:
                    if t.task_id == task_id_val:
                        task_type = t.task_type
                        break
                
                # Set current_task_id here (after checking prev_task_id for retry detection)
                self.current_task_id[robot_id] = task_id_val
                # reset retry counter when a task actually starts
                self.retry_counts[robot_id] = 0
                
                # Initialize or reset task progress tracking
                progress_key = (robot_id, task_id_val)
                if is_retry:
                    # Retry: keep current carried_weight and existing progress
                    if progress_key not in self.task_progress:
                        self.task_progress[progress_key] = {"picks": set(), "drops": set()}
                    # Keep existing waypoint_start if any (from previous retry)
                    # If not exists, it will be set from task_failed event
                    rid_str = self._color_robot(robot_id)
                    self.get_logger().info(
                        f"[scheduler] {rid_str} task {task_id_val} ({task_type}) retry started: carried_weight kept at {state.carried_weight:.2f} kg"
                    )
                else:
                    # New task: reset carried_weight to 0 and clear progress
                    state.carried_weight = 0.0
                    self.task_progress[progress_key] = {"picks": set(), "drops": set()}
                    self.task_waypoint_start[progress_key] = 0  # New task starts at waypoint 0
                    rid_str = self._color_robot(robot_id)
                    self.get_logger().info(
                        f"[scheduler] {rid_str} task {task_id_val} ({task_type}) started: carried_weight reset to 0.0"
                    )
            # Cancel pending start timeout for this robot (we received start)
            try:
                if robot_id in self.pending_task_start:
                    del self.pending_task_start[robot_id]
            except Exception:
                pass
            
            # CRITICAL: Increment sequence index when task actually starts
            # This ensures we move to the next task only after current task begins executing
            seq = self.current_sequences.get(robot_id, [])
            current_idx = self.current_seq_index.get(robot_id, 0)
            
            # Verify that the started task matches the current index task
            if current_idx < len(seq):
                expected_task = seq[current_idx]
                if expected_task.task_id == task_id_val:
                    # Task matches current index - increment to next task
                    self.current_seq_index[robot_id] = current_idx + 1
                    rid_str = self._color_robot(robot_id)
                    self.get_logger().info(
                        f"[scheduler] {rid_str} task {task_id_val} started: sequence index {current_idx} -> {current_idx + 1}"
                    )
                else:
                    # Task doesn't match - this shouldn't happen but log for debugging
                    rid_str = self._color_robot(robot_id)
                    self.get_logger().warn(
                        f"[scheduler] {rid_str} task mismatch: started {task_id_val} but expected {expected_task.task_id} at index {current_idx}"
                    )
        elif base == 'pick_complete':
            # Pick completed: add weight of picked item (only if not already completed)
            if task_id_val is not None and wp_idx is not None:
                progress_key = (robot_id, task_id_val)
                progress = self.task_progress.get(progress_key, {"picks": set(), "drops": set()})
                
                # For PICK_DROP/DUAL_PICK_MULTI_DROP: wp_idx from executor is already the pick index (0-based)
                # For other types (MULTI_PICK_DROP), wp_idx is the waypoint index which needs conversion
                active_seq = self.current_sequences.get(robot_id, [])
                for t in active_seq:
                    if t.task_id == task_id_val:
                        # wp_idx is already the pick index for PICK_DROP/DUAL_PICK_MULTI_DROP
                        pick_idx = wp_idx
                        
                        # Check if this pick was already completed (avoid duplicate on retry)
                        if pick_idx in progress["picks"]:
                            rid_str = self._color_robot(robot_id)
                            self.get_logger().warn(
                                f"[scheduler] {rid_str} pick {pick_idx} already completed for task {task_id_val}, skipping duplicate weight update"
                            )
                        else:
                            weight = self._resolve_pick_weight(t, pick_idx)
                            
                            state.carried_weight += weight
                            progress["picks"].add(pick_idx)
                            self.task_progress[progress_key] = progress
                            
                            # Publish waypoint completion for visualization
                            current_time = self.get_clock().now().nanoseconds / 1e9
                            task_start_time = self.task_start_times.get(progress_key, current_time)
                            waypoint_times = self.task_waypoint_times.get(progress_key, [])
                            
                            # Calculate waypoint index based on task type
                            if t.task_type == "PICK_DROP":
                                waypoint_idx = pick_idx * 2  # pick at 0, drop at 1
                            elif t.task_type == "DUAL_PICK_MULTI_DROP":
                                waypoint_idx = pick_idx * 2  # pick 0 at 0, drop 0 at 1, pick 1 at 2, etc.
                            else:
                                waypoint_idx = pick_idx
                            
                            # Start time is previous waypoint end time, or task start time for first waypoint
                            if waypoint_times:
                                start_time = waypoint_times[-1][2]  # last waypoint end time
                            else:
                                start_time = task_start_time
                            
                            end_time = current_time
                            waypoint_times.append((waypoint_idx, start_time, end_time))
                            self.task_waypoint_times[progress_key] = waypoint_times
                            
                            # Get SOC: start_soc from previous waypoint end_soc or task start_soc, end_soc from current state
                            waypoint_end_socs = self.task_waypoint_end_soc.get(progress_key, [])
                            if waypoint_end_socs:
                                start_soc = waypoint_end_socs[-1]
                            else:
                                start_soc = self.task_start_soc.get(progress_key, state.soc)
                            end_soc = state.soc
                            waypoint_end_socs.append(end_soc)
                            self.task_waypoint_end_soc[progress_key] = waypoint_end_socs
                            
                            self._publish_task_execution(robot_id, task_id_val, t.task_type, 
                                                        waypoint_idx, start_time, end_time, start_soc, end_soc)
                            
                            rid_str = self._color_robot(robot_id)
                            self.get_logger().info(
                                f"[scheduler] {rid_str} pick {pick_idx} completed: +{weight:.2f} kg, total={state.carried_weight:.2f} kg"
                            )
                        break
        elif base == 'drop_complete':
            # Drop completed: subtract weight of dropped item(s) (only if not already completed)
            if task_id_val is not None and wp_idx is not None:
                progress_key = (robot_id, task_id_val)
                progress = self.task_progress.get(progress_key, {"picks": set(), "drops": set()})
                
                # Check if this drop was already completed (avoid duplicate on retry)
                if wp_idx in progress["drops"]:
                    rid_str = self._color_robot(robot_id)
                    self.get_logger().warn(
                        f"[scheduler] {rid_str} drop {wp_idx} already completed for task {task_id_val}, skipping duplicate weight update"
                    )
                else:
                    active_seq = self.current_sequences.get(robot_id, [])
                    for t in active_seq:
                        if t.task_id == task_id_val:
                            # wp_idx is the drop index (0-based)
                            # Calculate weight to subtract based on deliveries
                            dropped_weight = 0.0
                            if hasattr(t, 'deliveries') and t.deliveries and wp_idx < len(t.deliveries) and len(t.deliveries[wp_idx]) > 0:
                                # Deliveries[wp_idx] contains list of pick indices to unload at this drop
                                for pick_idx in t.deliveries[wp_idx]:
                                    dropped_weight += self._resolve_pick_weight(t, pick_idx)
                            else:
                                # If no deliveries spec, drop weight follows the drop index
                                dropped_weight = self._resolve_pick_weight(t, wp_idx)
                            
                            state.carried_weight = max(0.0, state.carried_weight - dropped_weight)
                            progress["drops"].add(wp_idx)
                            self.task_progress[progress_key] = progress
                            
                            # Publish waypoint completion for visualization
                            current_time = self.get_clock().now().nanoseconds / 1e9
                            task_start_time = self.task_start_times.get(progress_key, current_time)
                            waypoint_times = self.task_waypoint_times.get(progress_key, [])
                            
                            # Calculate waypoint index based on task type
                            if t.task_type == "PICK_DROP":
                                waypoint_idx = wp_idx * 2 + 1  # drop at 1
                            elif t.task_type == "DUAL_PICK_MULTI_DROP":
                                waypoint_idx = wp_idx * 2 + 1  # drop 0 at 1, drop 1 at 3, etc.
                            else:
                                waypoint_idx = wp_idx
                            
                            # Start time is previous waypoint end time, or task start time for first waypoint
                            if waypoint_times:
                                start_time = waypoint_times[-1][2]  # last waypoint end time
                            else:
                                start_time = task_start_time
                            
                            end_time = current_time
                            waypoint_times.append((waypoint_idx, start_time, end_time))
                            self.task_waypoint_times[progress_key] = waypoint_times
                            
                            # Get SOC: start_soc from previous waypoint end_soc or task start_soc, end_soc from current state
                            waypoint_end_socs = self.task_waypoint_end_soc.get(progress_key, [])
                            if waypoint_end_socs:
                                start_soc = waypoint_end_socs[-1]
                            else:
                                start_soc = self.task_start_soc.get(progress_key, state.soc)
                            end_soc = state.soc
                            waypoint_end_socs.append(end_soc)
                            self.task_waypoint_end_soc[progress_key] = waypoint_end_socs
                            
                            self._publish_task_execution(robot_id, task_id_val, t.task_type, 
                                                        waypoint_idx, start_time, end_time, start_soc, end_soc)
                            
                            rid_str = self._color_robot(robot_id)
                            self.get_logger().info(
                                f"[scheduler] {rid_str} drop {wp_idx} completed: -{dropped_weight:.2f} kg, total={state.carried_weight:.2f} kg"
                            )
                            break
        elif base == 'waypoint_reached':
            # MULTI_PICK_DROP: determine if this waypoint is pick or drop based on waypoint index
            if task_id_val is not None and wp_idx is not None:
                progress_key = (robot_id, task_id_val)
                progress = self.task_progress.get(progress_key, {"picks": set(), "drops": set()})
                # Convert relative waypoint index (from retry) to absolute waypoint index
                waypoint_start = self.task_waypoint_start.get(progress_key, 0)
                absolute_wp_idx = waypoint_start + wp_idx
                
                active_seq = self.current_sequences.get(robot_id, [])
                for t in active_seq:
                    if t.task_id == task_id_val and t.task_type == "MULTI_PICK_DROP":
                        num_picks = len(t.picks) if t.picks else 0
                        # First num_picks waypoints are picks, rest are drops
                        # Use absolute waypoint index to determine if this is pick or drop
                        if absolute_wp_idx < num_picks:
                            # This is a pick waypoint (use absolute index for pick_idx)
                            pick_idx = absolute_wp_idx
                            if pick_idx in progress["picks"]:
                                rid_str = self._color_robot(robot_id)
                                self.get_logger().warn(
                                    f"[scheduler] {rid_str} pick {pick_idx} already completed for task {task_id_val}, skipping duplicate weight update"
                                )
                            else:
                                weight = self._resolve_pick_weight(t, pick_idx)
                                state.carried_weight += weight
                                progress["picks"].add(pick_idx)
                                self.task_progress[progress_key] = progress
                                
                                # Publish waypoint completion for visualization
                                current_time = self.get_clock().now().nanoseconds / 1e9
                                task_start_time = self.task_start_times.get(progress_key, current_time)
                                waypoint_times = self.task_waypoint_times.get(progress_key, [])
                                
                                # For MULTI_PICK_DROP, waypoint index is absolute_wp_idx
                                waypoint_idx = absolute_wp_idx
                                
                                # Start time is previous waypoint end time, or task start time for first waypoint
                                if waypoint_times:
                                    start_time = waypoint_times[-1][2]  # last waypoint end time
                                else:
                                    start_time = task_start_time
                                
                                end_time = current_time
                                waypoint_times.append((waypoint_idx, start_time, end_time))
                                self.task_waypoint_times[progress_key] = waypoint_times
                                
                                # Get SOC: start_soc from previous waypoint end_soc or task start_soc, end_soc from current state
                                waypoint_end_socs = self.task_waypoint_end_soc.get(progress_key, [])
                                if waypoint_end_socs:
                                    start_soc = waypoint_end_socs[-1]
                                else:
                                    start_soc = self.task_start_soc.get(progress_key, state.soc)
                                end_soc = state.soc
                                waypoint_end_socs.append(end_soc)
                                self.task_waypoint_end_soc[progress_key] = waypoint_end_socs
                                
                                self._publish_task_execution(robot_id, task_id_val, t.task_type, 
                                                            waypoint_idx, start_time, end_time, start_soc, end_soc)
                                
                                rid_str = self._color_robot(robot_id)
                                self.get_logger().info(
                                    f"[scheduler] {rid_str} pick {pick_idx} completed: +{weight:.2f} kg, total={state.carried_weight:.2f} kg"
                                )
                        else:
                            # This is a drop waypoint (use absolute index to compute drop_idx)
                            drop_idx = absolute_wp_idx - num_picks
                            if drop_idx in progress["drops"]:
                                rid_str = self._color_robot(robot_id)
                                self.get_logger().warn(
                                    f"[scheduler] {rid_str} drop {drop_idx} already completed for task {task_id_val}, skipping duplicate weight update"
                                )
                            else:
                                dropped_weight = 0.0
                                if hasattr(t, 'deliveries') and t.deliveries and drop_idx < len(t.deliveries):
                                    for pick_idx in t.deliveries[drop_idx]:
                                        dropped_weight += self._resolve_pick_weight(t, pick_idx)
                                else:
                                    # Use the resolved weight for the drop index
                                    dropped_weight = self._resolve_pick_weight(t, drop_idx)
                                
                                state.carried_weight = max(0.0, state.carried_weight - dropped_weight)
                                progress["drops"].add(drop_idx)
                                self.task_progress[progress_key] = progress
                                
                                # Publish waypoint completion for visualization
                                current_time = self.get_clock().now().nanoseconds / 1e9
                                task_start_time = self.task_start_times.get(progress_key, current_time)
                                waypoint_times = self.task_waypoint_times.get(progress_key, [])
                                
                                # For MULTI_PICK_DROP, waypoint index is absolute_wp_idx
                                waypoint_idx = absolute_wp_idx
                                
                                # Start time is previous waypoint end time, or task start time for first waypoint
                                if waypoint_times:
                                    start_time = waypoint_times[-1][2]  # last waypoint end time
                                else:
                                    start_time = task_start_time
                                
                                end_time = current_time
                                waypoint_times.append((waypoint_idx, start_time, end_time))
                                self.task_waypoint_times[progress_key] = waypoint_times
                                
                                # Get SOC: start_soc from previous waypoint end_soc or task start_soc, end_soc from current state
                                waypoint_end_socs = self.task_waypoint_end_soc.get(progress_key, [])
                                if waypoint_end_socs:
                                    start_soc = waypoint_end_socs[-1]
                                else:
                                    start_soc = self.task_start_soc.get(progress_key, state.soc)
                                end_soc = state.soc
                                waypoint_end_socs.append(end_soc)
                                self.task_waypoint_end_soc[progress_key] = waypoint_end_socs
                                
                                self._publish_task_execution(robot_id, task_id_val, t.task_type, 
                                                            waypoint_idx, start_time, end_time, start_soc, end_soc)
                                
                                rid_str = self._color_robot(robot_id)
                                self.get_logger().info(
                                    f"[scheduler] {rid_str} drop {drop_idx} completed: -{dropped_weight:.2f} kg, total={state.carried_weight:.2f} kg"
                                )
                        break
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

            # CRITICAL: Verify and update sequence index when task completes
            # This ensures we don't skip tasks or execute them multiple times
            seq = self.current_sequences.get(robot_id, [])
            current_idx = self.current_seq_index.get(robot_id, 0)
            
            # Check if completed task matches the expected task at current index
            # (accounting for fact that index was already incremented in task_started)
            if current_idx > 0 and current_idx <= len(seq):
                prev_idx = current_idx - 1
                if prev_idx < len(seq):
                    prev_task = seq[prev_idx]
                    if task_id_val is not None and task_id_val != -1:
                        if prev_task.task_id != task_id_val:
                            # Completed task doesn't match previous task - potential duplicate execution
                            rid_str = self._color_robot(robot_id)
                            self.get_logger().warn(
                                f"[scheduler] {rid_str} task_done mismatch: completed {task_id_val} but expected {prev_task.task_id} at index {prev_idx}"
                            )
                            # Try to find the actual task in sequence and adjust index
                            for check_idx, check_task in enumerate(seq):
                                if check_task.task_id == task_id_val:
                                    if check_idx >= current_idx:
                                        # Task is in future - this shouldn't happen, but update index
                                        self.current_seq_index[robot_id] = check_idx + 1
                                        self.get_logger().warn(
                                            f"[scheduler] {rid_str} adjusted index to {check_idx + 1} based on completed task"
                                        )
                                    break

            # Reset carried_weight when task is done (all items delivered)
            # Also clear task progress tracking and waypoint start index
            if task_id_val is not None and task_id_val != -1:
                state.carried_weight = 0.0
                progress_key = (robot_id, task_id_val)
                
                # Get task type for final message
                task_type = "UNKNOWN"
                active_seq = self.current_sequences.get(robot_id, [])
                for t in active_seq:
                    if t.task_id == task_id_val:
                        task_type = t.task_type
                        break
                
                # Clear tracking dictionaries
                if progress_key in self.task_progress:
                    del self.task_progress[progress_key]
                if progress_key in self.task_waypoint_start:
                    del self.task_waypoint_start[progress_key]
                if progress_key in self.task_start_times:
                    del self.task_start_times[progress_key]
                if progress_key in self.task_start_soc:
                    del self.task_start_soc[progress_key]
                if progress_key in self.task_waypoint_times:
                    del self.task_waypoint_times[progress_key]
                if progress_key in self.task_waypoint_end_soc:
                    del self.task_waypoint_end_soc[progress_key]
                
                rid_str = self._color_robot(robot_id)
                self.get_logger().info(
                    f"[scheduler] {rid_str} task {task_id_val} done: carried_weight reset to 0.0"
                )

            if task_id_val == -1:
                # CHARGE task done: publish execution message
                progress_key = (robot_id, task_id_val)
                current_time = self.get_clock().now().nanoseconds / 1e9
                task_start_time = self.task_start_times.get(progress_key, current_time)
                start_soc = self.task_start_soc.get(progress_key, state.soc)
                end_soc = state.soc
                self._publish_task_execution(robot_id, task_id_val, "CHARGE", -1, task_start_time, current_time, start_soc, end_soc)
                
                # Clean up tracking
                if progress_key in self.task_start_times:
                    del self.task_start_times[progress_key]
                if progress_key in self.task_start_soc:
                    del self.task_start_soc[progress_key]
                if progress_key in self.task_waypoint_times:
                    del self.task_waypoint_times[progress_key]
                if progress_key in self.task_waypoint_end_soc:
                    del self.task_waypoint_end_soc[progress_key]
                
                # Remove this robot from all charger queues where it appears.
                for cid, q in self.charger_queues.items():
                    if robot_id in q:
                        q.remove(robot_id)
                        # Apply charging effect based on absolute Wh/s rate.
                        # Use the actual elapsed time of this CHARGE task
                        # (current_time - task_start_time) and
                        # self.charge_rate_wh_per_s. Example:
                        #   charge_rate_wh_per_s=0.1 → 40Wh needs 400s,
                        #   60Wh needs 600s from empty to full.
                        cap_wh = self.robot_cap_wh.get(robot_id, 120.0)
                        now_sec = self.get_clock().now().nanoseconds / 1e9
                        start_time = self.task_start_times.get((robot_id, -1), now_sec)
                        elapsed = max(0.0, now_sec - start_time)
                        gain_per_sec = max(0.0, self.charge_rate_wh_per_s)
                        total_gain = gain_per_sec * elapsed
                        before = state.soc
                        state.soc = min(cap_wh, state.soc + total_gain)
                        rid_str = self._color_robot(robot_id)
                        self.get_logger().info(
                            f"[charge] {rid_str} finished CHARGE at charger {cid}, "
                            f"+{total_gain:.2f}Wh over {elapsed:.1f}s ({before:.2f}Wh -> {state.soc:.2f}Wh), queue={q}"
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
            # Note: carried_weight is NOT reset here because the robot continues
            # with the items it already picked, so weight should be maintained.
                
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
                            # Charge duration based on remaining energy and absolute Wh/s charge rate
                            cap_wh = self.robot_cap_wh.get(robot_id, 120.0)
                            remaining_wh = max(0.0, cap_wh - state.soc)
                            if self.charge_rate_wh_per_s <= 0.0 or remaining_wh <= 0.0:
                                charge_duration_s = 0.0
                            else:
                                charge_duration_s = remaining_wh / self.charge_rate_wh_per_s

                            charge_task = TaskSpec(
                                task_id=-1,
                                task_type="CHARGE",
                                picks=[(state.x, state.y)],
                                drops=[(ch.x, ch.y)],
                                pick_wait_s=0.0,
                                drop_wait_s=0.0,
                                charge_duration_s=charge_duration_s,
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
                    # PICK_DROP (or other) failure: recover Nav2 state and resend starting from the
                    # point nearest the robot so we resume rather than restart.
                    # If a waypoint fails multiple times, skip it to avoid getting stuck.
                    if failed_task is not None:
                        # Perform Nav2 recovery steps before retrying
                        self._recover_nav2_for_robot(robot_id, state)
                        
                        # Always rebuild waypoints from original TaskSpec to ensure consistency
                        cmd = self._make_task_command_from_spec(failed_task)

                        waypoints = getattr(cmd, 'waypoints', []) or []
                        waits = getattr(cmd, 'waits', []) or []
                        
                        # Get the waypoint start index from previous retry, or use failure point
                        progress_key = (robot_id, failed_tid)
                        if progress_key in self.task_waypoint_start:
                            # Use the start index from previous retry to maintain consistency
                            start_idx = self.task_waypoint_start[progress_key]
                        else:
                            # First failure: start from the failed waypoint index
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

                                # Check progress to avoid retrying completed waypoints
                                progress = self.task_progress.get(progress_key, {"picks": set(), "drops": set()})
                                
                                if failed_task.task_type == "PICK_DROP":
                                    # PICK_DROP: waypoint 0=pick, waypoint 1=drop
                                    # Check if pick was already completed
                                    pick_completed = (0 in progress["picks"])
                                    # Check if drop was already completed
                                    drop_completed = (0 in progress["drops"])
                                    
                                    # If drop is already completed, task should be done - skip retry
                                    if drop_completed:
                                        state.available = True
                                        self.get_logger().warn(
                                            f"[scheduler] {rid_str} task {failed_tid} retry: drop already completed, marking task done"
                                        )
                                    # If we're retrying from pick waypoint (start_idx=0) but pick was already completed,
                                    # skip to drop waypoint (index 1) since pick operation already finished
                                    elif start_idx == 0 and pick_completed:
                                        self.get_logger().info(
                                            f"[scheduler] {rid_str} task {failed_tid} retry: pick already completed (progress={progress}), "
                                            f"failure at waypoint {wp_idx}, skipping to drop waypoint"
                                        )
                                        if len(waypoints) > 1:
                                            start_idx = 1
                                            self.task_waypoint_start[progress_key] = 1
                                        else:
                                            # No more waypoints, task should be done - skip retry
                                            state.available = True
                                            self.get_logger().warn(
                                                f"[scheduler] {rid_str} task {failed_tid} retry: pick completed but no drop waypoint, marking done"
                                            )
                                    # If start_idx=1 (drop waypoint) and drop not completed, that's normal - continue retry
                                
                                elif failed_task.task_type == "DUAL_PICK_MULTI_DROP":
                                    # DUAL_PICK_MULTI_DROP: waypoint 0=pick 0, waypoint 1=drop 0, waypoint 2=pick 1, waypoint 3=drop 1
                                    # Find the first incomplete waypoint starting from start_idx
                                    # Check which picks and drops are already completed
                                    max_pick_idx = max(progress["picks"]) if progress["picks"] else -1
                                    max_drop_idx = max(progress["drops"]) if progress["drops"] else -1
                                    
                                    # If all drops are completed, task should be done
                                    if len(progress["drops"]) >= 2:
                                        state.available = True
                                        self.get_logger().warn(
                                            f"[scheduler] {rid_str} task {failed_tid} retry: all drops already completed, marking task done"
                                        )
                                    else:
                                        # Find the first incomplete waypoint
                                        # waypoint 0 = pick 0, waypoint 1 = drop 0, waypoint 2 = pick 1, waypoint 3 = drop 1
                                        target_start_idx = start_idx
                                        
                                        # Check each waypoint from start_idx to find the first incomplete one
                                        for wp_i in range(start_idx, len(waypoints)):
                                            if wp_i % 2 == 0:
                                                # Even waypoint = pick
                                                pick_idx = wp_i // 2
                                                if pick_idx not in progress["picks"]:
                                                    # This pick is not completed, start from here
                                                    target_start_idx = wp_i
                                                    break
                                            else:
                                                # Odd waypoint = drop
                                                drop_idx = (wp_i - 1) // 2
                                                if drop_idx not in progress["drops"]:
                                                    # This drop is not completed, start from here
                                                    target_start_idx = wp_i
                                                    break
                                        
                                        if target_start_idx != start_idx:
                                            self.get_logger().info(
                                                f"[scheduler] {rid_str} task {failed_tid} retry: skipping completed waypoints "
                                                f"(progress={progress}), adjusting start_idx from {start_idx} to {target_start_idx}"
                                            )
                                            start_idx = target_start_idx
                                            self.task_waypoint_start[progress_key] = target_start_idx
                                
                                elif failed_task.task_type == "MULTI_PICK_DROP":
                                    # MULTI_PICK_DROP: waypoint 0=pick 0, waypoint 1=pick 1, waypoint 2=drop 0, waypoint 3=drop 1
                                    # Find the first incomplete waypoint starting from start_idx
                                    num_picks = len(failed_task.picks) if failed_task.picks else 0
                                    num_drops = len(failed_task.drops) if failed_task.drops else 0
                                    
                                    # If all drops are completed, task should be done
                                    if len(progress["drops"]) >= num_drops:
                                        state.available = True
                                        self.get_logger().warn(
                                            f"[scheduler] {rid_str} task {failed_tid} retry: all drops already completed, marking task done"
                                        )
                                    else:
                                        # Find the first incomplete waypoint
                                        # First num_picks waypoints are picks, rest are drops
                                        target_start_idx = start_idx
                                        
                                        # Check each waypoint from start_idx to find the first incomplete one
                                        for wp_i in range(start_idx, len(waypoints)):
                                            if wp_i < num_picks:
                                                # This is a pick waypoint
                                                pick_idx = wp_i
                                                if pick_idx not in progress["picks"]:
                                                    # This pick is not completed, start from here
                                                    target_start_idx = wp_i
                                                    break
                                            else:
                                                # This is a drop waypoint
                                                drop_idx = wp_i - num_picks
                                                if drop_idx not in progress["drops"]:
                                                    # This drop is not completed, start from here
                                                    target_start_idx = wp_i
                                                    break
                                        
                                        if target_start_idx != start_idx:
                                            self.get_logger().info(
                                                f"[scheduler] {rid_str} task {failed_tid} retry: skipping completed waypoints "
                                                f"(progress={progress}), adjusting start_idx from {start_idx} to {target_start_idx}"
                                            )
                                            start_idx = target_start_idx
                                            self.task_waypoint_start[progress_key] = target_start_idx
                                    
                                
                                # Only proceed with retry if task is not done
                                if not state.available:
                                    # Retry from the same waypoint (start_idx) using original waypoints
                                    # Always slice from original waypoints to avoid accumulating slices
                                    new_waypoints = waypoints[start_idx:]
                                    new_waits = [float(w) for w in waits[start_idx:]]

                                resumed.waypoints = new_waypoints
                                resumed.waits = new_waits
                                resumed.charge_duration_s = float(getattr(cmd, 'charge_duration_s', 0.0))

                                # Track the waypoint start index for retry (needed to convert relative waypoint indices in waypoint_reached events)
                                # This ensures subsequent retries use the same start_idx from the original waypoints
                                self.task_waypoint_start[progress_key] = start_idx

                                pub.publish(resumed)
                                state.available = False
                                self.get_logger().warn(
                                        f"[scheduler] {rid_str} task_failed for id={failed_tid}: resumed at ordered stage {start_idx} "
                                        f"(retrying from original waypoint, total waypoints={len(waypoints)})"
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
            pass  # self.get_logger().info('[scheduler] Robot SOCs: ' + ', '.join(parts))

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

        # Per-robot color map (RGB) - supports up to 10 robots
        color_map = {
            'tb1': (1.0, 0.0, 0.0),   # red
            'tb2': (1.0, 0.5, 0.0),   # orange
            'tb3': (1.0, 1.0, 0.0),   # yellow
            'tb4': (0.0, 1.0, 0.0),   # green
            'tb5': (0.0, 0.0, 1.0),   # blue
            'tb6': (1.0, 0.0, 1.0),   # magenta
            'tb7': (0.0, 1.0, 1.0),   # cyan
            'tb8': (1.0, 0.3, 0.3),   # light red
            'tb9': (0.3, 1.0, 0.3),   # light green
            'tb10': (0.3, 0.3, 1.0),  # light blue
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

        # Makespan-dominant weighting (w_t > w_e)
        w_t = 1.0
        w_e = 0.1

        # ---------- objective ----------
        def _objective_function(solution: np.ndarray) -> float:
            # 1) Random-key decoding: sort by the real-valued keys to build the sequence.
            keys = solution
            ordered = sorted(zip(keys, candidate_tasks), key=lambda x: x[0])
            seq = [t for _, t in ordered]

            # Interpret the energy state in Wh
            energy = robot_state.soc

            # Assume the robot departs for the first task from its current pose
            sim_robot = RobotState(
                robot_id=robot_state.robot_id,
                namespace=robot_state.namespace,
                x=robot_state.x,
                y=robot_state.y,
                soc=robot_state.soc,
                available=robot_state.available,
            )

            total_energy = 0.0  # Sum of energy consumption
            total_time = 0.0    # Sum of elapsed time (including charging)
            penalty = 0.0

            for t in seq:
                # Step 1: decide whether the current pose can execute this task immediately (charge decision)
                # Approximate the prev_pose -> pick segment as its own TaskSpec to compute energy/time
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
                    # Step 2: if charging is required before the task, insert a charge event
                    try:
                        charger = self._nearest_available_charger(sim_robot.x, sim_robot.y)
                    except Exception:
                        charger = None

                    if charger is None:
                        # No available charger means infeasible, so apply a large penalty
                        penalty += 1e6
                        break

                    # Energy/time to travel from the previous pose to the charger
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
                        # Not enough energy to reach the charger is infeasible
                        penalty += 1e6
                        break

                    # Move to the charger
                    energy -= e_to_charger
                    total_energy += e_to_charger
                    total_time += t_to_charger

                    # Charging duration at the station (energy resets to capacity)
                    charge_time = 10.0  # Fixed charge duration (parameterize later if needed)
                    total_time += charge_time
                    energy = cap_wh  # full capacity 로 리셋

                    # Update pose to the charger location
                    sim_robot.x = charger.x
                    sim_robot.y = charger.y

                    # After charging, recompute energy/time from the charger pose to the pick/task
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
                        # Still infeasible after charging
                        penalty += 1e6
                        break

                # Step 3: execute the task after any required charging
                energy -= required_energy          # Energy drop from executing the task
                total_energy += required_energy    # Accumulate energy consumption
                total_time += t_to_pick            # Accumulate travel plus task time

                # Update robot pose to the final drop position
                sim_robot.x = t.drop_x
                sim_robot.y = t.drop_y

            # Objective: minimize J = w_t * T_total + w_e * E_total + penalty
            return w_t * total_time + w_e * total_energy + penalty

        # ---------- mealpy problem ----------
        problem = {
            "obj_func": _objective_function,
            "bounds": FloatVar(lb=(-1.0,) * K, ub=(1.0,) * K),
            "minmax": "min",
            # Use console logging to disable tqdm progress bar and enable epoch logs
            "log_to": "console",
        }

        # Create optimizer (use SMA for task sequence optimization)
        epoch = self.optimization_epoch
        pop_size = self.optimization_pop_size
        model = SMA.OriginalSMA(epoch=epoch, pop_size=pop_size, pr=0.03)
        
        self.get_logger().info(
            f"[optimization] Starting mealpy optimization for task sequence with SMA: epoch={epoch}, pop_size={pop_size}"
        )
        
        best = model.solve(problem)
        
        # Log best solution objective value
        # Target object has fitness property that returns the objective value
        try:
            if hasattr(best, 'target'):
                if hasattr(best.target, 'fitness'):
                    best_obj_value = float(best.target.fitness)
                elif hasattr(best.target, 'objectives') and len(best.target.objectives) > 0:
                    best_obj_value = float(best.target.objectives[0])
                elif isinstance(best.target, (int, float)):
                    best_obj_value = float(best.target)
                else:
                    best_obj_value = float(best.target)
            else:
                best_obj_value = float('inf')
        except (ValueError, TypeError, AttributeError) as e:
            self.get_logger().warn(f"Could not convert best.target to float: {e}, using inf")
            best_obj_value = float('inf')
        
        self.get_logger().info(
            f"[optimization] Best task sequence solution: objective={best_obj_value:.6f}"
        )

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
        # charge for a duration based on remaining energy and Wh/s rate.
        # The actual SOC gain is applied in the task_done handler for CHARGE tasks.
        remaining_wh = max(0.0, cap_wh - state.soc)
        if self.charge_rate_wh_per_s <= 0.0 or remaining_wh <= 0.0:
            charge_duration_s = 0.0
        else:
            charge_duration_s = remaining_wh / self.charge_rate_wh_per_s
        charge_task = TaskSpec(
            task_id=-1,
            task_type="CHARGE",
            picks=[(charger.x, charger.y)],
            drops=[(charger.x, charger.y)],
            pick_wait_s=0.0,
            drop_wait_s=0.0,
            charge_duration_s=charge_duration_s,
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

        # Record robot start time when first task is assigned (not when task actually starts)
        if self.robot_start_time.get(rid) is None:
            self.robot_start_time[rid] = self.get_clock().now()

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
                # Robot is currently busy (task assigned but not yet completed)
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

            # CRITICAL: Do NOT increment index here - wait for task_started event
            # to ensure we only increment when task actually starts executing.
            # Otherwise, if task publication fails or robot doesn't start, we'd skip tasks.
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

            # For CHARGE tasks, select charger based on current robot position at execution time
            if task_spec.task_type == "CHARGE":
                charge_duration = self._compute_charge_duration(rid, state.soc)
                charger = self._nearest_charger(state.x, state.y)
                if charger:
                    self.current_goals[rid] = (charger.x, charger.y)
                    task_spec = TaskSpec(
                        task_id=-1,
                        task_type="CHARGE",
                        picks=[(charger.x, charger.y)],
                        drops=[(charger.x, charger.y)],
                        pick_wait_s=0.0,
                        drop_wait_s=0.0,
                        charge_duration_s=charge_duration,
                    )
                else:
                    # Fall back to existing task but ensure duration reflects SOC
                    task_spec.charge_duration_s = charge_duration

            cmd = self._make_task_command_from_spec(task_spec)
            pub = self.task_pubs.get(rid)
            if pub is None:
                continue

            # Record robot start time when first task is assigned (not when task actually starts)
            if self.robot_start_time.get(rid) is None:
                self.robot_start_time[rid] = self.get_clock().now()

            pub.publish(cmd)
            state.available = False
            # record that we are waiting for a task_started event
            try:
                self.pending_task_start[rid] = self.get_clock().now().nanoseconds / 1e9
            except Exception:
                self.pending_task_start[rid] = time.time()
            # Note: current_task_id is set in task_started event handler, not here
            # This allows us to distinguish between new tasks and retries

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
            # MULTI_PICK_DROP: concatenate all picks then all drops
            # Do NOT collapse same-location picks - each pick needs its own waypoint
            # so that waypoint_reached events can distinguish pick 0, pick 1, etc.
            for (px, py) in spec.picks:
                ps = PoseStamped()
                ps.header.frame_id = 'map'
                ps.pose.position.x = float(px)
                ps.pose.position.y = float(py)
                waypoints.append(ps)
                waits.append(float(spec.pick_wait_s))

            # Add all drops as separate waypoints
            for (dx, dy) in spec.drops:
                ds = PoseStamped()
                ds.header.frame_id = 'map'
                ds.pose.position.x = float(dx)
                ds.pose.position.y = float(dy)
                waypoints.append(ds)
                waits.append(float(spec.drop_wait_s))

        
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
