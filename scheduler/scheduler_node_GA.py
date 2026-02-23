from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import copy
import math
import random
import time
import yaml

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
from mealpy.bio_based import SMA
from mealpy.evolutionary_based import GA, CRO
from mealpy.swarm_based import PSO


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
        # Provide a non-empty string array as default for type inference
        # If parameter file provides values, they will override this default
        param_value = self.declare_parameter(
            'robot_ids', 
            ['tb1'],  # Non-empty default ensures STRING_ARRAY type, will be overridden by param file
            param_descriptor
        )
        robot_ids_param = param_value.get_parameter_value().string_array_value
        self.robot_ids: List[str] = [str(rid) for rid in robot_ids_param] if robot_ids_param else []
        
        if not self.robot_ids:
            self.get_logger().warn("No robot_ids provided! Please set robot_ids parameter.")
        
        self.get_logger().info(f"Initialized scheduler with {len(self.robot_ids)} robots: {self.robot_ids}")
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
        # Task execution timeline publisher for visualization
        self.task_execution_pub = self.create_publisher(TaskExecution, '/task_execution_log', 10)
        # Optimization time publisher for metrics
        self.optimization_time_pub = self.create_publisher(Float32, '/optimization_time', 10)
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
        # Track completed pick/drop indices per (robot_id, task_id) to avoid duplicate weight updates on retry
        # Format: (robot_id, task_id) -> {"picks": set of completed pick indices, "drops": set of completed drop indices}
        self.task_progress: Dict[Tuple[str, int], Dict[str, set]] = {}
        # Track the waypoint start index for retried tasks (to convert relative waypoint indices to absolute)
        # Format: (robot_id, task_id) -> waypoint_start_idx (0 if new task, >0 if retry)
        self.task_waypoint_start: Dict[Tuple[str, int], int] = {}
        # Track waypoint start/end times for visualization: (robot_id, task_id) -> list of (waypoint_index, start_time, end_time)
        self.task_waypoint_times: Dict[Tuple[str, int], List[Tuple[int, float, float]]] = {}
        # Track waypoint end SOC: (robot_id, task_id) -> list of end_soc (for next waypoint start_soc)
        self.task_waypoint_end_soc: Dict[Tuple[str, int], List[float]] = {}
        # Track task start time: (robot_id, task_id) -> start_time
        self.task_start_times: Dict[Tuple[str, int], float] = {}
        # Track task start SOC: (robot_id, task_id) -> start_soc
        self.task_start_soc: Dict[Tuple[str, int], float] = {}
        # Distance threshold (meters) used to detect that a robot reached
        # Per-robot retry counters for failed tasks (avoid infinite retries)
        self.retry_counts: Dict[str, int] = {rid: 0 for rid in self.robot_ids}
        # Max number of automatic retries before giving up / marking dead
        self.retry_limit: int = 3
        # Per-robot current task sequence (queue of TaskSpec) and index into that sequence.
        # A robot must finish all tasks in its current sequence before receiving a new one.
        self.current_sequences: Dict[str, List[TaskSpec]] = {rid: [] for rid in self.robot_ids}
        self.current_seq_index: Dict[str, int] = {rid: 0 for rid in self.robot_ids}
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
        self.default_item_weight_kg = 1.0
        # Multiplier per kg applied on top of loaded_factor when computing energy
        # e.g., energy_multiplier = loaded_factor + load_weight_factor * carried_kg
        self.load_weight_factor = 0.05

        # Acceleration/inertia-based energy consumption parameters
        # Base robot mass (kg) - affects inertia-based energy consumption
        # Higher mass = more energy needed to accelerate/decelerate
        self.base_robot_mass_kg = 5.0  # Typical TurtleBot3 mass (~5kg)f
        # Store last velocity per robot for calculating acceleration energy
        self._last_velocity: Dict[str, Tuple[float, float]] = {}  # (vx, vy) in m/s
        
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
            ChargerSpec(charger_id=1, x=-3.0, y=-3.0),
            ChargerSpec(charger_id=2, x=0.0, y=-3.0),
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

        # Optimization parameters (algorithm selection removed - now runs all 4 algorithms)
        self.optimization_epoch = self.declare_parameter("optimization_epoch", 250).get_parameter_value().integer_value
        self.optimization_pop_size = self.declare_parameter("optimization_pop_size", 150).get_parameter_value().integer_value

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
        # Log optimizer configuration (mealpy) once at startup for visibility
        self.get_logger().info(
            f"SchedulerNode started for robots={self.robot_ids}; "
            f"task allocation: running all 4 algorithms (SMA, GA, PSO, CRO) and selecting best (epoch={self.optimization_epoch}, pop_size={self.optimization_pop_size})"
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
    # Path planning helper for accurate distance calculation
    # ------------------------------------------------------------------

    def _get_path_planning_client(self, robot_id: Optional[str]) -> Optional[ActionClient]:
        """Get or create action client for path planning."""
        if robot_id is None:
            robot_id = self.robot_ids[0] if self.robot_ids else None
            if robot_id is None:
                return None
        
        if robot_id not in self.path_planning_clients:
            action_name = f'/{robot_id}/compute_path_to_pose'
            client = ActionClient(self, ComputePathToPose, action_name)
            self.path_planning_clients[robot_id] = client
            self.get_logger().info(f"Initialized Nav2 path planning client for {robot_id}: {action_name}")
        
        client = self.path_planning_clients[robot_id]
        # Wait longer for action server to be ready (some robots may start slower)
        if not client.wait_for_server(timeout_sec=3.0):
            # Only log warning once per robot to avoid spam
            if robot_id not in getattr(self, '_logged_missing_action_server', set()):
                if not hasattr(self, '_logged_missing_action_server'):
                    self._logged_missing_action_server = set()
                self._logged_missing_action_server.add(robot_id)
                self.get_logger().warn(
                    f"Nav2 action server for {robot_id} not available: "
                    f"/{robot_id}/compute_path_to_pose action server not found. "
                    f"Will use euclidean distance for this robot. "
                    f"Check if Nav2 is properly configured for {robot_id}."
                        )
            # Don't disable Nav2 globally - just return None for this robot
            # Fallback to first robot if available, otherwise return None
            if robot_id != (self.robot_ids[0] if self.robot_ids else None):
                fallback_client = self._get_path_planning_client(self.robot_ids[0])
                if fallback_client is not None:
                    self.get_logger().debug(f"Using fallback client for {self.robot_ids[0]} instead of {robot_id}")
                return fallback_client
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
        try:
            send_goal_future = client.send_goal_async(goal_msg)
            rclpy.spin_until_future_complete(self, send_goal_future, timeout_sec=2.0)
            if not send_goal_future.done():
                self.get_logger().debug(
                    f"Nav2 path planning goal send timeout for {robot_id}: "
                    f"({start_x:.2f}, {start_y:.2f}) -> ({goal_x:.2f}, {goal_y:.2f})"
                )
                return None
            goal_handle = send_goal_future.result()
            if not goal_handle or not goal_handle.accepted:
                self.get_logger().debug(
                    f"Nav2 path planning goal rejected for {robot_id}: "
                    f"({start_x:.2f}, {start_y:.2f}) -> ({goal_x:.2f}, {goal_y:.2f})"
                )
                return None
            
            result_future = goal_handle.get_result_async()
            rclpy.spin_until_future_complete(self, result_future, timeout_sec=2.0)
            if not result_future.done():
                self.get_logger().debug(
                    f"Nav2 path planning result timeout for {robot_id}: "
                    f"({start_x:.2f}, {start_y:.2f}) -> ({goal_x:.2f}, {goal_y:.2f})"
                )
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
        
        ⭐ 핵심 원칙: Gazebo pose → TF로 반영 → Nav2가 자연스럽게 따라오게
        
        베스트 구조 (Recovery 시):
        Step 1. Gazebo에서 ground truth pose 획득 (/gazebo/model_states)
        Step 2. odom → base_link는 그대로 둔다 (Gazebo가 자동 업데이트)
        Step 3. map → odom TF를 재설정 (TF broadcaster로 직접 발행)
        
        이렇게 하면 costmap, planner, controller 전부 자동으로 정합됨.
        
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
            # ⭐ 핵심: Gazebo pose → TF로 반영 → Nav2가 자연스럽게 따라오게
            # odom->base_footprint는 그대로 두고, map->odom만 재설정
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
                        charge_task = TaskSpec(
                            task_id=-1,
                            task_type="CHARGE",
                            picks=[(x, y)],
                            drops=[(charger.x, charger.y)],
                            pick_wait_s=0.0,
                            drop_wait_s=0.0,
                            # One CHARGE action corresponds to instant charge (1s)
                            # Actual executor charges instantly, so we use 1 second
                            charge_duration_s=1.0,
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
        ]

        # Type 2: MULTI_PICK_DROP tasks (2 picks -> drop -> drop, 3점)
        # picks = [pick_loc, pick_loc] (같은 위치에서 2번 pick)
        # drops = [drop1, drop2]
        # waypoint 순서: pick_loc -> pick_loc -> drop1 -> drop2
        # Each entry is (pick_x, pick_y, drop1_x, drop1_y, drop2_x, drop2_y)
        demo_pairs_multi = [
            # group 1
            (-4.50,  7.05,  -6.35,  6.05,  -6.05, -2.05),
            # group 2
            (-3.00, -9.45,   3.45, -8.65,   3.75, -0.75),
            # group 3
            ( 4.65,  1.70,  -4.95,  4.20,  -6.40, -1.80),
            # group 4
            (-6.65,  5.75,  -2.35, -5.35,  -5.35, -8.35),
            # group 5
            ( 1.40,  5.00,   1.70,  7.35,   0.85, -4.45),
            # group 6
            ( 3.15, -9.35,  -5.95, -5.15,  -4.85,  6.25),
            # group 7
            (-6.45, -0.70,  -3.55, -1.75,  -3.75,  4.75),
            # group 8
            (-6.10,  6.15,  -5.80,  7.00,   1.70, -3.35),
            # group 9
            ( 3.15, -3.40,   5.40, -7.65,   5.10, -2.45),
            # group 10
            (-1.35, -0.65,   1.85, -2.20,  -0.55,  1.85),
            # group 11
            (-3.20, -4.75,  -5.10,  9.25,  -3.50, -6.00),
            # group 12
            ( 1.30,  1.25,   4.70,  1.95,   0.45, -4.05),
            # group 13
            (-4.10,  0.45,  -0.85, -0.05,  -0.45,  5.80),
            # group 14
            (-4.05, -5.65,  -5.60,  0.60,   0.45,  3.70),
            # group 15
            (-3.35,  5.10,  -3.30, -3.45,   2.40,  1.25),
            # group 16
            ( 0.65,  1.40,   0.25, -2.95,  -2.50,  3.65),
            # group 17
            (-1.05, -1.20,   3.95, -3.45,   0.40, -7.80),
            # group 18
            ( 4.00, -0.40,  -4.80,  4.70,   1.10, -3.90),
            # group 19
            ( 1.40,  3.50,   5.40, -9.05,  -6.35, -2.30),
            # group 20
            ( 1.50, -4.15,   5.15,  0.70,  -5.90, -8.00),
        ]

        # Type 3: DUAL_PICK_MULTI_DROP tasks
        # picks = [pick_loc, drop], drops = [drop, pick_loc]
        # waypoint 순서: pick_loc -> drop -> drop -> pick_loc
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
            # (-1.55, -3.95, 3.75, 1.15),   # task 81
            # (-2.85, 8.20, 4.80, -2.45),   # task 82
            # (-2.75, 8.75, -0.50, 5.65),   # task 83
            # (-2.55, -2.05, -0.35, 1.55),  # task 84
            # (-1.05, -1.45, -6.50, 6.05),  # task 85
            # (-1.50, -0.45, -0.45, -0.95), # task 86
            # (-3.95, -1.70, 6.00, -8.25),  # task 87
            # (-5.85, -4.25, -4.15, 5.30),  # task 88
            # (-5.05, -4.30, 0.95, -2.65),  # task 89
            # (4.90, -7.70, -3.10, -9.05),  # task 90
            # (-1.65, -4.30, -2.65, 7.00),  # task 91
            # (-5.45, -3.75, -2.50, 3.50),  # task 92
            # (-6.15, -4.35, 5.25, -4.35),  # task 93
            # (-5.25, 8.90, 2.85, -2.20),   # task 94
            # (-3.45, 1.50, 3.70, -8.35),   # task 95
            # (-5.00, -7.30, 0.40, 6.65),   # task 96
            # (-6.35, -4.30, -1.05, -5.70), # task 97
            # (0.10, 6.00, 5.80, -6.35),    # task 98
            # (-4.95, 6.75, -6.55, -3.15),  # task 99
            # (-6.45, -0.20, 0.90, -5.40)   # task 100
        ]

        # Combine all task data and randomly select task types
        self.pending_tasks = []
        tid = 1
        
        # Create tasks from demo_specs (PICK_DROP)
        pick_drop_tasks = []
        for (px, py, dx, dy) in demo_specs:
            pick_drop_tasks.append(('PICK_DROP', (px, py), (dx, dy), None, None))
        
        # Create tasks from demo_pairs_multi (MULTI_PICK_DROP: 2 picks -> drop -> drop)
        # picks = [pick_loc, pick_loc] (같은 위치에서 2번 pick)
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
        # waypoint 순서: pick_loc -> drop -> drop -> pick_loc
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
        random.shuffle(all_tasks)
        
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

    def _optimize_with_algorithm(self, candidate_tasks: List[TaskSpec], algorithm_name: str) -> Tuple[Dict[str, List[TaskSpec]], float]:
        """Use mealpy to optimally assign all tasks to robots and optimize task sequences.
        
        This function optimizes both:
        1. Which robot gets which task (allocation)
        2. The order of tasks for each robot (sequencing)
        
        Encoding: For each task, we use two values:
        - robot_key: [0, num_robots) → determines which robot
        - seq_key: random key → determines order within that robot
        
        Returns:
            Tuple of (Dict mapping robot_id to list of assigned tasks in optimal order, max_makespan)
        """
        if not candidate_tasks:
            return {rid: [] for rid in self.robot_ids}, 0.0
        
        if not self.robot_ids:
            self.get_logger().warn("Cannot allocate tasks: no robots available")
            return {}, 0.0
        
        K = len(candidate_tasks)
        num_robots = len(self.robot_ids)
        robot_list = list(self.robot_ids)  # Fixed order for indexing
        
        # makespan-dominant 가중치 (w_t > w_e)
        w_t = 1.0
        w_e = 0.1
        
        # ---------- objective ----------
        def _objective_function(solution: np.ndarray) -> float:
            # Decode solution: [robot_key_1, seq_key_1, charge_key_1, robot_key_2, seq_key_2, charge_key_2, ...]
            # Each task has 3 values: robot assignment, sequence key, and charge decision
            assignments: Dict[str, List[Tuple[float, float, TaskSpec]]] = {rid: [] for rid in robot_list}
            
            for i, task in enumerate(candidate_tasks):
                robot_key = solution[3 * i]  # First value: robot assignment [-1.0, 1.0]
                seq_key = solution[3 * i + 1]  # Second value: sequence key
                charge_key = solution[3 * i + 2]  # Third value: charge decision [-1.0, 1.0]
                
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
            
            # Track per-robot costs for load balancing
            robot_times: List[float] = []
            robot_energies: List[float] = []
            
            # Global charger occupancy tracker (shared across all robots for realistic simulation)
            charger_next_free: Dict[int, float] = {i: 0.0 for i in range(num_chargers)}
            
            # ChargeEvent list for time-based event-driven charging simulation
            # Format: (robot_id, task_idx_in_robot, reachable_chargers, time_before_charge, simulated_charge_end_time, charge_duration)
            # reachable_chargers: List[Tuple[int, float, float]] = (charger_idx, arrive_time, energy_cost)
            # arrive_time: 해당 충전소에 도착하는 실제 시간 (1차 시뮬레이션 기준)
            # energy_cost: 충전소까지 이동하는 데 필요한 에너지
            # time_before_charge: 충전 전까지의 누적 시간
            # simulated_charge_end_time: Phase 1에서 시뮬레이션된 충전 완료 시간 (충전 후 작업 시간 계산에 사용)
            # charge_duration: 이 충전 이벤트에 대해 필요한 실제 충전 시간 (초)
            charge_events: List[Tuple[str, int, List[Tuple[int, float, float]], float, float, float]] = []

            # Track global charging statistics for objective shaping
            # 총 충전 시간(대기 제외)과 충전 이벤트 개수를 집계해서
            # feasible 해들 사이에서 "충전 덜 하는 해"를 선호하도록 사용한다.
            total_charge_time: float = 0.0
            total_charge_count: int = 0
            
            for rid in robot_list:
                if not assignments[rid]:
                    continue
                
                state = self.robot_states.get(rid)
                if state is None:
                    # Infeasibility penalty: robot not available
                    total_cost += 1e6
                    continue

                # Sort tasks by sequence key
                sorted_tasks = sorted(assignments[rid], key=lambda x: x[0])
                seq_with_charge = [(charge_key, t) for _, charge_key, t in sorted_tasks]
                seq = [t for _, t in seq_with_charge]
                charge_keys = [charge_key for charge_key, _ in seq_with_charge]
                
                # Simulate robot execution
                sim_robot = RobotState(
                    robot_id=state.robot_id,
                    namespace=state.namespace,
                    x=state.x,
                    y=state.y,
                    soc=state.soc,
                    available=state.available,
                )

                cap_wh = self.robot_cap_wh.get(rid, 120.0)
                energy = state.soc
                robot_energy = 0.0 #사용한 에너지
                robot_time = 0.0
                penalty = 0.0  # Infeasibility penalty: accumulates for infeasible solutions
                total_penalty = 0.0
                
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
                        # Soft penalty: count a violation instead of killing the solution
                        required_energy, t_to_pick = 0.0, 0.0
                        penalty += 1.0
                        break

                    # Check if charging is needed based on charge_key
                    # charge_key > 0: charge before task, charge_key ≤ 0: do not charge
                    should_charge = (charge_key > 0.0)
                    
                    # Penalty: charge_key ≤ 0 but insufficient energy
                    if not should_charge and energy < required_energy:
                        # Soft penalty: insufficient energy without charging → count violation
                        penalty += 1.0
                        break
                    
                    # If should_charge, process charging (1st phase: simplified simulation)
                    if should_charge:
                        # Collect all charger candidates
                        if not chargers_list:
                            # Soft penalty: no charger available when requested
                            penalty += 1.0
                            break
                        
                        # Store time before charge (for Phase 2 time propagation)
                        time_before_charge = robot_time
                        
                        # Compute required charge duration based on remaining SOC.
                        # Charging rule: 0.5초당 2% → 초당 4% (0.04) 충전.
                        # 남은 에너지 비율 remaining_frac 만큼을 채우는 데 걸리는 시간:
                        #   charge_duration = remaining_frac / 0.04
                        remaining_frac = max(0.0, (cap_wh - energy) / cap_wh) if cap_wh > 0.0 else 0.0
                        charge_rate_frac_per_sec = 0.04  # 4% per second
                        if remaining_frac <= 0.0:
                            charge_duration = 0.0
                        else:
                            charge_duration = remaining_frac / charge_rate_frac_per_sec

                        # Accumulate charging statistics (충전 시간 및 횟수)
                        if charge_duration > 0.0:
                            total_charge_time += charge_duration
                            total_charge_count += 1
                        
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
                            # Soft penalty: cannot reach any charger when charging is requested
                            penalty += 1.0
                            break
                        
                        # Select best charger (Phase 1): ignore contention, pick the earliest
                        # arrive_time + charge_duration. Phase 2에서 실제 충전 대기/경합을 반영한다.
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
                            # Soft penalty: failed to select a charger
                            penalty += 1.0
                            break
                        
                        first_ch = chargers_list[best_ch_idx]
                        
                        # Update energy and time
                        energy -= best_e_cost
                        robot_energy += best_e_cost
                        
                        # Move to charger: update robot_time to arrival time
                        # best_arrive_time = robot_time + t_to_ch (calculated at line 2012)
                        robot_time = best_charge_end  # This includes travel time to charger

                        energy = cap_wh  # Instant full charge
                        
                        # Record charge event for Phase 2 processing (including time_before_charge,
                        # simulated_charge_end_time, and per-event charge_duration)
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
                        # Soft penalty 4: insufficient energy after charging (task will fail)
                        penalty += 1.0
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
                
                # Store robot time (will be updated in Phase 2 based on actual charge end times)
                robot_times.append(robot_time)
                robot_energies.append(robot_energy)
                
                # Add infeasibility penalty: penalize solutions where tasks will fail
                # This includes:
                # - Energy estimation failures
                # - No charger available when needed
                # - Insufficient energy to reach charger
                # - Insufficient energy after charging to complete task
                total_penalty += penalty
            
            # ========================================================================
            # Phase 2: Event-driven time-based charger selection and time update
            # ========================================================================
            
            # Track charger availability: charger_idx -> next_free_time
            charger_next_free: Dict[int, float] = {i: 0.0 for i in range(num_chargers)}
            
            # Track robot current time: robot_id -> current_time
            # This will be updated as charge events are processed
            robot_current_time: Dict[str, float] = {}
            for rid_idx, rid in enumerate(robot_list):
                if rid_idx < len(robot_times):
                    robot_current_time[rid] = robot_times[rid_idx]
                else:
                    robot_current_time[rid] = 0.0
            
            # Track charge events per robot: robot_id -> list of (task_idx, charge_end_time)
            # Used to update robot completion time
            robot_charge_end_times: Dict[str, List[Tuple[int, float]]] = {rid: [] for rid in robot_list}
            
            # Sort charge events by earliest arrive_time (across all reachable chargers)
            # This ensures we process events in chronological order
            if charge_events:
                # Create event list with arrive_time for sorting
                # reachable_chargers_list 요소는 (charger_idx, arrive_time, energy_cost) 형태의 3튜플이다.
                event_list: List[Tuple[float, str, int, List[Tuple[int, float, float]], float, float, float]] = []
                for robot_id, task_idx, reachable_chargers_list, time_before_charge, simulated_charge_end_time, charge_duration in charge_events:
                    # Use earliest arrive_time for sorting
                    if reachable_chargers_list:
                        # reachable_chargers_list 원소 구조가 (ch_idx, arrive_time, energy_cost)이므로
                        # 언패킹도 3개로 맞춰준다.
                        earliest_arrive = min(arrive_time for _, arrive_time, _ in reachable_chargers_list)
                        event_list.append(
                            (earliest_arrive, robot_id, task_idx, reachable_chargers_list, time_before_charge, simulated_charge_end_time, charge_duration)
                        )
                
                # Sort by earliest arrive_time
                event_list.sort(key=lambda x: x[0])
                
                # Process each event: select best charger and update robot time immediately
                for earliest_arrive, robot_id, task_idx, reachable_chargers_list, time_before_charge, simulated_charge_end_time, charge_duration in event_list:
                    # Select best charger: earliest charge_end = max(arrive_time, charger_next_free) + charge_duration
                    best_charger_idx = None
                    best_charge_end = float('inf')
                    best_arrive_time = 0.0
                    
                    for ch_idx, arrive_time, _ in reachable_chargers_list:
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
                    
                    # Find robot index to get Phase 1 final time
                    rid_idx = robot_list.index(robot_id) if robot_id in robot_list else -1
                    if rid_idx < 0 or rid_idx >= len(robot_times):
                        continue
                    
                    phase1_final_time = robot_times[rid_idx]
                    
                    # Calculate time after charge in Phase 1: remaining task execution time
                    time_after_charge_phase1 = phase1_final_time - simulated_charge_end_time
                    
                    # Update robot current time: actual charge_end + remaining task execution time
                    # This propagates the charging delay to all subsequent tasks
                    robot_current_time[robot_id] = best_charge_end + time_after_charge_phase1
                    
                    # Track charge end time for this robot
                    robot_charge_end_times[robot_id].append((task_idx, best_charge_end))
            
            # Update robot_times based on event-driven time updates from Phase 2
            # For robots with charges, use robot_current_time (which was updated to charge_end)
            # For robots without charges, use the original robot_time (robot_current_time == robot_times[rid_idx])
            for rid_idx, rid in enumerate(robot_list):
                if rid_idx < len(robot_times):
                    # robot_current_time was initialized for all robots in Phase 2
                    # For robots with charges: updated to charge_end in Phase 2
                    # For robots without charges: remains as Phase 1 completion time
                    robot_times[rid_idx] = robot_current_time[rid]
                    max_makespan = max(max_makespan, robot_times[rid_idx])
            
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
            
            # ========================================================================
            # OBJECTIVE FUNCTION
            # ========================================================================
            # Soft-penalty objective:
            # - total_penalty      : 에너지 제약을 어긴 로봇(또는 위반 횟수) 개수
            # - total_charge_time  : 전체 충전 시간(대기 제외)
            # - total_charge_count : 전체 충전 이벤트 수
            # - robot_times        : 각 로봇의 완료 시간 (Phase 2까지 반영)
            #
            # 추가로, 한 로봇만 과도하게 늦게 끝나는 해를 줄이기 위해
            # 로봇 완료 시간의 불균형(imbalance)도 아주 작게 패널티로 넣는다.
            # imbalance = sum_i |T_i - mean(T)|

            active_robot_times = [t for t in robot_times if t > 0.0]
            if active_robot_times:
                avg_time = sum(active_robot_times) / float(len(active_robot_times))
                imbalance = sum(abs(t - avg_time) for t in active_robot_times)
            else:
                imbalance = 0.0

            lambda_violation = 50.0    # 에너지 위반 1회(또는 1로봇)당 패널티
            lambda_charge_time = 0.1   # 1초 충전당 작은 패널티
            lambda_charge_count = 20.0 # 충전 1회당 추가 패널티
            lambda_balance = 1.0       # 로봇 간 완료 시간 불균형 패널티 (sum |T_i - mean(T)|)

            total_cost = (
                max_makespan
                + lambda_violation * total_penalty
                + lambda_charge_time * total_charge_time
                + lambda_charge_count * float(total_charge_count)
                + lambda_balance * imbalance
            )

            # Final objective: minimize total_cost
            return total_cost
        
        # ---------- mealpy problem ----------
        # Solution dimension: 3 * K (robot_key + seq_key + charge_key for each task)
        problem = {
            "obj_func": _objective_function,
            "bounds": FloatVar(lb=(-1.0,) * (3 * K), ub=(1.0,) * (3 * K)),
            "minmax": "min",
            "log_to": None,
        }

        # Create optimizer based on algorithm_name
        epoch = self.optimization_epoch
        pop_size = self.optimization_pop_size
        
        if algorithm_name == 'GA':
            model = GA.OriginalGA(epoch=epoch, pop_size=pop_size, pc=0.9, pm=0.05)
        elif algorithm_name == 'PSO':
            model = PSO.OriginalPSO(epoch=epoch, pop_size=pop_size, c1=2.05, c2=2.05, w=0.4)
        elif algorithm_name == 'CRO':
            model = CRO.OriginalCRO(epoch=epoch, pop_size=pop_size, po=0.4, Fb=0.9, Fa=0.1, Fd=0.1, Pd=0.1, GCR=0.1, G=0.02)
        else:  # Default to SMA
            model = SMA.OriginalSMA(epoch=epoch, pop_size=pop_size, pr=0.03)
        
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
            return {rid: [] for rid in robot_list}, float('inf')
        
        # Check if best solution is valid
        if best is None or not hasattr(best, 'solution'):
            self.get_logger().error(
                f"[optimization] {algorithm_name} returned invalid solution"
            )
            return {rid: [] for rid in robot_list}, float('inf')
        
        # ---------- decode best solution ----------
        solution = best.solution
        
        # Verify solution size matches expected dimension
        expected_size = 3 * K  # [robot_key, seq_key, charge_key] per task
        actual_size = len(solution) if hasattr(solution, '__len__') else 0
        if actual_size != expected_size:
            self.get_logger().warn(
                f"[optimization] Solution size mismatch: expected {expected_size}, got {actual_size}"
            )
        
        assignments: Dict[str, List[Tuple[float, TaskSpec]]] = {rid: [] for rid in robot_list}
        
        for i, task in enumerate(candidate_tasks):
            if 3 * i + 2 >= len(solution):
                self.get_logger().warn(
                    f"[optimization] Solution too short for task {i}: need index {3 * i + 2}, have {len(solution)}"
                )
                break
            robot_key = solution[3 * i]  # [-1.0, 1.0]
            seq_key = solution[3 * i + 1]
            charge_key = solution[3 * i + 2]  # [-1.0, 1.0]
            
            # Map robot_key from [-1.0, 1.0] to [0, num_robots) for robot_id
            # Step 1: Convert [-1.0, 1.0] to [0.0, 1.0]
            normalized_key = (robot_key + 1.0) / 2.0
            # Step 2: Scale to [0.0, num_robots) and convert to integer index
            robot_idx = int(np.clip(normalized_key * num_robots, 0, num_robots - 1))
            robot_id = robot_list[robot_idx]
            
            assignments[robot_id].append((seq_key, charge_key, task))
        
        # Verify all tasks were assigned
        total_assigned_in_decode = sum(len(tasks) for tasks in assignments.values())
        if total_assigned_in_decode != K:
            self.get_logger().warn(
                f"[optimization] Task count mismatch after decode: expected {K}, got {total_assigned_in_decode}"
            )
        
        # Sort each robot's tasks by sequence key and insert CHARGE events
        # Use same charging logic as objective function, including charger conflict handling
        num_chargers = max(1, len(self.chargers))
        chargers_list = list(self.chargers)
        charger_reservations: Dict[int, List[Tuple[float, float]]] = {
            i: [] for i in range(num_chargers)
        }
        
        result: Dict[str, List[TaskSpec]] = {}
        robot_times: Dict[str, float] = {}  # Track time for each robot to calculate reservations
        max_makespan = 0.0  # Initialize max_makespan for result generation
        
        for rid in robot_list:
            if not assignments[rid]:
                result[rid] = []
                robot_times[rid] = 0.0
                continue
            
            sorted_tasks = sorted(assignments[rid], key=lambda x: x[0])
            seq_with_charge = [(charge_key, t) for _, charge_key, t in sorted_tasks]
            seq = [t for _, t in seq_with_charge]
            charge_keys = [charge_key for charge_key, _ in seq_with_charge]
            
            # Insert CHARGE events based on charge_key (same logic as objective function)
            state = self.robot_states.get(rid)
            if state is None:
                result[rid] = seq
                robot_times[rid] = 0.0
                continue

            augmented_seq = []
            sim_robot = RobotState(
                robot_id=state.robot_id,
                namespace=state.namespace,
                x=state.x,
                y=state.y,
                soc=state.soc,
                available=state.available,
            )
            cap_wh = self.robot_cap_wh.get(rid, 120.0)
            energy = state.soc
            robot_time = 0.0
            
            # Process tasks with charging insertion based on charge_key
            for task_idx, t in enumerate(seq):
                charge_key = charge_keys[task_idx]  # Get charge decision for this task
                should_charge = (charge_key > 0.0)
                
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
                    required_energy = float('inf')
                    t_to_pick = 0.0
                
                # Insert charge if charge_key > 0
                if should_charge:
                    try:
                        charger = self._nearest_available_charger(sim_robot.x, sim_robot.y)
                    except Exception:
                        charger = None
                    
                    if charger is None or not chargers_list:
                        # Skip charging if charger not available, but continue with task
                        # (This should not happen in valid solutions, but handle gracefully)
                        should_charge = False
                    
                    # Find charger index
                    charger_idx = None
                    for i, ch in enumerate(chargers_list):
                        if ch.x == charger.x and ch.y == charger.y:
                            charger_idx = i
                            break
                    if charger_idx is None:
                        charger_idx = 0
                    
                    # Calculate time to charger
                    try:
                        _, t_to_charger = self.estimate_task_energy(
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
                        t_to_charger = 0.0
                    
                    robot_time += t_to_charger
                    arrive_time = robot_time
                    
                    # Check for waiting time due to existing reservations
                    reservations = charger_reservations[charger_idx]
                    wait_time = 0.0
                    if reservations:
                        latest_free = arrive_time
                        for (s0, e0) in reservations:
                            if e0 > latest_free and s0 < arrive_time:
                                latest_free = e0
                        if latest_free > arrive_time:
                            wait_time = latest_free - arrive_time
                    
                    # Charging model: SOC-dependent duration.
                    # Rule: 0.5초당 2% → 초당 4% (0.04) 충전.
                    # 현재 에너지 energy 기준 남은 비율을 모두 채우는 데 필요한 시간으로 charge_time을 설정.
                    remaining_frac = max(0.0, (cap_wh - energy) / cap_wh) if cap_wh > 0.0 else 0.0
                    charge_rate_frac_per_sec = 0.04  # 4% per second
                    if remaining_frac <= 0.0:
                        charge_time = 0.0
                    else:
                        charge_time = remaining_frac / charge_rate_frac_per_sec
                    energy = cap_wh  # After charging, assume full SOC
                    
                    # Register reservation
                    charge_start = arrive_time + wait_time
                    charge_end = charge_start + charge_time
                    reservations.append((charge_start, charge_end))
                    charger_reservations[charger_idx] = reservations
                    
                    # Update robot time and position
                    robot_time = charge_end
                    sim_robot.x = charger.x
                    sim_robot.y = charger.y
                    
                    # Create charge task
                    charge_task = TaskSpec(
                        task_id=-1,
                        task_type="CHARGE",
                        picks=[(charger.x, charger.y)],
                        drops=[(charger.x, charger.y)],
                        pick_wait_s=wait_time,  # Store wait time if needed
                            drop_wait_s=0.0,
                        charge_duration_s=charge_time,
                    )
                    augmented_seq.append(charge_task)
                    
                    # Re-estimate after charging (same logic as objective function)
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
                        required_energy = float('inf')
                        t_to_pick = 0.0
                    
                    # After charging, execute the task (energy should be sufficient after charging)
                    augmented_seq.append(t)
                    energy -= required_energy
                    robot_time += t_to_pick
                    
                    # Update robot position
                    if t.drops:
                        sim_robot.x, sim_robot.y = t.drops[-1]
                    else:
                        sim_robot.x = t.drop_x
                        sim_robot.y = t.drop_y
                else:
                    # Enough energy, execute task directly (or last task even if energy insufficient)
                    augmented_seq.append(t)
                    energy -= required_energy
                    robot_time += t_to_pick
                    
                    # Update robot position
                    if t.drops:
                        sim_robot.x, sim_robot.y = t.drops[-1]
                    else:
                        sim_robot.x = t.drop_x
                        sim_robot.y = t.drop_y
            
            result[rid] = augmented_seq
            robot_times[rid] = robot_time
        
        # Verify all tasks are in result before logging
        total_tasks_in_result = sum(len([t for t in seq if t.task_type != "CHARGE"]) for seq in result.values())
        if total_tasks_in_result != K:
            # Find which tasks are missing
            assigned_task_ids = set()
            for seq in result.values():
                for t in seq:
                    if t.task_type != "CHARGE":
                        assigned_task_ids.add(t.task_id)
            candidate_task_ids = {t.task_id for t in candidate_tasks}
            missing_task_ids = sorted(candidate_task_ids - assigned_task_ids)
            self.get_logger().warn(
                f"[assignment-{algorithm_name}] WARNING: Task count mismatch in result: expected {K}, got {total_tasks_in_result} "
                f"(missing {K - total_tasks_in_result} tasks: {missing_task_ids})"
            )
        
        # Log assignment summary with task type breakdown and estimated completion times
        assignment_summary = []
        task_type_counts: Dict[str, int] = {}
        total_charge_count = 0
        max_makespan = 0.0
        
        # Track task distribution across robots for load balancing analysis
        robot_task_counts: List[int] = []
        robots_with_tasks = 0
        
        for rid in robot_list:
            tasks = result[rid]
            non_charge_tasks = [t for t in tasks if t.task_type != "CHARGE"]
            charge_count = sum(1 for t in tasks if t.task_type == "CHARGE")
            total_charge_count += charge_count
            
            # Get estimated completion time for this robot
            estimated_time = robot_times.get(rid, 0.0)
            max_makespan = max(max_makespan, estimated_time)
            
            if non_charge_tasks:
                robots_with_tasks += 1
                robot_task_counts.append(len(non_charge_tasks))
                
                # Count task types for this robot
                robot_type_counts: Dict[str, int] = {}
                for t in non_charge_tasks:
                    task_type = t.task_type
                    robot_type_counts[task_type] = robot_type_counts.get(task_type, 0) + 1
                    task_type_counts[task_type] = task_type_counts.get(task_type, 0) + 1
                
                # Build type summary string
                type_summary_parts = [f"{task_type}={count}" for task_type, count in sorted(robot_type_counts.items())]
                type_summary = ", ".join(type_summary_parts)
                
                charge_str = f", {charge_count} charges" if charge_count > 0 else ""
                time_str = f", ETA={estimated_time:.1f}s" if estimated_time > 0.0 else ""
                assignment_summary.append(f"{rid}({len(non_charge_tasks)} tasks: {type_summary}{charge_str}{time_str})")
            else:
                robot_task_counts.append(0)
        
        if assignment_summary:
            self.get_logger().info(
                f"[assignment-{algorithm_name}] Task allocation result: {', '.join(assignment_summary)}"
            )
            
            # Log overall task type statistics and makespan
            if task_type_counts:
                total_tasks = sum(task_type_counts.values())
                overall_summary_parts = [f"{task_type}: {count}" for task_type, count in sorted(task_type_counts.items())]
                overall_summary = ", ".join(overall_summary_parts)
                charge_summary = f", {total_charge_count} CHARGE events" if total_charge_count > 0 else ""
                makespan_str = f", makespan={max_makespan:.1f}s" if max_makespan > 0.0 else ""
                
                # Log load balancing statistics
                if robot_task_counts:
                    max_tasks_per_robot = max(robot_task_counts)
                    min_tasks_per_robot = min([c for c in robot_task_counts if c > 0]) if any(c > 0 for c in robot_task_counts) else 0
                    avg_tasks_per_robot = total_tasks / robots_with_tasks if robots_with_tasks > 0 else 0
                    load_balance_info = f", load: {robots_with_tasks}/{num_robots} robots active (max={max_tasks_per_robot}, min={min_tasks_per_robot}, avg={avg_tasks_per_robot:.1f})"
                    
                    # Warn if only one robot is assigned tasks
                    if robots_with_tasks == 1:
                        self.get_logger().warn(
                            f"[assignment-{algorithm_name}] WARNING: All tasks assigned to only 1 robot (should be distributed across {num_robots} robots)"
                        )
                else:
                    load_balance_info = ""
                
                self.get_logger().info(
                    f"[assignment-{algorithm_name}] Overall: {total_tasks} tasks ({overall_summary}{charge_summary}{makespan_str}{load_balance_info})"
                )
        
        # Return result and max_makespan
        return result, max_makespan

    def _optimize_multi_robot_allocation(self, candidate_tasks: List[TaskSpec]) -> Dict[str, List[TaskSpec]]:
        """Run optimization using GA only and return its result.

        This variant disables SMA/PSO/CRO and always uses GA for stage-1
        multi-robot task allocation and sequencing.
        """
        if not candidate_tasks:
            return {rid: [] for rid in self.robot_ids}

        if not self.robot_ids:
            self.get_logger().warn("Cannot allocate tasks: no robots available")
            return {}

        algo_name = 'GA'
        try:
            self.get_logger().info(f"[multi-algo] Running {algo_name} only...")
            result, makespan = self._optimize_with_algorithm(candidate_tasks, algo_name)
            self.get_logger().info(
                f"[multi-algo] {algo_name} completed: makespan={makespan:.2f}s"
            )
            return result
        except Exception as e:
            self.get_logger().error(f"[multi-algo] {algo_name} failed: {e}")
            # Fall back to empty allocation on failure
            return {rid: [] for rid in self.robot_ids}

    def _offline_initialize_sequences(self) -> None:
        """Offline scheduling step: build one fixed optimized sequence per robot.

        First, optimally assign all pending tasks to robots based on energy/time cost.
        Then, for each robot, optimize the sequence of assigned tasks (stage 1).
        Finally, run a joint multi-robot charging optimization (stage 2) to
        insert CHARGE events.
        """

        if not self.pending_tasks:
            return

        # Track total optimization time
        total_optimization_t0 = time.time()
        self.optimization_time = 0.0  # Will be set after optimization completes
        
        # Set flag to prevent energy consumption during optimization
        # Reset all robots to initial 100% SOC for fair allocation
        self.optimization_in_progress = True
        for robot_id in self.robot_ids:
            state = self.robot_states.get(robot_id)
            if state is not None:
                cap_wh = self.robot_cap_wh.get(robot_id, 120.0)
                state.soc = cap_wh  # Reset to 100% (full capacity)
        
        self.get_logger().info(
            "[optimization] Starting precompute and allocation - robots reset to 100% SOC, "
            "energy consumption paused until allocation completes"
        )

        # Precompute Nav2 path cache: calculate only task-internal paths (not robot-to-task paths)
        # Path cache is coordinate-based (start_x, start_y, goal_x, goal_y), so it's shared across robots
        # We only compute paths within tasks (pick->drop, pick->pick, drop->drop) to avoid
        # including unnecessary robot initial position dependencies
        total_paths = 0
        precompute_t0 = time.time()
        self.get_logger().info("[precompute] Starting Nav2 path cache warming for task-internal paths...")
        
        if self.robot_states:
            # Use first available robot for Nav2 client (cache is coordinate-based, so robot doesn't matter)
            first_robot_id = list(self.robot_states.keys())[0]
            
            for task in self.pending_tasks:
                if task.task_type not in ("PICK_DROP", "MULTI_PICK_DROP", "DUAL_PICK_MULTI_DROP"):
                    continue

                picks = list(task.picks) if task.picks else []
                drops = list(task.drops) if task.drops else []
                
                # Compute paths within task: pick->drop, pick->pick, drop->drop
                # PICK_DROP: single pick -> drop
                if task.task_type == "PICK_DROP":
                    if picks and drops:
                        px, py = picks[0]
                        dx, dy = drops[0]
                        _ = self._nav2_distance(px, py, dx, dy, first_robot_id)
                        total_paths += 1
                
                # MULTI_PICK_DROP: all picks -> all drops
                elif task.task_type == "MULTI_PICK_DROP":
                    # Pick to pick paths
                    for i in range(len(picks) - 1):
                        px1, py1 = picks[i]
                        px2, py2 = picks[i + 1]
                        _ = self._nav2_distance(px1, py1, px2, py2, first_robot_id)
                        total_paths += 1
                    
                    # Pick to drop paths (first pick to first drop)
                    if picks and drops:
                        px, py = picks[0]
                        dx, dy = drops[0]
                        _ = self._nav2_distance(px, py, dx, dy, first_robot_id)
                        total_paths += 1
                    
                    # Drop to drop paths
                    for i in range(len(drops) - 1):
                        dx1, dy1 = drops[i]
                        dx2, dy2 = drops[i + 1]
                        _ = self._nav2_distance(dx1, dy1, dx2, dy2, first_robot_id)
                        total_paths += 1
                
                # DUAL_PICK_MULTI_DROP: interleaved pick->drop pairs
                elif task.task_type == "DUAL_PICK_MULTI_DROP":
                    # Pick to corresponding drop paths
                    n = min(len(picks), len(drops))
                    for i in range(n):
                        px, py = picks[i]
                        dx, dy = drops[i]
                        _ = self._nav2_distance(px, py, dx, dy, first_robot_id)
                        total_paths += 1
                    
                    # Pick to next pick paths
                    for i in range(len(picks) - 1):
                        px1, py1 = picks[i]
                        px2, py2 = picks[i + 1]
                        _ = self._nav2_distance(px1, py1, px2, py2, first_robot_id)
                        total_paths += 1
        
        precompute_dt = time.time() - precompute_t0
        if total_paths > 0:
            avg_time_per_path = precompute_dt / total_paths if total_paths > 0 else 0.0
            self.get_logger().info(
                f"[precompute] Nav2 path cache warming: {total_paths} task-internal paths computed in {precompute_dt:.3f}s"
            )

        # Stage 1: Multi-robot task allocation and sequencing using mealpy
        # This optimizes both which robot gets which task AND the order of tasks for each robot
        candidate_tasks = [
            t for t in self.pending_tasks 
            if t.task_type in ("PICK_DROP", "MULTI_PICK_DROP", "DUAL_PICK_MULTI_DROP")
                ]
        
        self.get_logger().info(
            f"[optimization] Starting allocation: {len(self.pending_tasks)} pending tasks, "
            f"{len(candidate_tasks)} candidate tasks (types: PICK_DROP, MULTI_PICK_DROP, DUAL_PICK_MULTI_DROP)"
        )
        
        if not candidate_tasks:
            base_sequences: Dict[str, List[TaskSpec]] = {rid: [] for rid in self.robot_ids}
            dt = 0.0  # No optimization needed
        else:
            t0 = time.time()
            base_sequences = self._optimize_multi_robot_allocation(candidate_tasks)
            dt = time.time() - t0
            total_assigned = sum(len([t for t in seq if t.task_type != "CHARGE"]) for seq in base_sequences.values())
            self.get_logger().info(
                f"[stage1] Multi-robot allocation and sequencing finished in {dt:.3f}s "
                f"({total_assigned} tasks assigned to {len([r for r in base_sequences.values() if r])} robots)"
            )

        # One-shot: initial energy estimation for all tasks using Nav2 paths
        total_tasks = 0
        energy_t0 = time.time()
        for rid, seq in base_sequences.items():
            state = self.robot_states.get(rid)
            if state is None:
                continue
            for spec in seq:
                if spec.task_type in ("PICK_DROP", "MULTI_PICK_DROP", "DUAL_PICK_MULTI_DROP"):
                    _ = self.estimate_task_energy(state, spec)
                    total_tasks += 1
        energy_dt = time.time() - energy_t0
        self.get_logger().info(
            f"[energy-estimate] initial energy precompute for all robots ({total_tasks} tasks) took {energy_dt:.3f}s"
        )

        # Stage 2: CHARGE events are already inserted in Stage 1 during decoding
        # So we use base_sequences directly (they already contain CHARGE events)
        augmented_sequences = base_sequences

        for rid in self.robot_ids:
            final_seq = augmented_sequences.get(rid, [])
            self.current_sequences[rid] = final_seq
            self.current_seq_index[rid] = 0
            if not final_seq:
                continue
            seq_ids = [t.task_id for t in final_seq]
            non_charge_tasks = [t for t in final_seq if t.task_type != "CHARGE"]
            charge_count = sum(1 for t in final_seq if t.task_type == "CHARGE")
            self.get_logger().info(
                f"[offline] initialized sequence for {rid}: "
                f"tasks={seq_ids} (total_len={len(seq_ids)}, tasks={len(non_charge_tasks)}, charges={charge_count})"
            )
        
        # Optimization complete - allow energy consumption to resume
        self.optimization_in_progress = False
        
        # Calculate and log total optimization time
        total_optimization_dt = time.time() - total_optimization_t0
        self.get_logger().info(
            f"[optimization] Total optimization time: {total_optimization_dt:.3f}s "
            f"(precompute: {precompute_dt:.3f}s, stage1: {dt:.3f}s, energy_estimate: {energy_dt:.3f}s)"
        )
        
        self.get_logger().info(
            "[optimization] Allocation complete - energy consumption resumed, "
            "robots will now consume energy during task execution"
            )
        
        # Publish optimization time for metrics collection
        opt_time_msg = Float32()
        opt_time_msg.data = float(total_optimization_dt)
        self.optimization_time_pub.publish(opt_time_msg)

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

        self.robot_states[robot_id] = RobotState(
            robot_id=robot_id,
            namespace=f'/{robot_id}',
            x=0.0,
            y=0.0,
            soc=init_soc,
            available=True,
            carried_weight=0.0,
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
                            # Get weight from pick_weights or use default
                            if hasattr(t, 'pick_weights') and t.pick_weights and pick_idx < len(t.pick_weights):
                                weight = float(t.pick_weights[pick_idx]) if t.pick_weights[pick_idx] is not None else 0.0
                            else:
                                # Use default weight if not specified
                                weight = float(getattr(self, 'default_item_weight_kg', 1.0))
                            
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
                                    if hasattr(t, 'pick_weights') and t.pick_weights and pick_idx < len(t.pick_weights):
                                        w = float(t.pick_weights[pick_idx]) if t.pick_weights[pick_idx] is not None else 0.0
                                        dropped_weight += w
                            else:
                                # If no deliveries spec, assume FIFO: drop first picked item based on actual pick_weights
                                if hasattr(t, 'pick_weights') and t.pick_weights and len(t.pick_weights) > 0:
                                    # Use the first pick weight (FIFO)
                                    dropped_weight = float(t.pick_weights[0]) if t.pick_weights[0] is not None else 0.0
                                else:
                                    # Fallback to default weight
                                    dropped_weight = float(getattr(self, 'default_item_weight_kg', 1.0))
                            
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
                                if hasattr(t, 'pick_weights') and t.pick_weights and pick_idx < len(t.pick_weights):
                                    weight = float(t.pick_weights[pick_idx]) if t.pick_weights[pick_idx] is not None else 0.0
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
                                        if hasattr(t, 'pick_weights') and t.pick_weights and pick_idx < len(t.pick_weights):
                                            w = float(t.pick_weights[pick_idx]) if t.pick_weights[pick_idx] is not None else 0.0
                                            dropped_weight += w
                                else:
                                    # Default: assume dropping first remaining item
                                    dropped_weight = float(getattr(self, 'default_item_weight_kg', 1.0))
                                
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

                # Apply charging effect based on elapsed time.
                # Rule: 0.5초당 2% → 초당 4% (0.04) 충전.
                elapsed = max(0.0, current_time - task_start_time)
                cap_wh = self.robot_cap_wh.get(robot_id, 120.0)
                charge_rate_frac_per_sec = 0.04  # 4% per second
                # Fraction of capacity that could be charged during this duration
                gained_frac = charge_rate_frac_per_sec * elapsed
                # Max possible gain until full from start_soc
                remaining_frac_from_start = 0.0
                if cap_wh > 0.0:
                    remaining_frac_from_start = max(0.0, (cap_wh - start_soc) / cap_wh)
                applied_frac = min(gained_frac, remaining_frac_from_start)
                total_gain = cap_wh * applied_frac

                before = state.soc
                # Set SOC based on start_soc plus gained energy, capped at capacity
                state.soc = min(cap_wh, start_soc + total_gain)
                end_soc = state.soc

                rid_str = self._color_robot(robot_id)
                self.get_logger().info(
                    f"[charge] {rid_str} finished CHARGE, +{total_gain:.2f}Wh over {elapsed:.1f}s "
                    f"({before:.2f}Wh -> {state.soc:.2f}Wh)"
                )

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
                        rid_str = self._color_robot(robot_id)
                        self.get_logger().info(
                            f"[charge] {rid_str} freed charger {cid}, queue={q}"
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
                            charge_task = TaskSpec(
                                task_id=-1,
                                task_type="CHARGE",
                                picks=[(state.x, state.y)],
                                drops=[(ch.x, ch.y)],
                                pick_wait_s=0.0,
                                drop_wait_s=0.0,
                                charge_duration_s=1.0,  # Instant charge (matches actual executor)
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
        # charge for a duration that depends on remaining SOC.
        # Rule: 0.5초당 2% → 초당 4% (0.04) 충전.
        remaining_frac = 0.0
        if cap_wh > 0.0:
            remaining_frac = max(0.0, (cap_wh - state.soc) / cap_wh)
        charge_rate_frac_per_sec = 0.04  # 4% per second
        if remaining_frac <= 0.0:
            charge_duration_s = 0.0
        else:
            charge_duration_s = remaining_frac / charge_rate_frac_per_sec

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
