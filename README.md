# An integrated framework for joint optimization of task allocation and charging scheduling in multi-robot systems

[![ROS 2](https://img.shields.io/badge/ROS-2_Humble-blue)](https://docs.ros.org/en/humble/index.html)
[![Gazebo](https://img.shields.io/badge/Simulation-Gazebo-orange)](https://gazebosim.org/)
[![Docker](https://img.shields.io/badge/Docker-jmk9%2Fmrta--charging--scheduler-blue?logo=docker)](https://hub.docker.com/r/jmk9/mrta-charging-scheduler)

This repository contains the implementation of an integrated framework for **Multi-Robot Task Allocation (MRTA)** and **Charging Scheduling**. By jointly optimizing task sequences and charging maneuvers, this framework minimizes makespan and prevents dock contention in shared infrastructure environments.

## 📺 Demo Video
Check out the simulation demo on YouTube:
<div align="center">
  <a href="https://youtu.be/TR0gkmdvi8w">
    <img src="https://img.youtube.com/vi/TR0gkmdvi8w/0.jpg" alt="MRTA Simulation Demo" width="600">
  </a>
</div>

## 📌 Key Features
- ■ **Jointly optimizes** task assignment, sequencing, and charging under shared docks.
- ■ **Models dock occupancy** and charge-wait to reduce contention-driven delays.
- ■ **Validated in ROS 2 and Gazebo** with execution-level navigation and charging effects.
- ■ **Achieves 7.6% makespan reduction** and 0.8% lower fleet-wide energy consumption compared to heuristic baselines.

## 🏗 System Architecture
![Architecture Diagram](./figures/main.png)

## 💻 Environment & Installation
- **OS:** Ubuntu 24.04
- **Middleware:** ROS 2 Humble
- **Docker Image:** `jmk9/mrta-charging-scheduler`

You can run the environment directly using Docker:
```bash
docker pull jmk9/mrta-charging-scheduler
docker run -it jmk9/mrta-charging-scheduler
```

## 🚀 Usage

### 1. Launch Gazebo and Navigation Stack

Execute the multi-robot world and navigation system:

```bash
ros2 launch turtlebot3_multi_robot gazebo_multi_nav2_world_waypoint.launch.py \
  enable_drive:=True \
  headless:=false \
  num_robots:=4
```

### 2. Run Scheduler Node

Run the optimization-based scheduler using the RIME algorithm:

```bash
ros2 run scheduler scheduler_node_1stage_RIME \
  --ros-args --log-level scheduler_node:=debug \
  -p charge:=optimized \
  --params-file /root/ros2_ws/install/scheduler/share/scheduler/config/scheduler_params.yaml
```

## 📊 Performance Analysis

### Comparative Analysis (Table 4)

The following table summarizes the performance of the proposed Joint Optimization approach compared to standard heuristic baselines, based on the experimental results.

| Method | # of Chg. | Chg. Time (s) | Wait Time (s) | Total Travel (m) | Energy (kJ) | Makespan (s) |
|--------|----------:|--------------:|--------------:|-----------------:|------------:|-------------:|
| Threshold-based | 1 | 1779.6 | 472.9 | 497.1 | 354.3 | 1026.3 |
| Feasibility-based | 7 | 1121.2 | 211.8 | 483.7 | 349.5 | 995.8 |
| **Optimized (Ours)** | **7** | **1146.9** | **0.0** | **488.9** | **343.5** | **948.3** |


### Table 7

Large-scale scalability results for Cases 1–8 (50 tasks). For each case, three policies are evaluated and the table reports the mean.  
*M* denotes the maximum makespan among robots (s), *E* the total energy consumption (Wh), `Σ_T` the sum of makespan over all robots (s), *W* the average waiting time (s), and *D* the number of battery depletion events (count).

| Spec. | Case 1 | Case 2 | Case 3 | Case 4 | Case 5 | Case 6 | Case 7 | Case 8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Task | 50 | 50 | 50 | 50 | 50 | 50 | 50 | 50 |
| Robot | 4 | 4 | 5 | 5 | 6 | 6 | 7 | 7 |
| Charger | 2 | 3 | 2 | 3 | 2 | 3 | 2 | 3 |

| Policy | Metric | Case 1 | Case 2 | Case 3 | Case 4 | Case 5 | Case 6 | Case 7 | Case 8 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Threshold | M (s) | 1991.31 | 1677.65 | 1750.41 | 1627.24 | 1403.76 | 1304.70 | 1218.10 | 1104.70 |
| Threshold | E (Wh) | 353.58 | 347.33 | 357.25 | 351.28 | 353.60 | 356.02 | 354.25 | 348.02 |
| Threshold | Σ_T (s) | 6265.12 | 6088.41 | 6257.63 | 6150.42 | 6333.16 | 5742.74 | 5262.23 | 5042.50 |
| Threshold | W (s) | 1188.78 | 149.29 | 1273.96 | 343.72 | 1221.16 | 325.10 | 427.49 | 0.00 |
| Threshold | D (count) | 0 | 2 | 1 | 1 | 2 | 0 | 0 | 0 |
| Feasibility | M (s) | 1963.23 | 1671.98 | 1798.16 | 1600.96 | 1436.68 | 1337.62 | 1242.85 | 1137.62 |
| Feasibility | E (Wh) | 352.24 | 346.78 | 354.77 | 355.50 | 352.42 | 356.83 | 355.42 | 348.83 |
| Feasibility | Σ_T (s) | 6267.21 | 6160.99 | 6382.77 | 6044.47 | 6196.05 | 5503.28 | 5423.70 | 5067.81 |
| Feasibility | W (s) | 899.43 | 140.07 | 1337.48 | 537.49 | 1066.29 | 300.66 | 539.79 | 0.00 |
| Feasibility | D (count) | 2 | 2 | 2 | 1 | 4 | 3 | 0 | 1 |
| Optimized | M (s) | 1620.36 | 1606.21 | 1511.25 | 1468.56 | 1356.97 | 1257.81 | 1134.61 | 1127.81 |
| Optimized | E (Wh) | 350.25 | 345.69 | 349.82 | 350.03 | 352.34 | 355.06 | 350.33 | 346.00 |
| Optimized | Σ_T (s) | 5971.38 | 5901.48 | 5993.47 | 5774.66 | 5732.28 | 5304.40 | 4890.98 | 4804.14 |
| Optimized | W (s) | 194.44 | 0.00 | 45.77 | 21.26 | 214.80 | 0.00 | 61.47 | 0.00 |
| Optimized | D (count) | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

### Schedule Timeline Comparison

The figure below illustrates the timeline comparison between Threshold-based, Feasibility-based, and our Optimized Joint Scheduling. Our approach proactively manages charging intervals to ensure zero wait time at the docks, leading to the fastest mission completion.

![Schedule Timeline Comparison](/figures/timeline_policy.png)


