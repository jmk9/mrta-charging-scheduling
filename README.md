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

### Table 6. Compact Summary

| Policy | Total Travel (s) | Total Charge (s) | Total Wait (s) | Total Makespan (s) |
|---|---:|---:|---:|---:|
| Threshold | 844.68 | 2434.00 | 947.33 | 6604.22 |
| Feasibility | 896.25 | 2607.26 | 1619.77 | 7500.50 |
| **Optimized (RIME)** | **855.88** | **1575.03** | **57.20** | **4903.10** |

### Schedule Timeline Comparison

The figure below illustrates the timeline comparison between Threshold-based, Feasibility-based, and our Optimized Joint Scheduling. Our approach proactively manages charging intervals to ensure zero wait time at the docks, leading to the fastest mission completion.

![Schedule Timeline Comparison](/figures/timeline_policy.png)


