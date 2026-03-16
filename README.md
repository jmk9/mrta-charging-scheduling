# Joint-MRTA-Charging: Integrated Task Allocation and Charging Scheduling

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
![Architecture Diagram](./images/architecture.png) 
*(Note: [IMAGE PLACEHOLDER] - Upload your architecture diagram to /images/architecture.png)*

## 💻 Environment & Installation
- **OS:** Ubuntu 24.04
- **Middleware:** ROS 2 Humble
- **Docker Image:** `jmk9/mrta-charging-scheduler`

You can run the environment directly using Docker:
```bash
docker pull jmk9/mrta-charging-scheduler
docker run -it --rm --net=host --env="DISPLAY" --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" jmk9/mrta-charging-scheduler
