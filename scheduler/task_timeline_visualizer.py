import rclpy
from rclpy.node import Node
from multi_robot_msgs.msg import TaskExecution
from std_msgs.msg import Float32
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import random
import json
import csv
import os
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import numpy as np

class TaskTimelineVisualizer(Node):

    def __init__(self):
        super().__init__("task_timeline_visualizer")

        self.sub = self.create_subscription(
            TaskExecution,
            "/task_execution_log",
            self.cb,
            10
        )
        
        # Subscribe to optimization time
        self.optimization_time_sub = self.create_subscription(
            Float32,
            "/optimization_time",
            self._optimization_time_cb,
            10
        )

        self.records = []
        # Robot order will be dynamically determined from received messages
        # Supports up to 10 robots (tb1~tb10)
        self.robot_order = []
        self.task_colors = {}
        
        # Metrics tracking
        self.robot_finish_times: Dict[str, float] = {}
        self.robot_start_times: Dict[str, float] = {}
        self.robot_task_counts: Dict[str, int] = defaultdict(int)
        self.robot_energy_consumption: Dict[str, float] = defaultdict(float)
        self.robot_wait_times: Dict[str, float] = defaultdict(float)
        self.robot_critical_soc_times: Dict[str, float] = defaultdict(float)
        self.robot_path_lengths: Dict[str, float] = defaultdict(float)
        self.robot_soc_history: Dict[str, List[Tuple[float, float]]] = defaultdict(list)  # (time, soc)
        self.robot_capacity: Dict[str, float] = defaultdict(lambda: 120.0)  # Default capacity
        
        # Simulation timing
        self.simulation_start_time: Optional[float] = None
        self.optimization_start_time: Optional[float] = None
        self.optimization_time: float = 0.0  # Received from scheduler_node via /optimization_time topic
        
        # Output directory
        self.output_dir = self.declare_parameter("output_dir", "metrics_output").get_parameter_value().string_value
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Save metrics on shutdown flag
        self.save_on_shutdown = True

        # 🔑 matplotlib 초기화 (한 번만)
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(14, 6))
        
        # SOC plot figure (separate window)
        self.soc_fig, self.soc_ax = plt.subplots(figsize=(12, 6))

        # 🔑 1초마다 갱신
        self.timer = self.create_timer(1.0, self.draw)
        
        self.get_logger().info(f"Task timeline visualizer initialized. Output directory: {self.output_dir}")

    def _optimization_time_cb(self, msg: Float32):
        """Callback for optimization time messages."""
        self.optimization_time = float(msg.data)

    def cb(self, msg):
        record = {
            "robot_id": msg.robot_id,
            "task_id": msg.task_id,
            "task_type": msg.task_type,
            "waypoint_index": msg.waypoint_index,
            "start_time": msg.start_time,
            "end_time": msg.end_time,
        }
        
        # Add SOC if available (backward compatible)
        if hasattr(msg, 'start_soc') and hasattr(msg, 'end_soc'):
            record["start_soc"] = msg.start_soc
            record["end_soc"] = msg.end_soc
            
            # Track SOC history
            self.robot_soc_history[msg.robot_id].append((msg.start_time, msg.start_soc))
            if msg.end_time > msg.start_time:
                self.robot_soc_history[msg.robot_id].append((msg.end_time, msg.end_soc))
        
        self.records.append(record)
        
        # Dynamically update robot_order when new robots are encountered
        if msg.robot_id not in self.robot_order:
            self.robot_order.append(msg.robot_id)
            # Sort robot IDs (tb1, tb2, ..., tb10)
            self.robot_order.sort(key=lambda x: (len(x), x))
        
        # Track robot start/finish times
        if msg.robot_id not in self.robot_start_times or msg.start_time < self.robot_start_times[msg.robot_id]:
            self.robot_start_times[msg.robot_id] = msg.start_time
        
        if msg.robot_id not in self.robot_finish_times or msg.end_time > self.robot_finish_times[msg.robot_id]:
            self.robot_finish_times[msg.robot_id] = msg.end_time
        
        # Count tasks (excluding CHARGE) - count each task_id only once
        # Track which (robot_id, task_id) combinations we've already counted
        if not hasattr(self, '_counted_tasks'):
            self._counted_tasks = set()
        
        if msg.task_type != "CHARGE":
            task_key = (msg.robot_id, msg.task_id)
            if task_key not in self._counted_tasks:
                self._counted_tasks.add(task_key)
                self.robot_task_counts[msg.robot_id] += 1
        
        # Track waiting time (CHARGE task duration)
        if msg.task_type == "CHARGE":
            wait_duration = msg.end_time - msg.start_time
            self.robot_wait_times[msg.robot_id] += wait_duration

    def draw(self):
        try:
            # Draw task timeline
            self.ax.clear()

            yticks = []
            ylabels = []

            for i, robot in enumerate(self.robot_order):
                yticks.append(i)
                ylabels.append(robot)

                # Group records by (robot_id, task_id) and sort by waypoint_index
                robot_records = [r for r in self.records if r["robot_id"] == robot]
                
                # Group by task_id and combine waypoints into single task bars
                task_groups = {}
                for r in robot_records:
                    task_id = r["task_id"]
                    if task_id not in task_groups:
                        task_groups[task_id] = []
                    task_groups[task_id].append(r)
                
                # Draw each task as a single bar (combine all waypoints)
                for task_id, segments in task_groups.items():
                    if not segments:
                        continue
                    
                    # Find task start (earliest waypoint start) and end (latest waypoint end)
                    valid_segments = [s for s in segments if s["start_time"] > 0 and s["end_time"] > s["start_time"]]
                    if not valid_segments:
                        continue
                    
                    task_start = min(s["start_time"] for s in valid_segments)
                    task_end = max(s["end_time"] for s in valid_segments)
                    task_type = valid_segments[0]["task_type"]
                    
                    try:
                        if task_type == "CHARGE":
                            color = "black"
                            hatch = "//"
                            alpha = 1.0
                        else:
                            if task_id not in self.task_colors:
                                self.task_colors[task_id] = (
                                    random.random(),
                                    random.random(),
                                    random.random()
                                )
                            color = self.task_colors[task_id]
                            hatch = None
                            alpha = 0.8
                        
                        # Draw single bar for the entire task
                        self.ax.barh(
                            i,
                            task_end - task_start,
                            left=task_start,
                            height=0.8,
                            color=color,
                            edgecolor="white",
                            hatch=hatch,
                            alpha=alpha,
                            linewidth=0.5
                        )
                    except Exception as ex:
                        # Skip this task if there's an error drawing it
                        self.get_logger().debug(f"Error drawing task {task_id}: {ex}")
                        continue

            self.ax.set_yticks(yticks)
            self.ax.set_yticklabels(ylabels)
            self.ax.set_xlabel("Time (sec)")
            self.ax.set_title("Multi-Robot Task Timeline (Tasks)")
            self.ax.grid(True, alpha=0.3, axis='x')

            charge_patch = mpatches.Patch(
                facecolor="black", hatch="//", label="CHARGE"
            )
            self.ax.legend(handles=[charge_patch])

            plt.tight_layout()
            plt.pause(0.001)   # 🔑 GUI 이벤트 처리
            
            # Draw SOC plot
            self.draw_soc_plot()
            
        except Exception as ex:
            self.get_logger().warn(f"Error in draw(): {ex}")
            return
    
    def draw_soc_plot(self):
        """Draw SOC history for each robot."""
        try:
            self.soc_ax.clear()
            
            for robot_id in self.robot_order:
                soc_history = self.robot_soc_history.get(robot_id, [])
                if not soc_history:
                    continue
                
                # Sort by time
                soc_history = sorted(soc_history, key=lambda x: x[0])
                times = [t for t, _ in soc_history]
                socs = [s for _, s in soc_history]
                
                self.soc_ax.plot(times, socs, label=robot_id, linewidth=2, marker='o', markersize=3)
            
            self.soc_ax.set_xlabel("Time (sec)")
            self.soc_ax.set_ylabel("SOC (Wh)")
            self.soc_ax.set_title("Robot SOC History")
            self.soc_ax.legend()
            self.soc_ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.pause(0.001)
        except Exception as ex:
            self.get_logger().debug(f"Error in draw_soc_plot(): {ex}")
    
    def calculate_metrics(self) -> Dict:
        """Calculate all metrics from collected data."""
        if not self.records:
            return {}
        
        # Find simulation start time (earliest robot start time, not task start time)
        # This should be when tasks are assigned, not when they actually start
        robot_start_time_values = [t for t in self.robot_start_times.values() if t > 0]
        simulation_start = min(robot_start_time_values) if robot_start_time_values else 0.0
        
        # Calculate makespan: time from earliest robot start to latest robot finish
        # This is the true makespan from task assignment to completion
        robot_finish_time_values = [t for t in self.robot_finish_times.values() if t > 0]
        if robot_start_time_values and robot_finish_time_values:
            earliest_start = min(robot_start_time_values)
            latest_finish = max(robot_finish_time_values)
            makespan = latest_finish - earliest_start
        else:
            makespan = 0.0
        
        # Per-robot metrics
        robot_metrics = {}
        for robot_id in self.robot_order:
            robot_records = [r for r in self.records if r["robot_id"] == robot_id]
            if not robot_records:
                continue
            
            # Robot completion time
            robot_finish = self.robot_finish_times.get(robot_id, 0.0)
            robot_start = self.robot_start_times.get(robot_id, 0.0)
            robot_completion_time = robot_finish - robot_start if robot_finish > robot_start else 0.0
            
            # Task count (excluding CHARGE)
            task_count = self.robot_task_counts.get(robot_id, 0)
            
            # Energy consumption (from SOC changes if available)
            energy_consumption = 0.0
            soc_history = self.robot_soc_history.get(robot_id, [])
            if soc_history:
                # Calculate total energy consumed (initial SOC - final SOC + recharges)
                sorted_soc = sorted(soc_history, key=lambda x: x[0])
                if sorted_soc:
                    initial_soc = sorted_soc[0][1]
                    final_soc = sorted_soc[-1][1]
                    
                    # Sum up all SOC increases (charging)
                    charge_gains = 0.0
                    for i in range(len(sorted_soc) - 1):
                        if sorted_soc[i+1][1] > sorted_soc[i][1]:
                            charge_gains += (sorted_soc[i+1][1] - sorted_soc[i][1])
                    
                    # Energy consumed = initial - final + gains from charging
                    capacity = self.robot_capacity[robot_id]
                    initial_normalized = initial_soc
                    final_normalized = final_soc
                    energy_consumption = initial_normalized - final_normalized + charge_gains
            
            # Waiting time
            wait_time = self.robot_wait_times.get(robot_id, 0.0)
            
            # Critical SOC time (SOC < 10% of capacity)
            critical_soc_time = 0.0
            capacity = self.robot_capacity[robot_id]
            critical_threshold = capacity * 0.1
            
            if soc_history:
                sorted_soc = sorted(soc_history, key=lambda x: x[0])
                for i in range(len(sorted_soc) - 1):
                    t1, soc1 = sorted_soc[i]
                    t2, soc2 = sorted_soc[i + 1]
                    if soc1 < critical_threshold or soc2 < critical_threshold:
                        critical_soc_time += (t2 - t1)
            
            # Path length (would need distance data - placeholder for now)
            path_length = self.robot_path_lengths.get(robot_id, 0.0)
            
            robot_metrics[robot_id] = {
                "completion_time": robot_completion_time,
                "task_count": task_count,
                "energy_consumption": energy_consumption,
                "wait_time": wait_time,
                "critical_soc_time": critical_soc_time,
                "path_length": path_length,
            }
        
        # Overall metrics
        total_energy = sum(m["energy_consumption"] for m in robot_metrics.values())
        total_wait_time = sum(m["wait_time"] for m in robot_metrics.values())
        total_tasks = sum(m["task_count"] for m in robot_metrics.values())
        total_path_length = sum(m["path_length"] for m in robot_metrics.values())
        
        # Energy failures (SOC reaches 0 or very low)
        energy_failures = 0
        for robot_id in self.robot_order:
            soc_history = self.robot_soc_history.get(robot_id, [])
            for _, soc in soc_history:
                if soc <= 0.01:  # Very low SOC (essentially 0)
                    energy_failures += 1
                    break  # Count once per robot
        
        # Use optimization_time received from scheduler_node via /optimization_time topic
        optimization_time = self.optimization_time
        
        metrics = {
            "makespan": makespan,
            "simulation_time": simulation_start,
            "optimization_time": optimization_time,
            "total_energy_consumption": total_energy,
            "total_wait_time": total_wait_time,
            "total_tasks": total_tasks,
            "total_path_length": total_path_length,
            "energy_failures": energy_failures,
            "robot_metrics": robot_metrics,
        }
        
        return metrics
    
    def save_metrics_json(self, filename: str = None):
        """Save metrics to JSON file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"metrics_{timestamp}.json"
        
        filepath = os.path.join(self.output_dir, filename)
        metrics = self.calculate_metrics()
        
        with open(filepath, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        self.get_logger().info(f"Metrics saved to {filepath}")
        return filepath
    
    def save_timeline_csv(self, filename: str = None):
        """Save task timeline to CSV file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"timeline_{timestamp}.csv"
        
        filepath = os.path.join(self.output_dir, filename)
        
        fieldnames = [
            "robot_id", "task_id", "task_type", "waypoint_index",
            "start_time", "end_time", "duration"
        ]
        
        # Add SOC fields if available
        if self.records and "start_soc" in self.records[0]:
            fieldnames.extend(["start_soc", "end_soc", "soc_change"])
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for record in self.records:
                row = record.copy()
                row["duration"] = row["end_time"] - row["start_time"]
                
                if "start_soc" in row and "end_soc" in row:
                    row["soc_change"] = row["end_soc"] - row["start_soc"]
                elif "start_soc" not in row:
                    row["start_soc"] = ""
                    row["end_soc"] = ""
                    row["soc_change"] = ""
                
                writer.writerow(row)
        
        self.get_logger().info(f"Timeline saved to {filepath}")
        return filepath
    
    def save_plots(self):
        """Save timeline and SOC plots as images."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save timeline plot
        timeline_path = os.path.join(self.output_dir, f"timeline_{timestamp}.png")
        self.fig.savefig(timeline_path, dpi=150, bbox_inches='tight')
        self.get_logger().info(f"Timeline plot saved to {timeline_path}")
        
        # Save SOC plot
        soc_path = os.path.join(self.output_dir, f"soc_history_{timestamp}.png")
        self.soc_fig.savefig(soc_path, dpi=150, bbox_inches='tight')
        self.get_logger().info(f"SOC plot saved to {soc_path}")
        
        return timeline_path, soc_path

    def destroy(self):
        if self.save_on_shutdown and self.records:
            self.get_logger().info("Saving metrics and plots on shutdown...")
            self.save_metrics_json()
            self.save_timeline_csv()
            self.save_plots()
        
        plt.ioff()
        plt.close(self.fig)
        plt.close(self.soc_fig)


def main():
    rclpy.init()
    node = TaskTimelineVisualizer()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
