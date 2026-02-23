# Pull
docker pull <dockerhub_user>/<image_name>:<tag>

# Run container in background (keep it alive)
xhost +local:docker
docker run -d --name mrta \
  --net=host \
  --ipc=host \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  jmk9/mrta-charging-scheduler:humble \

# Start simulation (Gazebo + Nav2) inside container
  "ros2 launch turtlebot3_multi_robot gazebo_multi_nav2_world_waypoint.launch.py enable_drive:=True headless:=false num_robots:=4"

# In another terminal, run the scheduler (same container)
# (Copy-paste this line into a second terminal)
  "ros2 run scheduler scheduler_node_1stage_RIME --ros-args --log-level scheduler_node:=debug -p charge:=optimized --params-file /root/ros2_ws/install/scheduler/share/scheduler/config/scheduler_params.yaml"
