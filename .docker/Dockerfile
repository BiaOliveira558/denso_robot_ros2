FROM ros:humble

ENV COLCON_WS=/root/denso_ws

RUN apt-get update && apt-get install -y git terminator

RUN apt-get update && apt-get install -y --no-install-recommends \
      ros-$ROS_DISTRO-moveit \
      ros-$ROS_DISTRO-moveit-servo \
      ros-$ROS_DISTRO-ros2-control \
      ros-$ROS_DISTRO-ros2-controllers \
      ros-$ROS_DISTRO-realsense2-camera \
      build-essential \
      git \
      nano \
      terminator && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p $COLCON_WS/src
WORKDIR $COLCON_WS

COPY . src/denso_robot_ros2/

RUN apt-get update && rosdep install --ignore-src --from-paths src -y -r

RUN rm -rf src/denso_robot_ros2/

# Carrega o ROS e o workspace automaticamente em todo shell
RUN echo "source /opt/ros/$ROS_DISTRO/setup.bash" >> /root/.bashrc && \
    echo "[ -f $COLCON_WS/install/setup.bash ] && source $COLCON_WS/install/setup.bash" >> /root/.bashrc

CMD ["bash"]