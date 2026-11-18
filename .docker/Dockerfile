FROM ros:humble

ENV COLCON_WS=/root/denso_ws

RUN apt-get update && apt-get install -y --no-install-recommends \
      ros-$ROS_DISTRO-moveit \
      ros-$ROS_DISTRO-moveit-servo \
      ros-$ROS_DISTRO-ros2-control \
      ros-$ROS_DISTRO-ros2-controllers \
      ros-$ROS_DISTRO-realsense2-camera \
      build-essential \
      python3-pip \
      git \
      wget \
      nano \
      terminator && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p $COLCON_WS/src
WORKDIR $COLCON_WS

COPY . src/denso_robot_ros2/

RUN apt-get update && rosdep install --ignore-src --from-paths src -y -r && \
    rm -rf /var/lib/apt/lists/*

RUN rm -rf src/denso_robot_ros2/

# ---- Python: torch CPU + XFeat + SAM2 (sem build, via PYTHONPATH) ----
RUN pip3 install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cpu && \
    pip3 install "setuptools==58.2.0" && \
    pip3 install "numpy==1.26.4" "hydra-core==1.3.8" "iopath==0.1.10" "omegaconf==2.3.1" "tqdm"

RUN git clone https://github.com/verlab/accelerated_features.git /root/accelerated_features && \
    git clone https://github.com/facebookresearch/sam2.git /root/sam2 && \
    mkdir -p /root/sam2/checkpoints && \
    wget -q -P /root/sam2/checkpoints https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt

ENV PYTHONPATH=/root/sam2:/root/accelerated_features

# Carrega o ROS e o workspace automaticamente em todo shell
RUN echo "source /opt/ros/$ROS_DISTRO/setup.bash" >> /root/.bashrc && \
    echo "[ -f $COLCON_WS/install/setup.bash ] && source $COLCON_WS/install/setup.bash" >> /root/.bashrc

CMD ["bash"]