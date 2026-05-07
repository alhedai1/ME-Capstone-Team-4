

# COPY requirements.txt .
# RUN pip3 install --no-cache-dir -r requirements.txt
# COPY ..


FROM ros:humble-ros-base

SHELL ["/bin/bash", "-c"]

RUN apt update && apt install -y \
    git \
    nano \
    python3-pip \
    python3-colcon-common-extensions \
    python3-rosdep \
    v4l-utils \
    ros-humble-cv-bridge \
    ros-humble-image-transport \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt

RUN echo "source /opt/ros/humble/setup.bash" >> /root/.bashrc

WORKDIR /root/capstone_robot/ros2_ws