

1. build docker image
from repo root, run:
    docker build -t my-ros2-humble-rpi -f docker/ros2_humble/Dockerfile .

2. run docker container
docker run -it --rm \
  --net=host \
  --ipc=host \
  --privileged \
  -v ~/projects/capstone_robot:/root/capstone_robot \
  my-ros2-humble-rpi