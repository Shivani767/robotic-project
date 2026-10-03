# The live Gazebo demo image.
#
# Deliberately a separate image from docker/ros2.Dockerfile, and deliberately
# self-contained. The demo needs Gazebo, the gz Python bindings and a web
# server, but none of MuJoCo, OSMesa or colcon: it runs one simulation, not the
# regression suite. Starting from ros:jazzy-ros-base keeps the demo image small
# and keeps `docker compose up` a one-step build on a fresh Mac, with no
# pre-built base image to remember.
#
# Why Jazzy and not Harmonic's own image: Harmonic's image is Ubuntu 22.04 /
# CPython 3.10, which is below this project's requires-python, and
# `ros-jazzy-ros-gz` installs Gazebo Harmonic (gz-sim8) as the default anyway.
FROM ros:jazzy-ros-base

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3-pip \
        ros-jazzy-ros-gz \
        ros-jazzy-ros-gz-bridge \
        ros-jazzy-ros-gz-sim \
        ros-jazzy-ros-gz-image \
    && rm -rf /var/lib/apt/lists/*

# No OSMesa or GL packages here: Gazebo's sensor system renders with Ogre2 on the
# CPU, and the viewer renders in the browser on the Mac. That is what keeps this
# demo free of any GPU or CUDA requirement.
#
# websockets and Pillow are installed via pip rather than apt so they land in the
# same interpreter ROS 2 uses, alongside the numpy that the gz bindings already
# work with.
RUN python3 -m pip install --no-cache-dir --break-system-packages --ignore-installed \
        "pydantic>=2.7,<3" \
        "numpy>=1.26" \
        "typer>=0.12" \
        "rich>=13.7" \
        "PyYAML>=6.0" \
        "websockets>=13" \
        "Pillow>=10" \
        "imageio-ffmpeg>=0.5"

ENV ROS_DISTRO=jazzy
ENV ROBOT_CI_ROOT=/workspace/robot-ci
ENV PYTHONPATH=/workspace/robot-ci/src

WORKDIR /workspace/robot-ci

# The repository is mounted at /workspace/robot-ci by docker-compose, so it is
# not copied in. That keeps the image citable and the demo a property of the
# checkout you are looking at.
CMD ["bash", "/workspace/robot-ci/scripts/gazebo_demo.sh"]
