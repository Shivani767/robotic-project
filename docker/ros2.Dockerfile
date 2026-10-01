# A ROS 2 environment capable of running the ROBOT CI episode live.
#
# Why a container: the project's `requires-python` is >=3.11,<3.13, and ROS 2
# Jazzy ships CPython 3.12 on Ubuntu 24.04. Humble ships 3.10 and cannot import
# this package at all. Pinning the image by distro *and* by the interpreter it
# carries is what makes "it builds" and "it runs" the same claim.
#
# Contents:
#   * ros-jazzy-ros-base        the client library, launch, rosbag2
#   * python3-mujoco + pip deps the simulator and the project
#   * robot-ci installed editable so the container runs this checkout
#
# MuJoCo needs no GPU: it runs on CPU with its bundled software renderer, so the
# camera frames are produced without CUDA or a display.
FROM ros:jazzy-ros-base

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3-pip \
        python3-venv \
        python3-colcon-common-extensions \
        ros-jazzy-rosbag2 \
        ros-jazzy-rosbag2-storage-mcap \
        git \
    && rm -rf /var/lib/apt/lists/*

# MuJoCo's classic renderer is imported at `import mujoco`, so a headless
# container still needs a real GL implementation. OSMesa gives software
# rendering with no GPU and no display, which is what keeps this image free of
# CUDA requirements. Without these libraries `import mujoco` raises inside
# PyOpenGL's loader and the run fails for a reason unrelated to ROS 2.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libosmesa6 \
        libgl1-mesa-dri \
        libglx-mesa0 \
        libglfw3 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY . /workspace

# Everything goes into the *system* interpreter, which is the one ROS 2 uses.
# `--ignore-installed` is required because pip refuses to uninstall files it did
# not create (Debian owns pluggy and friends) and those are exactly the ones
# that need upgrading. An earlier revision used a --system-site-packages venv;
# that failed at the last step, because colcon generates the node entry points
# with a `#!/usr/bin/python3` shebang regardless of which interpreter ran the
# build, so the nodes could not see the package they were launched to run.
RUN python3 -m pip install --no-cache-dir --break-system-packages --ignore-installed \
        "mujoco==3.14.0" \
        "pydantic>=2.7,<3" \
        "numpy>=1.26" \
        "typer>=0.12" \
        "rich>=13.7" \
        "PyYAML>=6.0" \
        "pytest>=8.0" \
    && python3 -m pip install --no-cache-dir --break-system-packages --ignore-installed -e . \
    && python3 -c "import mujoco, robot_ci; print('mujoco', mujoco.__version__)"

ENV MUJOCO_GL=osmesa
ENV ROS_DISTRO=jazzy

CMD ["/bin/bash"]
