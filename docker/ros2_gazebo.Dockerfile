# ROS 2 Jazzy + Gazebo Harmonic, for the cross-simulator backend.
#
# Extends the ROS 2 image rather than starting from osrf/ros:harmonic, because
# Harmonic's own image is Ubuntu 22.04 / CPython 3.10 and this package requires
# >=3.11,<3.13. Jazzy is Ubuntu 24.04 / CPython 3.12 and `ros-jazzy-ros-gz`
# installs Gazebo Harmonic (gz-sim8) as the default, so this pairs the supported
# distro with the required interpreter.
#
# headless: gz sim -s runs the server with no GUI, which is how CI and this
# container run. The GUI is never required by an experiment.
FROM robot-ci-ros2:jazzy

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        ros-jazzy-ros-gz \
        ros-jazzy-ros-gz-bridge \
        ros-jazzy-ros-gz-sim \
        ros-jazzy-ros-gz-image \
    && rm -rf /var/lib/apt/lists/*

# No verification step here on purpose. Two earlier revisions asserted
# `import rclpy` and `gz sim --versions` during the build and both failed for
# reasons unrelated to the packages: ROS exposes its Python side on PYTHONPATH
# via setup.bash rather than in the system site-packages, and Gazebo's `gz` tool
# is not on the default PATH. Assertions belong in the runtime checks
# (scripts/gazebo_evidence.sh), where ROS is sourced and PATH is real.

CMD ["/bin/bash"]
