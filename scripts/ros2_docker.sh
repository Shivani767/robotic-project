#!/usr/bin/env bash
# Run a command inside the ROS 2 environment this project verifies against.
#
# Why this exists: ROS 2 Jazzy ships CPython 3.12 (Ubuntu 24.04), which is the
# only ROS 2 line whose interpreter satisfies this package's
# `requires-python = ">=3.11,<3.13"`. Humble's 3.10 cannot import robot_ci at
# all. Rather than document "install ROS 2" and hope, the image is pinned by
# distro *and* interpreter so that "it builds" and "it runs" are the same claim.
#
# Usage:
#   scripts/ros2_docker.sh bash                                  # interactive shell
#   scripts/ros2_docker.sh pytest tests/integration/test_ros2_runtime.py -v
#   scripts/ros2_docker.sh ros2 launch robot_ci_bringup panda_pick_place.launch.py
#   scripts/ros2_docker.sh ./scripts/ros2_evidence.sh             # capture proof
#
# The workspace is bind-mounted, so edits on the host are what runs inside.
set -euo pipefail

cd "$(dirname "$0")/.."

IMAGE="${ROBOT_CI_ROS2_IMAGE:-robot-ci-ros2:jazzy}"
DOCKERFILE="docker/ros2.Dockerfile"

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  echo "building $IMAGE (first run; this takes a few minutes)" >&2
  docker build -f "$DOCKERFILE" -t "$IMAGE" .
fi

# ROS needs a shared memory / network domain that works inside the container.
# ROS_LOCALHOST_ONLY keeps discovery on loopback so the host's own ROS daemon
# cannot inject nodes into the run.
exec docker run --rm -it \
  --network host \
  -e ROS_LOCALHOST_ONLY=1 \
  -e ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}" \
  -e MUJOCO_GL=osmesa \
  -e DISPLAY="${DISPLAY:-}" \
  -v "$PWD:/workspace" \
  -w /workspace \
  --entrypoint /bin/bash \
  "$IMAGE" -lc "source /opt/ros/\$ROS_DISTRO/setup.bash && if [ -f ros2_ws/install/setup.bash ]; then source ros2_ws/install/setup.bash; fi && $*"
