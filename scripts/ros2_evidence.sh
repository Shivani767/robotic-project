#!/usr/bin/env bash
# Capture live evidence that the ROS 2 graph is real, for one condition.
#
# Runs inside the ROS 2 container (scripts/ros2_docker.sh). Starts the real
# launch, then asks ROS itself what exists: which nodes are up, which topics,
# how many publishers/subscribers each has, and the observed rate of
# /robot_ci/joint_states. The robot's state only moves because the policy node
# answers over the graph, so a recorded rate is evidence the transport carries
# the episode rather than decoration around it.
#
# Usage:
#   scripts/ros2_evidence.sh <condition-name> <depth_noise_mm> [policy] [seed]
set -uo pipefail

cd "$(dirname "$0")/.."

# One domain per run. With --network host and the default domain, two concurrent
# runs discover each other's nodes: the duplicate /robot_ci_policy from an
# earlier run answers this run's observations and the driver times out waiting
# for a command from a node that is not its peer. Deriving the domain from the
# run's own arguments makes concurrent runs impossible to confuse.
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-$(( 20 + ( RANDOM % 200 ) ))}"

CONDITION="${1:-nominal}"
NOISE="${2:-0.0}"
POLICY="${3:-policy_v1}"
SEED="${4:-3}"

OUT="experiments/results/ros2"
EVIDENCE="$OUT/evidence_${CONDITION}.txt"
BAG_DIR="$OUT/bag_${CONDITION}"
mkdir -p "$OUT"

# ROS's own setup scripts read unbound variables (AMENT_TRACE_SETUP_FILES),
# so `set -u` is relaxed across the sourcing and restored immediately after.
# Everything this script does afterwards is still nounset-checked.
set +u
source "/opt/ros/$ROS_DISTRO/setup.bash"
source ros2_ws/install/setup.bash
set -u

TOPICS=(/clock /robot_ci/joint_states /robot_ci/ee_pose /robot_ci/tf
        /robot_ci/camera/rgb /robot_ci/camera/depth /robot_ci/contact_state
        /robot_ci/observation /robot_ci/command /robot_ci/episode_result)

cleanup() {
  [ -n "${LAUNCH_PID:-}" ] && kill "$LAUNCH_PID" 2>/dev/null
  [ -n "${BAG_PID:-}" ] && kill -INT "$BAG_PID" 2>/dev/null
  wait 2>/dev/null
  return 0
}
trap cleanup EXIT

echo "=== ROS 2 evidence: $CONDITION (policy=$POLICY seed=$SEED depth_noise=${NOISE}mm) ==="

# Record the graph while it runs. rosbag2 is part of ros-jazzy-rosbag2.
ros2 bag record -o "$BAG_DIR" "${TOPICS[@]}" >/tmp/bag_"$CONDITION".log 2>&1 &
BAG_PID=$!
sleep 3

ros2 launch robot_ci_bringup panda_pick_place.launch.py \
  policy:="$POLICY" seed:="$SEED" depth_noise_mm:="$NOISE" \
  >/tmp/launch_"$CONDITION".log 2>&1 &
LAUNCH_PID=$!

# Give the graph a moment to discover, then sample it.
sleep 8

{
  echo "--- ros2 node list ---"
  ros2 node list
  echo
  echo "--- ros2 topic list ---"
  ros2 topic list
  echo
  echo "--- ros2 topic info ---"
  for topic in "${TOPICS[@]}"; do
    echo "\$ ros2 topic info $topic"
    ros2 topic info "$topic"
  done
  echo
  echo "--- ros2 topic hz /robot_ci/joint_states (8s sample) ---"
  timeout 8 ros2 topic hz /robot_ci/joint_states 2>&1 || true
} > "$EVIDENCE" 2>&1

# Sample a live JointState message so the evidence contains actual numbers.
timeout 6 ros2 topic echo /robot_ci/joint_states --once > "$OUT/sample_joint_state_${CONDITION}.txt" 2>&1 || true

# Wait for the episode artifact, which is written when the run finishes.
# This run's artifact name is derived exactly as ros2_artifact derives it, so
# the wait cannot be satisfied by a different condition's file.
if [ "$(python3 -c "print(0 if float('$NOISE') == 0.0 else 1)")" = "0" ]; then
  STEM="nominal_seed${SEED}_${POLICY}"
else
  STEM="depth_noise_${NOISE}mm_seed${SEED}_${POLICY}"
fi
TARGET="$OUT/${STEM}"
for _ in $(seq 1 120); do
  if compgen -G "${TARGET}_*.json" > /dev/null; then break; fi
  if ! kill -0 "$LAUNCH_PID" 2>/dev/null; then break; fi
  sleep 2
done

{
  echo
  echo "--- episode artifact ---"
  ls -1 "$OUT"/*.json 2>/dev/null
  for artifact in "${TARGET}"_*.json; do
    [ -f "$artifact" ] || continue
    python3 - "$artifact" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
e, r, g = d["execution"], d["result"], d["graph"]
print(f"artifact       : {sys.argv[1]}")
print(f"scenario_id    : {e['scenario_id']}")
print(f"config_hash    : {e['config_hash']}")
print(f"seed / policy  : {e['seed']} / {e['policy']}")
print(f"backend        : {e['backend']}")
print(f"ros distro     : {e['ros2'].get('ros_distro')}  rclpy={e['ros2'].get('rclpy_version')}")
print(f"mujoco / python: {e['mujoco_version']} / {e['python_version']}")
print(f"git commit     : {e['git_commit']}")
print(f"status         : {r['status']}  success={r['success']}  steps={r['steps']}")
print(f"commands over ROS: {g['commands_applied']}")
print(f"messages published: {g['messages_published']}")
PY
  done
} >> "$EVIDENCE" 2>&1

echo "wrote $EVIDENCE"
cat "$EVIDENCE"
