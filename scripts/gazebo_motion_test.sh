#!/usr/bin/env bash
# Drive the Panda in a live Gazebo world and record what physics actually did.
#
# This is the control-path proof. It publishes position commands and reads the
# joint state back off Gazebo's own topic, so the evidence is measured motion
# rather than the mere existence of a command topic. Every joint's position is
# printed before and after each move, along with the end-effector pose, so a
# reviewer can see the arm physically travel.
#
# Run inside the container after `source /opt/ros/$ROS_DISTRO/setup.bash`.
set -uo pipefail

WORLD=robot_ci_pick_place
STATE="/world/${WORLD}/model/panda/joint_state"
POSES="/world/${WORLD}/dynamic_pose/info"
OUT="${1:-/workspace/experiments/results/gazebo/evidence_joint_motion.txt}"

# Print joint name/position pairs from one joint_state message. The controller
# publishes every joint, so a single message is a full snapshot of physics.
snapshot() {
    gz topic -e -t "$STATE" -n 1 2>/dev/null \
        | awk '/^ *name: /{n=$2} /^ *position: /{printf "      %-14s % .6f\n", n, $2}'
}

# End-effector position, straight from the pose stream, as independent evidence
# that the links move in the world and not just in the joint readout.
ee_pose() {
    gz topic -e -t "$POSES" -n 1 2>/dev/null \
        | awk '/name: "panda_link7"/{f=1} f&&/position \{/{p=1;next} p&&/^ *[xyz]: /{printf " %s", $2; if($1=="z:"){print ""; exit}}'
}

{
    echo "Panda live motion evidence"
    echo "world=${WORLD}"
    echo "state_topic=${STATE}"
    echo "date=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo
    echo "=== baseline joint state (settled under gravity) ==="
    snapshot
    echo "    link7 position:$(ee_pose)"
    echo
} > "$OUT"

command_joint() {
    local joint="$1" target="$2"
    {
        echo "--- command ${joint} -> ${target} rad ---"
        gz topic -t "/model/panda/joint/${joint}/0/cmd_pos" \
            -m gz.msgs.Double -p "data: ${target}" >/dev/null 2>&1
        sleep 2
        echo "    measured:"
        snapshot
        echo "    link7 position:$(ee_pose)"
    } >> "$OUT" 2>&1
}

# A sweep through every arm joint, then both fingers, each returned to zero.
for j in joint1 joint2 joint3 joint4 joint5 joint6 joint7; do
    command_joint "$j" 0.6
    command_joint "$j" 0.0
done
command_joint left_finger 0.04
command_joint right_finger 0.04

{
    echo
    echo "=== final joint state ==="
    snapshot
    echo "    link7 position:$(ee_pose)"
} >> "$OUT" 2>&1
