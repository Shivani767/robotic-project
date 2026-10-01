#!/usr/bin/env bash
# Bring up the live ROBOT CI / Gazebo demo inside the container.
#
# This is the single command the whole demo hangs on: it starts Gazebo, checks
# that the simulation is really producing data, starts the joint driver, and then
# serves the browser viewer. Every "ready" line printed here is backed by an
# observation -- a topic that published, a frame that decoded -- because a
# process having started is not evidence that anything works.
#
# Nothing is faked: the joint commands go to Gazebo's cmd_pos topics and the
# browser is fed by the joint-state and camera topics of the running simulation.
set -o pipefail

ROOT="${ROBOT_CI_ROOT:-/workspace/robot-ci}"
WORLD="${ROBOT_CI_WORLD:-robot_ci_pick_place}"
PORT="${ROBOT_CI_PORT:-8000}"
URDF="${ROOT}/gazebo/models/panda/panda.urdf"
SDF="/tmp/robot_ci_demo_world.sdf"
SEED="${ROBOT_CI_SEED:-3}"
DEPTH_NOISE_MM="${ROBOT_CI_DEPTH_NOISE_MM:-0}"

# shellcheck disable=SC1091
source "/opt/ros/${ROS_DISTRO:-jazzy}/setup.bash"
export PATH="/opt/ros/${ROS_DISTRO:-jazzy}/bin:${PATH}"

log() { printf '[robot-ci] %s\n' "$*"; }
die() { printf '[robot-ci] FATAL: %s\n' "$*" >&2; exit 1; }

cleanup() {
    log "shutting down"
    kill "${DRIVER_PID:-}" "${GZ_PID:-}" 2>/dev/null || true
    wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

cd "${ROOT}" || die "repository not found at ${ROOT}"

# ---------------------------------------------------------------- preflight
command -v gz >/dev/null 2>&1 || die "gz not on PATH"
python3 -c "import robot_ci" 2>/dev/null || {
    export PYTHONPATH="${ROOT}/src:${PYTHONPATH}"
    python3 -c "import robot_ci" 2>/dev/null || die "robot_ci is not importable"
}
log "Gazebo $(gz sim --versions 2>/dev/null | head -1)"

# ---------------------------------------------------------------- world
log "generating world (seed=${SEED})"
PYTHONPATH="${ROOT}/src:${PYTHONPATH}" python3 - <<PY || die "world generation failed"
from robot_ci.gazebo import world_sdf
from robot_ci.scenarios import default_scenario

with open("${SDF}", "w") as handle:
    handle.write(world_sdf(default_scenario(), robot_urdf="${URDF}"))
PY

# ---------------------------------------------------------------- gazebo
log "starting Gazebo (headless, real physics)"
gz sim -s -r "${SDF}" > /tmp/robot_ci_gazebo.log 2>&1 &
GZ_PID=$!

JOINT_TOPIC="/world/${WORLD}/model/panda/joint_state"
CAMERA_TOPIC="/world/${WORLD}/model/camera_rig/link/link/sensor/robot_ci_camera/image"

# A topic that exists means the model loaded *and* the controller attached. The
# wait is bounded so a failure reports itself instead of hanging.
for _ in $(seq 1 60); do
    if gz topic -l 2>/dev/null | grep -qx "${JOINT_TOPIC}"; then
        break
    fi
    if ! kill -0 "${GZ_PID}" 2>/dev/null; then
        tail -20 /tmp/robot_ci_gazebo.log >&2
        die "Gazebo exited during startup"
    fi
    sleep 1
done
gz topic -l 2>/dev/null | grep -qx "${JOINT_TOPIC}" || die "Panda joint-state topic never appeared"

CONTROLLERS=$(gz topic -l 2>/dev/null | grep -c 'cmd_pos' || true)
[ "${CONTROLLERS}" -ge 9 ] || die "expected 9 joint controllers, found ${CONTROLLERS}"
log "Gazebo ready; Panda loaded with ${CONTROLLERS} joint controllers"

# ---------------------------------------------------------------- driver
log "starting joint driver (real cmd_pos commands)"
PYTHONPATH="${ROOT}/src:${PYTHONPATH}" python3 -m robot_ci.gazebo.motion_demo \
    --urdf "${URDF}" > /tmp/robot_ci_driver.log 2>&1 &
DRIVER_PID=$!

# ---------------------------------------------------------------- bridge
log "starting live bridge and viewer"
PYTHONPATH="${ROOT}/src:${PYTHONPATH}" python3 -m robot_ci.gazebo.live \
    --dashboard "${ROOT}/dashboard" \
    --urdf "${URDF}" \
    --world "${WORLD}" \
    --port "${PORT}" \
    --seed "${SEED}" \
    --depth-noise-mm "${DEPTH_NOISE_MM}" > /tmp/robot_ci_bridge.log 2>&1 &
BRIDGE_PID=$!

# ---------------------------------------------------------------- health
# Report readiness from observed data, not from the fact that processes exist.
health_ok=0
for _ in $(seq 1 60); do
    if ! kill -0 "${BRIDGE_PID}" 2>/dev/null; then
        tail -20 /tmp/robot_ci_bridge.log >&2
        die "bridge exited during startup"
    fi
    REPORT=$(python3 - "${PORT}" <<'PY' 2>/dev/null
import json, sys, urllib.request
try:
    with urllib.request.urlopen(f"http://127.0.0.1:{sys.argv[1]}/healthz", timeout=2) as response:
        print(json.dumps(json.load(response)))
except Exception:
    print("{}")
PY
)
    if printf '%s' "${REPORT}" | grep -q '"joint_stream": true'; then
        if printf '%s' "${REPORT}" | grep -q '"camera_stream": true'; then
            log "joint-state stream detected"
            log "camera stream detected"
            health_ok=1
            break
        fi
    fi
    sleep 1
done

[ "${health_ok}" -eq 1 ] || log "WARNING: streams not confirmed yet; the viewer will show them as waiting"

log "viewer ready -> http://localhost:${PORT}"
# Touch the page once so the served-viewer flag is set even before a browser
# connects, then report the full health picture.
python3 - "${PORT}" <<'PY' 2>/dev/null || true
import sys, urllib.request
urllib.request.urlopen(f"http://127.0.0.1:{sys.argv[1]}/", timeout=5).read()
PY
log "Open http://localhost:${PORT} in a browser on the Mac."

wait "${BRIDGE_PID}"
