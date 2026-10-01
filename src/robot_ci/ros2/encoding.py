r"""Conversion between ROBOT CI models and ROS 2 message payloads.

Split deliberately in two:

* **This module** turns ROBOT CI models into plain Python/NumPy values and
  back. It imports no ROS, so it is unit-testable on a machine with no ROS 2
  installed, and the conversion arithmetic is covered by the ordinary test
  suite rather than only by a runtime test that most machines skip.
* :mod:`robot_ci.ros2.nodes` does the thin step of writing those values into
  real ``rclpy`` message objects and publishing them.

Time
----
ROS 2 simulation time is nanoseconds since the epoch *on the simulation
clock*. MuJoCo time is seconds since episode start. The clock published on
``/clock`` is therefore the episode clock, and every message stamp is derived
from the same MuJoCo time value that produced the data in it -- never from
``time.time()``. A frame captured at episode time ``t`` is stamped ``t`` even
if it is published later, so a subscriber that compares stamps is comparing
physics, not scheduling.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from robot_ci.models.state import Action, Observation, RobotState, SensorFrame

__all__ = [
    "NS_PER_S",
    "action_from_payload",
    "action_to_payload",
    "decode_payload",
    "depth_to_float32_bytes",
    "encode_payload",
    "joint_state_fields",
    "observation_from_payload",
    "observation_to_payload",
    "quaternion_from_ros",
    "quaternion_to_ros",
    "rgb_to_bytes",
    "sim_time_to_ns",
    "stamp_to_sim_time",
]

#: Nanoseconds per second, for the ROS 2 ``builtin_interfaces/Time`` conversion.
NS_PER_S = 1_000_000_000


def sim_time_to_ns(sim_time_s: float) -> int:
    """Convert MuJoCo episode seconds to ROS 2 nanoseconds.

    Truncates toward zero so the mapping is monotonic: two distinct MuJoCo
    times never collapse into a decreasing nanosecond stamp.

    Args:
        sim_time_s: Time on the episode clock, in seconds.

    Returns:
        Nanoseconds on the simulation clock.
    """
    return int(max(0.0, float(sim_time_s)) * NS_PER_S)


def stamp_to_sim_time(stamp_ns: int) -> float:
    """Convert a ROS 2 nanosecond stamp back to episode seconds."""
    return float(stamp_ns) / NS_PER_S


def encode_payload(payload: dict[str, Any]) -> str:
    """Serialize a payload for a ``std_msgs/msg/String`` field.

    ``sort_keys`` keeps the encoding stable so two identical observations
    produce byte-identical messages, which is what makes a recorded bag
    diffable and a determinism test meaningful.
    """
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=True)


def decode_payload(text: str) -> dict[str, Any]:
    """Parse a payload produced by :func:`encode_payload`."""
    value = json.loads(text)
    if not isinstance(value, dict):  # pragma: no cover - defensive
        raise ValueError(f"expected a JSON object, got {type(value).__name__}")
    return value


def joint_state_fields(robot_state: RobotState) -> dict[str, Any]:
    """Extract the arrays that fill a ``sensor_msgs/msg/JointState``.

    The count of names and the count of positions must agree: a ROS joint
    state where they differ is rejected by consumers downstream, and silently
    padding is how a wrong joint gets commanded later. Names are synthesized
    only when the state carries none, and the synthesis is visible here rather
    than hidden in the node.

    Args:
        robot_state: The measured robot state.

    Returns:
        A mapping with ``name``, ``position``, ``velocity``, ``effort``.

    Raises:
        ValueError: If names are present but disagree in length with positions.
    """
    positions = [float(v) for v in robot_state.joint_positions]
    names = list(robot_state.joint_names)
    if names and len(names) != len(positions):
        raise ValueError(
            f"joint name/position length mismatch: {len(names)} names, {len(positions)} positions"
        )
    if not names:
        names = [f"panda_joint{i + 1}" for i in range(len(positions))]
    velocities = [float(v) for v in robot_state.joint_velocities]
    efforts = [float(v) for v in robot_state.joint_efforts]
    return {
        "name": names,
        "position": positions,
        "velocity": velocities,
        "effort": efforts,
    }


def observation_to_payload(observation: Observation) -> dict[str, Any]:
    """Encode an :class:`Observation` as a JSON-ready mapping.

    The pixel arrays in ``sensor_frame`` are **not** inlined. They are
    published as real ``sensor_msgs/msg/Image`` messages on the camera topics,
    which is where a ROS consumer expects to find them and what makes them
    viewable in ``rqt_image_view`` and recordable in a bag. What is inlined is
    the frame's metadata and its ``valid`` flag, so a subscriber can tell a
    dropped frame from a missing topic -- two different failures that a
    pixels-or-nothing message would conflate.

    Args:
        observation: The observation the policy would receive.

    Returns:
        A JSON-serializable mapping.
    """
    payload = observation.model_dump(mode="json", exclude={"sensor_frame"})
    frame = observation.sensor_frame
    if frame is not None:
        payload["sensor_frame"] = _frame_metadata(frame)
    return payload


def _frame_metadata(frame: SensorFrame) -> dict[str, Any]:
    """Return the array-free description of a sensor frame."""
    return {
        "camera_id": frame.camera_id,
        "timestamp_s": frame.timestamp_s,
        "width_px": frame.width_px,
        "height_px": frame.height_px,
        "intrinsics": list(frame.intrinsics) if frame.intrinsics is not None else None,
        "valid": frame.valid,
        "has_rgb": frame.rgb is not None,
        "has_depth": frame.depth is not None,
        "has_segmentation": frame.segmentation is not None,
    }


#: Markers that describe the wire, not the model. ``SensorFrame`` forbids extra
#: fields, and these flags say only whether pixels were present -- the pixels
#: themselves travel on the camera topics -- so they are dropped on receipt.
_FRAME_ONLY_KEYS = ("has_rgb", "has_depth", "has_segmentation")


def observation_from_payload(payload: dict[str, Any]) -> Observation:
    """Rebuild an :class:`Observation` from :func:`observation_to_payload`.

    The rebuilt frame carries metadata only. A policy that needs pixels reads
    them from the camera topics; ``PickAndPlacePolicy`` reads the fused object
    beliefs, which are carried in full.
    """
    data = dict(payload)
    frame_data = data.pop("sensor_frame", None)
    observation = Observation.model_validate(data)
    if frame_data:
        frame_fields = {k: v for k, v in frame_data.items() if k not in _FRAME_ONLY_KEYS}
        observation = observation.model_copy(update={"sensor_frame": SensorFrame(**frame_fields)})
    return observation


def action_to_payload(action: Action) -> dict[str, Any]:
    """Encode an :class:`Action` for the command topic."""
    return action.model_dump(mode="json")


def action_from_payload(payload: dict[str, Any]) -> Action:
    """Rebuild an :class:`Action` from :func:`action_to_payload`.

    ``model_validate`` re-runs the action's own payload checks, so a command
    that arrives malformed over the wire fails here with the same error a
    direct call would raise, rather than being applied as something else.
    """
    return Action.model_validate(payload)


#: Key holding the action inside a command envelope.
_COMMAND_ACTION_KEY = "action"
#: Key holding the policy's diagnostics inside a command envelope.
_COMMAND_DIAGNOSTICS_KEY = "diagnostics"


def command_to_payload(action: Action, diagnostics: dict[str, Any] | None = None) -> dict[str, Any]:
    """Encode a command and the policy's diagnostics as one payload.

    Diagnostics travel with the command because the experiment runner *decides
    the failure mode* from them: the stall detector reads ``waypoint_target``,
    ``reached_waypoint`` and ``waypoint`` out of ``PolicyOutput.diagnostics``.
    Dropping them did not change whether an episode succeeded -- it changed
    why a failing one failed, turning a detected stall into a step-cap timeout
    and moving the step count from 85 to 300 on the 8 mm counterexample. A
    diagnostic dropped on the wire is a behaviour change, not a lost message.
    """
    return {
        _COMMAND_ACTION_KEY: action_to_payload(action),
        _COMMAND_DIAGNOSTICS_KEY: dict(diagnostics or {}),
    }


def command_from_payload(payload: dict[str, Any]) -> tuple[Action, dict[str, Any]]:
    """Rebuild ``(action, diagnostics)`` from :func:`command_to_payload`.

    A bare action mapping is still accepted, so a payload written by an older
    producer decodes -- it simply carries no diagnostics, which is recorded as
    empty rather than invented.
    """
    if _COMMAND_ACTION_KEY in payload:
        action = Action.model_validate(payload[_COMMAND_ACTION_KEY])
        raw = payload.get(_COMMAND_DIAGNOSTICS_KEY) or {}
        return action, dict(raw) if isinstance(raw, dict) else {}
    return Action.model_validate(payload), {}


def quaternion_to_ros(quaternion: Any) -> tuple[float, float, float, float]:
    """Reorder a ROBOT CI quaternion into the ROS convention.

    ROBOT CI stores ``(w, x, y, z)``; ROS messages store ``(x, y, z, w)``.
    Getting this backwards produces a pose that looks plausible and is wrong,
    so the reorder lives here as a named function with a test rather than
    inline in the publisher.
    """
    return (
        float(quaternion.x),
        float(quaternion.y),
        float(quaternion.z),
        float(quaternion.w),
    )


def quaternion_from_ros(x: float, y: float, z: float, w: float) -> dict[str, float]:
    """Reorder a ROS ``(x, y, z, w)`` quaternion into ROBOT CI field names."""
    return {"w": float(w), "x": float(x), "y": float(y), "z": float(z)}


def rgb_to_bytes(rgb: Any) -> bytes:
    """Return a contiguous ``uint8`` RGB buffer for a ``rgb8`` Image message."""
    array = np.asarray(rgb)
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(array).tobytes()


def depth_to_float32_bytes(depth: Any) -> bytes:
    """Return a contiguous ``float32`` depth buffer for a ``32FC1`` Image.

    Depth is published in metres as the sensor produced it. It is *not*
    converted to millimetres for display: a scaling applied for the benefit of
    a viewer is exactly the kind of silent unit change that makes a downstream
    measurement wrong.
    """
    array = np.asarray(depth, dtype=np.float32)
    return np.ascontiguousarray(array).tobytes()
