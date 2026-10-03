"""Live, staged verification of the Gazebo backend against the running physics.

Phase 3 asks for evidence that can be checked stage by stage, because "the
episode failed" tells you nothing about *where*. Each stage below either passes
with a measured number or fails with the number that explains it:

  1. world loads, Panda exists, joint_state publishes
  2. all nine joints (7 arm + 2 gripper) respond to a commanded target
  3. arm joints settle within a stated tolerance
  4. URDF forward kinematics agree with Gazebo's own measured link poses (what
     validates the tool-centre-point frame the policy commands)
  5. the gripper reaches the commanded aperture

Nothing here fabricates a reading: every number comes off Gazebo's own
``joint_state`` or ``pose/info`` topics, or off a forward-kinematics evaluation
of the URDF that produced the world.

Run inside the container::

    scripts/gz_run.sh start 'source /opt/ros/jazzy/setup.bash && python3 scripts/gazebo_stage_check.py'

Exit code 0 means every stage passed. The JSON record is written to
``experiments/results/gazebo/stage_check.json``.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

RESULTS = Path("experiments/results/gazebo")

#: Control period the default scenario runs at (20 Hz). Every stage that
#: advances the simulator uses the backend's own ``step``, which waits for
#: exactly this much simulated time, so it is the common unit below.
PERIOD_S = 0.05

#: How far the tool may be from its commanded point once the joints have
#: settled, in metres.
#:
#: Judging the arm by a per-joint angle budget is the wrong question: the
#: policy never sees joint angles, it checks whether the *tool* is within 20 mm
#: of the waypoint it is driving to (``PickAndPlacePolicy``, the ``reached``
#: test). So the settled joint errors are propagated through the Jacobian into a
#: bound on the tool error and compared with that tolerance. Per-joint errors are
#: still reported, but they are diagnostics rather than the verdict.
TOOL_TOLERANCE_M = 0.02

#: Aperture tolerance for the gripper, in metres of per-finger travel.
GRIPPER_TOLERANCE_M = 0.004


def stage1_load(backend: Any, scenario: Any, handle: Any) -> dict[str, Any]:
    """The world is up, the robot publishes state, and the reader keeps up."""
    from robot_ci.models.state import Action

    reader = backend._state(handle)["joints"]
    reader.health = reader._health
    deadline = time.monotonic() + 20.0
    names: tuple[str, ...] = ()
    while time.monotonic() < deadline:
        positions, _sim_time = reader.latest()
        if positions:
            names = tuple(positions)
            break
        time.sleep(0.2)
    expected = set(scenario.robot.all_joint_names)
    missing = sorted(expected - set(names))

    # How fast the reader actually parses, against how fast Gazebo publishes.
    # The joint state arrives on a 1 kHz physics-rate topic, but it is decoded
    # from the `gz topic -e` CLI's pretty-printed JSON by a brace-matching
    # reader. If that reader cannot keep up, every measurement this project
    # makes about the arm is stale by however many periods it falls behind --
    # including the trace used to judge whether a grasp is stable.
    start_messages = reader.health.joint_messages
    _, start_sim = reader.latest()
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        time.sleep(0.5)
        backend.step(handle, Action.noop())
    parsed = reader.health.joint_messages - start_messages
    sim_elapsed = backend._sim_time(handle) - start_sim
    measured_hz = parsed / sim_elapsed if sim_elapsed > 0 else 0.0
    return {
        "ok": not missing and bool(names),
        "sim_time_s": backend._sim_time(handle),
        "joints_reported": sorted(names),
        "missing": missing,
        "joint_messages_parsed": parsed,
        "sim_seconds_advanced": round(sim_elapsed, 3),
        "reader_rate_hz": round(measured_hz, 1),
        "publish_rate_hz": 1000.0,
        # At the physics rate, a reader keeping up would parse 1000 messages
        # per simulated second.
        "reader_keeps_up": measured_hz > 500.0,
        "periods_behind_at_20hz": round((1000.0 / measured_hz - 1) / 20.0, 2)
        if measured_hz > 0
        else None,
    }


def stage2_response(backend: Any, scenario: Any, handle: Any) -> dict[str, Any]:
    """Command every joint and show it moved towards the target.

    Two rounds, to distinct targets. One round cannot prove responsiveness: a
    joint already sitting at the commanded target would be recorded as "moved"
    without anything having been commanded, or recorded as "stuck" when it is
    simply already there. Two separated targets make both readings impossible.
    """
    from robot_ci.models.state import Action

    reader = backend._state(handle)["joints"]
    before, _ = reader.latest()
    lower = scenario.robot.limits.lower
    upper = scenario.robot.limits.upper
    arms = scenario.robot.arm_joint_names
    fingers = scenario.robot.gripper_joint_names
    max_width = float(scenario.robot.max_gripper_width_m)
    start = {name: before.get(name, 0.0) for name in (*arms, *fingers)}
    moved = dict.fromkeys(start, False)
    last_measured: dict[str, float] = dict(start)
    last_commanded: dict[str, float] = dict(start)

    for fraction in (0.3, 0.7):
        targets = [float(lower[i] + fraction * (upper[i] - lower[i])) for i in range(len(arms))]
        width = fraction * max_width
        last_commanded = dict(zip(arms, targets, strict=True))
        last_commanded[fingers[0]] = width / 2.0
        last_commanded[fingers[1]] = width / 2.0
        action = Action.arm_and_gripper(targets, width, arm_joint_count=len(arms))
        deadline = time.monotonic() + 15.0
        while time.monotonic() < deadline:
            backend.step(handle, action)
            now, _ = reader.latest()
            last_measured = {name: now.get(name, 0.0) for name in start}
            for name in start:
                if abs(last_measured[name] - start[name]) > 1e-4:
                    moved[name] = True
            if all(moved.values()):
                break
        if all(moved.values()):
            break

    return {
        "ok": all(moved.values()),
        "not_moving": sorted(name for name, ok in moved.items() if not ok),
        "commanded_last_round": last_commanded,
        "measured_last_round": last_measured,
        "start": start,
        "travel_rad": {name: round(last_measured[name] - start[name], 5) for name in start},
    }


def stage3_settle(backend: Any, scenario: Any, handle: Any) -> dict[str, Any]:
    """Hold the home configuration and measure the *trajectory* of the error.

    A single number cannot distinguish "still converging" from "settled on the
    wrong answer", and those two are completely different defects. The error is
    therefore sampled over a fixed simulation window and the slope is reported:
    a controller with no integral term settles at a finite error under gravity,
    and that shows up as a flat tail, whereas a slow controller shows a falling
    one. The fix depends entirely on which it is.
    """
    from robot_ci.models.state import Action

    reader = backend._state(handle)["joints"]
    home = [float(v) for v in scenario.robot.home_position[:7]]
    action = Action.arm_and_gripper(
        home, float(scenario.robot.max_gripper_width_m), arm_joint_count=7
    )
    # Long enough to distinguish "converging slowly" from "asymptotes on the
    # wrong answer". The first short window measured joints still falling after
    # 4.5 s, which cannot tell the two apart, and the fix depends on which.
    window_s = 25.0
    deadline = time.monotonic() + 90.0
    samples: list[dict[str, float]] = []
    sim_start = backend._sim_time(handle)
    while time.monotonic() < deadline and backend._sim_time(handle) - sim_start < window_s:
        backend.step(handle, action)
        positions, _ = reader.latest()
        samples.append(
            {
                name: abs(positions.get(name, 0.0) - target)
                for name, target in zip(scenario.robot.arm_joint_names, home, strict=True)
            }
        )
    if not samples:
        return {"ok": False, "reason": "no samples collected"}

    worst_final = max(samples[-1].values())
    per_joint: dict[str, Any] = {}
    for name in scenario.robot.arm_joint_names:
        series = [s[name] for s in samples]
        tail = series[-max(1, len(series) // 5) :]
        head = series[: max(1, len(series) // 5)]
        per_joint[name] = {
            "first_mrad": round(1000.0 * series[0], 2),
            "last_mrad": round(1000.0 * series[-1], 2),
            "max_mrad": round(1000.0 * max(series), 2),
            # Mean of the last fifth minus the first fifth: negative means the
            # error is still falling, ~zero means it has settled.
            "drift_mrad": round(1000.0 * (sum(tail) / len(tail) - sum(head) / len(head)), 2),
        }
    settled_flat = [
        name for name, info in per_joint.items() if abs(info["drift_mrad"]) < 1.0
    ]

    # The fingers settle separately, and they are not a formality: the pads can
    # only close onto the object if the commanded aperture actually arrives, and
    # a jaw that rests a few millimetres open simply does not touch it.
    finger_action = Action.arm_and_gripper(home, 0.0, arm_joint_count=len(home))
    finger_deadline = time.monotonic() + 20.0
    finger_error_m = {}
    while time.monotonic() < finger_deadline:
        backend.step(handle, finger_action)
        positions, _ = reader.latest()
        finger_error_m = {
            name: abs(positions.get(name, 0.0))
            for name in scenario.robot.gripper_joint_names
        }
    # The physically meaningful question about the jaw is not "did it hit an
    # arbitrary tolerance" but "can it close onto the object at all". That is
    # pad_separation = closed_separation + 2 * measured travel, compared with the
    # object's own width: if the pads cannot close past the object they never
    # touch it, and no amount of accuracy elsewhere changes that.
    closed_residual = max(finger_error_m.values()) if finger_error_m else float("inf")
    pad_separation = float(scenario.robot.gripper_closed_separation_m) + 2.0 * closed_residual
    object_width = 2.0 * float(scenario.target.size_m)
    finger_ok = pad_separation < object_width

    # Propagate the settled joint errors into a bound on the tool error: the
    # worst case a positional Jacobian permits is ``sum |J_ij| * |error_j|``.
    # That is the quantity the policy's own "have I reached the waypoint" test
    # is written against, so bounding it bounds what the task actually needs.
    import numpy as _np

    positions_now, _sim_now = reader.latest()
    q_now = _np.array(
        [positions_now.get(name, 0.0) for name in scenario.robot.arm_joint_names]
    )
    jac = _np.asarray(backend.jacobian_provider(handle).jacobian(q_now), dtype=float)
    vector = _np.array(
        [per_joint[name]["last_mrad"] / 1000.0 for name in scenario.robot.arm_joint_names]
    )
    tool_bound_m = float(_np.sum(_np.abs(jac[:3, :]) * vector))
    within = tool_bound_m <= TOOL_TOLERANCE_M

    return {
        "ok": within and finger_ok,
        "tool_tolerance_m": TOOL_TOLERANCE_M,
        "tool_error_bound_m": round(tool_bound_m, 5),
        "samples": len(samples),
        "window_sim_s": round(backend._sim_time(handle) - sim_start, 2),
        "final_max_error_rad": round(worst_final, 5),
        "settled_within_budget": within,
        "fingers": {
            "ok": finger_ok,
            "commanded_aperture_m": 0.0,
            "residual_per_finger_m": {k: round(v, 5) for k, v in finger_error_m.items()},
            "pad_separation_at_closed_m": round(pad_separation, 5),
            "object_width_m": round(object_width, 5),
            "interference_m": round(object_width - pad_separation, 5),
        },
        "still_converging": sorted(
            name
            for name, info in per_joint.items()
            if info["drift_mrad"] < -1.0
        ),
        "flat_tail": settled_flat,
        "per_joint": per_joint,
        # A coarse error trace for joint1, the joint that settles slowest, so
        # the shape of the convergence (exponential tail vs flat plateau) is
        # visible in the artifact rather than only in a summary statistic.
        "trace_joint1_mrad": [
            round(1000.0 * s[scenario.robot.arm_joint_names[0]], 2)
            for s in samples[:: max(1, len(samples) // 25)]
        ],
    }


def stage4_kinematics(backend: Any, handle: Any) -> dict[str, Any]:
    """URDF forward kinematics vs Gazebo's own measured link poses.

    This is the check that matters most for control. The policy commands a tool
    centre point that exists only in the URDF, so if the URDF's chain and
    Gazebo's links disagree, every commanded pose is wrong by that amount even
    though the arm visibly moves.
    """
    from robot_ci.gazebo.live import JOINT_NAMES
    from robot_ci.recording.render3d import forward_kinematics, load_kinematics

    reader = backend._state(handle)["joints"]
    poses = backend._state(handle)["poses"]
    positions, _ = reader.latest()
    kin = load_kinematics(backend._state(handle)["urdf_path"])
    q = np.array([positions.get(name, 0.0) for name in JOINT_NAMES[:7]])
    full = np.concatenate(
        [q, [positions.get("left_finger", 0.0), positions.get("right_finger", 0.0)]]
    )
    frames = forward_kinematics(kin, full)

    residuals: dict[str, float] = {}
    for link in ("panda_link7", "panda_hand"):
        measured = poses.position(link)
        if measured is None:
            residuals[link] = float("nan")
            continue
        residuals[link] = float(
            np.linalg.norm(frames[link] - np.asarray(measured))
        )

    provider = backend.jacobian_provider(handle)
    tcp_position, _tcp_rotation = provider.forward(q)
    tcp_beyond_hand = float(
        np.linalg.norm(np.asarray(frames["panda_hand"]) - np.asarray(tcp_position))
    )
    # What the simulator says the scene is, rather than what the SDF template
    # says it should be. The base's measured height is what decides whether the
    # URDF's own frame assumptions hold, and the table's top decides whether any
    # link is expected to be resting on it.
    scene = {
        name: tuple(round(v, 6) for v in (poses.position(name) or ()))
        for name in sorted(p for p in poses._poses if "panda" in p or "table" in p)
    }
    finite = {k: v for k, v in residuals.items() if v == v}
    worst = max(finite.values()) if finite else float("inf")
    return {
        "ok": bool(finite) and worst <= 0.005,
        "link_residual_m": {k: round(v, 5) for k, v in residuals.items()},
        "worst_residual_m": round(worst, 5),
        "bound_m": 0.005,
        "tcp_beyond_hand_m": round(tcp_beyond_hand, 5),
        "measured_links_available": sorted(
            name for name in ("panda_link7", "panda_hand") if poses.position(name) is not None
        ),
        "base_measured": scene.get("panda_link0"),
        "wrist_fk_minus_measured_m": residuals.get("panda_link7"),
        "scene_entities": len(scene),
        "scene": scene,
    }


def stage5_gripper(backend: Any, scenario: Any, handle: Any) -> dict[str, Any]:
    """Command several apertures and measure what the fingers actually reached.

    Sweeping the command, rather than reporting one closed position, is what
    separates the three candidate explanations for a residual: a constant offset
    would be flat across the sweep, a proportional term would scale with it, and
    a convergence still in progress would shrink as the window grows. The
    fingers matter more than they look: the pads close to 11 mm apart and the
    object is 16 mm across, so a jaw resting a few millimetres open does not
    touch the object at all, and the episode cannot grasp anything.
    """
    from robot_ci.models.state import Action

    reader = backend._state(handle)["joints"]
    names = scenario.robot.gripper_joint_names
    measured: dict[str, float] = {}
    settle_s = 30.0
    for width in (
        float(scenario.robot.max_gripper_width_m),
        0.5 * float(scenario.robot.max_gripper_width_m),
        0.0,
    ):
        action = Action.gripper(width)
        deadline = time.monotonic() + settle_s
        last = None
        while time.monotonic() < deadline:
            backend.step(handle, action)
            positions, _ = reader.latest()
            last = positions
            # No early exit: an earlier version broke out as soon as the
            # reading came within tolerance and therefore recorded the aperture
            # *on its way past*, which reported a 3.9 mm "residual" at one
            # command and a 3.9 mm one of the opposite sign at the next. Every
            # aperture gets the full window.
        positions = last if last is not None else reader.latest()[0]
        measured[f"{width:.4f}"] = {
            name: positions.get(name, 0.0) for name in names
        }
    errors = {
        key: {
            name: value[name] - float(key) / 2.0 for name in names
        }
        for key, value in measured.items()
    }
    worst = max(
        (abs(v) for entry in errors.values() for v in entry.values()), default=float("inf")
    )
    return {
        "ok": worst <= GRIPPER_TOLERANCE_M,
        "tolerance_m": GRIPPER_TOLERANCE_M,
        "commanded_aperture_m": [float(k) for k in measured],
        "measured_per_finger_m": measured,
        "residual_per_finger_m": errors,
        "max_residual_m": round(worst, 5),
        # A closed jaw's pad separation is closed_separation + 2 * travel, so the
        # residual is reported as the pad gap it produces at the grasp command.
        "pad_separation_at_closed_m": round(
            float(scenario.robot.gripper_closed_separation_m)
            + 2 * (errors["0.0000"][names[0]]),
            5,
        ),
        "object_width_m": round(2.0 * float(scenario.target.size_m), 5),
    }


def main() -> int:
    """Run every stage against a fresh live simulation."""
    from robot_ci.scenarios import default_scenario
    from robot_ci.simulation.gazebo import GazeboBackend

    scenario = default_scenario(seed=3)
    backend = GazeboBackend(startup_timeout_s=180)
    stages: dict[str, dict[str, Any]] = {}
    started = time.time()
    try:
        handle = backend.reset(scenario, seed=3)
        stages["1_world_loaded"] = stage1_load(backend, scenario, handle)
        stages["2_all_joints_respond"] = stage2_response(backend, scenario, handle)
        stages["3_joints_settle"] = stage3_settle(backend, scenario, handle)
        stages["4_kinematics_match"] = stage4_kinematics(backend, handle)
        stages["5_gripper_reaches_aperture"] = stage5_gripper(backend, scenario, handle)
    except Exception as exc:  # evidence run: record the failure, do not hide it
        stages["0_fatal"] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    finally:
        backend.close_all()

    passed = all(stage.get("ok") for stage in stages.values())
    payload = {
        "scenario_id": scenario.scenario_id,
        "seed": 3,
        "wall_clock_s": round(time.time() - started, 2),
        "passed": passed,
        "stages": stages,
        "tool_tolerance_m": TOOL_TOLERANCE_M,
        "gripper_tolerance_m": GRIPPER_TOLERANCE_M,
    }

    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "stage_check.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8"
    )
    with (RESULTS / "stage_check.txt").open("w", encoding="utf-8") as out:
        out.write("gazebo stage check seed=3\n" + "=" * 72 + "\n")
        for name, stage in stages.items():
            out.write(f"\n{name}: {'PASS' if stage.get('ok') else 'FAIL'}\n")
            for key, value in stage.items():
                out.write(f"  {key}: {value}\n")
        out.write(f"\noverall: {'PASS' if passed else 'FAIL'}\n")

    for name, stage in stages.items():
        print(f"{name}: {'PASS' if stage.get('ok') else 'FAIL'}")
        for key, value in stage.items():
            if key != "ok":
                print(f"   {key}: {value}")
    print(f"overall: {'PASS' if passed else 'FAIL'}")
    print(f"wrote {RESULTS / 'stage_check.json'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
