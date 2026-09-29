"""Grasp geometry, measured from the compiled model.

Every constant involved in grasping -- the finger actuators, the maximum pad
opening, the reachable grasp orientations -- is derived from the model rather
than assumed, and these tests pin that derivation. A regression here produces a
*silent* failure: the arm converges, the waypoints are hit on time, and the
gripper closes beside the object.
"""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from robot_ci.models.geometry import Pose, Quaternion, Vec3
from robot_ci.models.scenario import ScenarioConfig
from robot_ci.planning.kinematics import IKOptions, solve_ik
from robot_ci.policies import TOP_DOWN
from robot_ci.simulation import create_backend
from robot_ci.simulation.mujoco_env import object_geoms

pytestmark = pytest.mark.integration
pytest.importorskip("mujoco", reason="MuJoCo is not installed")

#: Maximum pad separation measured on the vendored Panda. Objects must be sized
#: against this, not against the intuitive "50 mm cube".
MAX_PAD_SEPARATION_M = 0.0262


@pytest.fixture
def compiled():
    backend = create_backend("mujoco")
    handle = backend.reset(ScenarioConfig(scenario_id="grasp", seed=5), seed=5)
    yield backend, handle._opaque["model"], handle._opaque["data"]
    backend.close_all()


def test_finger_actuators_exist_and_map_to_finger_joints(compiled) -> None:
    """Both fingers must be driven, and by distinct actuators."""
    _backend, model, _data = compiled
    driven: dict[str, int] = {}
    for aid in range(model.nu):
        jid = int(model.actuator_trnid[aid, 0])
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
        if name and name.startswith("finger_joint"):
            driven[name] = aid
    assert set(driven) == {"finger_joint1", "finger_joint2"}, driven
    assert driven["finger_joint1"] != driven["finger_joint2"]


def test_gripper_opens_and_closes() -> None:
    """Commanding the actuator must actually move the pads.

    Measured through the public ``Action`` interface rather than by writing
    ``ctrl`` directly, so the test covers the actuator mapping the backend
    actually uses.
    """
    from robot_ci.models.state import Action

    scenario = ScenarioConfig(scenario_id="g", seed=5)
    home = list(scenario.robot.home_position)
    widths: dict[float, float] = {}
    backend = create_backend("mujoco")
    try:
        for aperture in (0.0, 0.04):
            handle = backend.reset(scenario, seed=5)
            for _ in range(3):
                backend.step(
                    handle,
                    Action.arm_and_gripper(home, aperture, arm_joint_count=7),
                )
            model, data = handle._opaque["model"], handle._opaque["data"]
            widths[aperture] = sum(
                float(
                    data.qpos[
                        model.jnt_qposadr[
                            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
                        ]
                    ]
                )
                for name in ("finger_joint1", "finger_joint2")
            )
    finally:
        backend.close_all()
    assert widths[0.0] < 0.002, f"closed width {widths[0.0]:.4f} m"
    assert widths[0.04] > 0.02, f"open width {widths[0.04]:.4f} m"
    assert widths[0.04] > widths[0.0]


def test_objects_fit_inside_the_gripper_envelope() -> None:
    """An object wider than the jaw can never be grasped.

    This is the constraint that made a 50 mm cube ungraspable, and it is the
    kind of mismatch a task specification should never contain.
    """
    from robot_ci.models.environment import ObjectCategory

    for category in ObjectCategory:
        extents, _geom = object_geoms(category, 0.008)
        width = 2.0 * extents[0]
        assert width <= MAX_PAD_SEPARATION_M, (
            f"{category.value} is {width * 1000:.0f} mm wide but the gripper "
            f"opens only {MAX_PAD_SEPARATION_M * 1000:.0f} mm"
        )
        # And not so small that the grasp is unstable: a parallel gripper wants
        # the object at 50-70% of its maximum opening. The lifter is exempt --
        # it is the deliberately hard, thin object, and grasping it is expected
        # to fail, which is the point of including it.
        if category is not ObjectCategory.LIFTER:
            assert width >= 0.4 * MAX_PAD_SEPARATION_M, (
                f"{category.value} is too small to grasp reliably"
            )


def test_grasp_orientation_is_reachable() -> None:
    """The grasp orientation must be kinematically achievable.

    Which orientations are reachable is a property of the arm and the scene
    layout, and it has changed more than once during Phase 2. Rather than
    asserting a remembered verdict, this measures the reachable set and pins
    only the fact the policy depends on: the orientation it uses is reachable,
    and it is among the best available.
    """
    backend = create_backend("mujoco")
    try:
        scenario = ScenarioConfig(scenario_id="k", seed=5)
        handle = backend.reset(scenario, seed=5)
        model = handle._opaque["model"]
        data = handle._opaque["data"]
        lo = np.array(scenario.robot.limits.lower)
        hi = np.array(scenario.robot.limits.upper)
        provider = backend.jacobian_provider(handle)
        obj = np.array(data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target_cube")])
        target = Vec3.of(obj[0], obj[1], obj[2] + 0.01)
        options = IKOptions(
            max_iterations=120, position_tolerance=6e-3, orientation_tolerance=0.4, nullspace_gain=0.0
        )

        def best_error(orientation) -> float:
            return min(
                solve_ik(
                    provider,
                    Pose(position=target, orientation=orientation),
                    np.random.default_rng(s).uniform(lo, hi),
                    lower=lo, upper=hi, options=options,
                ).position_error
                for s in range(12)
            )

        # The orientation the policy actually uses.
        chosen = best_error(TOP_DOWN)
        assert chosen < 0.01, f"the chosen grasp orientation is not reachable ({chosen * 1000:.1f} mm)"

        # It must be competitive with the alternatives, so a better grasp does
        # not silently go unused.
        alternatives = {
            "top_down": Quaternion(w=0.0, x=1.0, y=0.0, z=0.0),
            "identity": Quaternion.identity(),
            "x_neg90": Quaternion.from_axis_angle(Vec3.of(1, 0, 0), -1.5708),
        }
        for name, orientation in alternatives.items():
            assert best_error(orientation) >= chosen - 0.005, (
                f"{name} is more accurate than the chosen orientation"
            )
    finally:
        backend.close_all()


def test_tcp_sits_at_the_pad_midpoint(compiled) -> None:
    """The TCP must be the point between the pads, not the flange.

    A TCP that is not at the pad midpoint makes the policy aim a few
    centimetres off, which again fails silently.
    """
    _backend, model, data = compiled
    home = np.array(ScenarioConfig().robot.home_position)
    for index in range(7):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"joint{index + 1}")
        data.qpos[model.jnt_qposadr[jid]] = home[index]
    for aid in range(model.nu):
        for name in ("finger_joint1", "finger_joint2"):
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid >= 0 and int(model.actuator_trnid[aid, 0]) == jid:
                data.ctrl[aid] = 0.04
    data.qvel[:] = 0.0
    for _ in range(200):
        mujoco.mj_step(model, data)
    mujoco.mj_forward(model, data)

    pad_positions = []
    for gid in range(model.ngeom):
        body = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[gid])) or ""
        if (
            body in {"left_finger", "right_finger"}
            and int(model.geom_group[gid]) == 3
            and model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_BOX
        ):
            pad_positions.append(np.array(data.geom_xpos[gid]))
    assert pad_positions, "no finger collision pads found"

    midpoint = np.mean(pad_positions, axis=0)
    tcp = np.array(data.site_xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tcp")])
    assert float(np.linalg.norm(midpoint - tcp)) < 0.01, (
        f"TCP is {np.linalg.norm(midpoint - tcp) * 1000:.1f} mm from the pad midpoint"
    )
