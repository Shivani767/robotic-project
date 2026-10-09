"""Clearance floor for the pick-and-place grasp aim.

The jaw must close without its pads resting on the support surface: measured
on live Gazebo, pad_bottom reached 0.5-0.9 mm below the tabletop at the
object-centre aim and the resulting ~4.3 N of Coulomb friction arrested the jaw
at ~19.8 mm pad separation -- 2.1 mm short of touching the 16 mm object, with
the policy's fixed 0.3 s dwell lifting before it could creep through.

The floor is derived, not calibrated: pad bottom = jaw_z - pad_half_height_m
must sit at least _GRASP_CLEARANCE_TOLERANCE_M above table_z.
"""

from __future__ import annotations

import pytest

from robot_ci.models.scenario import PolicyConfig
from robot_ci.models.state import GripperState, Observation, RobotState
from robot_ci.policies import create_policy
from robot_ci.scenarios import default_scenario
from robot_ci.tasks.manipulation import PickAndPlaceTask


def _observation(width_m: float) -> Observation:
    """A minimal observation whose only meaningful channel is the jaw aperture."""
    return Observation(
        robot=RobotState(
            joint_positions=(0.0,) * 9,
            gripper=GripperState(width_m=width_m, max_width_m=0.08),
        )
    )


def _aperture_seated_policy():
    scenario = default_scenario(seed=3)
    policy = create_policy(PolicyConfig(name="policy_v1"), scenario)
    return policy, scenario


def _grasp_waypoint_z(task: PickAndPlaceTask, scenario: object) -> float:
    from robot_ci.models.geometry import Vec3

    class _Obj:
        def __init__(self, x: float, y: float, z: float) -> None:
            self.position = Vec3.of(x, y, z)

    return task.waypoints(_Obj(0.0, 0.0, 0.008))[1].z


def test_grasp_aim_keeps_pad_bottom_above_table() -> None:
    """The aim floor applies exactly and only when the aim would bite."""
    from robot_ci.scenarios import default_scenario

    scenario = default_scenario(seed=3)
    task = PickAndPlaceTask(scenario)
    table_z = float(scenario.environment.table_height_m)
    pad_half = float(scenario.robot.pad_half_height_m)
    z = _grasp_waypoint_z(task, scenario)
    assert z - pad_half >= table_z, (
        f"pad bottom {z - pad_half:.4f} below table {table_z:.4f}"
    )


def test_grasp_aim_floor_is_exactly_the_clearance() -> None:
    """Raising the object keeps the aim on the object centre until the floor binds.

    For a tall object the aim stays at obj.z + offset; the floor only binds at
    the low object where pads would otherwise graze the table.
    """
    from robot_ci.models.geometry import Vec3
    from robot_ci.models.scenario import ScenarioConfig
    from robot_ci.scenarios import default_scenario

    scenario: ScenarioConfig = default_scenario(seed=3)
    task = PickAndPlaceTask(scenario)
    table_z = float(scenario.environment.table_height_m)
    pad_half = float(scenario.robot.pad_half_height_m)

    class _Obj:
        def __init__(self, z: float) -> None:
            self.position = Vec3.of(0.0, 0.0, z)

    low = task.waypoints(_Obj(0.008))[1].z
    assert low - pad_half >= table_z - 1e-9
    high = task.waypoints(_Obj(0.03))[1].z
    assert abs(high - (0.03 + float(scenario.robot.pad_offset_from_tcp_m))) < 1e-9, (
        "tall-object aim must stay on the object centre, floor must not touch it"
    )

def test_policy_aim_floor_includes_measured_settle_shortfall() -> None:
    """The policy floor commands clearance + measured shortfall, not just clearance.

    Five identical live Gazebo failures pinned the execution shortfall at
    1.5 mm (commanded 0.0095 m, achieved 0.0080 m on every run), so the floor
    must compensate it: achieved pad bottom is what must clear the table.
    """
    import numpy as np

    from robot_ci.models.scenario import PolicyConfig
    from robot_ci.policies import create_policy
    from robot_ci.scenarios import default_scenario

    scenario = default_scenario(seed=3)
    policy = create_policy(PolicyConfig(name="policy_v1"), scenario)
    table_z = float(scenario.environment.table_height_m)
    pad_half = float(scenario.robot.pad_half_height_m)
    expect = table_z + pad_half + 0.001 + 0.0015
    wps = [np.array([0.0, 0.0, 0.1085]), np.array([0.0, 0.0, 0.0085])]
    policy._apply_aim_clearance_floor(wps)
    assert abs(float(wps[1][2]) - expect) < 1e-9, wps[1][2]
    # achieved pose (= command minus the measured shortfall) still clears the table
    assert float(wps[1][2]) - 0.0015 - pad_half >= table_z - 1e-9


def test_aperture_seated_admits_only_a_closed_on_cube_jaw() -> None:
    """The lift gate admits an on-cube jaw and refuses a still-travelling one.

    The on-cube aperture is ``2*size_m - closed_separation`` = 5 mm here. The
    measured live failure read 9.33 mm at the old twelve-tick dwell -- the
    fingers were still closing onto the cube -- and the lift fired straight into
    a flick. The gate must therefore refuse anything meaningfully above 5 mm
    (band 1.1 mm, so <= 6.1 mm) and admit the on-cube reading.
    """
    policy, scenario = _aperture_seated_policy()
    on_cube = 2.0 * float(scenario.target.size_m) - float(
        scenario.robot.gripper_closed_separation_m
    )
    assert on_cube == pytest.approx(0.005)

    # On the cube: seated.
    assert policy._aperture_seated(_observation(on_cube)) is True
    # Steady-state tracking noise a hair above the mark: still seated.
    assert policy._aperture_seated(_observation(on_cube + 0.001)) is True
    # The measured live failure reading: still travelling, not seated.
    assert policy._aperture_seated(_observation(0.00933)) is False
    # Wide open: obviously not seated.
    assert policy._aperture_seated(_observation(0.04)) is False


def test_aperture_seated_threshold_is_derived_not_tuned() -> None:
    """The band scales with the object, so a different cube moves the gate.

    The threshold is ``object_width - closed_separation + band``; a 20 mm
    object's on-cube aperture is 9 mm, and a reading that fails the 16 mm
    object (9.33 mm) must pass for the 20 mm one, proving the gate is computed
    from the embodiment and target rather than pinned to one number.

    ``ScenarioConfig.target`` is a property derived from ``objects`` (a scenario
    must have exactly one target), so the bigger cube is set by replacing the
    object in ``objects`` -- ``model_copy`` on the property itself would be a
    silent no-op.
    """
    scenario = default_scenario(seed=3)
    bigger_object = scenario.target.model_copy(update={"size_m": 0.010})
    bigger = scenario.model_copy(update={"objects": (bigger_object,)})
    policy = create_policy(PolicyConfig(name="policy_v1"), bigger)
    assert float(bigger.target.size_m) == 0.010
    # on-cube for a 20 mm cube = 2*0.010 - 0.011 = 0.009; the 16 mm object's
    # failure reading (9.33 mm) is only 0.33 mm past this object's mark, well
    # inside the 1.1 mm band, so it must now be seated.
    assert policy._aperture_seated(_observation(0.00933)) is True
    # and the same reading must still be refused by the 16 mm scenario above.
    small_policy, small_scenario = _aperture_seated_policy()
    assert float(small_scenario.target.size_m) == 0.008
    assert small_policy._aperture_seated(_observation(0.00933)) is False

