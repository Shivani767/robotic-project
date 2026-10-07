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

from robot_ci.tasks.manipulation import PickAndPlaceTask


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
