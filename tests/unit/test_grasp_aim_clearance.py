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

from robot_ci.models.scenario import PolicyConfig, ScenarioConfig
from robot_ci.models.state import GripperState, Observation, RobotState
from robot_ci.policies import create_policy
from robot_ci.policies.rule_based import PickAndPlacePolicy
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


def _aperture_seated_policy() -> tuple[PickAndPlacePolicy, ScenarioConfig]:
    scenario = default_scenario(seed=3)
    policy = create_policy(PolicyConfig(name="policy_v1"), scenario)
    assert isinstance(policy, PickAndPlacePolicy)
    return policy, scenario


def _grasp_waypoint_z(task: PickAndPlaceTask, scenario: object) -> float:
    from robot_ci.models.geometry import Pose, Vec3

    return task.waypoints(Pose.from_position(Vec3.of(0.0, 0.0, 0.008)))[1].z


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
    from robot_ci.models.geometry import Pose, Vec3
    from robot_ci.scenarios import default_scenario

    scenario = default_scenario(seed=3)
    task = PickAndPlaceTask(scenario)
    table_z = float(scenario.environment.table_height_m)
    pad_half = float(scenario.robot.pad_half_height_m)

    low = task.waypoints(Pose.from_position(Vec3.of(0.0, 0.0, 0.008)))[1].z
    assert low - pad_half >= table_z - 1e-9
    high = task.waypoints(Pose.from_position(Vec3.of(0.0, 0.0, 0.03)))[1].z
    assert abs(high - (0.03 + float(scenario.robot.pad_offset_from_tcp_m))) < 1e-9, (
        "tall-object aim must stay on the object centre, floor must not touch it"
    )

def test_policy_aim_floor_defaults_to_pure_derived_clearance() -> None:
    """The default floor is exactly the derived clearance, with no compensation.

    The live-Gazebo tracking deficit is a property of *that* backend (measured:
    commanded 0.0095 m, achieved 0.0080 m, five for five runs). Adding it
    unconditionally overshoots on MuJoCo, which executes commands accurately:
    the pads grip 1.5 mm high and placement orientation error grows past
    tolerance (commit d7026af, nominal seeds failing).
    """
    import numpy as np

    from robot_ci.models.scenario import PolicyConfig
    from robot_ci.policies import create_policy
    from robot_ci.policies.rule_based import PickAndPlacePolicy
    from robot_ci.scenarios import default_scenario

    scenario = default_scenario(seed=3)
    policy = create_policy(PolicyConfig(name="policy_v1"), scenario)
    assert isinstance(policy, PickAndPlacePolicy)
    table_z = float(scenario.environment.table_height_m)
    pad_half = float(scenario.robot.pad_half_height_m)
    expect = table_z + pad_half + 0.001
    wps = [np.array([0.0, 0.0, 0.1085]), np.array([0.0, 0.0, 0.0085])]
    policy._apply_aim_clearance_floor(wps)
    assert abs(float(wps[1][2]) - expect) < 1e-9, wps[1][2]
    # achieved pose (= command, on an accurate backend) clears the table
    assert float(wps[1][2]) - pad_half >= table_z - 1e-9


def test_policy_aim_floor_adds_configured_settle_shortfall() -> None:
    """A backend that measured a tracking deficit opts in via configuration.

    The Gazebo evidence scripts set ``settle_shortfall_m=0.0015``; the floor
    then commands clearance + shortfall so the *achieved* pose clears the
    table. The parameter is recorded in the config like every other policy
    hyperparameter, so an experiment can sweep it.
    """
    import numpy as np

    from robot_ci.models.scenario import PolicyConfig
    from robot_ci.policies import create_policy
    from robot_ci.policies.rule_based import PickAndPlacePolicy
    from robot_ci.scenarios import default_scenario

    scenario = default_scenario(seed=3)
    policy = create_policy(
        PolicyConfig(name="policy_v1", parameters={"settle_shortfall_m": 0.0015}),
        scenario,
    )
    assert isinstance(policy, PickAndPlacePolicy)
    table_z = float(scenario.environment.table_height_m)
    pad_half = float(scenario.robot.pad_half_height_m)
    expect = table_z + pad_half + 0.001 + 0.0015
    wps = [np.array([0.0, 0.0, 0.1085]), np.array([0.0, 0.0, 0.0085])]
    policy._apply_aim_clearance_floor(wps)
    assert abs(float(wps[1][2]) - expect) < 1e-9, wps[1][2]
    # achieved pose (= command minus the measured shortfall) still clears the table
    assert float(wps[1][2]) - 0.0015 - pad_half >= table_z - 1e-9


def test_policy_aim_floor_raises_carry_plane_by_the_same_amount() -> None:
    """Raising the grasp aim raises the carry waypoints, never the place pose.

    The pads close on the object at a higher point, so the object hangs lower
    in the grip by exactly the floor's ``raise_by``. The lift and transit
    waypoints (indices 2 and 3) are derived under the unreraised assumption, so
    they must rise by the same amount or the carried object travels below its
    clearance -- measured at seed 8: object peak 49.3 mm against the 50 mm
    lift criterion after a 1 mm floor raise, the whole margin. The place
    waypoint (index 4) stays where the task put it: the object is released
    from the place pose, so raising it would change placement, not protection.
    """
    import numpy as np

    from robot_ci.models.scenario import PolicyConfig
    from robot_ci.policies import create_policy
    from robot_ci.policies.rule_based import PickAndPlacePolicy
    from robot_ci.scenarios import default_scenario

    scenario = default_scenario(seed=3)
    policy = create_policy(PolicyConfig(name="policy_v1"), scenario)
    assert isinstance(policy, PickAndPlacePolicy)
    table_z = float(scenario.environment.table_height_m)
    pad_half = float(scenario.robot.pad_half_height_m)
    expect = table_z + pad_half + 0.001
    raise_by = expect - 0.0085
    wps = [
        np.array([0.0, 0.0, 0.1085]),
        np.array([0.0, 0.0, 0.0085]),
        np.array([0.0, 0.0, 0.058]),
        np.array([0.0, 0.0, 0.058]),
        np.array([0.0, 0.0, 0.0]),
    ]
    policy._apply_aim_clearance_floor(wps)
    assert abs(float(wps[1][2]) - expect) < 1e-9, wps[1][2]
    # the carry plane follows the raise exactly
    assert abs(float(wps[2][2]) - (0.058 + raise_by)) < 1e-9, wps[2][2]
    assert abs(float(wps[3][2]) - (0.058 + raise_by)) < 1e-9, wps[3][2]
    # the release pose is untouched
    assert abs(float(wps[4][2]) - 0.0) < 1e-9, wps[4][2]


def test_aperture_settled_refuses_a_still_travelling_jaw() -> None:
    """The lift gate refuses while the aperture is still moving.

    The live Gazebo failure mode this gate exists for: the close creeps at
    ~4 mm/s (~1.2 mm/tick at the 0.3 s control period), so a sequence of
    still-creeping readings must never count as settled, however each reading
    compares to any absolute on-cube mark.
    """
    policy, _scenario = _aperture_seated_policy()
    state: dict = {}
    # Creep: 1.2 mm per tick, the measured live rate. Alternate samples so at
    # least one windowed change is large on every call.
    readings = [0.0200, 0.0212, 0.0224, 0.0236, 0.0248, 0.0260]
    for reading in readings:
        assert policy._aperture_settled(_observation(reading), state) is False
    # It settles only once the readings stop changing.
    for _ in range(6):
        policy._aperture_settled(_observation(0.0116), state)
    assert policy._aperture_settled(_observation(0.0116), state) is True


def test_aperture_settled_accepts_any_settled_width_including_mujoco_on_cube() -> None:
    """A settled jaw passes regardless of where the backend's width sits.

    Measured: MuJoCo settles a successful 16 mm grasp at ``width_m`` = 11.6 mm
    (never at the 5 mm the old Gazebo-derived mark predicted), with
    steady-state drift ~1e-7 m/tick. The old absolute-mark gate never opened
    on that reading and every nominal MuJoCo episode stalled at the grasp
    (commit d7026af, 0/12 seeds, caught by the pinned-seed integration test).
    """
    policy, _scenario = _aperture_seated_policy()
    state: dict = {}
    # Settle at the MuJoCo on-cube reading with float-level jitter.
    for reading in (0.0116, 0.0116000001, 0.0116, 0.0115999999, 0.0116):
        policy._aperture_settled(_observation(reading), state)
    assert policy._aperture_settled(_observation(0.0116), state) is True


def test_aperture_settled_needs_a_full_window_of_stability() -> None:
    """Fewer than the required consecutive stable changes never settles.

    A bounce resets the window, so a single quiet reading after a jump cannot
    masquerade as convergence: the full run of stable changes must be present.
    """
    policy, _scenario = _aperture_seated_policy()
    state: dict = {}
    for _ in range(5):
        policy._aperture_settled(_observation(0.0116), state)
    assert policy._aperture_settled(_observation(0.0140), state) is False  # bounce
    assert policy._aperture_settled(_observation(0.0140), state) is False  # window filling
    assert policy._aperture_settled(_observation(0.0140), state) is False  # still filling
    assert policy._aperture_settled(_observation(0.0140), state) is True  # full window


def test_aperture_settled_resets_between_episodes() -> None:
    """A new episode's gate starts empty; old convergence cannot leak in."""
    from robot_ci.models.scenario import PolicyConfig

    policy, scenario = _aperture_seated_policy()
    config = PolicyConfig(name="policy_v1")
    policy.reset(config, 1, scenario)
    state = policy._state
    for _ in range(5):
        policy._aperture_settled(_observation(0.0116), state)
    assert policy._aperture_settled(_observation(0.0116), state) is True
    policy.reset(config, 2, scenario)
    assert policy._state["aperture_trace"] == []
    assert policy._aperture_settled(_observation(0.0116), policy._state) is False

