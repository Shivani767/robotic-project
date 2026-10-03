"""Tests for the Gazebo backend's logic that needs no Gazebo runtime.

The parts exercised here are the ones that would otherwise only be tested by
running an episode: action decoding, the joint-mapping contract, capability
reporting, and the depth-perception maths fed a synthetic point cloud. A
backend that only worked when someone watched a real episode would be untested
on every machine that is not the container, which is most of them.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from xml.etree import ElementTree

import numpy as np
import pytest

from robot_ci.core.errors import BackendUnavailableError
from robot_ci.models.state import Action, ActionType
from robot_ci.scenarios import default_scenario, set_by_path
from robot_ci.simulation import SimulatorBackend, create_backend, probe_backend
from robot_ci.simulation.gazebo import ARM_JOINTS, FINGER_JOINTS, GazeboBackend

URDF = Path(__file__).resolve().parents[2] / "gazebo" / "models" / "panda" / "panda.urdf"


# ------------------------------------------------------------- registration
def test_gazebo_is_registered_and_resolves_to_the_real_class() -> None:
    """The registry's gazebo entry must now import, not raise."""
    backend = create_backend("gazebo")
    try:
        assert isinstance(backend, GazeboBackend)
        assert isinstance(backend, SimulatorBackend)
    finally:
        backend.close_all()


def test_probe_reports_constructible_even_without_a_gazebo_runtime() -> None:
    """Constructible and runnable are different facts and must not be merged.

    ``GazeboBackend`` constructs anywhere. Only running it needs Gazebo. A
    machine without Gazebo must still be able to see the backend's capabilities,
    which is what makes ``robot-ci validate`` useful before a container exists.
    """
    result = probe_backend("gazebo")
    assert result["available"] is True
    assert result["capabilities"] is not None
    if not result["usable"]:
        assert result["reason"]


def test_unknown_options_are_rejected_rather_than_ignored() -> None:
    """A silently ignored option is a configuration error that surfaces later."""
    with pytest.raises(BackendUnavailableError):
        GazeboBackend(nonexistent_option=1)


def test_capabilities_describe_real_gazebo_not_a_wish_list() -> None:
    """The capability flags are what let the engine refuse bad configurations."""
    caps = GazeboBackend().capabilities()
    assert caps.physics and caps.collision, "Gazebo runs contact dynamics"
    assert caps.rgb and caps.depth, "the generated world has an rgbd_camera"
    assert caps.segmentation is False, "no segmentation sensor is generated"
    assert caps.requires_gpu is False, "the demo runs on CPU-only containers"
    assert caps.deterministic is False, "ODE integrates against wall-clock"


# ------------------------------------------------------------ action decode
def test_joint_position_action_decodes_to_targets_and_aperture() -> None:
    joints, width = GazeboBackend._decode(
        Action.joint_positions([0.1, 0.2, 0.3, -1.5, 0.4, 0.5, 0.6])
    )
    assert joints == [pytest.approx(v) for v in (0.1, 0.2, 0.3, -1.5, 0.4, 0.5, 0.6)]
    assert width is None, "no aperture was commanded, so none is invented"


def test_arm_and_gripper_action_carries_both_channels() -> None:
    action = Action(
        type=ActionType.ARM_AND_GRIPPER,
        joint_position=(0.0,) * 7,
        gripper_width_m=0.04,
    )
    joints, width = GazeboBackend._decode(action)
    assert len(joints) == 7
    assert width == pytest.approx(0.04)


def test_gripper_only_action_leaves_arm_targets_to_the_backend() -> None:
    joints, width = GazeboBackend._decode(Action(type=ActionType.GRIPPER, gripper_width_m=0.02))
    assert joints == []
    assert width == pytest.approx(0.02)


def test_an_action_the_backend_cannot_execute_is_refused_by_name() -> None:
    """Cartesian deltas need a Jacobian this backend does not provide."""
    action = Action(type=ActionType.CARTESIAN_DELTA, cartesian_delta=(0.0,) * 6)
    with pytest.raises(Exception) as excinfo:
        GazeboBackend._decode(action)
    assert "cartesian_delta" in str(excinfo.value)


# ------------------------------------------------------- joint name contract
def test_arm_joint_names_match_the_scenario_robot() -> None:
    """The command mapping must be the robot the scenario actually describes.

    The scenario's arm order is what a policy's joint vector is indexed by, so
    a mismatch here would silently command the wrong joints rather than fail.
    """
    assert list(ARM_JOINTS) == list(default_scenario().robot.all_joint_names[:7])
    assert list(FINGER_JOINTS) == list(default_scenario().robot.all_joint_names[7:])


# --------------------------------------------------------------- perception
def test_largest_cluster_picks_the_bigger_of_two_blobs() -> None:
    """Perception must choose the object, not a stray smaller point."""
    small = np.array([[0.0, 0.0, 0.02], [0.01, 0.0, 0.02], [0.0, 0.01, 0.02]])
    big = np.array([[0.2, 0.2, 0.03]] * 40)
    cluster = GazeboBackend._largest_cluster(np.vstack([small, big]))
    assert cluster.shape[0] == 40
    assert cluster[:, 0].mean() == pytest.approx(0.2)


def test_perception_reports_nothing_when_there_is_no_depth() -> None:
    """Silence must produce no belief, not a default one.

    An object reported when nothing was seen would be a fabricated target for
    the policy to aim at, which is the exact failure the ground-truth boundary
    in :class:`Observation` exists to prevent.
    """
    backend = GazeboBackend()
    depth = type("D", (), {"latest": lambda _self: (None, None, 0, 0)})()
    backend._episodes["fake"] = {"depth": depth}
    handle = type(
        "H",
        (),
        {
            "episode_id": "fake",
            "scenario": default_scenario(),
            "step_index": 0,
            "sim_time_s": 0.0,
        },
    )()
    try:
        assert backend._detect_objects(handle) == ()
    finally:
        backend._episodes.clear()


def test_camera_geometry_is_read_from_the_same_helper_that_writes_the_sdf() -> None:
    """Perception must not drift from the pose Gazebo was actually given."""
    from robot_ci.gazebo.world import _camera_pose

    scenario = default_scenario()
    eye, forward, right, up = GazeboBackend()._camera_geometry(scenario)
    x, y, z, _roll, _pitch, _yaw = (float(v) for v in _camera_pose(scenario).split())
    assert (eye[0], eye[1], eye[2]) == pytest.approx((x, y, z))
    # The basis must be orthonormal: a broken basis shears the back-projection
    # and quietly biases every belief the policy acts on.
    basis = np.vstack([forward, right, up])
    assert basis @ basis.T == pytest.approx(np.eye(3), abs=1e-9)


def test_back_projection_recovers_an_object_from_a_synthetic_frame() -> None:
    """A cube rendered at its true pixel must be believed at its true position.

    This is the regression test for the row-sign bug found against a real
    Gazebo frame: the row term was double-negated, mirroring every point
    vertically, and the cube at z = 0.02 m was believed at z = 0.197 m -- in
    mid-air. The synthetic depth image here is built from the *same* camera
    geometry the SDF uses (table plane via ray-cast, cube as a blob at its
    projected pixel), so a mirror-sign back-projection places the blob below
    the table and fails this test rather than passing a self-consistent pair
    of wrong formulas.
    """
    from robot_ci.gazebo.world import camera_intrinsics_from_scenario, object_pose_from_scenario

    scenario = default_scenario(seed=42)
    backend = GazeboBackend()
    eye, forward, right, up = backend._camera_geometry(scenario)
    width, height, fx, fy, cx, cy = camera_intrinsics_from_scenario(scenario)

    # Ray-cast the table plane (z = 0) for every pixel: the background of the
    # synthetic frame, exactly as the camera would measure it.
    u = np.arange(width, dtype=np.float64) + 0.5
    v = np.arange(height, dtype=np.float64) + 0.5
    du = (u - cx) / fx
    dv = (v - cy) / fy
    direction = (
        forward[None, None, :]
        - du[None, :, None] * right[None, None, :]
        - dv[:, None, None] * up[None, None, :]
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        table_depth = -eye[2] / direction[:, :, 2]
    table_depth = np.where(np.isfinite(table_depth) & (table_depth > 0), table_depth, 8.0)
    depth = table_depth.astype(np.float32)

    # Stamp the cube at its projected pixel, at the cube's own centre depth.
    cube = np.array(object_pose_from_scenario(scenario))
    local = cube - eye
    along = float(local @ forward)
    col = int(round(cx + (-(local @ right) / along) * fx))
    row = int(round(cy + (-(local @ up) / along) * fy))
    assert 3 <= row < height - 3 and 3 <= col < width - 3, "cube must be in frame"
    depth[row - 2 : row + 3, col - 2 : col + 3] = along

    class _FakeDepth:
        def latest(self) -> tuple[np.ndarray | None, None, int, int]:
            return depth, None, width, height

    backend._episodes["synthetic"] = {"depth": _FakeDepth()}
    handle = type(
        "H",
        (),
        {
            "episode_id": "synthetic",
            "scenario": scenario,
            "step_index": 0,
            "sim_time_s": 0.0,
        },
    )()
    try:
        beliefs = backend._detect_objects(handle)
    finally:
        backend._episodes.clear()
    assert len(beliefs) == 1, "the stamped cube must be detected as exactly one object"
    believed = beliefs[0].pose.position.array
    assert believed == pytest.approx(cube, abs=0.04), (
        f"belief {believed} must be at the cube {cube}; a mirrored row sign "
        "puts the belief on the other side of the optical axis"
    )
def test_support_plane_ignores_what_stands_on_the_table() -> None:
    """An object and a distractor must not tilt the surface estimate.

    This is the measured defect in miniature: a least-squares fit over every
    point returned ``|slope| = 0.151`` on a table that is genuinely flat, which
    selected half the tabletop as "the object". The support-plane fit must
    recover the flat table even when most outliers stand above it.
    """
    rng = np.random.default_rng(0)
    # A flat table at z = 0 across the workspace, sampled on a fine grid.
    xs = np.linspace(-0.3, 0.3, 41)
    ys = np.linspace(-0.2, 0.4, 41)
    grid_x, grid_y = np.meshgrid(xs, ys)
    table = np.column_stack([grid_x.ravel(), grid_y.ravel(), np.zeros(grid_x.size)])
    # A 40 mm cube standing on it, plus a large tilted distractor above.
    cube = np.column_stack(
        [
            rng.uniform(-0.12, -0.08, 60),
            rng.uniform(0.21, 0.25, 60),
            rng.uniform(0.0, 0.04, 60),
        ]
    )
    link = np.column_stack(
        [
            rng.uniform(0.2, 0.43, 400),
            rng.uniform(-0.12, 0.10, 400),
            rng.uniform(0.06, 0.20, 400),
        ]
    )
    cloud = np.vstack([table, cube, link])
    coefficients = GazeboBackend._support_plane(cloud)
    assert abs(coefficients[0]) < 0.01
    assert abs(coefficients[1]) < 0.01
    assert coefficients[2] == pytest.approx(0.0, abs=0.005)


def test_clusters_come_back_with_usable_indices() -> None:
    """Every group is returned, and its indices select it."""
    near = np.array([[0.0, 0.0, 0.03 + 0.001 * i] for i in range(12)])
    far = np.array([[0.3, 0.0, 0.12 + 0.001 * i] for i in range(30)])
    cloud = np.vstack([near, far])
    groups = GazeboBackend._clusters(cloud)
    assert len(groups) == 2
    assert groups[0].size == 30  # largest first
    assert cloud[groups[0]][:, 0].mean() == pytest.approx(0.3)
    assert groups[1].size == 12
    assert cloud[groups[1]][:, 0].mean() == pytest.approx(0.0)


def test_detector_prefers_the_cube_over_a_bigger_arm_link() -> None:
    """The graspable object wins even when the robot's arm is the bigger cloud.

    A synthetic frame carries a stamped cube *and* a large distractor floating
    above the table. The detector must believe the cube. This is the regression
    test for the failure that reported the arm at 492 mm from the truth on a
    real frame.
    """
    from robot_ci.gazebo.world import camera_intrinsics_from_scenario, object_pose_from_scenario

    scenario = default_scenario(seed=42)
    backend = GazeboBackend()
    eye, forward, right, up = backend._camera_geometry(scenario)
    width, height, fx, fy, cx, cy = camera_intrinsics_from_scenario(scenario)

    # The same flat-table background the back-projection test trusts.
    u = np.arange(width, dtype=np.float64) + 0.5
    v = np.arange(height, dtype=np.float64) + 0.5
    du = (u - cx) / fx
    dv = (v - cy) / fy
    direction = (
        forward[None, None, :]
        - du[None, :, None] * right[None, None, :]
        - dv[:, None, None] * up[None, None, :]
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        table_depth = -eye[2] / direction[:, :, 2]
    table_depth = np.where(np.isfinite(table_depth) & (table_depth > 0), table_depth, 8.0)
    depth = table_depth.astype(np.float32)

    # The real cube, stamped at its true pixel at the cube's own centre depth.
    cube = np.array(object_pose_from_scenario(scenario))
    local = cube - eye
    along = float(local @ forward)
    col = int(round(cx + (-(local @ right) / along) * fx))
    row = int(round(cy + (-(local @ up) / along) * fy))
    depth[row - 2 : row + 3, col - 2 : col + 3] = along

    # A distractor bigger than the cube and floating above the table -- the
    # shape an arm link makes in the depth image.
    link_depth = along * 0.75
    depth[85:112, 100:150] = link_depth

    class _FakeDepth:
        def latest(self) -> tuple[np.ndarray | None, None, int, int]:
            return depth, None, width, height

    backend._episodes["synthetic"] = {"depth": _FakeDepth()}
    handle = type(
        "H",
        (),
        {
            "episode_id": "synthetic",
            "scenario": scenario,
            "step_index": 0,
            "sim_time_s": 0.0,
        },
    )()
    try:
        beliefs = backend._detect_objects(handle)
    finally:
        backend._episodes.clear()
    assert len(beliefs) == 1, "exactly one graspable object must be believed"
    believed = beliefs[0].pose.position.array
    assert believed == pytest.approx(cube, abs=0.04), (
        f"belief {believed} must be at the cube {cube}, not at the distractor"
    )
    assert believed[2] < 0.10, "the cube sits on the table; believing it in mid-air is the bug"


# ------------------------------------------------------------- IK provider
def test_urdf_jacobian_fk_matches_the_recording_renderer() -> None:
    """The IK provider and the recorded 3-D view must describe the same robot.

    The provider's ``forward`` returns the **tool centre point**, not
    ``panda_link7``: it is the frame the policy solves IK for, and the frame
    MuJoCo's injected ``tcp`` site sits in. The renderer is an independent
    implementation of the same URDF, so it is what the TCP is checked against:
    the TCP must be rigidly attached to the hand, ``TCP_SITE_IN_HAND_M`` beyond
    the hand frame and nothing else.
    """
    from robot_ci.gazebo.kinematics import TCP_SITE_IN_HAND_M, UrdfJacobianProvider
    from robot_ci.recording.render3d import forward_kinematics, load_kinematics

    provider = UrdfJacobianProvider(URDF)
    kin = load_kinematics(URDF)
    home = np.asarray(default_scenario().robot.home_position[:7], dtype=np.float64)
    for q in (home, np.zeros(7), home * 0.5):
        full = np.concatenate([q, np.zeros(len(kin.joints) - 7)])
        position, rotation = provider.forward(q)
        frames = forward_kinematics(kin, full)
        # The wrist-to-hand link is fixed at 0.107 m in this URDF.
        assert np.linalg.norm(frames["panda_hand"] - frames["panda_link7"]) == pytest.approx(
            0.107, abs=1e-9
        )
        # The TCP is exactly the site offset beyond the hand, for every
        # configuration. A provider that stopped at the wrist was 0.2104 m out.
        assert np.linalg.norm(position - frames["panda_hand"]) == pytest.approx(
            TCP_SITE_IN_HAND_M, abs=1e-9
        )
        # A rotation matrix must stay orthonormal through the cached chain.
        assert rotation @ rotation.T == pytest.approx(np.eye(3), abs=1e-9)


def test_tcp_constant_matches_the_site_mujoco_injects() -> None:
    """The Gazebo tool frame and MuJoCo's ``tcp`` site must be the same frame.

    The two backends are compared against each other, so the point they command
    has to be the same point. MuJoCo injects the site with a literal position;
    if the two numbers drift apart the policy quietly drives different parts of
    the robot in the two simulators.
    """
    from robot_ci.gazebo.kinematics import TCP_SITE_IN_HAND_M
    from robot_ci.simulation.mujoco_env import _inject_tcp_site

    patched = _inject_tcp_site('<body name="hand"><geom name="g"/></body>')
    assert f'pos="0 0 {TCP_SITE_IN_HAND_M}"' in patched, (
        "MuJoCo's injected tcp site and the Gazebo URDF provider's TCP must be "
        f"the same offset; MuJoCo injects {patched!r}"
    )


def test_urdf_gripper_geometry_matches_the_robot_config() -> None:
    """The jaw Gazebo loads must be the jaw the config says it is.

    ``RobotConfig`` derives the grasp aim and the feasibility check from
    ``pad_half_height_m``, ``pad_offset_from_tcp_m`` and
    ``gripper_closed_separation_m``, measured from the MuJoCo model. The Gazebo
    URDF is hand-written primitive geometry, so those three quantities are
    recomputed here from the URDF itself and pinned against the config.
    """
    from robot_ci.gazebo.kinematics import TCP_SITE_IN_HAND_M

    robot = default_scenario().robot
    root = ElementTree.parse(URDF).getroot()
    links = {link.get("name"): link for link in root.findall("link")}
    joints = {joint.get("name"): joint for joint in root.findall("joint")}

    axes = {
        name: [float(v) for v in joints[name].find("axis").get("xyz").split()]
        for name in ("left_finger", "right_finger")
    }
    assert axes["left_finger"] == [0.0, 1.0, 0.0]
    assert axes["right_finger"] == [0.0, -1.0, 0.0], "the jaws must be mirror images"

    centres: dict[str, tuple[float, float, float]] = {}
    # Finger-joint origins in the hand frame: these are the points the
    # prismatic axes pass through, and the closed separation is measured from
    # them.
    joint_origin = {
        link_name: [
            float(v) for v in joints[joint_name].find("origin").get("xyz").split()
        ]
        for link_name, joint_name in (
            ("panda_left_finger", "left_finger"),
            ("panda_right_finger", "right_finger"),
        )
    }
    half_height = 0.0
    # Pad-centre y in the hand frame, measured through the fixture: finger
    # joint y + prismatic displacement at q = 0 + pad y in the link.
    centres = {
        link_name: joint_origin[link_name][1]
        + float(links[link_name].find("collision/origin").get("xyz").split()[1])
        for link_name in ("panda_left_finger", "panda_right_finger")
    }
    for link_name in ("panda_left_finger", "panda_right_finger"):
        box = links[link_name].find("collision/geometry/box")
        size = [float(v) for v in box.get("size").split()]
        assert half_height in (0.0, size[2] / 2.0)
        half_height = size[2] / 2.0
    # Pad-centre z in the hand frame: finger joint z + pad z in the link.
    # (0.0445 + 0.0584 = 0.1029, 0.5 mm behind the 0.1034 TCP -- exactly the
    # measured pad_offset_from_tcp_m the shared task uses as its aim offset.)
    pad_centre_hand_z = {
        link_name: joint_origin[link_name][2]
        + float(links[link_name].find("collision/origin").get("xyz").split()[2])
        for link_name in ("panda_left_finger", "panda_right_finger")
    }

    # Pad half-height, measured from the URDF's own collision boxes.
    assert half_height == pytest.approx(float(robot.pad_half_height_m), abs=1e-6)
    # Closed separation: the inner faces at q = 0. The pads' hand-frame y
    # origins place the pad centres at y = +/-0.013 (measured), and each pad
    # extends thickness/2 inboard, so the faces sit at +/-0.0055 and the
    # separation is 0.011 m. The pad centres are asserted symmetric about the
    # hand axis first: a test that asserts on a quantity it never derived is
    # a tautology.
    pad_half = 0.015 / 2.0
    left_face = centres["panda_left_finger"] - pad_half
    right_face = centres["panda_right_finger"] + pad_half
    assert left_face == pytest.approx(-right_face, abs=1e-9), (
        "the jaw must be symmetric about the hand axis; "
        f"faces at {left_face:+.4f} and {right_face:+.4f}"
    )
    separation = left_face - right_face
    assert separation == pytest.approx(float(robot.gripper_closed_separation_m), abs=1e-4)
    # Pad centres in the hand frame are measured through the same fixture, so
    # the composition must agree with the raw recomputation below, and must sit
    # ``pad_offset_from_tcp_m`` behind the TCP -- the aim offset the task adds.
    tcp_behind_pad_left = TCP_SITE_IN_HAND_M - pad_centre_hand_z["panda_left_finger"]
    tcp_behind_pad_right = TCP_SITE_IN_HAND_M - pad_centre_hand_z["panda_right_finger"]
    assert tcp_behind_pad_left == pytest.approx(tcp_behind_pad_right, abs=1e-9)
    assert tcp_behind_pad_left == pytest.approx(
        float(robot.pad_offset_from_tcp_m), abs=1e-4
    )


def test_urdf_jacobian_position_rows_match_central_differences() -> None:
    """An independent differencing scheme must agree with the provider's."""
    from robot_ci.gazebo.kinematics import UrdfJacobianProvider

    provider = UrdfJacobianProvider(URDF)
    q = np.asarray(default_scenario().robot.home_position[:7], dtype=np.float64)
    jac = provider.jacobian(q)
    assert jac.shape == (6, 7)
    assert np.all(np.isfinite(jac))
    step = 1e-4
    for column in range(7):
        delta = np.zeros(7)
        delta[column] = step
        plus, _ = provider.forward(q + delta)
        minus, _ = provider.forward(q - delta)
        central = (plus - minus) / (2 * step)
        assert jac[:3, column] == pytest.approx(central, abs=1e-4)


# ------------------------------------------------------------------ scenario
@pytest.mark.skipif(
    shutil.which("gz") is None,
    reason=(
        "generating a world needs Gazebo's own URDF converter ('gz sdf -p'), "
        "which only exists in the container. Run in Docker: "
        "docker compose exec gazebo-demo pytest tests/unit"
    ),
)
def test_generated_world_carries_a_nine_joint_robot() -> None:
    """The backend's own world must contain a commandable robot."""
    from robot_ci.gazebo.world import world_sdf

    sdf = world_sdf(default_scenario(), robot_urdf=str(URDF))
    assert sdf.count("JointPositionController") == 9
    assert "JointStatePublisher" in sdf


@pytest.mark.skipif(
    shutil.which("gz") is None,
    reason=(
        "generating a world needs Gazebo's own URDF converter ('gz sdf -p'), "
        "which only exists in the container. Run in Docker: "
        "docker compose exec gazebo-demo pytest tests/unit"
    ),
)
def test_depth_noise_condition_changes_the_generated_world() -> None:
    """0 mm and 8 mm must be different worlds, not one relabelled."""
    from robot_ci.gazebo.world import world_sdf

    nominal = world_sdf(default_scenario(), robot_urdf=str(URDF))
    noisy = world_sdf(
        set_by_path(default_scenario(), "sensors.depth.sigma_mm", 8.0),
        robot_urdf=str(URDF),
    )
    assert nominal != noisy
