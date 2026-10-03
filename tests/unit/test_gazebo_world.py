"""Tests for the Gazebo world generator.

These run with **no Gazebo installed**, which is the point: the questions worth
asking about the generated world -- is it well-formed, does it carry the
scenario's noise, is placement reproducible, is `scenario_id` left alone -- are
all answerable from the SDF text, and answering them on every machine is better
than answering them once in a container.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

import pytest

from robot_ci.gazebo.world import (
    GAZEBO_WORLD_NAME,
    camera_intrinsics_from_scenario,
    object_pose_from_scenario,
    unsupported_conditions,
    world_sdf,
)
from robot_ci.models.scenario import ScenarioConfig
from robot_ci.scenarios import default_scenario, set_by_path


# -- structure ---------------------------------------------------------------
def test_world_is_well_formed_sdf() -> None:
    """The generated document parses as XML and names the expected world."""
    root = ET.fromstring(world_sdf(default_scenario()))
    assert root.tag == "sdf"
    assert root.find("world").get("name") == GAZEBO_WORLD_NAME


def test_world_contains_the_pick_and_place_parts() -> None:
    """Table, robot, object, target and camera are all present."""
    root = ET.fromstring(world_sdf(default_scenario()))
    names = {m.get("name") for m in root.iter("model")}
    assert {"table", "target_cube", "target_zone", "camera_rig"} <= names
    assert root.find(".//sensor").get("name") == "robot_ci_camera"


def test_physics_timestep_matches_the_control_rate() -> None:
    """A 1 ms step is the documented control timestep, not a default."""
    root = ET.fromstring(world_sdf(default_scenario()))
    assert float(root.find(".//max_step_size").text) == pytest.approx(0.001, abs=1e-9)


# -- the camera, which is where a unit mistake would hide -------------------
def test_camera_fov_is_in_radians_and_within_gazebo_limits() -> None:
    """Gazebo rejects a FOV above 2*pi, so this must not be degrees.

    An earlier revision emitted 59.49 (degrees) where 1.038 (radians) was meant.
    The SDF schema accepted it and the Gazebo server refused to load the world,
    which is how the mistake was found -- and is why this is a test rather than
    a comment.
    """
    import math

    root = ET.fromstring(world_sdf(default_scenario()))
    fov = float(root.find(".//horizontal_fov").text)
    assert 0.0 < fov <= 2.0 * math.pi

    scenario = default_scenario()
    width, _height, fx, _fy, _cx, _cy = camera_intrinsics_from_scenario(scenario)
    assert fov == pytest.approx(2.0 * math.atan(width / (2.0 * fx)), abs=1e-6)


def test_camera_resolution_matches_the_scenario() -> None:
    """Gazebo is told the scenario's own resolution, not a default one."""
    scenario = default_scenario()
    root = ET.fromstring(world_sdf(scenario))
    image = root.find(".//camera/image")
    assert int(image.find("width").text) == scenario.resolution[1]
    assert int(image.find("height").text) == scenario.resolution[0]


def test_depth_noise_reaches_the_gazebo_sensor() -> None:
    """8 mm of depth noise must appear in the SDF, not be dropped.

    A backend that silently ignored the corruption would produce a success rate
    that looks comparable with the MuJoCo one and is not.
    """
    nominal = world_sdf(default_scenario()).replace(" ", "")
    noisy = world_sdf(set_by_path(default_scenario(), "sensors.depth.sigma_mm", 8.0)).replace(
        " ", ""
    )
    assert "<stddev>0.000000</stddev>" in nominal
    assert "<stddev>0.008000</stddev>" in noisy


def test_depth_clipping_comes_from_the_scenario() -> None:
    """Near/far planes match the depth config, so the range is comparable."""
    scenario = default_scenario()
    root = ET.fromstring(world_sdf(scenario))
    clip = root.find(".//camera/clip")
    assert float(clip.find("near").text) == pytest.approx(scenario.sensors.depth.min_range_m)
    assert float(clip.find("far").text) == pytest.approx(scenario.sensors.depth.max_range_m)


# -- determinism -------------------------------------------------------------
def test_object_placement_is_reproducible() -> None:
    """The same scenario yields the same placement, every time."""
    scenario = default_scenario()
    assert object_pose_from_scenario(scenario) == object_pose_from_scenario(scenario)


def test_object_placement_follows_the_scenario_seed() -> None:
    """A different seed moves the object, so the seed is actually used."""
    base = default_scenario()
    moved = base.model_copy(update={"seed": base.seed + 1})
    assert object_pose_from_scenario(base) != object_pose_from_scenario(moved)


def test_object_placement_follows_the_scenario_id() -> None:
    """`scenario_id` seeds the placement, exactly as it does for MuJoCo.

    This is why the ROS 2 renderer and the Gazebo generator both refuse to rename
    it: the identifier is an input to the physics, not a label. Renaming a
    condition to describe a generated artifact would move the target object.
    """
    base = default_scenario()
    renamed = base.model_copy(update={"scenario_id": "nominal__depth_noise_mm=8"})
    assert object_pose_from_scenario(base) != object_pose_from_scenario(renamed)


def test_generated_sdf_records_the_scenario_id_it_came_from() -> None:
    """Provenance is written into the file, read from the scenario unchanged."""
    scenario = set_by_path(default_scenario(), "sensors.depth.sigma_mm", 8.0)
    sdf = world_sdf(scenario)
    assert f"scenario_id={scenario.scenario_id}" in sdf
    assert f"seed={scenario.seed}" in sdf


# -- honesty about gaps ------------------------------------------------------
def test_nominal_and_eight_mm_report_no_gaps() -> None:
    """The two conditions this project runs in are fully reproducible."""
    assert unsupported_conditions(default_scenario()) == []
    noisy = set_by_path(default_scenario(), "sensors.depth.sigma_mm", 8.0)
    assert unsupported_conditions(noisy) == []


def test_unreproducible_conditions_are_named_not_ignored() -> None:
    """A feature Gazebo cannot reproduce is listed by name, never dropped."""
    scenario = set_by_path(default_scenario(), "sensors.depth.dropout_percent", 12.0)
    gaps = unsupported_conditions(scenario)
    assert any("dropout" in gap for gap in gaps)


def test_the_two_conditions_generate_different_sensor_noise() -> None:
    """Nominal and the counterexample differ in the SDF, not only in metadata."""
    nominal = world_sdf(default_scenario())
    counterexample = world_sdf(set_by_path(default_scenario(), "sensors.depth.sigma_mm", 8.0))
    assert nominal != counterexample


# -- camera aiming -----------------------------------------------------------
def _camera_ray(
    scenario: ScenarioConfig,
) -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...], tuple[float, ...]]:
    """Return the camera's ``(eye, forward, right, up)`` from the SDF pose.

    Gazebo cameras look along +X with +Z up, and the SDF pose applies
    ``Rz(yaw) * Ry(pitch) * Rx(roll)``. The full basis is returned, not just the
    forward vector, because a test that only checks "is it pointing roughly the
    right way" would pass a pose whose up-vector was mirrored, and a mirrored up
    vector silently mirrors the image.
    """
    import math
    from xml.etree import ElementTree as ET

    root = ET.fromstring(world_sdf(scenario))
    pose = next(model for model in root.iter("model") if model.get("name") == "camera_rig").find(
        "pose"
    )
    assert pose is not None and pose.text is not None
    x, y, z, _roll, pitch, yaw = (float(v) for v in pose.text.split())
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return (
        (x, y, z),
        (cy * cp, sy * cp, -sp),
        (-sy, cy, 0.0),
        (cy * sp, sy * sp, cp),
    )


def test_camera_aims_at_the_workspace_not_past_it() -> None:
    """The camera must see the table it exists to observe.

    Regression guard for a real failure: the pose was once hard-coded with pitch
    only and no yaw, so the ray travelled along y = -0.25 -- outside the table's
    y range -- and crossed z = 0 at x = 0.824, well beyond the table's x edge.
    Every frame came back exactly the background colour with all-inf depth, which
    reads as a renderer fault but was an unwatched camera.
    """
    import math

    scenario = default_scenario()
    eye, forward, right, up = _camera_ray(scenario)
    cx = float(scenario.environment.workspace_center[0])
    cy = float(scenario.environment.workspace_center[1])
    table_z = float(scenario.environment.table_height_m)

    assert forward[2] < 0, "the camera must be pitched down, not level"

    # Horizontal field of view from the scenario's own intrinsics, so the
    # frustum test uses the same numbers the sensor is configured with.
    width, _height, fx, _fy, _cx, _cy = camera_intrinsics_from_scenario(scenario)
    half_fov = 2.0 * math.atan(width / (2.0 * fx)) / 2.0
    cos_limit = math.cos(half_fov)

    def in_frustum(point: tuple[float, ...]) -> bool:
        dx, dy, dz = (point[i] - eye[i] for i in range(3))
        length = math.sqrt(dx * dx + dy * dy + dz * dz)
        dot = forward[0] * dx + forward[1] * dy + forward[2] * dz
        return dot / length > cos_limit

    # Three things the camera exists to show. The object check is the strict
    # one: the camera is aimed at the table, so the object standing on it must
    # be in frustum, and a small margin is right. The arm is above the top of
    # frame by design (see _camera_pose), so it is only checked for not being
    # behind the camera, which would mean a sign error in the pose maths.
    object_pose = tuple(float(v) for v in object_pose_from_scenario(scenario))
    required = {
        "table": (cx, cy, table_z),
        "object": object_pose,
    }
    for label, point in required.items():
        assert in_frustum(point), (
            f"{label} at {point} is outside the camera frustum "
            f"(eye={eye}, forward={forward}, half_fov={half_fov:.3f} rad)"
        )

    arm = (0.0, 0.0, table_z + 0.5)
    dx, dy, dz = (arm[i] - eye[i] for i in range(3))
    depth = sum(a * b for a, b in zip(forward, (dx, dy, dz), strict=True))
    assert depth > 0, "the arm is behind the camera, so the pose maths are wrong"

    # The table's far edge must be inside the image. When it is not, the camera
    # is aimed past the table and every depth pixel comes back as background,
    # which is exactly how this bug presented the first time.
    width_px, height_px, fx_, fy_, pcx, pcy = camera_intrinsics_from_scenario(scenario)
    # table_extent_m is a (x, y) half-extent pair, not a single number.
    half_x, half_y = (float(v) / 2.0 for v in scenario.environment.table_extent_m)
    for label, corner in {
        "near": (cx - half_x, cy - half_y, table_z),
        "far": (cx + half_x, cy + half_y, table_z),
        "left": (cx - half_x, cy + half_y, table_z),
        "right": (cx + half_x, cy - half_y, table_z),
    }.items():
        vx, vy, vz = (corner[i] - eye[i] for i in range(3))
        along = forward[0] * vx + forward[1] * vy + forward[2] * vz
        assert along > 0, f"table {label} corner is behind the camera"
        px = pcx - fx_ * ((right[0] * vx + right[1] * vy + right[2] * vz) / along)
        py = pcy - fy_ * ((up[0] * vx + up[1] * vy + up[2] * vz) / along)
        assert 0 <= px <= width_px, f"table {label} edge is off-frame (px={px:.0f})"
        assert 0 <= py <= height_px, f"table {label} edge is off-frame (py={py:.0f})"


def test_explicit_object_position_wins_over_the_seeded_draw() -> None:
    """An explicit ``objects[0].position`` must reach the generated world.

    The ``pose_perturbation`` family expresses object-position perturbation as
    an explicit position. MuJoCo honours it (``mujoco_env.py``); Gazebo used to
    ignore it and place the cube from the seed alone, which would have run a
    *different* condition under the perturbed condition's name.
    """
    from robot_ci.scenarios import get_family

    perturbed = get_family("pose_perturbation").build(5.0, seed=42)
    assert object_pose_from_scenario(perturbed) == pytest.approx((0.35, 0.0, 0.025))

    seeded = default_scenario(seed=42)
    explicit_free = object_pose_from_scenario(seeded)
    assert object_pose_from_scenario(seeded) == pytest.approx(explicit_free)


def test_every_material_sets_diffuse_so_surfaces_are_not_black() -> None:
    """A material with only <ambient> renders black under Gazebo's lights.

    Regression guard for the first rendered frame, in which the table came back
    pure black against the background. SDF defaults ``<diffuse>`` to
    ``0 0 0 1`` and diffuse is the term the scene lights multiply, so naming a
    colour as ambient only is enough to load and enough to see nothing.
    """
    root = ET.fromstring(world_sdf(default_scenario()))
    checked = 0
    for material in root.iter("material"):
        if material.find("ambient") is None:
            continue
        assert material.find("diffuse") is not None, (
            "material declares <ambient> but no <diffuse>; it will render black"
        )
        checked += 1
    # The three world visuals (table, object, zone) plus the Panda's own.
    assert checked >= 3, f"expected the world's own materials to be checked, saw {checked}"
