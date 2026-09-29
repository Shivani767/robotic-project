"""Unit tests for the bounded-contact gripper controller.

All run without MuJoCo: the controller is pure logic over a stiffness, a force
target and two measurements, so every claim about what it does with a request
can be checked arithmetically. The model-derived quantities it depends on
(stiffness, closed separation) are covered by the integration tests, which read
them from the compiled scene.
"""

from __future__ import annotations

import pytest

from robot_ci.control.gripper import GripperController
from robot_ci.models.robot import GripperControlMode

STIFFNESS = 800.0
CLOSED = 0.011
OPEN = 0.08
OBJECT = 0.016


def bounded(force_n: float = 1.0, **overrides: object) -> GripperController:
    params = {
        "mode": GripperControlMode.BOUNDED,
        "stiffness_n_per_m": STIFFNESS,
        "target_contact_force_n": force_n,
        "max_closure_speed_m_s": 1.0,
        "control_dt_s": 0.05,
        "open_width_m": OPEN,
        "closed_width_m": CLOSED,
        **overrides,
    }
    return GripperController(**params)  # type: ignore[arg-type]


def test_position_mode_passes_the_request_through():
    controller = GripperController(
        mode=GripperControlMode.POSITION, stiffness_n_per_m=STIFFNESS
    )
    controller.reset(contact_width_m=OBJECT)
    assert controller.step(requested_width_m=0.0002, measured_width_m=0.08) == pytest.approx(0.0002)
    assert controller.step(requested_width_m=0.08, measured_width_m=0.0002) == pytest.approx(0.08)


def test_bounded_mode_floors_the_close_at_force_over_stiffness():
    controller = bounded(force_n=1.0)
    controller.reset(contact_width_m=OBJECT)
    # 1.0 N / 800 N/m per finger, both sides: 16 mm - 2 * 1.25 mm = 13.5 mm.
    assert controller.engagement_limit_m == pytest.approx(1.0 / STIFFNESS)
    assert controller.step(requested_width_m=0.0002, measured_width_m=OPEN) == pytest.approx(0.0135)
    # A request above the floor is untouched.
    assert controller.step(requested_width_m=0.05, measured_width_m=OPEN) == pytest.approx(0.05)


def test_bounded_opening_is_unrestricted():
    controller = bounded()
    controller.reset(contact_width_m=OBJECT)
    assert controller.step(requested_width_m=OPEN, measured_width_m=0.0135) == pytest.approx(OPEN)


def test_bounded_mode_never_commands_below_the_mechanism():
    controller = bounded(force_n=100.0)
    controller.reset(contact_width_m=0.012)
    # Even a force that would demand the pads pass each other stops at the
    # measured closed separation.
    assert controller.step(requested_width_m=0.0, measured_width_m=OPEN) == pytest.approx(CLOSED)


def test_no_contact_width_means_the_request_stands():
    controller = bounded()
    controller.reset()
    assert controller.step(requested_width_m=0.0002, measured_width_m=OPEN) == pytest.approx(0.0002)


def test_opening_forgets_the_object():
    controller = bounded()
    controller.reset(contact_width_m=OBJECT)
    controller.command_open()
    assert controller.step(requested_width_m=0.0002, measured_width_m=OPEN) == pytest.approx(0.0002)
    assert controller.contact_detected is False


def test_contact_detection_is_geometric_not_tuned():
    controller = bounded(force_n=1.0)
    controller.reset(contact_width_m=OBJECT)
    controller.step(requested_width_m=0.0002, measured_width_m=OBJECT)
    assert controller.contact_detected is True
    controller.step(requested_width_m=0.0002, measured_width_m=OBJECT + 0.01)
    assert controller.contact_detected is False


def test_ramp_off_by_default_steps_fully():
    controller = bounded()
    assert controller.closure_ramp_enabled is False
    controller.reset(contact_width_m=OBJECT)
    out = controller.step(requested_width_m=0.0002, measured_width_m=OPEN)
    assert out == pytest.approx(0.0135), "default must reach the target in one step"
