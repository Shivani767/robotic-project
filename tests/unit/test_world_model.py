"""Tests for the world model's reconstructibility and staleness handling.

The properties tested here are the ones that justify a world model existing at
all. A dictionary of beliefs would pass most of them, which is the point: the
tests are written so that a dict fails them, because the value is in the
guarantees rather than the storage.
"""

from __future__ import annotations

import numpy as np
import pytest

from robot_ci.models.geometry import Covariance, Pose, Vec3
from robot_ci.models.state import GroundTruthObject, ObjectState, Source
from robot_ci.world_model import TaskState, WorldModel

TRUTH = np.array([0.30, 0.10, 0.008])


def _belief(step: int, *, offset: float = 0.0, sigma: float = 0.004) -> ObjectState:
    """One object belief, timestamped at ``step``."""
    return ObjectState(
        id="cube",
        pose=Pose(position=Vec3.of(*(TRUTH + np.array([offset, 0.0, 0.0])))),
        covariance=Covariance.isotropic(sigma, 6),
        confidence=0.85,
        source=Source.DEPTH,
        timestamp_s=step * 0.05,
        last_seen_s=step * 0.05,
    )


def test_an_episode_can_be_reconstructed_from_its_snapshots() -> None:
    """Initial state, per-step history and final state must all be reachable."""
    model = WorldModel()
    for step in range(12):
        model.observe(
            step=step,
            timestamp_s=step * 0.05,
            objects={"cube": _belief(step, offset=0.001 * np.sin(step))},
            task=TaskState(name="pick_and_place", phase="approach", waypoint_index=step % 3),
        )
    assert model.initial() is not None and model.initial().step == 0
    assert model.final() is not None and model.final().step == 11
    assert model.snapshot_at(5) is not None
    assert model.snapshot_at(5).task.waypoint_index == 2
    assert model.snapshot_at(99) is None, "an unrecorded step must not be invented"


def test_a_snapshot_does_not_change_when_the_belief_moves_on() -> None:
    """Snapshots are the record of what was believed *at the time*.

    A model that mutated its history in place could not answer "what did the
    policy think at step 4", which is the question a replayable artifact exists
    to answer.
    """
    model = WorldModel()
    model.observe(step=0, timestamp_s=0.0, objects={"cube": _belief(0, offset=0.0)})
    first = model.initial()
    assert first is not None
    recorded = np.array(first.object("cube").pose.position.array)
    for step in range(1, 10):
        model.observe(step=step, timestamp_s=step * 0.05, objects={"cube": _belief(step, offset=0.05)})
    assert np.allclose(np.array(model.initial().object("cube").pose.position.array), recorded), (
        "an early snapshot was rewritten by later observations"
    )


def test_staleness_is_visible_rather_than_silent() -> None:
    """An object that stopped being seen must stop looking current.

    Copying a belief forward without advancing its timestamp would make a stale
    estimate indistinguishable from a fresh one, and a policy would act on a
    belief the world no longer supports.
    """
    model = WorldModel()
    for step in range(10):
        model.observe(step=step, timestamp_s=step * 0.05, objects={"cube": _belief(step)})
    assert model.visible_objects(max_age_s=0.5) != {}, "0.5 s old is still current"
    # The clock only advances when the model is told the time, so staleness has
    # to be produced by a step in which the object is not reported. That is
    # exactly the situation staleness exists for.
    model.observe(step=10, timestamp_s=10.0, objects={})
    assert model.visible_objects(max_age_s=2.0) == {}, (
        "an object not seen for 9.5 s must not be reported as current"
    )


def test_a_disappearing_object_stays_in_the_belief_set() -> None:
    """Losing sight of an object is different from believing it gone.

    Dropping it silently would make a later re-appearance look like a new
    object, and a world model that forgets cannot answer "was it ever there?".
    """
    model = WorldModel()
    model.observe(step=0, timestamp_s=0.0, objects={"cube": _belief(0)})
    model.observe(step=1, timestamp_s=3.0, objects={})
    assert "cube" in model.state.objects, "the belief must survive losing sight"
    assert model.visible_objects(max_age_s=0.5) == {}, "but it must not look current"
    # And the belief it kept is the one from when it was last actually seen.
    assert model.state.objects["cube"].last_seen_s == 0.0


def test_ground_truth_is_never_folded_into_the_beliefs() -> None:
    """Beliefs and truth must stay separable.

    A model that read the simulator's state would be indistinguishable from one
    that works, and that difference is exactly what a policy consumes.
    """
    model = WorldModel()
    model.observe(step=0, timestamp_s=0.0, objects={"cube": _belief(0, offset=0.01)})
    model.note_ground_truth(
        {"cube": GroundTruthObject(id="cube", pose=Pose(position=Vec3.of(*TRUTH)))}
    )
    believed = np.array(model.state.objects["cube"].pose.position.array)
    assert not np.allclose(believed, TRUTH), "the belief was overwritten by ground truth"
    # The error series is the sanctioned way to compare the two.
    errors = model.belief_error("cube")
    assert errors and errors[0] == pytest.approx(0.01, abs=1e-6)


def test_belief_error_reports_a_series_not_a_single_number() -> None:
    """A per-step series distinguishes a briefly-wild filter from a mediocre one."""
    model = WorldModel()
    for step in range(6):
        model.observe(step=step, timestamp_s=step * 0.05, objects={"cube": _belief(step)})
        model.note_ground_truth(
            {"cube": GroundTruthObject(id="cube", pose=Pose(position=Vec3.of(*TRUTH)))}
        )
    errors = model.belief_error("cube")
    assert len(errors) == 6
    assert all(value == pytest.approx(0.0, abs=1e-6) for value in errors)


def test_snapshot_retention_is_bounded() -> None:
    """A long episode must not hold every step in memory.

    The limit keeps the tail, because the tail is what a diagnosis reads; the
    count is checked rather than assumed so the bound cannot silently stop
    working.
    """
    model = WorldModel(max_snapshots=20)
    for step in range(100):
        model.observe(step=step, timestamp_s=step * 0.05, objects={"cube": _belief(step)})
    assert len(model.state.snapshots) == 20
    assert model.state.snapshots[-1].step == 99, "the newest snapshots must be the ones kept"


def test_sensor_degraded_is_derived_from_the_beliefs() -> None:
    """A degraded sensor state must have numbers behind it.

    Asserting it would make the flag a decoration; deriving it from the fused
    sigma means a reader can check the claim against the object estimates.
    """
    model = WorldModel()
    model.observe(step=0, timestamp_s=0.0, objects={"cube": _belief(0, sigma=0.002)})
    assert not model.state.sensors.degraded
    model.observe(step=1, timestamp_s=0.05, objects={"cube": _belief(1, sigma=0.05)})
    assert model.state.sensors.degraded, "50 mm of uncertainty is not a healthy sensor"
