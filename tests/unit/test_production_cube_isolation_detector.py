"""Regression tests for the Part 1F contact detector.

The detector bug these pin was concrete and bidirectional: a previous sweep run
concatenated the cube's and the table's contact messages before substring
matching, so a pair made of ``panda_right_finger`` from ONE message and
``target_cube`` from the OTHER was reported as if Gazebo had published it.
Anything that re-opens that hole reintroduces a false "grasp contact" claim, so
these tests fail if the parser ever reads across sensor boundaries or matches
collision names as loose substrings.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
_spec = importlib.util.spec_from_file_location(
    "production_cube_isolation", _SCRIPTS / "production_cube_isolation.py"
)
assert _spec is not None and _spec.loader is not None
mod = importlib.util.module_from_spec(_spec)
sys.modules["production_cube_isolation"] = mod
_spec.loader.exec_module(mod)

CUBE_ONLY = """header {
  stamp {
    sec: 23
    nsec: 665000000
  }
}
contact {
  header {
    stamp {
      sec: 23
      nsec: 665000000
    }
  }
  collision1 {
    id: 55
    name: "target_cube::link::collision"
  }
  collision2 {
    id: 7
    name: "table::link::collision"
  }
  depth: 0.0012
}
"""

TABLE_WITH_FINGER = """header {
  stamp {
    sec: 24
    nsec: 1000000
  }
}
contact {
  collision1 {
    id: 7
    name: "table::link::collision"
  }
  collision2 {
    id: 41
    name: "panda::panda_right_finger::panda_right_finger_collision"
  }
}
"""

CUBE_WITH_FINGER = """contact {
  collision1 {
    id: 55
    name: "target_cube::link::collision"
  }
  collision2 {
    id: 40
    name: "panda::panda_left_finger::panda_left_finger_collision"
  }
}
"""


def test_no_pair_is_manufactured_across_sensor_topics() -> None:
    """The exact bug: cube text + table text must not invent cube<->finger.

    Concatenating these two messages is what made the archived sweep report
    ``right_finger_vs_cube=True`` while Gazebo's cube sensor had only ever said
    ``target_cube <-> table``.
    """
    hits = mod.contact_hits(CUBE_ONLY, TABLE_WITH_FINGER)
    assert hits["right_finger_vs_cube"] is False
    assert hits["left_finger_vs_cube"] is False
    assert hits["cube_vs_table"] is True
    assert hits["any_finger_vs_table"] is True


def test_finger_cube_contact_is_reported_from_the_cube_sensor_itself() -> None:
    """When Gazebo really publishes the pair, both orientations are recognised."""
    hits = mod.contact_hits(CUBE_WITH_FINGER, "")
    assert hits["left_finger_vs_cube"] is True
    assert hits["right_finger_vs_cube"] is False


def test_orientation_is_ignored_within_one_message() -> None:
    """collision1/collision2 order must not matter."""
    flipped = CUBE_WITH_FINGER.replace(
        'name: "target_cube::link::collision"', 'name: "TMP"'
    ).replace(
        'name: "panda::panda_left_finger::panda_left_finger_collision"',
        'name: "target_cube::link::collision"',
    ).replace('name: "TMP"', 'name: "panda::panda_left_finger::panda_left_finger_collision"')
    hits = mod.contact_hits(flipped, "")
    assert hits["left_finger_vs_cube"] is True


def test_exact_names_prevent_substring_forgery() -> None:
    """A collision merely containing the name text must not count as a contact."""
    fake = """contact {
  collision1 {
    id: 1
    name: "old_recorded::target_cube_placeholder"
  }
  collision2 {
    id: 2
    name: "archive::panda_left_finger_note"
  }
}
"""
    hits = mod.contact_hits(fake, "")
    assert hits["left_finger_vs_cube"] is False


def test_empty_topics_fail_soft_to_false_not_exception() -> None:
    """A missing topic is null-equivalent: no raise, all False."""
    hits = mod.contact_hits("", "")
    assert hits["left_finger_vs_cube"] is False
    assert hits["cube_sensor_pairs"] == []
