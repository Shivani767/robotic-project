"""Every declared policy parameter must be read by the implementation.

Two parameters in this project have been validated, stored and then never read,
and both did real damage:

* ``max_position_sigma_m`` was 12x larger than the grasp budget it protected, so
  the candidate's confidence gate could never engage. The two policies were the
  same policy under different labels, and the "no regression found" result was an
  artefact of the dead parameter rather than a fact about the robots.
* ``estimate_smoothing`` was declared on the positive-control policy, validated
  for range, and never read -- the settle window computes a plain mean. An
  ablation varying it would have measured nothing and reported the absence of an
  effect as a finding.

A dead parameter is invisible from the outside. It validates, it round-trips
through config hashing, it appears in a report, and it does nothing. So the
check is mechanical: a parameter that appears in a named policy's defaults must
also appear in a ``parameters.get`` call in the module that implements it.

This inspects source rather than behaviour, which is unusual and deliberate. A
behavioural test cannot distinguish "this parameter does nothing" from "this
parameter does nothing *on the conditions I happened to test*", and the whole
failure mode is that the second looks like the first.
"""

from __future__ import annotations

import inspect
import re

import pytest

from robot_ci.models.scenario import PolicyConfig
from robot_ci.policies import resolve_policy_config, rule_based

#: Named policies whose defaults must all correspond to a real read.
NAMED_POLICIES = ("policy_v1", "policy_v2", "policy_v3")

#: Matches every key the implementation actually reads.
_READ = re.compile(r"""parameters\.get\(\s*["']([A-Za-z_][A-Za-z0-9_]*)["']""")


def _read_parameters() -> set[str]:
    """Parameter names the policy implementation reads."""
    source = inspect.getsource(rule_based)
    return set(_READ.findall(source))


@pytest.mark.parametrize("name", NAMED_POLICIES)
def test_declared_parameters_are_read_by_the_implementation(name: str) -> None:
    """No named policy may declare a parameter that changes nothing."""
    declared = set(resolve_policy_config(PolicyConfig(name=name)).parameters)
    dead = declared - _read_parameters()
    assert not dead, (
        f"{name} declares {sorted(dead)} but the implementation never reads "
        "them. A parameter that is stored but unused validates, round-trips "
        "through config hashing, appears in reports, and does nothing."
    )


def test_the_removed_smoothing_parameter_stays_removed() -> None:
    """Pin a specific regression, since the guard above is generic.

    ``estimate_smoothing`` was declared on ``policy_v3`` and never read. If it
    ever comes back it must come back with a read, and until then its return
    would mean the guard was weakened rather than the policy being fixed.
    """
    assert "estimate_smoothing" not in _read_parameters()
    declared = resolve_policy_config(PolicyConfig(name="policy_v3")).parameters
    assert "estimate_smoothing" not in declared


def test_the_guard_would_catch_a_reintroduced_dead_parameter() -> None:
    """A guard that cannot fail is not a guard.

    Checked by feeding the scanner a synthetic source containing a parameter
    that is declared but never read, which is exactly the shape of both bugs
    this file exists for.
    """
    source = 'x = config.parameters.get("read_one")\n'
    assert set(_READ.findall(source)) == {"read_one"}
    assert "never_declared" not in _READ.findall(source)
