"""Failure taxonomy, ablation and regression diagnosis.

Ordered as an investigation proceeds: :mod:`taxonomy` names what went wrong,
:mod:`ablation` establishes which component is associated with the change, and
:mod:`diagnose` writes the conclusion down with the hedging the evidence
supports.
"""

from robot_ci.diagnosis.ablation import (
    AblationArm,
    AblationFactor,
    AblationResult,
    build_ofat,
    build_pairwise,
    run_ablation,
)
from robot_ci.diagnosis.diagnose import Diagnosis, Evidence, diagnose
from robot_ci.diagnosis.taxonomy import FailureCategory, FailureEvent, classify_failure

__all__ = [
    "AblationArm",
    "AblationFactor",
    "AblationResult",
    "Diagnosis",
    "Evidence",
    "FailureCategory",
    "FailureEvent",
    "build_ofat",
    "build_pairwise",
    "classify_failure",
    "diagnose",
    "run_ablation",
]
