"""Regression detection: conditional comparison of two policies."""

from robot_ci.regression.detector import (
    ConditionCell,
    RegressionConfig,
    RegressionFinding,
    RegressionReport,
    compare_conditions,
    detect_regressions,
)
from robot_ci.regression.matrix import RegressionMatrix, build_matrix

__all__ = [
    "ConditionCell",
    "RegressionConfig",
    "RegressionFinding",
    "RegressionMatrix",
    "RegressionReport",
    "build_matrix",
    "compare_conditions",
    "detect_regressions",
]
