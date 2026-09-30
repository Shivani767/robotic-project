"""Sensor fusion: combine repeated object-pose estimates with real uncertainty.

Deliberately small. The simulator presents one object at a time, so a factor graph
or pose-graph optimiser would be machinery with nothing behind it.
"""

from robot_ci.sensor_fusion.kalman import (
    FusedEstimate,
    FusionConfig,
    ObjectPoseFilter,
)

__all__ = ["FusedEstimate", "FusionConfig", "ObjectPoseFilter"]
