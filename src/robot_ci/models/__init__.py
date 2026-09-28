"""Versioned data schemas for ROBOT CI.

Every persisted object in this project is a Pydantic v2 model deriving from
:class:`~robot_ci.models.base.VersionedModel`. Schemas are frozen (a running
experiment cannot mutate the conditions it claimed to be testing) and reject
unknown fields (a typo in a YAML file is an error, not a silent no-op).

Import order in this module is irrelevant to the dependency graph; the
submodules are the unit of dependency.
"""

from robot_ci.models.base import VersionedModel
from robot_ci.models.environment import (
    BackgroundKind,
    EnvironmentConfig,
    GraspObjectParams,
    ObjectCategory,
    PickAndPlaceParams,
    TaskConfig,
    TaskName,
)
from robot_ci.models.geometry import Covariance, Pose, Quaternion, Vec3
from robot_ci.models.robot import ArmModel, ControllerType, JointLimits, LinkPrimitive, RobotConfig
from robot_ci.models.scenario import (
    ExperimentConfig,
    ObjectSpec,
    PolicyComponent,
    PolicyConfig,
    ScenarioConfig,
    VariableSpec,
)
from robot_ci.models.sensors import (
    CameraCorruptionConfig,
    CameraIntrinsics,
    DepthCorruptionConfig,
    ExtrinsicPerturbationMode,
    ForceTorqueCorruptionConfig,
    LatencyConfig,
    RGBCorruptionConfig,
    RobotStateCorruptionConfig,
    SensorBundleConfig,
)
from robot_ci.models.state import (
    Action,
    ActionType,
    ContactState,
    GripperState,
    GroundTruthObject,
    GroundTruthState,
    JointState,
    ObjectState,
    Observation,
    PoseEstimate,
    RobotState,
    SensorFrame,
    Source,
)

__all__ = [
    "Action",
    "ActionType",
    "ArmModel",
    "BackgroundKind",
    "CameraCorruptionConfig",
    "CameraIntrinsics",
    "ContactState",
    "ControllerType",
    "Covariance",
    "DepthCorruptionConfig",
    "EnvironmentConfig",
    "ExperimentConfig",
    "ExtrinsicPerturbationMode",
    "ForceTorqueCorruptionConfig",
    "GraspObjectParams",
    "GripperState",
    "GroundTruthObject",
    "GroundTruthState",
    "JointLimits",
    "JointState",
    "LatencyConfig",
    "LinkPrimitive",
    "ObjectCategory",
    "ObjectSpec",
    "ObjectState",
    "Observation",
    "PickAndPlaceParams",
    "PolicyComponent",
    "PolicyConfig",
    "Pose",
    "PoseEstimate",
    "Quaternion",
    "RGBCorruptionConfig",
    "RobotConfig",
    "RobotState",
    "RobotStateCorruptionConfig",
    "ScenarioConfig",
    "SensorBundleConfig",
    "SensorFrame",
    "Source",
    "TaskConfig",
    "TaskName",
    "VariableSpec",
    "Vec3",
    "VersionedModel",
]
