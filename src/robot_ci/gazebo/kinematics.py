"""URDF forward kinematics for the Gazebo backend's IK provider.

The policy solves IK through the ``JacobianProvider`` interface
(:mod:`robot_ci.planning.kinematics`), which needs a ``forward(q)`` and a
``jacobian(q)``. MuJoCo answers both from its live model. Gazebo has no local
model -- the simulation lives in another process -- so this module answers
from the URDF the world was generated from: the same tree the live browser
viewer (``dashboard/viewer.js``) and the recorded 3-D view
(:mod:`robot_ci.recording.render3d`) use.

The chain is evaluated in URDF file order from the robot base, applying each
joint's fixed origin then its motion. Positioning rows of the Jacobian are
finite differences of that map; orientation rows use the rotation-vector
difference. The differencing step (1e-6 rad) is small enough that truncation
error is far below the IK tolerance (0.5 mm) and large enough to avoid float
cancellation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from robot_ci.planning.kinematics import JacobianProvider, rotation_error
from robot_ci.recording.render3d import load_kinematics

__all__ = ["UrdfJacobianProvider", "urdf_forward"]


def urdf_forward(urdf_path: str | Path, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the end-effector ``(position, rotation)`` for arm joints ``q``.

    Args:
        urdf_path: Path to the robot URDF.
        q: Seven arm joint values, in URDF order.

    Returns:
        The ``panda_link7`` origin and its rotation matrix, in the base frame.
    """
    return UrdfJacobianProvider(urdf_path).forward(q)


class UrdfJacobianProvider(JacobianProvider):
    """Finite-difference Jacobian of the URDF forward kinematics.

    The kinematic chain is parsed **once** in ``__init__`` into a cached list of
    per-joint transforms. An earlier version parsed the URDF on every
    ``forward()`` call, and measured that cost 58.6 s of planning in a 61-tick
    episode -- the IK solver calls ``forward`` several times per iteration, and
    each call re-read the file from disk. The cache turns the per-call cost into
    seven small matrix products.

    Attributes:
        urdf_path: URDF the kinematics were read from.
        step: Differencing step in radians.
    """

    def __init__(self, urdf_path: str | Path, *, step: float = 1e-6) -> None:
        from robot_ci.recording.render3d import _axis_matrix, _homogeneous, _rpy_matrix

        self.urdf_path = str(urdf_path)
        self.step = float(step)
        kin = load_kinematics(self.urdf_path)
        # The arm chain is the first seven joints of this URDF (joint1..joint7);
        # the eighth is the fixed hand_joint and the rest are the fingers. The
        # backend's EE_LINK is panda_link7, so the chain stops there.
        self._chain: list[tuple[np.ndarray, np.ndarray, str, np.ndarray]] = []
        for joint in kin.joints[:7]:
            rotation = _rpy_matrix(*[float(v) for v in joint["rpy"]])
            translation = np.asarray(joint["xyz"], dtype=np.float64)
            axis = np.asarray(joint.get("axis", [0.0, 0.0, 1.0]), dtype=np.float64)
            self._chain.append((rotation, translation, str(joint.get("type", "revolute")), axis))
        if len(self._chain) != 7:
            raise ValueError(
                f"expected a 7-joint arm chain in {self.urdf_path}, found {len(self._chain)}"
            )
        self._axis_matrix = _axis_matrix
        self._homogeneous = _homogeneous

    def forward(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return the end-effector position and rotation at ``q``."""
        values = np.asarray(q, dtype=np.float64).reshape(-1)
        transform = np.eye(4)
        for index, (rotation, translation, kind, axis) in enumerate(self._chain):
            transform = transform @ self._homogeneous(rotation, translation)
            value = float(values[index]) if index < values.size else 0.0
            if kind == "revolute":
                transform = transform @ self._homogeneous(
                    self._axis_matrix(axis, value), np.zeros(3)
                )
            elif kind == "prismatic":
                norm = float(np.linalg.norm(axis))
                direction = axis / norm if norm > 1e-12 else axis
                transform = transform @ self._homogeneous(np.eye(3), direction * value)
        return transform[:3, 3].copy(), transform[:3, :3].copy()

    def jacobian(self, q: np.ndarray) -> np.ndarray:
        """Return the 6xN task-space Jacobian at ``q`` via finite differences."""
        q = np.asarray(q, dtype=np.float64)
        position0, rotation0 = self.forward(q)
        n = int(q.size)
        jac = np.zeros((6, n))
        for column in range(n):
            dq = np.zeros(n)
            dq[column] = self.step
            position1, rotation1 = self.forward(q + dq)
            jac[:3, column] = (position1 - position0) / self.step
            jac[3:, column] = rotation_error(rotation0, rotation1) / self.step
        return jac

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return f"UrdfJacobianProvider({self.urdf_path!r})"

    def __getstate__(self) -> dict[str, Any]:
        return {"urdf_path": self.urdf_path, "step": self.step}

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__init__(state["urdf_path"], step=float(state["step"]))  # type: ignore[misc]
