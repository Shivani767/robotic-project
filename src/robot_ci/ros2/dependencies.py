"""ROS 2 availability, detected rather than assumed.

The rule this module exists to enforce: **a missing ROS 2 is an environment
fact, not a silent fallback.** There is no stubbed transport here, no mock
publisher, and no "ROS-like" shim. If ``rclpy`` cannot be imported, the caller
gets an error naming the distro it needs and the command that installs it, and
no experiment runs.

Why the distro list is not "any ROS 2"
--------------------------------------
``pyproject.toml`` pins ``requires-python = ">=3.11,<3.13"``. A ROS 2
distribution ships against one specific CPython:

======================  ==================  ====================
Distro                  Ubuntu              Bundled CPython
======================  ==================  ====================
Humble Hawksbill        22.04               3.10  -- too old
Jazzy Jalisco           24.04               3.12  -- compatible
Kilted Kaiju            24.10               3.12  -- compatible
======================  ==================  ====================

So "ROS 2 is installed" is not sufficient; the distro has to be one whose
interpreter can import this package. A Humble install fails at
``import robot_ci``, and that is reported as a version mismatch rather than as
a broken robot, because the two need different fixes.
"""

from __future__ import annotations

import importlib.util
import os
import platform
import sys
from dataclasses import dataclass, field
from typing import Any

from robot_ci.core.errors import BackendUnavailableError

__all__ = [
    "SUPPORTED_DISTROS",
    "UNSUPPORTED_DISTROS",
    "Ros2Environment",
    "detect",
    "install_hint",
    "require_rclpy",
]

#: Distros whose bundled CPython satisfies ``pyproject.toml``.
SUPPORTED_DISTROS: tuple[str, ...] = ("jazzy", "kilted")

#: Distros that exist but cannot import this package. Named explicitly so the
#: error can say *why* rather than "ROS 2 is missing".
UNSUPPORTED_DISTROS: dict[str, str] = {
    "humble": "Humble bundles CPython 3.10; robot-ci requires >=3.11",
    "iron": "Iron bundles CPython 3.10; robot-ci requires >=3.11",
    "rolling": "Rolling moves; pin a release distro for a reproducible run",
    "foxy": "Foxy bundles CPython 3.8; robot-ci requires >=3.11",
}

#: Modules that must be importable for a real node graph. ``rclpy`` is the
#: client library; ``rclpy.node`` is probed separately because a partial
#: install can expose one without the other.
_REQUIRED_MODULES: tuple[str, ...] = ("rclpy", "rclpy.node")


@dataclass(frozen=True)
class Ros2Environment:
    """What ROS 2 is actually present on this machine.

    Attributes:
        available: True only when every required module imports.
        distro: Value of ``ROS_DISTRO``, if set.
        python_version: The running interpreter's version, for the compat check.
        rclpy_version: ``rclpy`` version string when importable.
        missing: Required modules that could not be imported.
        reason: Short human-readable explanation when ``available`` is False.
        detail: Structured facts for an artifact, never for prose.
    """

    available: bool
    distro: str | None
    python_version: str
    rclpy_version: str | None = None
    missing: tuple[str, ...] = ()
    reason: str = ""
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable record for the execution artifact."""
        return {
            "ros2_available": self.available,
            "ros_distro": self.distro,
            "ros2_reason": self.reason,
            "rclpy_version": self.rclpy_version,
            "missing_modules": list(self.missing),
            "python_version": self.python_version,
            "platform": platform.platform(),
            "machine": platform.machine(),
            **self.detail,
        }


def install_hint(distro: str = "jazzy") -> str:
    """Return the command sequence that produces a working ROS 2 environment.

    The Linux sequence is the one this project can actually verify, so it is
    the one printed. The macOS/conda path exists but is marked unverified
    rather than presented as equivalent.
    """
    return (
        "ROS 2 was not found on this machine.\n"
        "\n"
        "On Ubuntu 24.04 (the configuration this project verifies against):\n"
        f"  sudo apt install ros-{distro}-ros-base python3-colcon-common-extensions\n"
        f"  source /opt/ros/{distro}/setup.bash\n"
        f"  echo $ROS_DISTRO   # must print {distro}\n"
        "\n"
        "Then install the project and build the ROS 2 workspace:\n"
        '  .venv/bin/pip install -e ".[dev]"   # or: pip install -e .\n'
        "  colcon build --symlink-install\n"
        "  source ros2_ws/install/setup.bash\n"
        "\n"
        "Inside a container, `scripts/ros2_docker.sh` performs exactly this and\n"
        "is the path used for the recorded live runs.\n"
        "\n"
        f"Note: Humble is not supported -- {UNSUPPORTED_DISTROS['humble']}."
    )


def detect() -> Ros2Environment:
    """Probe the environment. Never raises; the caller decides what to do.

    Returns:
        A :class:`Ros2Environment` describing what is present.
    """
    missing: list[str] = []
    for module in _REQUIRED_MODULES:
        try:
            if importlib.util.find_spec(module) is None:
                missing.append(module)
        except (ImportError, ValueError):
            missing.append(module)

    distro = os.environ.get("ROS_DISTRO") or None
    python_version = platform.python_version()
    rclpy_version: str | None = None
    detail: dict[str, Any] = {
        "ros_prefix": os.environ.get("ROS_PREFIX") or os.environ.get("AMENT_PREFIX_PATH") or None,
    }

    if not missing:
        try:
            import rclpy

            rclpy_version = getattr(rclpy, "__version__", None) or "present"
        except Exception as exc:  # pragma: no cover - defensive; spec said present
            missing.append("rclpy")
            detail["rclpy_import_error"] = f"{type(exc).__name__}: {exc}"

    if missing:
        reason = (
            "rclpy is not importable, so no ROS 2 process graph can run. "
            f"Missing: {', '.join(missing)}."
        )
        if distro in UNSUPPORTED_DISTROS:
            reason += f" {UNSUPPORTED_DISTROS[distro]}."
        return Ros2Environment(
            available=False,
            distro=distro,
            python_version=python_version,
            missing=tuple(missing),
            reason=reason,
            detail=detail,
        )

    minor = sys.version_info[:2]
    if minor < (3, 11) or minor >= (3, 13):
        return Ros2Environment(
            available=False,
            distro=distro,
            python_version=python_version,
            rclpy_version=rclpy_version,
            reason=(
                f"rclpy imports but this interpreter is {python_version}; "
                "robot-ci requires >=3.11,<3.13."
            ),
            detail=detail,
        )

    return Ros2Environment(
        available=True,
        distro=distro,
        python_version=python_version,
        rclpy_version=rclpy_version,
        detail=detail,
    )


def require_rclpy() -> Ros2Environment:
    """Return the environment or raise with an actionable message.

    Returns:
        The detected :class:`Ros2Environment`, guaranteed ``available``.

    Raises:
        BackendUnavailableError: When ROS 2 cannot be used, carrying the
            install hint so a caller can print it verbatim.
    """
    environment = detect()
    if environment.available:
        return environment
    raise BackendUnavailableError(
        environment.reason,
        code="ros2.unavailable",
        distro=environment.distro,
        missing=list(environment.missing),
        python_version=environment.python_version,
        hint=install_hint(environment.distro or "jazzy"),
    )
