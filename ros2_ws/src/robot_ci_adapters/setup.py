"""Packaging for robot_ci_adapters.

The nodes themselves come from the `robot-ci` distribution, which must be
installed (``pip install -e .``) before this workspace builds; the container in
``scripts/ros2_docker.sh`` does exactly that. Declaring the dependency here
means a bare ``colcon build`` on a fresh machine fails with a sentence saying
what to install, rather than producing a package whose nodes cannot start.
"""

from setuptools import find_packages, setup

package_name = "robot_ci_adapters"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools", "robot-ci"],
    zip_safe=True,
    maintainer="Robotics CI contributors",
    maintainer_email="noreply@example.com",
    description=(
        "ROS 2 nodes exposing the ROBOT CI MuJoCo experiment as a live process graph."
    ),
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "robot_ci_simulation_node = robot_ci_adapters.nodes:main_simulation",
            "robot_ci_policy_node = robot_ci_adapters.nodes:main_policy",
        ],
    },
)
