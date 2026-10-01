"""Packaging for robot_ci_bringup: the launch entry point."""

from glob import glob
from os.path import join

from setuptools import find_packages, setup

package_name = "robot_ci_bringup"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        (join("share", package_name, "launch"), glob("launch/*.launch.py")),
        (join("share", package_name), ["package.xml"]),
    ],
    install_requires=["setuptools", "launch"],
    zip_safe=True,
    maintainer="Robotics CI contributors",
    maintainer_email="noreply@example.com",
    description="Launch entry point for the ROBOT CI episode over ROS 2.",
    license="MIT",
    tests_require=["pytest"],
)
