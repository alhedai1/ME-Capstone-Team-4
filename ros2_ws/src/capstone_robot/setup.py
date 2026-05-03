from setuptools import find_packages, setup

package_name = "capstone_robot"

setup(
    name=package_name,
    version="0.0.1",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Capstone Team",
    maintainer_email="team@example.com",
    description="ROS2 nodes for the capstone pole-climbing robot.",
    license="TODO",
    entry_points={
        "console_scripts": [
            "state_machine_node = capstone_robot.state_machine_node:main",
            "motor_driver_node = capstone_robot.motor_driver_node:main",
            "vision_node = capstone_robot.vision_node:main",
        ],
    },
)
