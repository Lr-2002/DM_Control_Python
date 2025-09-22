#!/usr/bin/env python3
"""
Setup script for IC ARM Control package
"""

from setuptools import setup, find_packages

setup(
    name="ic_arm_control",
    version="0.1.0",
    description="Unified motor control interface for IC ARM",
    author="lr-2002",
    packages=find_packages(),
    python_requires=">=3.7",
    install_requires=[
        # Add your dependencies here
        # "numpy",
        # "matplotlib",
        # etc.
    ],
    entry_points={
        'console_scripts': [
            'ic-position-monitor=ic_arm_control.tools.position_monitor:main',
        ],
    },
)
