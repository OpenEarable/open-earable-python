from setuptools import setup, find_packages

setup(
    name="open-earable-python",
    version="0.1.0",
    description="Reader and utilities for multi-sensor OpenEarable recordings.",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
    ],
    python_requires=">=3.9",
)