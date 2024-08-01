from setuptools import setup, find_packages

setup(
    name="modularl",
    version="0.1.0",
    description="A modular reinforcement learning library",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.24",
        "torch>=2.0",
        "torchrl>=0.4.0",
        "tensorboard>=2.17.0",
    ],
    extras_require={
        "dev": [
            "black",
            "flake8",
            "mypy",
            "pytest>=6.2.5",
        ],
    },
)
