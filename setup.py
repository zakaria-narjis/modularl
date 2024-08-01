from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="modularl",
    version="0.1.0",
    author="Zakaria Narjis",
    author_email="zakaria.narjis.97@gmail.com",
    description="A modular reinforcement learning library",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/zakaria-narjis/modularl",
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
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.7",
)
