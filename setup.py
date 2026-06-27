"""
SUMO Work Zone Traffic Control - Setup Configuration

This file configures the package for installation via pip.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [
        line.strip() 
        for line in fh 
        if line.strip() and not line.startswith("#")
    ]

setup(
    name="sumo-workzone-rl",
    version="1.0.0",
    author="Israel Afriyie",
    author_email="",
    description="Deep Q-Network reinforcement learning for SUMO work zone traffic control",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/IsraelAfriyie-dev/Sumo-Reinforcement-Learning-for-WorkZone-Traffic-Condition",
    packages=find_packages(exclude=["tests", "tests.*", "docs"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
        ],
        "gpu": [
            "torch>=2.0.0+cu118",
        ],
    },
    entry_points={
        "console_scripts": [
            "train-dqn=training.train_dqn:main",
            "evaluate-policy=evaluation.evaluate_policy:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": [
            "configs/*.yaml",
            "data/networks/workzone/*.xml",
            "docs/*.md",
        ],
    },
    zip_safe=False,
)