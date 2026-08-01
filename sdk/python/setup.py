"""
Setup script for DistribAI Python SDK
"""

import os

from setuptools import find_packages, setup

here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()
setup(
    name="distribai",
    version="0.1.0",
    description="Python SDK for DistribAI Distributed Compute Network",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="EnderchefCoder",
    author_email="contact@distribai.io",
    url="https://github.com/naxium-oss/DistribAI",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "aiohttp>=3.9.0",
        "pydantic>=2.0.0",
        "typing-extensions>=4.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "mypy>=1.5.0",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: System :: Distributed Computing",
    ],
    keywords="distributed-compute machine-learning training ai",
    project_urls={
        "Bug Reports": "https://github.com/naxium-oss/DistribAI/issues",
        "Source": "https://github.com/naxium-oss/DistribAI",
        "Documentation": "https://docs.distribai.io",
    },
)
