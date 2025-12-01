from setuptools import setup, find_packages
from pathlib import Path


setup(
    name="torchbenchx",
    version="0.1.0",
    author="Syeed Mohd Saquib",
    author_email="syeedmsaquib@gmail.com",
    description="Unified Deep Learning Benchmark Library for PyTorch and TensorFlow",
    long_description = Path("README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    url="https://github.com/syeedsaquib/torchbenchx",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "torch",
        "torchvision",
        "tensorflow",
        "plotly",
        "pandas",
        "psutil"
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)

