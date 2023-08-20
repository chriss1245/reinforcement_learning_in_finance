import io
from pathlib import Path

from setuptools import find_packages, setup

# Package meta-data.
NAME = "portfolio-management-rl"
DESCRIPTION = "My short description for my project."
URL = "https://github.com/chriss1245/reinforcement_learning_in_finance.git"
EMAIL = "christopher.manzano.vimos@gmail.com"
AUTHOR = "Christopher Manzano"
REQUIRES_PYTHON = ">=3.10.0"
VERSION = "23.8.20"


# Constants
PROJECT_DIR = Path(__file__).parent.resolve()

# Requirements
with open(PROJECT_DIR / "requirements.txt", "r", encoding="utf-8") as f:
    REQUIRED = f.read().splitlines()


# Import the README and use it as the long-description.
# Note: this will only work if 'README.md' is present in your MANIFEST.in file!

try:
    with io.open(PROJECT_DIR / "README.md", encoding="utf-8") as f:
        long_description = "\n" + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

# Setup
setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type="text/markdown",
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=find_packages(exclude=["tests", "*.tests", "*.tests.*", "tests.*"]),
    install_requires=REQUIRED,
    include_package_data=True,
    license="Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International Public License",
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        "License :: Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International Public License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: Implementation :: CPython",
        "Programming Language :: Python :: Implementation :: PyPy",
        "Operating System :: Unix",
        "Environment :: Conda Environment",
    ],
)
