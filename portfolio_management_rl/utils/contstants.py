"""
This file contains all the constants used in the project.
"""

from pathlib import Path

# Path to the root directory of the project
PROJECT_DIR = Path(__file__).resolve().parents[2]

# Path to the root package directory
PACKAGE_DIR = PROJECT_DIR / "portfolio_managemaner_rl"
DATA_DIR = PROJECT_DIR / "data"
TESTS_DIR = PROJECT_DIR / "tests"

MLFLOW_TRACKING_URI = (PROJECT_DIR / "mlflow").as_uri()

# Number of stocks of the training dataset
N_STOCKS = 100

# Used as an overcost for all the stocks when buying in order to avoid floating point precision errors
PRICE_EPSILON = 1e-3


if __name__ == "__main__":
    print("Project dir:", PROJECT_DIR)
    print("Package dir:", PACKAGE_DIR)
    print("Tests dir:", TESTS_DIR)
    print("Data dir:", DATA_DIR)
    print("MLFLOW tracking uri:", MLFLOW_TRACKING_URI)
    print("N_STOCKS:", N_STOCKS)
    print("PRICE_EPSILON:", PRICE_EPSILON)
