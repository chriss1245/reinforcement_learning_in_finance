"""
This file contains all the constants used in the project.
"""

from pathlib import Path

# Path to the root directory of the project
PROJECT_DIR = Path(__file__).resolve().parents[2]

# Path to the root package directory
PACKAGE_DIR = PROJECT_DIR / "portfolio_managemaner_rl"
DATA_DIR = PROJECT_DIR / "data"
BUFFER_DIR = PROJECT_DIR / "buffer"
TESTS_DIR = PROJECT_DIR / "tests"
LOGS_DIR = PROJECT_DIR / "logs"

MLFLOW_TRACKING_URI = (PROJECT_DIR / "mlflow").as_uri()
# Used as an overcost for all the stocks when buying in order to avoid floating point precision errors
PRICE_EPSILON = 1e-6


# Number of stocks of the training da   taset
N_STOCKS = 101
WINDOW_SIZE = 3 * 252  # 3 years of trading days 3 * 252
FORECAST_HORIZON = 22  # 1 month of trading days 21-22 without holidays
INITIAL_DATE = "1980-01-01"
END_DATE = "2023-08-01"


if __name__ == "__main__":
    print("Project dir:", PROJECT_DIR)
    print("Package dir:", PACKAGE_DIR)
    print("Tests dir:", TESTS_DIR)
    print("Data dir:", DATA_DIR)
    print("MLFLOW tracking uri:", MLFLOW_TRACKING_URI)
    print("N_STOCKS:", N_STOCKS)
    print("PRICE_EPSILON:", PRICE_EPSILON)
