"""Configuration module for the Medical Insurance ML & MongoDB project.

Defines all filesystem paths, database connection parameters, feature definitions,
validation constraints, and reproducibility seeds.
"""

from pathlib import Path
import os
import logging
from dotenv import load_dotenv

# Base Directory Resolution
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent

# Load environment variables from .env if present
load_dotenv(PROJECT_ROOT / ".env")

# Directories
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
TESTS_DIR = PROJECT_ROOT / "tests"

# Ensure runtime directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

# Data File Paths
RAW_CSV_PATH = DATA_DIR / "insurance.csv"
RAW_JSON_PATH = DATA_DIR / "insurance.json"
CLEANED_CSV_PATH = DATA_DIR / "insurance_cleaned.csv"
PREPROCESSING_SUMMARY_PATH = OUTPUTS_DIR / "preprocessing_summary.json"

# Model and Metric File Paths
LINEAR_REGRESSION_MODEL_PATH = MODELS_DIR / "linear_regression_pipeline.joblib"
RANDOM_FOREST_MODEL_PATH = MODELS_DIR / "random_forest_pipeline.joblib"
BEST_MODEL_PATH = MODELS_DIR / "best_model_pipeline.joblib"
METRICS_JSON_PATH = OUTPUTS_DIR / "model_metrics.json"
METRICS_CSV_PATH = OUTPUTS_DIR / "model_metrics.csv"
BEST_MODEL_METADATA_PATH = OUTPUTS_DIR / "best_model_metadata.json"

# MongoDB Settings
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
DATABASE_NAME = os.getenv("DATABASE_NAME", "medical_insurance_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "insurance_records")
MONGO_TIMEOUT_MS = int(os.getenv("MONGO_TIMEOUT_MS", "3000"))

# Dataset Source URL
DATASET_URL = (
    "https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv"
)

# Feature and Target Definitions
NUMERICAL_FEATURES = ["age", "bmi", "children"]
CATEGORICAL_FEATURES = ["sex", "smoker", "region"]
ALL_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
TARGET_COLUMN = "charges"

# Model Reproducibility and Train/Test Split
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Validation Constraints
VALID_SEX = ["female", "male"]
VALID_SMOKER = ["yes", "no"]
VALID_REGIONS = ["southwest", "southeast", "northwest", "northeast"]
AGE_MIN = 18
AGE_MAX = 120
BMI_MIN = 10.0
BMI_MAX = 65.0
CHILDREN_MIN = 0
CHILDREN_MAX = 20

# Configure root logger for the application
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("medical_insurance_ml")
