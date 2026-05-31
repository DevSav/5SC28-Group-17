from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

RESULTS_DIR = ROOT / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
MODELS_DIR = RESULTS_DIR / "models"
TEST_OUTPUTS_DIR = RESULTS_DIR / "test_outputs"

MAX_VOLTAGE = 3.0
MIN_VOLTAGE = -3.0

