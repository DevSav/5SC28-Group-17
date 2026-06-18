from pathlib import Path


ANN_DIR = Path(__file__).resolve().parent
ROOT = ANN_DIR.parents[1]

DATA_DIR = ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"

RESULTS_DIR = ANN_DIR / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
MODELS_DIR = RESULTS_DIR / "models"
TEST_OUTPUTS_DIR = RESULTS_DIR / "test_outputs"
