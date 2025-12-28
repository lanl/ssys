"""Configuration for BioModels benchmark suite."""

# Fetch settings
FETCH_STRATEGIES = ["random", "sequential"]
FETCH_MODES = ["initial", "expand", "update"]
DEFAULT_STRATEGY = "random"
DEFAULT_MODE = "expand"  # Safe default: won't duplicate

# Model limits for initial exploration
MAX_MODELS = None  # None = all available, or set integer for testing
START_INDEX = 0

# Complexity filters (models exceeding these will be noted but not rejected)
MAX_SPECIES = 100
MAX_REACTIONS = 200
MAX_PARAMETERS = 500

# Timeout settings (seconds)
FETCH_TIMEOUT = 30  # Per model download
CONVERT_TIMEOUT = 10  # SBML → Antimony conversion
RECAST_TIMEOUT = 120  # Per recast attempt (increased from 60 for complex models)
VALIDATION_TIMEOUT = 30  # Per validation

# Parallel processing
N_WORKERS = 4  # Number of parallel processes for batch operations

# Recast modes to test
RECAST_MODES = ["simplified", "canonical"]

# Features that block recast attempts
BLOCKING_FEATURES = [
    "events",
    "delays",
    "algebraic_rules",
    "sbml_l3_packages",  # Uses SBML L3 extensions (layout, fbc, etc.)
    "unsupported_trig",  # Uses tan/tanh (ssys only supports sin/cos)
]

# Features that flag as challenging but don't block
WARNING_FEATURES = [
    "piecewise_heavy",  # Multiple piecewise constructs
    "time_dependent",  # Explicit time in rate laws
    "sin_cos",  # Contains sin/cos (supported, needs transform)
    "negative_species",  # Species with negative initial values
]

# BioModels API settings
BIOMODELS_API_BASE = "https://www.ebi.ac.uk/biomodels"
BIOMODELS_REST_API = f"{BIOMODELS_API_BASE}/model/download"
REQUEST_TIMEOUT = 30
REQUEST_DELAY = 0.5  # Delay between requests to avoid hammering API

# Paths (relative to benchmarks/)
DATA_DIR = "data"
SBML_DOWNLOADS_DIR = f"{DATA_DIR}/sbml_downloads"  # All fetched SBML files from BioModels
SBML_CANDIDATES_DIR = f"{DATA_DIR}/sbml_candidates"  # Filtered candidates for recasting
FETCH_HISTORY = f"{DATA_DIR}/fetch_history.json"
MODEL_REGISTRY = f"{DATA_DIR}/model_registry.json"
METADATA_FILE = f"{DATA_DIR}/metadata.json"

RESULTS_DIR = "results"
CANDIDATES_CSV = f"{RESULTS_DIR}/candidates.csv"
RECASTS_DIR = f"{RESULTS_DIR}/recasts"
VALIDATION_DIR = f"{RESULTS_DIR}/validation"
FAILURES_DIR = f"{RESULTS_DIR}/failures"
SUMMARY_JSON = f"{RESULTS_DIR}/summary.json"
REPORT_NOTEBOOK = f"{RESULTS_DIR}/report.ipynb"

# Logging
LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR
LOG_FILE = "benchmark.log"
