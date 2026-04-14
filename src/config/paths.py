from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Directorios principales
# DATA
DATA_DIR = PROJECT_ROOT / "data"

# Subdirectorios de DATA
GROUPED_DATA_DIR = DATA_DIR / "grouped_data"
ORIGINAL_DATA_DIR = DATA_DIR / "original_data"
SPLITTED_DATA_DIR = DATA_DIR / "splitted_data"


# SRC
SRC_DIR = PROJECT_ROOT / "src"

# Subdirectorios de SRC
CONFIG_DIR = SRC_DIR / "config"
MODELS_DIR = SRC_DIR / "models"
TRAINING_DIR = SRC_DIR / "training"
UTILS_DIR = SRC_DIR / "utils"
EVALUATION_DIR = SRC_DIR / "evaluation"
DATA_SRC_DIR = SRC_DIR / "data"

# Subdirectorios de DATA_SRC_DIR
DATA_PROCESSED_DIR = DATA_SRC_DIR / "processed"

# Subdirectorios de processed
IMAGES_RGB_DIR = DATA_PROCESSED_DIR / "imagenes_RGB"
LOGS_DIR = DATA_PROCESSED_DIR / "logs"


# DOCS
DOCS_DIR = PROJECT_ROOT / "docs"


# EXPERIMENTS
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"

# Subdirectorios de EXPERIMENTS
CNN_RGB_DIR = EXPERIMENTS_DIR / "cnn_rgb_baseline"

#Subdirectorios de CNN_RGB_DIR
LOGS_DIR = CNN_RGB_DIR / "logs"
RESULTS_DIR = CNN_RGB_DIR / "results"


# IMAGES
IMAGES_DIR = PROJECT_ROOT / "images"

# Subdirectorios de IMAGES
IMAGES_ISIC_DIR = IMAGES_DIR / "images_ISIC"
IMAGES_MALIGNANT_DIR = IMAGES_DIR / "imagenes_Malignant"


# LICENSES
LICENSES_DIR = PROJECT_ROOT / "licenses"


# NOTEBOOKS
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
