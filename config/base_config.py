"""
Base configuration class for driver identification experiments.
"""
import os
from pathlib import Path


class BaseConfig:
    """Base configuration class with common parameters."""
    
    # Project paths
    PROJECT_ROOT = Path(__file__).parent.parent
    DATA_DIR = PROJECT_ROOT / "data"
    RAW_DATA_DIR = DATA_DIR / "raw"
    PROCESSED_DATA_DIR = DATA_DIR / "processed"
    FEATURES_DIR = DATA_DIR / "features"
    SPLITS_DIR = DATA_DIR / "splits"
    
    RESULTS_DIR = PROJECT_ROOT / "results"
    MODELS_DIR = RESULTS_DIR / "models"
    FIGURES_DIR = RESULTS_DIR / "figures"
    TABLES_DIR = RESULTS_DIR / "tables"
    LOGS_DIR = RESULTS_DIR / "logs"
    
    # Data parameters
    RANDOM_SEED = 42
    TEST_SIZE = 0.2
    VAL_SIZE = 0.2
    
    # Feature extraction parameters
    GPS_FEATURES = ["speed", "acceleration_est_1", "angular_acc"]
    TIME_FEATURES = ["orig_time"]
    
    # Evaluation parameters
    CV_FOLDS = 5
    METRICS = ["nmi", "ami", "silhouette", "accuracy"]
    
    # Ground truth mapping for car 460631
    GROUND_TRUTH_MAPPING = {
        1: ['Hadar Yosef', 'Neot Afeka A', 'Rom 2000', 'Herzliya B'],
        2: ['Tel Aviv University', 'HaOgen'],
        3: ['Naot Uzi']
    }
    
    def __init__(self):
        """Initialize configuration and create necessary directories."""
        self.create_directories()
    
    def create_directories(self):
        """Create necessary directories if they don't exist."""
        directories = [
            self.RAW_DATA_DIR,
            self.PROCESSED_DATA_DIR,
            self.FEATURES_DIR,
            self.SPLITS_DIR,
            self.MODELS_DIR,
            self.FIGURES_DIR,
            self.TABLES_DIR,
            self.LOGS_DIR
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True) 