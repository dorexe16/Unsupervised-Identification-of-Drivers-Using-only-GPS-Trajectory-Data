"""
Configuration for semi-supervised learning experiments.
"""
from .base_config import BaseConfig


class SemiSupervisedConfig(BaseConfig):
    """Configuration for semi-supervised learning approaches."""
    
    # Semi-supervised parameters
    INITIAL_LABELED_RATIO = 0.1
    ACTIVE_LEARNING_STRATEGY = "uncertainty"  # "uncertainty", "diversity", "hybrid"
    SELF_TRAINING_THRESHOLD = 0.9
    PSEUDO_LABELING_THRESHOLD = 0.8
    MAX_ITERATIONS = 50
    CONFIDENCE_MARGIN = 0.2
    
    # Active learning parameters
    QUERY_STRATEGY = "uncertainty_sampling"  # "uncertainty_sampling", "query_by_committee", "expected_model_change"
    BATCH_SIZE = 10
    POOL_SIZE = 100
    
    # Self-training parameters
    SELF_TRAINING_MODE = "threshold"  # "threshold", "top_k", "adaptive"
    TOP_K_CONFIDENT = 20
    ADAPTIVE_THRESHOLD_PERCENTILE = 90
    
    # Co-training parameters
    VIEW1_FEATURES = ["speed", "acceleration_est_1"]
    VIEW2_FEATURES = ["angular_acc", "jerk"]
    CO_TRAINING_AGREEMENT_THRESHOLD = 0.8
    
    # Model parameters
    MODEL_TYPE = "xgboost"  # "xgboost", "transformer", "mlp", "ensemble"
    
    # XGBoost parameters
    XGBOOST_PARAMS = {
        "n_estimators": 600,
        "learning_rate": 0.05,
        "max_depth": 10,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "eval_metric": "logloss",
        "n_jobs": -1,
        "random_state": 42
    }
    
    # Transformer parameters  
    TRANSFORMER_PARAMS = {
        "latent_dim": 128,
        "n_heads": 4,
        "n_layers": 3,
        "dropout_rate": 0.1,
        "max_seq_len": 100
    }
    
    # Feature extraction parameters
    FEATURE_TYPE = "minirocket"  # "minirocket", "autoencoder", "raw", "statistical"
    MINIROCKET_KERNELS = 4000
    AUTOENCODER_LATENT_DIM = 128
    
    # Evaluation parameters
    WILSON_CONFIDENCE_ALPHA = 0.05
    EVALUATION_FREQUENCY = 5  # Evaluate every N iterations 