"""
Configuration for unsupervised learning experiments.
"""
from .base_config import BaseConfig


class UnsupervisedConfig(BaseConfig):
    """Configuration for unsupervised learning approaches."""
    
    # Clustering parameters
    CLUSTERING_METHODS = ["dbscan", "spectral", "agglomerative", "kmeans", "hdbscan"]
    DEFAULT_CLUSTERING_METHOD = "spectral"
    
    # DBSCAN parameters
    DBSCAN_PARAMS = {
        "eps": 0.5,
        "min_samples": 5,
        "metric": "euclidean"
    }
    
    # Spectral clustering parameters
    SPECTRAL_PARAMS = {
        "n_clusters": 3,
        "gamma": 0.5,
        "assign_labels": "kmeans",
        "random_state": 42
    }
    
    # Agglomerative clustering parameters
    AGGLOMERATIVE_PARAMS = {
        "n_clusters": 3,
        "linkage": "ward"
    }
    
    # K-means parameters
    KMEANS_PARAMS = {
        "n_clusters": 3,
        "n_init": 20,
        "random_state": 42
    }
    
    # HDBSCAN parameters
    HDBSCAN_PARAMS = {
        "min_cluster_size": 10,
        "min_samples": 5
    }
    
    # Feature extraction parameters
    FEATURE_METHODS = ["minirocket", "autoencoder", "statistical", "raw"]
    DEFAULT_FEATURE_METHOD = "minirocket"
    
    # MiniRocket parameters
    MINIROCKET_PARAMS = {
        "num_kernels": 4000,
        "random_state": 42
    }
    
    # Autoencoder parameters
    AUTOENCODER_PARAMS = {
        "latent_dim": 128,
        "hidden_dim": 256,
        "n_layers": 3,
        "dropout": 0.1,
        "learning_rate": 1e-3,
        "epochs": 100,
        "batch_size": 32
    }
    
    # Statistical features
    STATISTICAL_FEATURES = ["mean", "std", "min", "max", "q25", "q50", "q75"]
    
    # Dimensionality reduction parameters
    PCA_COMPONENTS = 10
    TSNE_PARAMS = {
        "n_components": 2,
        "perplexity": 30,
        "random_state": 42
    }
    UMAP_PARAMS = {
        "n_components": 2,
        "n_neighbors": 15,
        "min_dist": 0.1,
        "random_state": 42
    }
    
    # Outlier detection parameters
    OUTLIER_DETECTION_METHODS = ["isolation_forest", "local_outlier_factor", "dbscan"]
    ISOLATION_FOREST_CONTAMINATION = 0.1
    LOF_N_NEIGHBORS = 20
    
    # Window parameters for time series analysis
    WINDOW_LENGTH = 30
    STRIDE = 15
    MIN_TRIP_LENGTH = 10
    
    # Pattern detection parameters
    MIN_PATTERN_FREQUENCY = 3
    PATTERN_SIMILARITY_THRESHOLD = 0.8 