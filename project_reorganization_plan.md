# Project Reorganization Plan: Semi-Supervised Driver Identification

## Current Issues with Project Structure
1. **Scattered experiment files** at root level
2. **Mixed naming conventions** (expirements vs experiments)
3. **Results stored inconsistently** across multiple directories
4. **Unclear separation** between different model approaches
5. **No clear distinction** between supervised, unsupervised, and semi-supervised approaches

## Proposed New Structure

```
driver-identification-project/
├── README.md
├── requirements.txt
├── setup.py
├── config/
│   ├── __init__.py
│   ├── base_config.py
│   ├── unsupervised_config.py
│   ├── semi_supervised_config.py
│   └── supervised_config.py
├── data/
│   ├── raw/                     # Original GPS trajectory files
│   ├── processed/               # Cleaned and preprocessed data
│   ├── features/                # Extracted features (MiniRocket, etc.)
│   └── splits/                  # Train/val/test splits for semi-supervised learning
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── preprocessing.py     # Data cleaning and preparation
│   │   ├── feature_extraction.py
│   │   └── data_loaders.py      # Dataset classes and loaders
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base_model.py
│   │   ├── unsupervised/
│   │   │   ├── __init__.py
│   │   │   ├── clustering.py    # DBSCAN, Spectral Clustering
│   │   │   ├── autoencoders.py  # RNN Autoencoders
│   │   │   └── rocket_features.py
│   │   ├── semi_supervised/
│   │   │   ├── __init__.py
│   │   │   ├── active_learning.py
│   │   │   ├── self_training.py
│   │   │   ├── co_training.py
│   │   │   └── pseudo_labeling.py
│   │   └── supervised/
│   │       ├── __init__.py
│   │       ├── transformers.py  # Driver identification transformers
│   │       └── classical_ml.py  # XGBoost, etc.
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py           # NMI, AMI, Silhouette, etc.
│   │   ├── visualization.py     # Plotting and analysis
│   │   └── statistical_tests.py
│   └── utils/
│       ├── __init__.py
│       ├── drives.py           # Drive data structures
│       ├── geo_utils.py        # Geographical calculations
│       └── common.py           # Common utilities
├── experiments/
│   ├── unsupervised/
│   │   ├── clustering_experiments.py
│   │   ├── feature_analysis.py
│   │   └── dimensionality_reduction.py
│   ├── semi_supervised/
│   │   ├── active_learning_experiments.py
│   │   ├── self_training_experiments.py
│   │   └── comparative_analysis.py
│   └── supervised/
│       ├── transformer_experiments.py
│       └── baseline_experiments.py
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_unsupervised_analysis.ipynb
│   ├── 04_semi_supervised_experiments.ipynb
│   ├── 05_model_comparison.ipynb
│   └── 06_results_visualization.ipynb
├── results/
│   ├── figures/                 # All plots and visualizations
│   ├── tables/                  # Result summaries and tables
│   ├── models/                  # Trained model checkpoints
│   │   ├── unsupervised/
│   │   ├── semi_supervised/
│   │   └── supervised/
│   └── logs/                    # Training logs and metrics
├── tests/
│   ├── __init__.py
│   ├── test_data_processing.py
│   ├── test_models.py
│   └── test_evaluation.py
└── docs/
    ├── methodology.md
    ├── api_reference.md
    └── experiment_protocols.md
```

## Migration Steps

### Phase 1: Create New Directory Structure
1. Create the new directory hierarchy
2. Initialize Python packages with `__init__.py` files
3. Set up configuration management

### Phase 2: Reorganize Source Code
1. Move and refactor preprocessing code
2. Separate models by learning paradigm
3. Consolidate utility functions
4. Create proper data loading infrastructure

### Phase 3: Reorganize Experiments and Results
1. Categorize experiments by learning approach
2. Migrate notebook files with clear naming
3. Organize results by experiment type
4. Create standardized result formats

### Phase 4: Documentation and Testing
1. Update documentation
2. Create proper testing infrastructure
3. Set up configuration management
4. Standardize experiment protocols

## Key Improvements

### 1. Clear Separation of Learning Paradigms
- **Unsupervised**: Clustering, pattern detection, feature learning
- **Semi-supervised**: Active learning, self-training, pseudo-labeling
- **Supervised**: Direct driver classification with labeled data

### 2. Standardized Experiment Management
- Consistent naming conventions
- Clear separation of concerns
- Reproducible experiment protocols
- Standardized result formats

### 3. Better Code Organization
- Modular design with clear interfaces
- Proper Python package structure
- Configuration management
- Testing infrastructure

### 4. Enhanced Documentation
- Clear methodology documentation
- API reference
- Experiment protocols
- Result interpretation guides

## Configuration Management

Create configuration files for different experimental setups:

```python
# config/semi_supervised_config.py
class SemiSupervisedConfig:
    # Data parameters
    data_dir = "data/processed"
    feature_type = "minirocket"  # or "autoencoder", "raw"
    
    # Semi-supervised parameters
    initial_labeled_ratio = 0.1
    active_learning_strategy = "uncertainty"  # or "diversity", "hybrid"
    self_training_threshold = 0.9
    max_iterations = 50
    
    # Model parameters
    model_type = "xgboost"  # or "transformer", "mlp"
    
    # Evaluation parameters
    cv_folds = 5
    random_seed = 42
```

This reorganization will make your project more maintainable, reproducible, and easier to extend with new semi-supervised learning approaches. 