"""
Base model class for driver identification.

This module provides a common interface for all model types
(supervised, unsupervised, and semi-supervised).
"""
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import pickle
import logging
from pathlib import Path


class BaseModel(ABC):
    """Abstract base class for all driver identification models."""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.is_trained = False
        self.model_params = {}
        
    @abstractmethod
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> 'BaseModel':
        """
        Train the model on the provided data.
        
        Args:
            X: Feature matrix
            y: Target labels (optional for unsupervised methods)
            **kwargs: Additional parameters
            
        Returns:
            Self (for method chaining)
        """
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X: Feature matrix
            
        Returns:
            Predictions
        """
        pass
    
    def save_model(self, filepath: Optional[Path] = None) -> Path:
        """
        Save the trained model to disk.
        
        Args:
            filepath: Path to save the model (optional)
            
        Returns:
            Path where model was saved
        """
        if filepath is None:
            model_name = f"{self.__class__.__name__}_{self.config.RANDOM_SEED}.pkl"
            filepath = self.config.MODELS_DIR / model_name
        
        # Create directory if it doesn't exist
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Save model state
        model_state = {
            'model': self,
            'config': self.config,
            'is_trained': self.is_trained,
            'model_params': self.model_params
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_state, f)
        
        self.logger.info(f"Model saved to {filepath}")
        return filepath
    
    @classmethod
    def load_model(cls, filepath: Path) -> 'BaseModel':
        """
        Load a trained model from disk.
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            Loaded model instance
        """
        with open(filepath, 'rb') as f:
            model_state = pickle.load(f)
        
        model = model_state['model']
        model.logger = logging.getLogger(cls.__name__)
        model.logger.info(f"Model loaded from {filepath}")
        
        return model
    
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters."""
        return self.model_params.copy()
    
    def set_params(self, **params) -> 'BaseModel':
        """Set model parameters."""
        self.model_params.update(params)
        return self
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(trained={self.is_trained})"


class SupervisedModel(BaseModel):
    """Base class for supervised learning models."""
    
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> 'SupervisedModel':
        """Train supervised model."""
        if y is None:
            raise ValueError("Supervised models require target labels (y)")
        
        self._fit_implementation(X, y, **kwargs)
        self.is_trained = True
        return self
    
    @abstractmethod
    def _fit_implementation(self, X: np.ndarray, y: np.ndarray, **kwargs) -> None:
        """Implementation-specific training logic."""
        pass
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities (if supported)."""
        raise NotImplementedError("Probability prediction not implemented")


class UnsupervisedModel(BaseModel):
    """Base class for unsupervised learning models."""
    
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> 'UnsupervisedModel':
        """Train unsupervised model."""
        self._fit_implementation(X, **kwargs)
        self.is_trained = True
        return self
    
    @abstractmethod
    def _fit_implementation(self, X: np.ndarray, **kwargs) -> None:
        """Implementation-specific training logic."""
        pass
    
    def fit_predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """Fit model and return predictions in one step."""
        self.fit(X, **kwargs)
        return self.predict(X)


class SemiSupervisedModel(BaseModel):
    """Base class for semi-supervised learning models."""
    
    def fit(self, X: np.ndarray, y: np.ndarray, labeled_mask: np.ndarray, 
            **kwargs) -> 'SemiSupervisedModel':
        """
        Train semi-supervised model.
        
        Args:
            X: Feature matrix
            y: Target labels (may contain unknown values for unlabeled data)
            labeled_mask: Boolean mask indicating which samples are labeled
            **kwargs: Additional parameters
        """
        self._fit_implementation(X, y, labeled_mask, **kwargs)
        self.is_trained = True
        return self
    
    @abstractmethod
    def _fit_implementation(self, X: np.ndarray, y: np.ndarray, 
                           labeled_mask: np.ndarray, **kwargs) -> None:
        """Implementation-specific training logic."""
        pass


class ModelEvaluator:
    """Utility class for model evaluation."""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def evaluate_model(self, model: BaseModel, X_test: np.ndarray, 
                      y_test: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance on test data."""
        from ..evaluation.metrics import compute_metrics
        
        predictions = model.predict(X_test)
        metrics = compute_metrics(y_test, predictions, self.config.METRICS)
        
        self.logger.info(f"Model evaluation: {metrics}")
        return metrics
    
    def cross_validate(self, model: BaseModel, X: np.ndarray, y: np.ndarray,
                      cv_folds: int = None) -> Dict[str, float]:
        """Perform cross-validation."""
        from sklearn.model_selection import cross_val_score
        from sklearn.metrics import accuracy_score, make_scorer
        
        if cv_folds is None:
            cv_folds = self.config.CV_FOLDS
        
        # Wrap model for sklearn compatibility
        sklearn_model = SklearnModelWrapper(model)
        
        scores = cross_val_score(
            sklearn_model, X, y, 
            cv=cv_folds, 
            scoring=make_scorer(accuracy_score)
        )
        
        return {
            'cv_mean': np.mean(scores),
            'cv_std': np.std(scores),
            'cv_scores': scores.tolist()
        }


class SklearnModelWrapper:
    """Wrapper to make BaseModel compatible with sklearn."""
    
    def __init__(self, model: BaseModel):
        self.model = model
    
    def fit(self, X, y):
        self.model.fit(X, y)
        return self
    
    def predict(self, X):
        return self.model.predict(X)
    
    def predict_proba(self, X):
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        else:
            raise NotImplementedError("Model does not support probability prediction") 