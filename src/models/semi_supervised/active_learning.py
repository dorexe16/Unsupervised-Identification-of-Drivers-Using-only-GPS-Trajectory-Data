"""
Active Learning module for semi-supervised driver identification.

This module implements various active learning strategies for selecting
the most informative unlabeled samples for annotation.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
from statsmodels.stats.proportion import proportion_confint
from xgboost import XGBClassifier
from typing import Dict, List, Tuple, Optional
import logging

from ...utils.drives import drives
from ..base_model import BaseModel


class ActiveLearningStrategy:
    """Base class for active learning strategies."""
    
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def select_samples(self, features: np.ndarray, predictions: np.ndarray, 
                      n_samples: int) -> List[int]:
        """Select most informative samples for labeling."""
        raise NotImplementedError


class UncertaintySampling(ActiveLearningStrategy):
    """Uncertainty-based active learning strategy."""
    
    def select_samples(self, features: np.ndarray, predictions: np.ndarray, 
                      n_samples: int) -> List[int]:
        """Select samples with highest prediction uncertainty."""
        # Calculate uncertainty (distance from 0.5 for binary classification)
        uncertainty = 1 - np.abs(predictions - 0.5) * 2
        
        # Select indices of most uncertain samples
        uncertain_indices = np.argsort(uncertainty)[-n_samples:]
        return uncertain_indices.tolist()


class DiversitySampling(ActiveLearningStrategy):
    """Diversity-based active learning strategy."""
    
    def select_samples(self, features: np.ndarray, predictions: np.ndarray, 
                      n_samples: int) -> List[int]:
        """Select diverse samples to maximize coverage."""
        # Use k-means++ style selection for diversity
        selected = []
        remaining = list(range(len(features)))
        
        # Select first sample randomly
        if remaining:
            first_idx = np.random.choice(remaining)
            selected.append(first_idx)
            remaining.remove(first_idx)
        
        # Select remaining samples to maximize distance
        for _ in range(min(n_samples - 1, len(remaining))):
            if not remaining:
                break
                
            distances = pairwise_distances(
                features[remaining], 
                features[selected], 
                metric='euclidean'
            )
            min_distances = np.min(distances, axis=1)
            next_idx = remaining[np.argmax(min_distances)]
            selected.append(next_idx)
            remaining.remove(next_idx)
        
        return selected


class ActiveLearner:
    """Main active learning class for driver identification."""
    
    def __init__(self, config):
        self.config = config
        self.model = None
        self.strategy = self._create_strategy()
        self.logger = logging.getLogger(__name__)
        
        # Active learning state
        self.labeled_indices = set()
        self.unlabeled_indices = set()
        self.iteration = 0
        self.history = []
    
    def _create_strategy(self) -> ActiveLearningStrategy:
        """Create active learning strategy based on config."""
        if self.config.ACTIVE_LEARNING_STRATEGY == "uncertainty":
            return UncertaintySampling(self.model, self.config)
        elif self.config.ACTIVE_LEARNING_STRATEGY == "diversity":
            return DiversitySampling(self.model, self.config)
        else:
            raise ValueError(f"Unknown strategy: {self.config.ACTIVE_LEARNING_STRATEGY}")
    
    def _create_model(self):
        """Create model based on configuration."""
        if self.config.MODEL_TYPE == "xgboost":
            return XGBClassifier(**self.config.XGBOOST_PARAMS)
        else:
            raise ValueError(f"Unknown model type: {self.config.MODEL_TYPE}")
    
    def initialize_labeled_set(self, features: np.ndarray, labels: np.ndarray,
                              trip_ids: np.ndarray) -> None:
        """Initialize with a small labeled set using pattern-based selection."""
        # Group by trip ID and find most representative trips
        trip_centroids = self._compute_trip_centroids(features, trip_ids)
        
        # Select initial labeled trips based on driver patterns
        initial_trips = self._select_initial_trips(trip_centroids, labels, trip_ids)
        
        # Convert trip IDs to sample indices
        for trip_id in initial_trips:
            trip_indices = np.where(trip_ids == trip_id)[0]
            self.labeled_indices.update(trip_indices)
        
        # Remaining samples are unlabeled
        all_indices = set(range(len(features)))
        self.unlabeled_indices = all_indices - self.labeled_indices
        
        self.logger.info(f"Initialized with {len(self.labeled_indices)} labeled samples")
    
    def _compute_trip_centroids(self, features: np.ndarray, 
                               trip_ids: np.ndarray) -> Dict:
        """Compute centroid features for each trip."""
        trip_centroids = {}
        for trip_id in np.unique(trip_ids):
            trip_mask = trip_ids == trip_id
            trip_features = features[trip_mask]
            centroid = np.mean(trip_features, axis=0)
            trip_centroids[trip_id] = centroid
        return trip_centroids
    
    def _select_initial_trips(self, trip_centroids: Dict, labels: np.ndarray,
                             trip_ids: np.ndarray) -> List:
        """Select initial trips for labeling using distance-based strategy."""
        # Get unique trip labels
        trip_labels = {}
        for trip_id in trip_centroids.keys():
            trip_mask = trip_ids == trip_id
            trip_labels[trip_id] = labels[trip_mask][0]  # First label for trip
        
        selected_trips = []
        
        # Select diverse trips from each driver
        for driver_id in np.unique(list(trip_labels.values())):
            driver_trips = [tid for tid, label in trip_labels.items() 
                          if label == driver_id]
            
            if driver_trips:
                # Compute distances from driver centroid
                driver_centroids = np.array([trip_centroids[tid] for tid in driver_trips])
                driver_center = np.mean(driver_centroids, axis=0)
                
                distances = np.linalg.norm(driver_centroids - driver_center, axis=1)
                n_select = max(1, int(self.config.INITIAL_LABELED_RATIO * len(driver_trips)))
                
                # Select most distant trips (outliers are more informative)
                selected_idx = np.argsort(distances)[-n_select:]
                selected_trips.extend([driver_trips[i] for i in selected_idx])
        
        return selected_trips
    
    def wilson_confidence_interval(self, p: float, n: int) -> Tuple[float, float]:
        """Compute Wilson confidence interval for proportion."""
        alpha = self.config.WILSON_CONFIDENCE_ALPHA
        return proportion_confint(int(p * n), n, method="wilson", alpha=alpha)
    
    def train_iteration(self, features: np.ndarray, labels: np.ndarray) -> Dict:
        """Perform one iteration of active learning."""
        # Train model on labeled data
        labeled_idx = list(self.labeled_indices)
        self.model = self._create_model()
        self.model.fit(features[labeled_idx], labels[labeled_idx])
        
        # Get predictions on all data
        predictions = self.model.predict_proba(features)[:, 1]
        
        # Store iteration results
        iteration_results = {
            'iteration': self.iteration,
            'labeled_count': len(self.labeled_indices),
            'unlabeled_count': len(self.unlabeled_indices),
            'predictions': predictions.copy()
        }
        
        return iteration_results
    
    def query_next_samples(self, features: np.ndarray, predictions: np.ndarray,
                          trip_ids: np.ndarray) -> List[int]:
        """Query next samples for labeling using confidence-based selection."""
        # Compute trip-level confidence
        trip_confidences = self._compute_trip_confidences(
            predictions, trip_ids, self.unlabeled_indices
        )
        
        newly_confident = []
        delta = self.config.CONFIDENCE_MARGIN
        
        for trip_id, (mean_prob, n_segments) in trip_confidences.items():
            if trip_id in [trip_ids[i] for i in self.labeled_indices]:
                continue  # Skip already labeled trips
                
            # Compute Wilson confidence interval
            lo, hi = self.wilson_confidence_interval(mean_prob, n_segments)
            
            # Check if trip is confidently classified
            if hi < 0.5 - delta or lo > 0.5 + delta:
                trip_mask = trip_ids == trip_id
                trip_indices = np.where(trip_mask)[0]
                newly_confident.extend(trip_indices)
        
        return newly_confident
    
    def _compute_trip_confidences(self, predictions: np.ndarray, trip_ids: np.ndarray,
                                 unlabeled_indices: set) -> Dict:
        """Compute confidence statistics for each trip."""
        trip_confidences = {}
        
        for trip_id in np.unique(trip_ids):
            trip_mask = trip_ids == trip_id
            trip_indices = np.where(trip_mask)[0]
            
            # Only consider unlabeled segments
            unlabeled_trip_indices = [i for i in trip_indices if i in unlabeled_indices]
            
            if unlabeled_trip_indices:
                trip_predictions = predictions[unlabeled_trip_indices]
                mean_prob = np.mean(trip_predictions)
                n_segments = len(trip_predictions)
                trip_confidences[trip_id] = (mean_prob, n_segments)
        
        return trip_confidences
    
    def fit(self, features: np.ndarray, labels: np.ndarray, 
            trip_ids: np.ndarray) -> List[Dict]:
        """Run complete active learning process."""
        # Initialize
        self.initialize_labeled_set(features, labels, trip_ids)
        
        history = []
        
        for iteration in range(self.config.MAX_ITERATIONS):
            self.iteration = iteration
            
            # Train model and get predictions
            results = self.train_iteration(features, labels)
            history.append(results)
            
            # Query next samples
            newly_confident = self.query_next_samples(
                features, results['predictions'], trip_ids
            )
            
            if not newly_confident:
                self.logger.info(f"No new confident samples found. Stopping at iteration {iteration}")
                break
            
            # Add newly confident samples to labeled set
            self.labeled_indices.update(newly_confident)
            self.unlabeled_indices -= set(newly_confident)
            
            self.logger.info(
                f"Iteration {iteration}: {len(newly_confident)} new samples, "
                f"{len(self.labeled_indices)} total labeled"
            )
        
        return history 