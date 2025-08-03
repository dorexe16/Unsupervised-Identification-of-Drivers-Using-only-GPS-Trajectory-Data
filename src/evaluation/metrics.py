"""
Evaluation metrics for driver identification models.

This module provides various metrics for evaluating clustering
and classification performance in driver identification tasks.
"""
import numpy as np
from sklearn.metrics import (
    normalized_mutual_info_score,
    adjusted_mutual_info_score,
    silhouette_score,
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_auc_score
)
from scipy.optimize import linear_sum_assignment
from typing import Dict, List, Optional, Union
import warnings


def clustering_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute clustering accuracy using optimal label assignment.
    
    Args:
        y_true: True cluster labels
        y_pred: Predicted cluster labels
        
    Returns:
        Accuracy score after optimal label assignment
    """
    # Create confusion matrix
    n_true = len(np.unique(y_true))
    n_pred = len(np.unique(y_pred))
    
    # Handle case where number of clusters differs
    max_clusters = max(n_true, n_pred)
    
    # Create cost matrix
    cost_matrix = np.zeros((max_clusters, max_clusters))
    
    for i, true_label in enumerate(np.unique(y_true)):
        for j, pred_label in enumerate(np.unique(y_pred)):
            mask = (y_true == true_label) & (y_pred == pred_label)
            cost_matrix[i, j] = np.sum(mask)
    
    # Find optimal assignment
    row_ind, col_ind = linear_sum_assignment(-cost_matrix)
    
    # Compute accuracy
    optimal_matches = cost_matrix[row_ind, col_ind].sum()
    accuracy = optimal_matches / len(y_true)
    
    return accuracy


def compute_clustering_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                             X: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Compute comprehensive clustering evaluation metrics.
    
    Args:
        y_true: True cluster labels
        y_pred: Predicted cluster labels
        X: Original features (for silhouette score)
        
    Returns:
        Dictionary of metric values
    """
    metrics = {}
    
    # Mutual information scores
    metrics['nmi'] = normalized_mutual_info_score(y_true, y_pred, average_method='arithmetic')
    metrics['ami'] = adjusted_mutual_info_score(y_true, y_pred, average_method='arithmetic')
    
    # Clustering accuracy
    metrics['accuracy'] = clustering_accuracy(y_true, y_pred)
    
    # Silhouette score (requires original features)
    if X is not None and len(np.unique(y_pred)) > 1:
        try:
            metrics['silhouette'] = silhouette_score(X, y_pred)
        except Exception as e:
            warnings.warn(f"Could not compute silhouette score: {e}")
            metrics['silhouette'] = np.nan
    else:
        metrics['silhouette'] = np.nan
    
    return metrics


def compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                                 y_proba: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Compute comprehensive classification evaluation metrics.
    
    Args:
        y_true: True class labels
        y_pred: Predicted class labels
        y_proba: Predicted class probabilities (optional)
        
    Returns:
        Dictionary of metric values
    """
    metrics = {}
    
    # Basic classification metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    
    # Precision, recall, F1-score
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )
    
    metrics['precision'] = precision
    metrics['recall'] = recall
    metrics['f1'] = f1
    
    # Per-class metrics
    precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    
    for i, (p, r, f) in enumerate(zip(precision_per_class, recall_per_class, f1_per_class)):
        metrics[f'precision_class_{i}'] = p
        metrics[f'recall_class_{i}'] = r
        metrics[f'f1_class_{i}'] = f
    
    # AUC (if probabilities provided)
    if y_proba is not None:
        try:
            if len(np.unique(y_true)) == 2:  # Binary classification
                metrics['auc'] = roc_auc_score(y_true, y_proba)
            else:  # Multi-class
                metrics['auc'] = roc_auc_score(y_true, y_proba, multi_class='ovr')
        except Exception as e:
            warnings.warn(f"Could not compute AUC: {e}")
            metrics['auc'] = np.nan
    
    return metrics


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                   metric_names: List[str], X: Optional[np.ndarray] = None,
                   y_proba: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Compute specified metrics for model evaluation.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        metric_names: List of metric names to compute
        X: Original features (for silhouette score)
        y_proba: Predicted probabilities (for AUC)
        
    Returns:
        Dictionary of computed metrics
    """
    all_metrics = {}
    
    # Determine if this is clustering or classification task
    has_continuous_predictions = len(np.unique(y_pred)) > 10
    
    if has_continuous_predictions or 'silhouette' in metric_names:
        # Treat as clustering task
        clustering_metrics = compute_clustering_metrics(y_true, y_pred, X)
        all_metrics.update(clustering_metrics)
    else:
        # Treat as classification task
        classification_metrics = compute_classification_metrics(y_true, y_pred, y_proba)
        all_metrics.update(classification_metrics)
    
    # Filter to requested metrics
    requested_metrics = {name: all_metrics.get(name, np.nan) for name in metric_names}
    
    return requested_metrics


def purity_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute purity score for clustering evaluation.
    
    Args:
        y_true: True cluster labels
        y_pred: Predicted cluster labels
        
    Returns:
        Purity score
    """
    # Confusion matrix
    contingency_matrix = confusion_matrix(y_true, y_pred)
    
    # Purity is the sum of max values in each column divided by total
    purity = np.sum(np.max(contingency_matrix, axis=0)) / np.sum(contingency_matrix)
    
    return purity


def inverse_purity_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute inverse purity score for clustering evaluation.
    
    Args:
        y_true: True cluster labels
        y_pred: Predicted cluster labels
        
    Returns:
        Inverse purity score
    """
    # Switch arguments to compute inverse purity
    return purity_score(y_pred, y_true)


def v_measure_score(y_true: np.ndarray, y_pred: np.ndarray, beta: float = 1.0) -> float:
    """
    Compute V-measure score (harmonic mean of homogeneity and completeness).
    
    Args:
        y_true: True cluster labels
        y_pred: Predicted cluster labels
        beta: Weight for completeness vs homogeneity
        
    Returns:
        V-measure score
    """
    from sklearn.metrics import v_measure_score as sklearn_v_measure
    return sklearn_v_measure(y_true, y_pred, beta=beta)


def driver_identification_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                                trip_ids: np.ndarray) -> Dict[str, float]:
    """
    Compute driver identification specific metrics.
    
    Args:
        y_true: True driver labels
        y_pred: Predicted driver labels
        trip_ids: Trip identifiers
        
    Returns:
        Dictionary of driver identification metrics
    """
    metrics = {}
    
    # Trip-level accuracy
    trip_true = []
    trip_pred = []
    
    for trip_id in np.unique(trip_ids):
        trip_mask = trip_ids == trip_id
        # Use majority vote for trip-level prediction
        trip_true.append(y_true[trip_mask][0])  # Assume all segments in trip have same label
        trip_pred.append(np.bincount(y_pred[trip_mask]).argmax())
    
    metrics['trip_accuracy'] = accuracy_score(trip_true, trip_pred)
    
    # Segment-level accuracy
    metrics['segment_accuracy'] = accuracy_score(y_true, y_pred)
    
    # Driver consistency (within-trip agreement)
    trip_consistencies = []
    for trip_id in np.unique(trip_ids):
        trip_mask = trip_ids == trip_id
        trip_predictions = y_pred[trip_mask]
        if len(trip_predictions) > 1:
            # Compute agreement ratio
            most_common = np.bincount(trip_predictions).argmax()
            agreement_ratio = np.mean(trip_predictions == most_common)
            trip_consistencies.append(agreement_ratio)
    
    metrics['driver_consistency'] = np.mean(trip_consistencies) if trip_consistencies else 1.0
    
    return metrics 