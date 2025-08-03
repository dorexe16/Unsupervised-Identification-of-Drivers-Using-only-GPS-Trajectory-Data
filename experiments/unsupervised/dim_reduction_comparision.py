from src.utils.Drives import drives
import pickle
import os
os.environ['OMP_NUM_THREADS'] = '1'
# ----------------------------------------------------------------------
# Load the drive instance from file
# ----------------------------------------------------------------------
car_id = 460631
file_path = fr"C:\Users\dorex\Desktop\Unsupervised Identification of Drivers Using only GPS Trajectory Data\Unsupervised-Identification-of-Drivers-Using-only-GPS-Trajectory-Data\data\{car_id}.pkl"

if os.path.exists(file_path):
    with open(file_path, "rb") as file:
        drive_instance = pickle.load(file)
    print(f"Loaded object for car_id: {drive_instance.car_id}")
else:
    print(f"File not found: {file_path}")
    exit()

import os
import pickle
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import DBSCAN, SpectralClustering
from sklearn.metrics import silhouette_score, roc_auc_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, trustworthiness
from sktime.transformations.panel.rocket import MiniRocketMultivariate
import umap.umap_ as umap  # Requires umap-learn package

# ----- Use your saved lst_car object -----
v = drive_instance
neigh_dict = v.neigh_dict
dict_length = v.dict_length


# ---------------------------
# Helper function: Convert list of DataFrames to arrays.
# ---------------------------
def convert_dfs_to_arrays(df_list):
    """
    Converts a list of DataFrames (one per drive) into a numpy array
    along with a list of true group labels.
    """
    arrays = []
    labels = []
    for df in df_list:
        arrays.append(df[['speed', 'acceleration_est_1', 'angular_acc']].values.T)
        labels.append(df.iloc[0]['group'] - 1)
    return np.stack(arrays, axis=0), labels


# ---------------------------
# Helper function: Get processed data (after MiniRocket & feature filtering)
# ---------------------------
def get_processed_data(dict_length, rand):
    """
    For a given random seed, run MiniRocket on all drives in dict_length,
    concatenate the results, and then apply two feature filtering steps:
      - Remove columns with very low standard deviation.
      - Remove columns with high pairwise correlation (abs(corr) > threshold).
    Returns the processed features (numpy array) and corresponding drive labels.
    """
    t_labels = []
    X_transformed_list = []

    for key in dict_length:
        arrays, true_labels = convert_dfs_to_arrays(dict_length[key])
        t_labels += true_labels

        rocket = MiniRocketMultivariate(num_kernels=10000, random_state=int(rand))
        rocket.fit(arrays)
        X_transformed = rocket.transform(arrays)
        X_transformed_list.append(X_transformed)

    X_transformed_all = np.concatenate(X_transformed_list, axis=0)

    # Feature Filtering Phase 1: Remove features with very low std
    stds = np.std(X_transformed_all, axis=0)
    low_std_threshold = 0.2  # adjust threshold as needed
    mask = stds > low_std_threshold
    X_transformed_all = X_transformed_all[:, mask]

    # Feature Filtering Phase 2: Remove features with high correlation
    df_features = pd.DataFrame(X_transformed_all)
    corr_matrix = df_features.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
    df_features_reduced = df_features.drop(columns=to_drop)

    X_processed = df_features_reduced.values
    return X_processed, t_labels


# ---------------------------
# Ground Truth Setup
# ---------------------------
ground_truth_clusters = {
    1: ['Hadar Yosef', 'Neot Afeka A', 'Rom 2000', 'Herzliya B'],
    2: ['Tel Aviv University', 'HaOgen'],
    3: ['Naot Uzi']
}
gt_mapping = {}
for clust, groups in ground_truth_clusters.items():
    for group in groups:
        gt_mapping[group] = clust


# ---------------------------
# Helper function: Compute group statistics (using DBSCAN)
# ---------------------------
def calculate_stats_for_largest_cluster(df):
    """
    For each unique group in df['group'], run DBSCAN on the features (all columns except 'group')
    and compute the mean and std of the largest cluster.
    Returns a dict mapping group to { 'mean': ..., 'std': ... }.
    """
    group_stats = {}
    for group in df['group'].unique():
        group_data = df[df['group'] == group].copy()
        features = group_data.drop(columns=['group'])
        dbscan = DBSCAN(eps=1, min_samples=5)
        dbscan_labels = dbscan.fit_predict(features)
        group_data['dbscan_label'] = dbscan_labels
        largest_cluster_label = group_data['dbscan_label'].value_counts().idxmax()
        largest_cluster = group_data[group_data['dbscan_label'] == largest_cluster_label]
        stats = largest_cluster.drop(columns=['group', 'dbscan_label']).agg(['mean', 'std'])
        group_stats[group] = {
            'mean': stats.loc['mean'].values,
            'std': stats.loc['std'].values
        }
    return group_stats


# ---------------------------
# Extended Experiment Function with Method Parameter
# ---------------------------
def run_experiment_on_processed(X_processed, t_labels, num_components, neigh_dict, method='PCA'):
    """
    Given processed features and drive labels, reduce dimensionality with the specified method
    (PCA, TSNE, or UMAP) using num_components. For PCA, both the explained variance and
    trustworthiness (embedding quality) are computed; for TSNE and UMAP only trustworthiness is computed.
    Then, the function builds a drive-level dataframe, computes group statistics via DBSCAN,
    clusters the groups using Spectral Clustering, and computes a ROC AUC (comparing the predicted
    group similarities to ground truth as defined by gt_mapping).

    Returns:
      - auc: ROC AUC score based on cluster similarity.
      - exp_metric: For PCA, a tuple (explained_variance, trustworthiness);
                    for TSNE and UMAP, the trustworthiness value.
    """
    if method == 'PCA':
        transformer = PCA(n_components=num_components)
        X_emb = transformer.fit_transform(X_processed)
        explained = transformer.explained_variance_ratio_.sum()
        trust_val = trustworthiness(X_processed, X_emb, n_neighbors=5)
        exp_metric = (explained, trust_val)
    elif method == 'TSNE':
        transformer = TSNE(n_components=num_components, method='exact', random_state=42)
        X_emb = transformer.fit_transform(X_processed)
        exp_metric = trustworthiness(X_processed, X_emb, n_neighbors=5)
    elif method == 'UMAP':
        transformer = umap.UMAP(n_components=num_components, random_state=42)
        X_emb = transformer.fit_transform(X_processed)
        exp_metric = trustworthiness(X_processed, X_emb, n_neighbors=5)
    else:
        raise ValueError("Method not recognized. Use 'PCA', 'TSNE', or 'UMAP'.")

    # Build a dataframe with the reduced features and corresponding drive labels.
    df_emb = pd.DataFrame(X_emb)
    df_emb['group'] = t_labels

    # Compute group-level statistics.
    group_stats = calculate_stats_for_largest_cluster(df_emb)
    unique_groups = sorted(group_stats.keys())
    X_means = np.array([group_stats[g]['mean'] for g in unique_groups])

    # Run spectral clustering on the group means.
    silhouette_scores = []
    max_k = len(neigh_dict)
    for k in range(2, max_k):
        spectral = SpectralClustering(n_clusters=k, random_state=42)
        labels_k = spectral.fit_predict(X_means)
        if len(set(labels_k)) > 1:
            silhouette_scores.append(silhouette_score(X_means, labels_k))
        else:
            silhouette_scores.append(-1)
    best_k = np.argmax(silhouette_scores) + 2
    spectral = SpectralClustering(n_clusters=best_k, random_state=42)
    spectral_labels = spectral.fit_predict(X_means)

    # Build ROC AUC using predicted similarity vs. ground truth.
    groups_names = [neigh_dict[g + 1] for g in unique_groups]
    y_true = []
    y_pred = []
    for i in range(len(groups_names)):
        for j in range(i + 1, len(groups_names)):
            true_sim = 1 if gt_mapping.get(groups_names[i]) == gt_mapping.get(groups_names[j]) else 0
            pred_sim = 1 if spectral_labels[i] == spectral_labels[j] else 0
            y_true.append(true_sim)
            y_pred.append(pred_sim)
    try:
        auc = roc_auc_score(y_true, y_pred)
    except Exception as e:
        auc = np.nan

    return auc, exp_metric


# ---------------------------
# Main Sensitivity Analysis Loop
# ---------------------------
# Save processed data tables (to avoid re-running MiniRocket each time).
num_iterations = 30
processed_results = []
seeds = np.random.randint(0, 10000, size=num_iterations)
print("Saving processed data tables over {} iterations...".format(num_iterations))
for seed in seeds:
    X_processed, t_labels = get_processed_data(dict_length, seed)
    processed_results.append((X_processed, t_labels))

# Define the methods to compare.
methods = ['PCA', 'TSNE', 'UMAP']

# Dictionaries to collect results.
results_auc = {m: {} for m in methods}
# For quality metrics, we'll store:
# - For PCA: separate dictionaries for explained variance and trustworthiness.
# - For TSNE and UMAP: a single dictionary (trustworthiness).
results_explained = {}  # Only for PCA.
results_trust = {m: {} for m in ['TSNE', 'UMAP']}

# For PCA, we want to collect both.
results_explained['PCA'] = {}
results_trust['PCA'] = {}

for num_components in range(2, 16):  # Try components from 2 to 15.
    print(f"\nRunning experiments for {num_components} components...")
    for method in methods:
        auc_list = []
        # For PCA, we collect two metrics.
        if method == 'PCA':
            explained_list = []
            trust_list = []
        else:
            metric_list = []
        for X_proc, t_labels in processed_results:
            auc, exp_metric = run_experiment_on_processed(X_proc, t_labels, num_components, neigh_dict, method=method)
            auc_list.append(auc)
            if method == 'PCA':
                # exp_metric is a tuple: (explained_variance, trustworthiness)
                explained_list.append(exp_metric[0])
                trust_list.append(exp_metric[1])
            else:
                metric_list.append(exp_metric)
        avg_auc = np.nanmean(auc_list)
        results_auc[method][num_components] = avg_auc

        if method == 'PCA':
            avg_explained = np.mean(explained_list)
            avg_trust = np.mean(trust_list)
            results_explained['PCA'][num_components] = avg_explained
            results_trust['PCA'][num_components] = avg_trust
        else:
            avg_metric = np.mean(metric_list)
            results_trust[method][num_components] = avg_metric

        print(f"Method: {method}, Components: {num_components}, Avg ROC AUC: {avg_auc:.3f}, ", end="")
        if method == 'PCA':
            print(f"Expl.Var: {avg_explained:.3f}, Trust: {avg_trust:.3f}")
        else:
            print(f"Trust: {avg_metric:.3f}")

# ---------------------------
# Plotting the Results
# ---------------------------
methods_colors = {'PCA': 'tab:blue', 'TSNE': 'tab:orange', 'UMAP': 'tab:green'}

# Graph 1: ROC AUC Comparison (all methods)
components = np.array(sorted(results_auc['PCA'].keys()))
plt.figure(figsize=(10, 6))
for method in methods:
    comp_vals = sorted(results_auc[method].keys())
    auc_vals = [results_auc[method][c] for c in comp_vals]
    plt.plot(comp_vals, auc_vals, marker='o', color=methods_colors[method],
             linestyle='-', label=f'{method} ROC AUC')
plt.xlabel('Number of Components')
plt.ylabel('ROC AUC')
plt.title('Sensitivity Analysis: ROC AUC Comparison')
plt.legend()
plt.grid(True)
plt.ylim(0, 1)
plt.tight_layout()
plt.show()

# Graph 2: Quality Metrics Comparison
# For PCA we plot both explained variance and trustworthiness.
plt.figure(figsize=(10, 6))
# PCA Explained Variance:
comp_vals = sorted(results_explained['PCA'].keys())
explained_vals = [results_explained['PCA'][c] for c in comp_vals]
plt.plot(comp_vals, explained_vals, marker='s', color=methods_colors['PCA'],
         linestyle='-', label='PCA Explained Variance')
# PCA Trustworthiness:
trust_vals = [results_trust['PCA'][c] for c in comp_vals]
plt.plot(comp_vals, trust_vals, marker='o', color=methods_colors['PCA'],
         linestyle='--', label='PCA Trustworthiness')
# TSNE Trustworthiness:
comp_vals_tsne = sorted(results_trust['TSNE'].keys())
tsne_trust_vals = [results_trust['TSNE'][c] for c in comp_vals_tsne]
plt.plot(comp_vals_tsne, tsne_trust_vals, marker='^', color=methods_colors['TSNE'],
         linestyle='--', label='TSNE Trustworthiness')
# UMAP Trustworthiness:
comp_vals_umap = sorted(results_trust['UMAP'].keys())
umap_trust_vals = [results_trust['UMAP'][c] for c in comp_vals_umap]
plt.plot(comp_vals_umap, umap_trust_vals, marker='d', color=methods_colors['UMAP'],
         linestyle='--', label='UMAP Trustworthiness')

plt.xlabel('Number of Components')
plt.ylabel('Embedding Quality Metric')
plt.title('Sensitivity Analysis: Explained Variance & Trustworthiness')
plt.legend()
plt.grid(True)
plt.ylim(0, 1)
plt.tight_layout()
plt.show()
