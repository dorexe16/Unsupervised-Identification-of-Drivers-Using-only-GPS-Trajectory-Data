from src.utils.Drives import drives
import pickle
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import DBSCAN, SpectralClustering
from sklearn.metrics import silhouette_score, roc_auc_score
from sklearn.decomposition import PCA
from sklearn.manifold import trustworthiness
from sktime.transformations.panel.rocket import MiniRocketMultivariate

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

# ----------------------------------------------------------------------
# Ground Truth Setup and Retrieval of Your Object:
# ----------------------------------------------------------------------
# The drive instance ("v") is assumed to contain:
#   - v.neigh_dict: A dictionary mapping from integer keys (1-indexed) to group names.
#   - v.dict_length: A dictionary mapping from group keys to lists of DataFrames.
# Each drive DataFrame must have the columns 'speed', 'acceleration_est_1',
# 'angular_acc', and 'group'.
v = drive_instance
neigh_dict = v.neigh_dict
dict_length = v.dict_length

# Ground truth clusters mapping.
ground_truth_clusters = {
    1: ['Hadar Yosef', 'Neot Afeka A', 'Rom 2000', 'Herzliya B'],
    2: ['Tel Aviv University', 'HaOgen'],
    3: ['Naot Uzi']
}
gt_mapping = {}
for clust, groups in ground_truth_clusters.items():
    for group in groups:
        gt_mapping[group] = clust

# ----------------------------------------------------------------------
# Helper Functions (from your original code)
# ----------------------------------------------------------------------
def convert_dfs_to_arrays(df_list):
    """
    Converts a list of DataFrames (one per drive) into a numpy array and
    extracts the corresponding group labels.
    """
    arrays = []
    labels = []
    for df in df_list:
        arrays.append(df[['speed', 'acceleration_est_1', 'angular_acc']].values.T)
        labels.append(df.iloc[0]['group'] - 1)  # subtract 1 to follow your convention
    return np.stack(arrays, axis=0), labels

def get_processed_data(dict_length, rand):
    """
    For a given random seed, transforms all drives in dict_length using MiniRocket,
    concatenates the results, and then applies two-step feature filtering:
      1. Remove features with very low standard deviation.
      2. Remove features with high pairwise correlation (absolute > 0.95).
    Returns the processed feature array along with the drive labels.
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
    # Phase 1: Remove features with low standard deviation.
    stds = np.std(X_transformed_all, axis=0)
    low_std_threshold = 0.2  # adjust as needed
    mask = stds > low_std_threshold
    X_transformed_all = X_transformed_all[:, mask]
    # Phase 2: Remove highly correlated features.
    df_features = pd.DataFrame(X_transformed_all)
    corr_matrix = df_features.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
    df_features_reduced = df_features.drop(columns=to_drop)
    X_processed = df_features_reduced.values
    return X_processed, t_labels

def calculate_stats_for_largest_cluster(df):
    """
    For each unique group (in df['group']), runs DBSCAN on the features (all columns
    except 'group'), finds the largest cluster, and computes the mean and std.
    Returns a dictionary mapping group id to a dictionary with keys 'mean' and 'std'.
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
        group_stats[group] = {'mean': stats.loc['mean'].values,
                              'std': stats.loc['std'].values}
    return group_stats

def run_experiment_on_processed(X_processed, t_labels, num_pca, neigh_dict):
    """
    Runs PCA with the given number of components on the processed data,
    computes group-level statistics using DBSCAN within the PCA space,
    applies spectral clustering on group means, and computes ROC AUC against
    the ground truth (from gt_mapping). Also returns the total explained variance.
    """
    pca = PCA(n_components=num_pca)
    X_pca = pca.fit_transform(X_processed)
    exp_var = pca.explained_variance_ratio_.sum()

    df_pca = pd.DataFrame(X_pca)
    df_pca['group'] = t_labels

    group_stats = calculate_stats_for_largest_cluster(df_pca)
    unique_groups = sorted(group_stats.keys())
    X_means = np.array([group_stats[g]['mean'] for g in unique_groups])

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

    # Convert numeric group keys back to group names.
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
    return auc, exp_var

# ----------------------------------------------------------------------
# Precompute Processed Data (only for PCA experiments) over 10 iterations:
# ----------------------------------------------------------------------
num_iterations = 10
processed_results = []
seeds = np.random.randint(0, 10000, size=num_iterations)
print("Saving processed data tables over {} iterations (each with a different seed).".format(num_iterations))
for seed in seeds:
    X_processed, t_labels = get_processed_data(dict_length, seed)
    processed_results.append((X_processed, t_labels))

# ----------------------------------------------------------------------
# Hard-coded Results Provided (for Components 2 to 13)
# For PCA, the provided numbers are:
#   - ROC AUC and Explained Variance (Avg Metric)
# For TSNE and UMAP, the provided numbers correspond to:
#   - ROC AUC and Trustworthiness ("New Avg Metric" for components 9–13)
# ----------------------------------------------------------------------
# PCA (Avg ROC AUC and Explained Variance)
pca_auc = {2: 0.632, 3: 0.836, 4: 0.929, 5: 0.929, 6: 0.911, 7: 0.911, 8: 0.893,
           9: 0.875, 10: 0.875, 11: 0.875, 12: 0.875, 13: 0.857}
pca_explained = {2: 0.232, 3: 0.290, 4: 0.336, 5: 0.376, 6: 0.410, 7: 0.439, 8: 0.465,
                 9: 0.498, 10: 0.518, 11: 0.537, 12: 0.553, 13: 0.568}

# TSNE (Avg ROC AUC and Trustworthiness)
tsne_auc = {2: 0.621, 3: 0.668, 4: 0.714, 5: 0.729, 6: 0.707, 7: 0.704, 8: 0.729,
            9: 0.718, 10: 0.718, 11: 0.729, 12: 0.718, 13: 0.718}
tsne_trust = {2: 0.965, 3: 0.973, 4: 0.977, 5: 0.978, 6: 0.980, 7: 0.981, 8: 0.981,
              9: 0.983, 10: 0.983, 11: 0.983, 12: 0.983, 13: 0.983}

# UMAP (Avg ROC AUC and Trustworthiness)
umap_auc = {2: 0.496, 3: 0.514, 4: 0.468, 5: 0.489, 6: 0.464, 7: 0.468, 8: 0.471,
            9: 0.500, 10: 0.493, 11: 0.475, 12: 0.468, 13: 0.461}
umap_trust = {2: 0.937, 3: 0.957, 4: 0.963, 5: 0.966, 6: 0.967, 7: 0.967, 8: 0.968,
              9: 0.969, 10: 0.968, 11: 0.968, 12: 0.969, 13: 0.969}

# ----------------------------------------------------------------------
# Compute PCA Trustworthiness from Processed Data for Components 2 to 13.
# (This value is computed from the PCA mapping using the trustworthiness function.)
# ----------------------------------------------------------------------
pca_trust = {}
for num_pca in range(2, 14):
    trust_list = []
    for X_processed, _ in processed_results:
        pca_model = PCA(n_components=num_pca)
        X_pca = pca_model.fit_transform(X_processed)
        tw = trustworthiness(X_processed, X_pca, n_neighbors=5)
        trust_list.append(tw)
    pca_trust[num_pca] = np.mean(trust_list)

# ----------------------------------------------------------------------
# Plotting
# We now create two separate graphs:
#   Graph 1: ROC AUC values (for PCA, TSNE, and UMAP)
#   Graph 2: Explained Variance (only for PCA) and Trustworthiness (for PCA, TSNE, and UMAP)
# ----------------------------------------------------------------------
components = np.array(sorted(pca_auc.keys()))

# ---- Graph 1: AUC Results ----
plt.figure(figsize=(10, 6))
plt.plot(components, [pca_auc[c] for c in components], marker='o', label='PCA ROC AUC')
plt.plot(components, [tsne_auc[c] for c in components], marker='s', linestyle='--', label='TSNE ROC AUC')
plt.plot(components, [umap_auc[c] for c in components], marker='^', linestyle=':', label='UMAP ROC AUC')
plt.xlabel('Number of Components')
plt.ylabel('ROC AUC')
plt.title('Sensitivity Analysis: ROC AUC Comparison (10 extractors)')
plt.legend()
plt.grid(True)
plt.ylim(0, 1.05)
plt.tight_layout()
plt.show()

# ---- Graph 2: Explained Variance & Trustworthiness ----
plt.figure(figsize=(10, 6))
# PCA Explained Variance (only available for PCA)
plt.plot(components, [pca_explained[c] for c in components], marker='o', label='PCA Explained Variance')
# Trustworthiness for PCA (computed) and for TSNE/UMAP (hard-coded)
plt.plot(components, [pca_trust[c] for c in components], marker='s', linestyle='-', label='PCA Trustworthiness')
plt.plot(components, [tsne_trust[c] for c in components], marker='^', linestyle='--', label='TSNE Trustworthiness')
plt.plot(components, [umap_trust[c] for c in components], marker='d', linestyle=':', label='UMAP Trustworthiness')
plt.xlabel('Number of Components')
plt.ylabel('Metric Value')
plt.title('Sensitivity Analysis: Explained Variance & Trustworthiness (10 extractors)')
plt.legend()
plt.grid(True)
plt.ylim(0, 1.05)
plt.tight_layout()
plt.show()
