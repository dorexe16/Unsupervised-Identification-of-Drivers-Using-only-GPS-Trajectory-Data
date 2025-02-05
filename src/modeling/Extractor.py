
from sklearn.cluster import DBSCAN, SpectralClustering
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sktime.transformations.panel.rocket import MiniRocketMultivariate
import pandas as pd

def calculate_stats_for_largest_cluster(df):
    group_stats = {}

    for group in df['group'].unique():
        group_data = df[df['group'] == group]
        dbscan = DBSCAN(eps=1, min_samples=5)
        dbscan_labels = dbscan.fit_predict(group_data[['PC1', 'PC2', 'PC3', 'PC4', 'PC5']])

        # Add DBSCAN labels to the DataFrame
        group_data['dbscan_label'] = dbscan_labels
        # Get the largest cluster
        largest_cluster_label = group_data['dbscan_label'].value_counts().idxmax()
        largest_cluster = group_data[group_data['dbscan_label'] == largest_cluster_label]

        # Calculate mean and std of features in the largest cluster
        group_stats[group] = {
            'mean': largest_cluster[['PC1', 'PC2', 'PC3', 'PC4', 'PC5']].mean().values,
            'std': largest_cluster[['PC1', 'PC2', 'PC3', 'PC4', 'PC5']].std().values
        }

    return group_stats




def compute_cluster(drive_obj, rand=100):
    """
    Computes clusters using MiniRocket transformation, PCA, and Spectral Clustering.
    Returns the labels for Spectral Clustering.
    """
    spectral_matrix = np.zeros((len(drive_obj.neighbors_dict), len(drive_obj.neighbors_dict)))

    def convert_dfs_to_arrays(df_list):
        """Convert list of DataFrames to numpy arrays."""
        arrays = []
        labels = []
        for df in df_list:
            arrays.append(
                df[['speed', 'acceleration_est_1', 'angular_acc']].values.T)  # Shape: (num_features, num_timesteps)
            labels.append(df.iloc[0]['group'] - 1)
        return np.stack(arrays, axis=0), labels

    t_labels = []  # True labels
    X_transformed_list = []  # List to store transformed arrays

    # Transform each DataFrame in dict_length using MiniRocket
    for key in drive_obj.dict_length:
        arrays, true_labels = convert_dfs_to_arrays(drive_obj.dict_length[key])
        t_labels += true_labels

        # Create and fit the MiniRocket transformer
        rocket = MiniRocketMultivariate(num_kernels=10000, random_state=int(rand))
        rocket.fit(arrays)

        # Transform the data and store it
        X_transformed = rocket.transform(arrays)
        X_transformed_list.append(X_transformed)

    # Concatenate all transformed arrays along the first axis
    X_transformed_all = np.concatenate(X_transformed_list, axis=0)

    # Apply PCA to reduce dimensions to 3 components
    pca = PCA(n_components=5)
    X_pca = pca.fit_transform(X_transformed_all)
    df_pca = pd.DataFrame(X_pca, columns=['PC1', 'PC2', 'PC3', 'PC4', 'PC5'])
    df_pca['group'] = t_labels

    # Calculate statistics for the largest cluster
    group_stats = calculate_stats_for_largest_cluster(df_pca)

    # Extract means for clustering
    X_means = np.array([stats['mean'] for stats in group_stats.values()])
    group_labels = list(group_stats.keys())

    def evaluate_spectral_clustering(X):

        silhouette_scores_spectral = []

        #         for k in range(2, len(neigh_dict)-1):
        for k in range(2, len(drive_obj.neighbors_dict)):
            # Spectral Clustering
            spectral = SpectralClustering(n_clusters=k, random_state=42)
            spectral_labels = spectral.fit_predict(X)

            # Check if there is more than one unique label
            if len(set(spectral_labels)) > 1:
                # Calculate silhouette score only if there is more than one unique cluster
                silhouette_scores_spectral.append(silhouette_score(X, spectral_labels))
            else:
                # Append a placeholder score (e.g., -1) if only one cluster is found
                silhouette_scores_spectral.append(-1)

        return silhouette_scores_spectral

    # Evaluate silhouette scores for Spectral Clustering
    silhouette_scores_spectral = evaluate_spectral_clustering(X_means)

    # Determine the best k for Spectral Clustering based on silhouette scores
    best_k_spectral = np.argmax(silhouette_scores_spectral) + 2

    # Apply Spectral Clustering with the best k
    spectral_labels = SpectralClustering(n_clusters=best_k_spectral, random_state=42).fit_predict(X_means)

    return spectral_labels