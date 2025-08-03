from src.utils.Drives import drives
import pickle
import os
import time  # Add at the top with other imports

# Set the environment variable to avoid MKL warnings.
os.environ['OMP_NUM_THREADS'] = '1'

import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import DBSCAN, SpectralClustering
from sklearn.metrics import silhouette_score, roc_auc_score
from sklearn.decomposition import PCA
from sktime.transformations.panel.rocket import MiniRocketMultivariate

# =============================================================================
# LOAD DRIVE INSTANCE
# =============================================================================
car_id = 460631
file_path = fr"C:\Users\dorex\Desktop\Unsupervised Identification of Drivers Using only GPS Trajectory Data\Unsupervised-Identification-of-Drivers-Using-only-GPS-Trajectory-Data\data\{car_id}.pkl"

if os.path.exists(file_path):
    with open(file_path, "rb") as file:
        drive_instance = pickle.load(file)
    print(f"Loaded object for car_id: {drive_instance.car_id}")
else:
    print(f"File not found: {file_path}")
    exit()

# Use the drive instance.
v = drive_instance
neigh_dict = v.neigh_dict
dict_length = v.dict_length

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

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

def get_processed_data(dict_length, rand, filter_type='both'):
    """
    Processes the drives in dict_length using MiniRocket.
    Optimized version that processes correlation in batches for better efficiency.
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

    if filter_type in ['std', 'both']:
        # Remove low std features
        stds = np.std(X_transformed_all, axis=0)
        low_std_threshold = 0.2
        mask = stds > low_std_threshold
        X_transformed_all = X_transformed_all[:, mask]

    if filter_type in ['corr', 'both']:
        # Optimized correlation filtering using batches
        df_features = pd.DataFrame(X_transformed_all)
        batch_size = 1000  # Process correlation in batches
        columns_to_drop = set()  # Use a set to avoid duplicates
        
        # Process correlations in batches
        for i in range(0, df_features.shape[1], batch_size):
            batch_end = min(i + batch_size, df_features.shape[1])
            batch_cols = df_features.iloc[:, i:batch_end]
            
            # Calculate correlation matrix for the batch
            corr_matrix = batch_cols.corr().abs()
            
            # Find highly correlated features within the batch
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            
            # Find columns with high correlation
            for col in upper.columns:
                corr_series = upper[col]
                if any(corr_series > 0.95):
                    # Add the column name to drop
                    columns_to_drop.add(col)
        
        # Drop highly correlated features
        if columns_to_drop:
            df_features = df_features.drop(columns=list(columns_to_drop))
            X_transformed_all = df_features.values

    return X_transformed_all, t_labels

def calculate_stats_for_largest_cluster(df):
    """
    For each unique group in df['group'], run DBSCAN on the features and compute
    the mean and std of the largest cluster.
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

def run_experiment_on_processed(X_processed, t_labels, num_components, neigh_dict):
    """
    Runs PCA with num_components on processed data and computes ROC AUC.
    """
    pca = PCA(n_components=num_components)
    X_pca = pca.fit_transform(X_processed)
    exp_var = pca.explained_variance_ratio_.sum()

    df_pca = pd.DataFrame(X_pca)
    df_pca['group'] = t_labels

    group_stats = calculate_stats_for_largest_cluster(df_pca)
    unique_groups = sorted(group_stats.keys())
    X_means = np.array([group_stats[g]['mean'] for g in unique_groups])

    # Run spectral clustering on the group means
    silhouette_scores = []
    max_k = len(neigh_dict)
    for k in range(2, max_k):
        spectral = SpectralClustering(n_clusters=k)
        labels_k = spectral.fit_predict(X_means)
        if len(set(labels_k)) > 1:
            silhouette_scores.append(silhouette_score(X_means, labels_k))
        else:
            silhouette_scores.append(-1)
    best_k = np.argmax(silhouette_scores) + 2
    spectral = SpectralClustering(n_clusters=best_k)
    spectral_labels = spectral.fit_predict(X_means)

    # Compute ROC AUC
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

# =============================================================================
# GROUND TRUTH SETUP
# =============================================================================
ground_truth_clusters = {
    1: ['Hadar Yosef', 'Neot Afeka A', 'Rom 2000', 'Herzliya B'],
    2: ['Tel Aviv University', 'HaOgen'],
    3: ['Naot Uzi']
}
gt_mapping = {}
for clust, groups in ground_truth_clusters.items():
    for group in groups:
        gt_mapping[group] = clust

# =============================================================================
# EXPERIMENT SETTINGS
# =============================================================================
N_RUNS = 30  # Number of independent runs
ITER_PER_RUN = 50  # Number of iterations per run
COMPONENTS = 10  # Fixed number of components
FILTER_OPTIONS = ['none', 'std', 'corr', 'both']  # Different filtering methods to compare
CHECKPOINT_FILE = 'clustering_results_checkpoint.pkl'  # File to save results

# =============================================================================
# RUN THE EXPERIMENT
# =============================================================================
print(f"Running {N_RUNS} runs with {ITER_PER_RUN} iterations each...")

# Try to load previous results
if os.path.exists(CHECKPOINT_FILE):
    print(f"Loading previous results from {CHECKPOINT_FILE}")
    with open(CHECKPOINT_FILE, 'rb') as f:
        checkpoint_data = pickle.load(f)
        results = checkpoint_data['results']
        timing_stats = checkpoint_data['timing_stats']
        completed_runs = checkpoint_data.get('completed_runs', 0)
else:
    # Dictionary to store results for each method and run
    results = {filt: {'runs': [[] for _ in range(N_RUNS)]} for filt in FILTER_OPTIONS}
    timing_stats = {filt: [] for filt in FILTER_OPTIONS}  # To store timing information
    completed_runs = 0

# Run the experiment
for run in range(completed_runs, N_RUNS):
    print(f"\nRun {run + 1}/{N_RUNS}")
    
    for iteration in range(ITER_PER_RUN):
        # Generate one seed per iteration to use across all filter types
        seed = random.randint(0, 10000)
        
        # First get the unfiltered data
        X_unfiltered, t_labels = get_processed_data(dict_length, seed, filter_type='none')
        
        for filt in FILTER_OPTIONS:
            if iteration == 0:
                print(f"Processing filter type: {filt}")
            
            start_time = time.time()
            
            if filt == 'none':
                # Reuse unfiltered data
                X_processed = X_unfiltered
            else:
                # Apply filtering to the unfiltered data
                X_processed = X_unfiltered.copy()
                
                if filt in ['std', 'both']:
                    # Remove low std features
                    stds = np.std(X_processed, axis=0)
                    low_std_threshold = 0.2
                    mask = stds > low_std_threshold
                    X_processed = X_processed[:, mask]
                    if iteration == 0 and run == completed_runs:
                        print(f"After std filtering: {X_processed.shape[1]} features")

                if filt in ['corr', 'both']:
                    initial_features = X_processed.shape[1]
                    # Optimized correlation filtering using batches
                    df_features = pd.DataFrame(X_processed)
                    batch_size = 1000
                    columns_to_drop = set()
                    
                    for i in range(0, df_features.shape[1], batch_size):
                        batch_end = min(i + batch_size, df_features.shape[1])
                        batch_cols = df_features.iloc[:, i:batch_end]
                        corr_matrix = batch_cols.corr().abs()
                        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
                        for col in upper.columns:
                            corr_series = upper[col]
                            if any(corr_series > 0.95):
                                columns_to_drop.add(col)
                    
                    if columns_to_drop:
                        df_features = df_features.drop(columns=list(columns_to_drop))
                        X_processed = df_features.values
                    if iteration == 0 and run == completed_runs:
                        print(f"After correlation filtering: {X_processed.shape[1]} features (removed {initial_features - X_processed.shape[1]})")
            
            auc, _ = run_experiment_on_processed(X_processed, t_labels, COMPONENTS, neigh_dict)
            results[filt]['runs'][run].append(auc)
            
            # Store timing
            timing_stats[filt].append(time.time() - start_time)
    
    # Save checkpoint after each run
    checkpoint_data = {
        'results': results,
        'timing_stats': timing_stats,
        'completed_runs': run + 1,
        'experiment_settings': {
            'N_RUNS': N_RUNS,
            'ITER_PER_RUN': ITER_PER_RUN,
            'COMPONENTS': COMPONENTS,
            'FILTER_OPTIONS': FILTER_OPTIONS
        }
    }
    with open(CHECKPOINT_FILE, 'wb') as f:
        pickle.dump(checkpoint_data, f)
    print(f"Saved checkpoint after run {run + 1}")

# Print timing statistics
print("\nTiming Statistics:")
print("-" * 50)
for filt in FILTER_OPTIONS:
    avg_time = np.mean(timing_stats[filt])
    std_time = np.std(timing_stats[filt])
    print(f"{filt}: {avg_time:.3f}s ± {std_time:.3f}s per iteration")

# =============================================================================
# PLOTTING THE RESULTS
# =============================================================================
# Calculate mean and std for each iteration
iterations = np.arange(ITER_PER_RUN)
colors = {'none': 'black', 'std': 'tab:blue', 'corr': 'tab:orange', 'both': 'tab:green'}

plt.figure(figsize=(12, 8))

for filt in FILTER_OPTIONS:
    # Convert runs to numpy array for easier calculation
    run_data = np.array(results[filt]['runs'])  # Shape: (N_RUNS, ITER_PER_RUN)
    
    # Calculate mean and std for each iteration
    means = np.mean(run_data, axis=0)
    stds = np.std(run_data, axis=0)
    
    # Plot mean line
    plt.plot(iterations, means, label=filt, color=colors[filt])
    
    # Plot std band
    plt.fill_between(iterations, means - stds, means + stds, 
                    alpha=0.2, color=colors[filt])

plt.xlabel('Extractors')
plt.ylabel('ROC-AUC (mean ± std)')
plt.title(f'Driver-ID clustering – {N_RUNS} runs × {ITER_PER_RUN} Extractors\nPCA components = {COMPONENTS}')
plt.legend(title='Pruning method')
plt.grid(True)
plt.ylim(0.5, 1)  # Adjusted y-axis range
plt.tight_layout()

# Save the figure
plt.savefig('clustering_results.png', dpi=300, bbox_inches='tight')
plt.show()

# Print final statistics
print("\nFinal Statistics (last iteration):")
print("-" * 50)
for filt in FILTER_OPTIONS:
    run_data = np.array(results[filt]['runs'])
    final_means = np.mean(run_data[:, -1])
    final_stds = np.std(run_data[:, -1])
    print(f"\n{filt.upper()} method:")
    print(f"ROC AUC: {final_means:.3f} ± {final_stds:.3f}")
