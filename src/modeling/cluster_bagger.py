from src.modeling.Extractor import *
from src.preprocessing.Data_Preparation2 import normalize_time_series

def cluster_bagging(drive_obj, extractor_heads=150):

    num_runs = extractor_heads
    random_states = np.random.randint(0, 10000, size=num_runs)
    spectral_matrix = np.zeros((len(drive_obj.neigh_dict), len(drive_obj.neigh_dict)))

    for rand in random_states:
        spectral_labels = compute_cluster(drive_obj.neigh_dict, drive_obj.dict_length, rand)

        # Dictionary to count occurrences of clustering
        s_cluster_dict = {}

        # Count occurrences of clustering
        for i, group in enumerate(drive_obj.neigh_dict.keys()):
            s_cluster_dict[group] = spectral_labels[i]

        # Update matrix based on clustering results
        for group in s_cluster_dict:
            for j in s_cluster_dict:
                if s_cluster_dict[group] == s_cluster_dict[j]:
                    spectral_matrix[int(group - 1), int(j - 1)] += 1

    rounded_spectral_matrix = np.round(spectral_matrix / num_runs, decimals=2)
    df_spectral = pd.DataFrame(rounded_spectral_matrix,
                               index=[drive_obj.neigh_dict[i + 1] for i in range(len(drive_obj.neigh_dict))],
                               columns=[drive_obj.neigh_dict[i + 1] for i in range(len(drive_obj.neigh_dict))])
    drive_obj.df_spectral = df_spectral
    return df_spectral


def classifier(drive_obj, threshold=0.9):
    """
    Clusters groups within a drive object based on spectral similarity.

    :param drive_obj: A `drives` object containing `df_spectral`.
    :param threshold: The similarity threshold for clustering.
    """
    drive_obj.clusters = {}
    df_spectral = drive_obj.df_spectral.data if hasattr(drive_obj.df_spectral, 'data') else drive_obj.df_spectral  # Handle styled DataFrame
    visited = set()

    try:
        # Iterate over groups (index) to find clusters
        for group in df_spectral.index:
            if group in visited:
                continue  # Skip already clustered groups

            # Find all groups with similarity >= threshold
            high_sim_groups = [g for g in df_spectral.columns if df_spectral.at[group, g] >= threshold and g not in visited]

            # Check if the group has no high similarity connections
            if len(high_sim_groups) == 1:  # The group itself
                # Check borderline similarities
                borderline_groups = [g for g in df_spectral.columns if (1 - threshold) < df_spectral.at[group, g] < threshold]

                if borderline_groups:
                    continue  # Ignore this group (as it has borderline connections)
                else:
                    # Form a single cluster for this group
                    cluster_id = len(drive_obj.clusters) + 1
                    drive_obj.clusters[cluster_id] = [group]
                    visited.add(group)
            elif len(high_sim_groups) > 1:
                # Validate all groups in high_sim_groups are compatible
                valid_cluster = all(
                    df_spectral.at[g1, g2] >= threshold
                    for g1 in high_sim_groups for g2 in high_sim_groups if g1 != g2
                )

                if valid_cluster:
                    # Form a cluster for groups with high similarity
                    cluster_id = len(drive_obj.clusters) + 1
                    drive_obj.clusters[cluster_id] = high_sim_groups
                    visited.update(high_sim_groups)
                else:
                    # Ignore the group as it doesn't fit a valid cluster
                    visited.add(group)
    except Exception as e:
        print(f"Error processing drive object {drive_obj.car_id}: {e}")
    return drive_obj
