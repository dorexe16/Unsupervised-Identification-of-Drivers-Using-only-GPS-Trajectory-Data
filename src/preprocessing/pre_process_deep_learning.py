# import glob
# import os
# import pickle
# import numpy as np
# import pandas as pd
# import torch
# from torch.nn.utils.rnn import pad_sequence
# import math
#
#
# class MergedDrives:
#     def __init__(self):
#         # This mimics data.splitted_ts_groups
#         self.splitted_ts_groups = {}
#         # New: We'll store a mapping from new group_id -> original group name
#         self.group_names = {}  # e.g., { 0: "some_group_name", 1: "another_name", ...}
#
#
# def load_and_merge_data(data_dir="data", neigh_dict=None):
#     """
#     Loads all .pkl files in `data_dir`, ensuring that 460631.pkl is loaded first.
#     Merges them into a single MergedDrives object, reindexing group IDs and
#     excluding trips of length < 6.
#
#     Additionally, creates a dictionary mapping each car (pkl file) to the list
#     of group IDs originating from that file.
#
#     If `neigh_dict` is provided, it should map compound keys (car_key, original group ID)
#     to group names. In other words, your neigh_dict should have keys like:
#         (filename, original_group_id)
#     We then store these names in `merged_data.group_names` using the new group ID.
#     """
#     # Get all .pkl files in data_dir
#     pkl_files = sorted(glob.glob(os.path.join(data_dir, "*.pkl")))
#
#     # Ensure '460631.pkl' is loaded first (if it exists)
#     main_file = os.path.join(data_dir, "460631.pkl")
#     if main_file in pkl_files:
#         pkl_files.remove(main_file)
#         pkl_files = [main_file] + pkl_files
#
#     print("Loading these files in order:")
#     for f in pkl_files:
#         print("   ", f)
#
#     merged_data = MergedDrives()
#     car_to_groups = {}  # Maps car (file) to its new group IDs.
#     group_offset = 0
#
#     for idx, pkl_file in enumerate(pkl_files):
#         print(f"\nLoading {pkl_file} ...")
#         with open(pkl_file, 'rb') as f:
#             data_temp = pickle.load(f)  # Expected to have data_temp.splitted_ts_groups
#
#         # Re-label group IDs from 0...N-1 for this file
#         temp_dict = {}
#         i = 0
#         # Sorting the original keys ensures consistency.
#         for original_gid in sorted(data_temp.splitted_ts_groups.keys()):
#             temp_dict[i] = data_temp.splitted_ts_groups[original_gid]
#             i += 1
#         data_temp.splitted_ts_groups = temp_dict
#         temp_keys = sorted(temp_dict.keys())
#
#         # Use the file's basename as the car identifier
#         car_key = os.path.basename(pkl_file)
#         new_groups = []  # List to store new group IDs for this car
#
#         if idx == 0:
#             # For the first file, keep group IDs as they are.
#             for g in temp_keys:
#                 # Initialize an empty list for each group
#                 merged_data.splitted_ts_groups[g] = []
#                 # Copy only trips with >=6 points (and <=60, as in your code)
#                 for df in data_temp.splitted_ts_groups[g]:
#                     if len(df) >= 6 and len(df) <= 60:
#                         merged_data.splitted_ts_groups[g].append(df)
#                 new_groups.append(g)
#
#                 # Use compound key (car_key, original group id) to get a name.
#                 compound_key = (car_key, g)
#                 if neigh_dict is not None and 0 in neigh_dict:
#                     merged_data.group_names[g] = neigh_dict[g]
#                 elif 1 in neigh_dict:
#                     merged_data.group_names[g] = neigh_dict[g-1]
#                 else:
#                     merged_data.group_names[g] = f"Group_{g}"
#             car_to_groups[car_key] = new_groups
#             group_offset = max(temp_keys)
#         else:
#             # For subsequent files, shift the group IDs by the current offset.
#             for g in temp_keys:
#                 new_g = g + group_offset
#                 if new_g not in merged_data.splitted_ts_groups:
#                     merged_data.splitted_ts_groups[new_g] = []
#                 for df in data_temp.splitted_ts_groups[g]:
#                     if len(df) >= 8 and len(df) <= 60:
#                         merged_data.splitted_ts_groups[new_g].append(df)
#                 new_groups.append(new_g)
#
#                 # Look up the group name using the compound key.
#                 compound_key = (car_key, g)
#                 if neigh_dict is not None and 0 in neigh_dict:
#                     merged_data.group_names[g] = neigh_dict[g]
#                 elif 1 in neigh_dict:
#                     merged_data.group_names[g] = neigh_dict[g - 1]
#                 else:
#                     merged_data.group_names[g] = f"Group_{g}"
#             car_to_groups[car_key] = new_groups
#             group_offset += max(temp_keys)
#
#     print("\nFinal merged group IDs:", sorted(merged_data.splitted_ts_groups.keys()))
#     print("\nCar to Groups Mapping:")
#     for car, groups in car_to_groups.items():
#         print(f"{car}: {groups}")
#
#     return merged_data, car_to_groups
#
#
# def compute_global_stats(merged_data, columns=None):
#     """
#     Compute global mean & std for specified columns across all data frames.
#     Returns two dicts: means and stds keyed by column name.
#     """
#     if columns is None:
#         columns = ['longitude', 'latitude', 'speed',
#                    'acceleration_est_1', 'angular_acc']
#
#     all_vals = {col: [] for col in columns}
#
#     # Gather all values across groups
#     for group_id, time_series_list in merged_data.splitted_ts_groups.items():
#         for df in time_series_list:
#             for col in columns:
#                 all_vals[col].extend(df[col].values.tolist())
#
#     # Compute means and standard deviations
#     means = {}
#     stds = {}
#     for col in columns:
#         arr = np.array(all_vals[col], dtype=np.float32)
#         means[col] = arr.mean()
#         stds[col] = arr.std() if arr.std() > 1e-6 else 1.0  # avoid division by zero
#     return means, stds
#
#
# def normalize_dataframe(df, means, stds, columns=None):
#     """
#     Normalize the specified columns of the dataframe in-place using the provided means and stds.
#     """
#     if columns is None:
#         columns = ['longitude', 'latitude', 'speed',
#                    'acceleration_est_1', 'angular_acc']
#     for col in columns:
#         df[col] = (df[col] - means[col]) / stds[col]
#     return df
#
#
# def preprocess_data(data, means=None, stds=None, use_cyclical_hour=True, normalize_road_speed=True):
#     """
#     Converts `data.splitted_ts_groups` dictionary into padded tensors and normalizes continuous columns
#     if means/stds are provided.
#
#     Returns:
#        X_padded: FloatTensor of shape (B, T, input_dim)
#        y_padded: FloatTensor of shape (B, T, 2)  [predicting speed, acceleration]
#        group_ids_tensor: LongTensor of shape (B,)
#        lengths_tensor: LongTensor of shape (B,)
#     """
#     columns_to_normalize = ['longitude', 'latitude', 'speed',
#                             'acceleration_est_1', 'angular_acc']
#
#     X_list, y_list = [], []
#     group_ids_list, lengths = [], []
#
#     for group_id, time_series_list in data.splitted_ts_groups.items():
#         for df in time_series_list:
#             df = df.sort_values("orig_time")  # Ensure correct time order
#
#             # Normalize continuous columns if provided
#             if (means is not None) and (stds is not None):
#                 df = normalize_dataframe(df, means, stds, columns=columns_to_normalize)
#
#             # Prepare hour-of-day (cyclical or not)
#             hour = pd.to_datetime(df['orig_time']).dt.hour.values.astype(float)
#             if use_cyclical_hour:
#                 sin_hour = np.sin(2.0 * math.pi * hour / 24.0)
#                 cos_hour = np.cos(2.0 * math.pi * hour / 24.0)
#             else:
#                 sin_hour = hour
#                 cos_hour = None
#
#             # Prepare road_speed (with optional normalization)
#             road_speed = df['road_speed'].values
#             road_speed = np.clip(road_speed, 0, 99)
#             if normalize_road_speed:
#                 rs_mean, rs_std = 50.0, 25.0  # Example placeholder values
#                 road_speed = (road_speed - rs_mean) / rs_std
#
#             # Next-step target for speed and acceleration (after normalization)
#             target = df[['speed', 'acceleration_est_1']].values
#
#             # Ensure there are at least 3 points to shift by one safely
#             if len(df) < 3:
#                 continue
#
#             # Build the design matrix for all timesteps:
#             # We use the 5 normalized columns, then add hour (or cyclical components) and road_speed.
#             feats_5 = df[['longitude', 'latitude', 'speed', 'acceleration_est_1', 'angular_acc']].values
#             if use_cyclical_hour:
#                 combined_feats = np.column_stack((feats_5, sin_hour, cos_hour, road_speed))
#             else:
#                 combined_feats = np.column_stack((feats_5, hour, road_speed))
#
#             # Shift so that the t-th row in X corresponds to the t+1-th row in y.
#             X_np = combined_feats[:-1]
#             y_np = target[1:]
#
#             # Convert to torch tensors
#             X_tensor = torch.tensor(X_np, dtype=torch.float32)
#             y_tensor = torch.tensor(y_np, dtype=torch.float32)
#
#             X_list.append(X_tensor)
#             y_list.append(y_tensor)
#             group_ids_list.append(torch.tensor(group_id, dtype=torch.long))
#             lengths.append(X_tensor.shape[0])
#
#     X_padded = pad_sequence(X_list, batch_first=True, padding_value=0.0)
#     y_padded = pad_sequence(y_list, batch_first=True, padding_value=0.0)
#     group_ids_tensor = torch.stack(group_ids_list)
#     lengths_tensor = torch.tensor(lengths, dtype=torch.long)
#
#     return X_padded, y_padded, group_ids_tensor, lengths_tensor
import glob
import os
import pickle
import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
import math


class MergedDrives:
    def __init__(self):
        # This mimics data.splitted_ts_groups
        self.splitted_ts_groups = {}
        # New: We'll store a mapping from new group_id -> original group name
        self.group_names = {}  # e.g., { 0: "some_group_name", 1: "another_name", ...}


def load_and_merge_data(data_dir="data", neigh_dict=None):
    """
    Loads all .pkl files in `data_dir`, ensuring that 460631.pkl is loaded first.
    Merges them into a single MergedDrives object, reindexing group IDs and
    excluding trips of length < 6.

    Additionally, creates a dictionary mapping each car (pkl file) to the list
    of group IDs originating from that file.

    If `neigh_dict` is provided, it should map compound keys (car_key, original group ID)
    to group names. In other words, your neigh_dict should have keys like:
        (filename, original_group_id)
    We then store these names in `merged_data.group_names` using the new group ID.
    """
    # Get all .pkl files in data_dir
    pkl_files = sorted(glob.glob(os.path.join(data_dir, "*.pkl")))

    # Ensure '460631.pkl' is loaded first (if it exists)
    main_file = os.path.join(data_dir, "460631.pkl")
    # pkl_files = [main_file]
    if main_file in pkl_files:
        pkl_files.remove(main_file)
        pkl_files = [main_file] + pkl_files

    print("Loading these files in order:")
    for f in pkl_files:
        print("   ", f)

    merged_data = MergedDrives()
    car_to_groups = {}  # Maps car (file) to its new group IDs.
    group_offset = 0

    for idx, pkl_file in enumerate(pkl_files):
        print(f"\nLoading {pkl_file} ...")
        with open(pkl_file, 'rb') as f:
            data_temp = pickle.load(f)  # Expected to have data_temp.splitted_ts_groups

        # Re-label group IDs from 0...N-1 for this file
        temp_dict = {}
        temp_neigh = {}
        i = 0
        # Sorting the original keys ensures consistency.
        for original_gid in sorted(data_temp.groups_ts.keys()):
            temp_dict[i] = data_temp.groups_ts[original_gid]
            temp_neigh[i] = data_temp.neigh_dict[original_gid]
            i += 1
        data_temp.groups_ts = temp_dict
        temp_keys = sorted(temp_dict.keys())

        # Use the file's basename as the car identifier
        car_key = os.path.basename(pkl_file)
        new_groups = []  # List to store new group IDs for this car

        if idx == 0:
            # For the first file, keep group IDs as they are.
            print(data_temp.groups_ts[1][0].columns)
            for g in temp_keys:
                # Initialize an empty list for each group
                merged_data.splitted_ts_groups[g] = []
                # Copy only trips with >=6 points (and <=60, as in your code)
                for df in data_temp.groups_ts[g]:
                    if len(df) >= 6 and len(df) <= 60:
                        merged_data.splitted_ts_groups[g].append(df)
                new_groups.append(g)

                # Use compound key (car_key, original group id) to get a name.
                compound_key = (car_key, g)
                if g in temp_neigh.keys():
                    merged_data.group_names[g] = temp_neigh[g]
                else:
                    merged_data.group_names[g] = f"Group_{g}"
            car_to_groups[car_key] = new_groups
            group_offset = max(temp_keys) + 1
        else:
            # For subsequent files, shift the group IDs by the current offset.
            for g in temp_keys:
                new_g = g + group_offset
                if new_g not in merged_data.splitted_ts_groups:
                    merged_data.splitted_ts_groups[new_g] = []
                for df in data_temp.groups_ts[g]:
                    if len(df) >= 8 and len(df) <= 60:
                        merged_data.splitted_ts_groups[new_g].append(df)
                new_groups.append(new_g)

                # Look up the group name using the compound key.
                compound_key = (car_key, g)
                if g in temp_neigh.keys():
                    merged_data.group_names[new_g] = temp_neigh[g]
                else:
                    merged_data.group_names[new_g] = f"Group_{g}"
            car_to_groups[car_key] = new_groups
            group_offset += max(temp_keys)

    print("\nFinal merged group IDs:", sorted(merged_data.splitted_ts_groups.keys()))
    print("\nCar to Groups Mapping:")
    for car, groups in car_to_groups.items():
        print(f"{car}: {groups} -> {[merged_data.group_names[g] for g in groups]}")

    return merged_data, car_to_groups


def compute_global_stats(merged_data, columns=None):
    """
    Compute global mean & std for specified columns across all data frames.
    Returns two dicts: means and stds keyed by column name.
    """
    if columns is None:
        columns = ['longitude', 'latitude', 'speed',
                   'acceleration_est_1', 'angular_acc']

    all_vals = {col: [] for col in columns}

    # Gather all values across groups
    for group_id, time_series_list in merged_data.splitted_ts_groups.items():
        for df in time_series_list:
            for col in columns:
                all_vals[col].extend(df[col].values.tolist())

    # Compute means and standard deviations
    means = {}
    stds = {}
    for col in columns:
        arr = np.array(all_vals[col], dtype=np.float32)
        means[col] = arr.mean()
        stds[col] = arr.std() if arr.std() > 1e-6 else 1.0  # avoid division by zero
    return means, stds


def get_trip_time_category(df):
    """
    Determines the time-of-day category for a trip based on the first timestamp.
    Categories:
      - 'morning': 5:00 <= hour < 11:00
      - 'rush_hour': 11:00 <= hour < 17:00
      - 'night': all other hours
    """
    hour = pd.to_datetime(df['orig_time']).dt.hour.iloc[0]
    if 5 <= hour < 11:
        return 'morning'
    elif 11 <= hour < 17:
        return 'rush_hour'
    else:
        return 'night'


def get_road_speed_bin(rs_value):
    """
    Bins the road_speed value into one of three categories.
    Example thresholds:
      - 'low' for road_speed < 30
      - 'medium' for 30 <= road_speed < 60
      - 'high' for road_speed >= 60
    """
    if rs_value < 30:
        return 'low'
    elif rs_value < 60:
        return 'medium'
    else:
        return 'high'


def compute_group_specific_stats(merged_data):
    """
    Computes group-specific statistics (mean and std) for the columns:
    'speed', 'acceleration_est_1', and 'angular_acc'.

    The groups are defined by a tuple (time_category, road_speed_bin).
    Uses the first road_speed value of the trip (instead of an average) for binning.
    Returns a dictionary where keys are group tuples and values are dicts mapping
    each column to (mean, std).
    """
    group_values = {}  # key: (time_category, road_speed_bin), value: dict of lists for each column
    for group_id, trip_list in merged_data.splitted_ts_groups.items():
        for df in trip_list:
            time_cat = get_trip_time_category(df)
            rs_val = df['road_speed'].iloc[0]  # using the first road_speed value
            rs_bin = get_road_speed_bin(rs_val)
            key = (time_cat, rs_bin)
            if key not in group_values:
                group_values[key] = {'speed': [], 'acceleration_est_1': [], 'angular_acc': []}
            for col in ['speed', 'acceleration_est_1', 'angular_acc']:
                group_values[key][col].extend(df[col].values.tolist())

    group_stats = {}
    for key, col_dict in group_values.items():
        group_stats[key] = {}
        for col, values in col_dict.items():
            arr = np.array(values, dtype=np.float32)
            mean = arr.mean()
            std = arr.std() if arr.std() > 1e-6 else 1.0
            group_stats[key][col] = (mean, std)
    return group_stats


def normalize_dataframe(df, means, stds, columns=None):
    """
    Normalize the specified columns of the dataframe in-place using the provided means and stds.
    Used for columns that do not receive group-specific normalization.
    """
    if columns is None:
        columns = ['longitude', 'latitude', 'speed',
                   'acceleration_est_1', 'angular_acc']
    for col in columns:
        df[col] = (df[col] - means[col]) / stds[col]
    return df


def normalize_dataframe_groupwise(df, global_means, global_stds, group_stats, group_key, columns_to_normalize):
    """
    Normalize the dataframe in-place.
    For the group-specific columns ('speed', 'acceleration_est_1', 'angular_acc'),
    use the statistics from group_stats (if available), otherwise fall back to global stats.
    Other columns (e.g., 'longitude', 'latitude') are normalized using global statistics.
    """
    group_cols = ['speed', 'acceleration_est_1', 'angular_acc']
    for col in group_cols:
        if group_key in group_stats:
            mean, std = group_stats[group_key][col]
        else:
            mean, std = global_means[col], global_stds[col]
        df[col] = (df[col] - mean) / std

    # Normalize remaining columns (if any)
    remaining_cols = set(columns_to_normalize) - set(group_cols)
    for col in remaining_cols:
        df[col] = (df[col] - global_means[col]) / global_stds[col]
    return df


def preprocess_data(data, means=None, stds=None, group_stats=None,
                    use_cyclical_hour=True, normalize_road_speed=True):
    """
    Converts `data.splitted_ts_groups` dictionary into padded tensors and normalizes continuous columns.

    If both global means/stds and group_stats are provided, then the columns:
       'speed', 'acceleration_est_1', 'angular_acc'
    are normalized in a group-specific fashion. The groups are defined by the trip’s
    time-of-day (morning, rush_hour, night) and its road_speed (binned as 'low', 'medium', or 'high')
    using the road_speed value from the first row of the trip.

    Returns:
       X_padded: FloatTensor of shape (B, T, input_dim)
       y_padded: FloatTensor of shape (B, T, 2)  [predicting speed, acceleration]
       group_ids_tensor: LongTensor of shape (B,)
       lengths_tensor: LongTensor of shape (B,)
    """
    columns_to_normalize = ['longitude', 'latitude', 'speed',
                            'acceleration_est_1', 'angular_acc']

    X_list, y_list = [], []
    group_ids_list, lengths = [], []

    for group_id, time_series_list in data.splitted_ts_groups.items():
        for df in time_series_list:
            df = df.sort_values("orig_time")  # Ensure correct time order

            # Normalize continuous columns if provided
            if (means is not None) and (stds is not None):
                if group_stats is not None:
                    # Compute group key for this trip based on its start time and first road_speed value.
                    time_cat = get_trip_time_category(df)
                    rs_val = df['road_speed'].iloc[0]
                    rs_bin = get_road_speed_bin(rs_val)
                    group_key = (time_cat, rs_bin)
                    df = normalize_dataframe_groupwise(df, means, stds, group_stats, group_key, columns_to_normalize)
                else:
                    df = normalize_dataframe(df, means, stds, columns=columns_to_normalize)

            # Prepare hour-of-day (cyclical or not)
            hour = pd.to_datetime(df['orig_time']).dt.hour.values.astype(float)
            if use_cyclical_hour:
                sin_hour = np.sin(2.0 * math.pi * hour / 24.0)
                cos_hour = np.cos(2.0 * math.pi * hour / 24.0)
            else:
                sin_hour = hour
                cos_hour = None

            # Prepare road_speed (with optional normalization)
            road_speed = df['road_speed'].values
            road_speed = np.clip(road_speed, 0, 99)
            if normalize_road_speed:
                # Example normalization for road_speed; adjust as needed.
                rs_mean, rs_std = 50.0, 25.0
                road_speed = (road_speed - rs_mean) / rs_std

            # Next-step target for speed and acceleration (after normalization)
            target = df[['speed', 'acceleration_est_1']].values

            # Ensure there are at least 3 points to shift by one safely
            if len(df) < 3:
                continue

            # Build the design matrix for all timesteps:
            # We use the 5 normalized columns, then add hour (or cyclical components) and road_speed.
            feats_5 = df[['longitude', 'latitude', 'speed', 'acceleration_est_1', 'angular_acc']].values
            if use_cyclical_hour:
                combined_feats = np.column_stack((feats_5, sin_hour, cos_hour, road_speed))
            else:
                combined_feats = np.column_stack((feats_5, hour, road_speed))

            # Shift so that the t-th row in X corresponds to the t+1-th row in y.
            X_np = combined_feats[:-1]
            y_np = target[1:]

            # Convert to torch tensors
            X_tensor = torch.tensor(X_np, dtype=torch.float32)
            y_tensor = torch.tensor(y_np, dtype=torch.float32)

            X_list.append(X_tensor)
            y_list.append(y_tensor)
            group_ids_list.append(torch.tensor(group_id, dtype=torch.long))
            lengths.append(X_tensor.shape[0])

    X_padded = pad_sequence(X_list, batch_first=True, padding_value=0.0)
    y_padded = pad_sequence(y_list, batch_first=True, padding_value=0.0)
    group_ids_tensor = torch.stack(group_ids_list)
    lengths_tensor = torch.tensor(lengths, dtype=torch.long)

    return X_padded, y_padded, group_ids_tensor, lengths_tensor
