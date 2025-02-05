
from src.preprocessing.Data_Preparation2 import *

vehicle_id = 460631


agg_df = aggregating_car_data_for_pattern(vehicle_id)
dict_sum, dict_cor, dict_drives = dict_creation_driving_groups(agg_df,vehicle_id)
all_drives = loading_ts_drives(vehicle_id)
df = concated_data_with_feat(vehicle_id)
dict_sum, dict_cor, dict_drives, neigh_dict = fixing_and_sorting_dicts(dict_sum, dict_cor, dict_drives)
groups_ts = create_dict_groups2ts(dict_drives, all_drives)
splitted_ts_groups = create_feat_(groups_ts)
scaling_parameters = scaling(splitted_ts_groups)

dict_length = {}
for group in groups_ts:
    for se in groups_ts[group]:
        curr_len = len(se)
        se['group'] = group
        dict_length[curr_len] = dict_length.get(curr_len,[])
        dict_length[curr_len].append(se)


new_dict = {key: scaling_parameters[key]  for key in scaling_parameters if len(scaling_parameters[key])>30}
groups_splitted_normlized_ts = {}
for key in new_dict:
    lst = normalize_time_series(new_dict[key])
    for series in lst:
        group_num = int(series.iloc[0][['group']])
        groups_splitted_normlized_ts[group_num] = groups_splitted_normlized_ts.get(group_num,[])
        groups_splitted_normlized_ts[group_num].append(series)

dict_length = {}
for group in groups_splitted_normlized_ts:
    for series in groups_splitted_normlized_ts[group]:
        curr_len = len(series)
        if curr_len>=9:
            dict_length[curr_len] = dict_length.get(curr_len,[])
            dict_length[curr_len].append(series)

drive_instance = drives(
                        car_id=vehicle_id,
                        agg_df=agg_df,
                        dict_sum=dict_sum,
                        dict_cor=dict_cor,
                        dict_drives=dict_drives,
                        all_drives=all_drives,
                        df=df,
                        neigh_dict=neigh_dict,
                        groups_ts=groups_ts,
                        splitted_ts_groups=splitted_ts_groups,
                        scaling_parameters=scaling_parameters,
                        dict_length=dict_length,
                        new_dict=new_dict,
                        groups_splitted_normlized_ts=groups_splitted_normlized_ts,
                        df_spectral=None
                                )

from src.modeling.cluster_bagger import cluster_bagging, classifier
df_spectral = cluster_bagging(drive_instance, extractor_heads=150)
drive_instance.df_spectral = df_spectral

print(classifier(drive_instance, threshold=0.9))