from Library import *
from Data_Preparation import *


def Prepare_metadata_by_vehicle_number(vehicle_ids=460631):
    PATH_raw_data = '/data/inputs/Year_Data'
    vehicle_ids = [vehicle_ids]

    for year_str in ['2018', '2019']:
        for month_index in tqdm(range(1, 13, 1)):
            if os.path.exists(
                    f"/bigdata/users-home/dor/transpotation research/agg_data/preprocessGlobalFeatures/{month_index}_{year_str}_{vehicle_ids}_trips.pickle") \
                    and os.path.exists(
                f"/bigdata/users-home/dor/transpotation research/agg_data/preprocessGlobalFeatures/{month_index}_{year_str}_{vehicle_ids}_car.csv"):
                continue

            months_metadata = pd.DataFrame(
                {'month_num_str': ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12'],
                 'month_shortcut_name': ['jan', 'feb', 'march', 'april', 'may', 'june', 'july', 'aug',
                                         'sep', 'oct', 'nov', 'dec'],
                 'days': [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]})
            if year_str == '2020':
                months_metadata['days'][1] = 29
            num_days = [int(np.floor((months_metadata.days[month_index - 1] - i) / 7 + 1)) for i in
                        range(1, 8, 1)]  # calculate number of days in each day of the week

            drives1, gps1, _ = read_same_trips('01', months_metadata.month_num_str[month_index - 1], year_str,
                                               num_days[0],
                                               directory=PATH_raw_data + '/')  # Saturdays of July
            drives1, gps1 = drives1.loc[drives1['VEHICLE_ID'].isin(vehicle_ids)], gps1.loc[
                gps1['vehicle_id'].isin(vehicle_ids)]
            drives1['START_DRIVE'] = pd.to_datetime(drives1['START_DRIVE'])
            drives1['END_DRIVE'] = pd.to_datetime(drives1['END_DRIVE'])
            gps1['orig_time'] = pd.to_datetime(gps1['orig_time'])

            drives2, gps2, _ = read_same_trips('02', months_metadata.month_num_str[month_index - 1], year_str,
                                               num_days[1],
                                               directory=PATH_raw_data + '/')  # Sundays of July
            drives2, gps2 = drives2.loc[drives2['VEHICLE_ID'].isin(vehicle_ids)], gps2.loc[
                gps2['vehicle_id'].isin(vehicle_ids)]
            drives2['START_DRIVE'] = pd.to_datetime(drives2['START_DRIVE'])
            drives2['END_DRIVE'] = pd.to_datetime(drives2['END_DRIVE'])
            gps2['orig_time'] = pd.to_datetime(gps2['orig_time'])

            drives3, gps3, _ = read_same_trips('03', months_metadata.month_num_str[month_index - 1], year_str,
                                               num_days[2],
                                               directory=PATH_raw_data + '/')  # Mondays of July
            drives3, gps3 = drives3.loc[drives3['VEHICLE_ID'].isin(vehicle_ids)], gps3.loc[
                gps3['vehicle_id'].isin(vehicle_ids)]
            drives3['START_DRIVE'] = pd.to_datetime(drives3['START_DRIVE'])
            drives3['END_DRIVE'] = pd.to_datetime(drives3['END_DRIVE'])
            gps3['orig_time'] = pd.to_datetime(gps3['orig_time'])

            drives4, gps4, _ = read_same_trips('04', months_metadata.month_num_str[month_index - 1], year_str,
                                               num_days[3],
                                               directory=PATH_raw_data + '/')  # Tuesdays of July
            drives4, gps4 = drives4.loc[drives4['VEHICLE_ID'].isin(vehicle_ids)], gps4.loc[
                gps4['vehicle_id'].isin(vehicle_ids)]
            drives4['START_DRIVE'] = pd.to_datetime(drives4['START_DRIVE'])
            drives4['END_DRIVE'] = pd.to_datetime(drives4['END_DRIVE'])
            gps4['orig_time'] = pd.to_datetime(gps4['orig_time'])

            drives5, gps5, _ = read_same_trips('05', months_metadata.month_num_str[month_index - 1], year_str,
                                               num_days[4],
                                               directory=PATH_raw_data + '/')  # Wendsdays of July
            drives5, gps5 = drives5.loc[drives5['VEHICLE_ID'].isin(vehicle_ids)], gps5.loc[
                gps5['vehicle_id'].isin(vehicle_ids)]
            drives5['START_DRIVE'] = pd.to_datetime(drives5['START_DRIVE'])
            drives5['END_DRIVE'] = pd.to_datetime(drives5['END_DRIVE'])
            gps5['orig_time'] = pd.to_datetime(gps5['orig_time'])

            drives6, gps6, _ = read_same_trips('06', months_metadata.month_num_str[month_index - 1], year_str,
                                               num_days[5],
                                               directory=PATH_raw_data + '/')  # Theursdays of July
            drives6, gps6 = drives6.loc[drives6['VEHICLE_ID'].isin(vehicle_ids)], gps6.loc[
                gps6['vehicle_id'].isin(vehicle_ids)]
            drives6['START_DRIVE'] = pd.to_datetime(drives6['START_DRIVE'])
            drives6['END_DRIVE'] = pd.to_datetime(drives6['END_DRIVE'])
            gps6['orig_time'] = pd.to_datetime(gps6['orig_time'])

            drives7, gps7, _ = read_same_trips('07', months_metadata.month_num_str[month_index - 1], year_str,
                                               num_days[6],
                                               directory=PATH_raw_data + '/')  # Fridays of July
            drives7, gps7 = drives7.loc[drives7['VEHICLE_ID'].isin(vehicle_ids)], gps7.loc[
                gps7['vehicle_id'].isin(vehicle_ids)]
            drives7['START_DRIVE'] = pd.to_datetime(drives7['START_DRIVE'])
            drives7['END_DRIVE'] = pd.to_datetime(drives7['END_DRIVE'])
            gps7['orig_time'] = pd.to_datetime(gps7['orig_time'])

            drives = pd.concat([drives1, drives2, drives3, drives4, drives5, drives6, drives7], ignore_index=True)
            gps = pd.concat([gps1, gps2, gps3, gps4, gps5, gps6, gps7], ignore_index=True)

            trip_len_min = 10
            drives_sub_group = drives.loc[drives['DRIVE_DURATION'] >= trip_len_min]

            trips_lst = separate_to_drives(drives_sub_group, gps)
            with open(
                    f"/bigdata/users-home/dor/transpotation research/agg_data/preprocessGlobalFeatures/{month_index}_{year_str}_{vehicle_ids}_trips.pickle",
                    'wb') as handle:
                pickle.dump(trips_lst, handle)

            drives_sub_group.to_csv(
                f"/bigdata/users-home/dor/transpotation research/agg_data/preprocessGlobalFeatures/{month_index}_{year_str}_{vehicle_ids}_car.csv",
                mode='w', index=False)