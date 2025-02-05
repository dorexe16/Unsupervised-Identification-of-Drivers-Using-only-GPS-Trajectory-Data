class drives:
    def __init__(self, car_id, agg_df, dict_sum, dict_cor, dict_drives, all_drives,
                 df, neigh_dict, groups_ts, splitted_ts_groups, scaling_parameters,
                 dict_length, new_dict, groups_splitted_normlized_ts, df_spectral):
        self.car_id = car_id #id of the car
        self.agg_df = agg_df #aggregated df for pattern detection
        self.dict_sum = dict_sum #shows how many drives for each group
        self.dict_cor = dict_cor #show the end coordinate for each group
        self.dict_drives = dict_drives #saves the drives as ts for each group
        self.all_drives = all_drives #list of all the drives as ts
        self.df = df #metadata of the drives
        self.neigh_dict = neigh_dict #map each group to its destination neighberhood
        self.groups_ts = groups_ts #dict that contains ts of each group
        self.splitted_ts_groups = splitted_ts_groups #dict that contains segmented ts of each group
        self.scaling_parameters = scaling_parameters #dict that helps standetize the ts
        self.dict_length = dict_length #ts sorted by their lengths
        self.new_dict = new_dict
        self.groups_splitted_normlized_ts = groups_splitted_normlized_ts #ts divided into groups after segmented and standetize
        self.df_spectral = df_spectral #metrix that shows the probability for each group to be the same driver
