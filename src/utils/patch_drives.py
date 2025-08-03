# -*- coding: utf-8 -*-
"""
Loop over every trip in one `Drive` instance and patch in the new columns.
"""
# src/utils/patch_drive.py
from src.utils.feature_engineering import add_motion_features_inplace

def patch_drive_instance(drive_inst):
    """
    Mutate every DataFrame held by groups_ts *and*
    (if present) splitted_ts_groups so all future loaders
    see the new columns.
    "groups_ts", "splitted_ts_groups",
    """
    for container_name in ('groups_splitted_normlized_ts'):
        if not hasattr(drive_inst, container_name):
            continue
        container = getattr(drive_inst, container_name)
        for trip_list in container.values():
            for df in trip_list:
                add_motion_features_inplace(df)