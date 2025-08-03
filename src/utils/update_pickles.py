# -*- coding: utf-8 -*-
"""
One-shot script: traverse the /data folder, load each car’s .pkl,
add the derived features, overwrite the pickle.

Run once, then every downstream loader (incl. ARNet dataset) finds the columns.
"""
import pickle, sys
from pathlib import Path
from src.utils.patch_drives import patch_drive_instance
# from src.arnet_unsup.train_arnet import data_dir
from src.utils.Drives import drives

data_dir = r"C:\Users\dorex\Desktop\Unsupervised Identification of Drivers Using only GPS Trajectory Data\Unsupervised-Identification-of-Drivers-Using-only-GPS-Trajectory-Data\data"

DATA_DIR = Path(data_dir)
      # <— adjust if your pickles live elsewhere
BACKUP   = True                # rename *.pkl → *.bak before overwrite
OVERWRITE_BACKUP = True         # keep .bak alongside new .pkl

for pkl in DATA_DIR.glob("*.pkl"):
    print("Patching", pkl.name)
    drive_inst = pickle.load(open(pkl, "rb"))
    patch_drive_instance(drive_inst)
    if BACKUP:
        pkl.rename(pkl.with_suffix(".bak"))
    pickle.dump(drive_inst, open(pkl, "wb"))
print("✓ All pickles updated with d_speed & jerk")