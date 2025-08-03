import numpy as np, torch, math
from torch.utils.data import Dataset
from pathlib import Path
from src.preprocessing.pre_process_deep_learning import load_and_merge_data
data_dir = r'C:\Users\dorex\Desktop\Unsupervised Identification of Drivers Using only GPS Trajectory Data\Unsupervised-Identification-of-Drivers-Using-only-GPS-Trajectory-Data\data'

# ── global seven-stat list stays the same ─────────────────────────
STATS = [np.mean, np.min, np.max,
         lambda x: np.percentile(x, 25),
         lambda x: np.percentile(x, 50),
         lambda x: np.percentile(x, 75),
         np.std]

# ── helper unchanged ─────────────────────────────────────────────
def segment_to_stat_matrix(seg_np: np.ndarray) -> np.ndarray:
    feats = []
    for fn in STATS:
        for ch in range(seg_np.shape[1]):
            feats.append([fn(seg_np[t:t+4, ch])
                          for t in range(0, seg_np.shape[0]-4)])
    return np.stack(feats)          # 35 × L

# ── NEW: load or fit channel-wise μ, σ once ──────────────────────
# SCALER_PATH = Path("src/arnet_unsup/arnet_scaler.npz")
# if SCALER_PATH.exists():
#     MU, SIG = np.load(SCALER_PATH)["mu"], np.load(SCALER_PATH)["sig"]
# else:                               # first run → compute from all trips
#     print("Fitting scaler …")
#     merged_all, _ = load_and_merge_data(data_dir)
#     cat = np.concatenate(
#         [df[["speed","d_speed","acceleration_est_1","jerk","angular_acc"]].to_numpy()
#          for trips in merged_all.splitted_ts_groups.values()
#          for df in trips], axis=0)
#     MU, SIG = cat.mean(0), cat.std(0) + 1e-6
#     # pick a home for the scaler file – next to the dataset module
#     SCALER_PATH = Path(__file__).parent / "arnet_scaler.npz"
#     SCALER_PATH.parent.mkdir(parents=True, exist_ok=True)  # <-- NEW
#
#     np.savez(SCALER_PATH, mu=MU, sig=SIG)
#     print("Saved channel scaler to", SCALER_PATH)

# ── DATASET class — only arr5 line changed ───────────────────────
class GPSWindowDataset(Dataset):
    def __init__(self, merged_data, win_len=10, stride=5):
        self.win_len, self.stride = win_len, stride
        self.windows = []
        for _, trips in merged_data.splitted_ts_groups.items():
            for df in trips:
                raw = df[["speed","d_speed","acceleration_est_1",
                          "jerk","angular_acc"]].to_numpy()
                arr5 = (raw - 0) / 1          # ← z-score here
                if len(df) < win_len:
                    continue
                for s in range(0, len(arr5) - win_len, stride):
                    seg = arr5[s:s+win_len]
                    mat = segment_to_stat_matrix(seg)   # 35 × L
                    self.windows.append(torch.from_numpy(mat).float())

    def __len__(self):   return len(self.windows)
    def __getitem__(self, idx):

        return self.windows[idx]                  # (35 × L) tensor
