#
# import pickle, torch, numpy as np
# from pathlib import Path
# from sklearn.cluster import AgglomerativeClustering
# from sklearn.metrics import normalized_mutual_info_score as NMI, \
#                             adjusted_mutual_info_score  as AMI
# from scipy.optimize import linear_sum_assignment
#
# from src.arnet_unsup.model   import ARNetUnsup
# import src.arnet_unsup.dataset as dset          # STATS list only
# from src.utils.Drives import drives
# # ── settings ──────────────────────────────────────────────────────
# CAR_ID   = 460631
# PKL      = Path(fr"C:\Users\dorex\Desktop\Unsupervised Identification of Drivers Using only GPS Trajectory Data\Unsupervised-Identification-of-Drivers-Using-only-GPS-Trajectory-Data\data_backup\{CAR_ID}.pkl")
# CHKPT    = Path("checkpoints/arnet_unsup_e.pth")
# device   = "cpu"
#
# WIN_LEN, STRIDE = 30, 15              # 5‑min windows
#
# # ground‑truth mapping: pattern‑group‑name → driver‑ID
# GT = {
#     1: ['Hadar Yosef', 'Neot Afeka A', 'Rom 2000', 'Herzliya B'],
#     2: ['Tel Aviv University', 'HaOgen'],
#     3: ['Naot Uzi']
# }
# name2drv = {n: d for d, names in GT.items() for n in names}
#
# # ── load model ────────────────────────────────────────────────────
# model = ARNetUnsup(latent_dim=32).to(device)
# model.load_state_dict(torch.load(CHKPT, map_location=device))
# model.eval()
#
# # ── helper: 5×30 → 35×L statistics grid ──────────────────────────
# def to_stats(seg5):
#     feats = []
#     for fn in dset.STATS:
#         for ch in range(seg5.shape[0]):
#             feats.append([fn(seg5[ch, t:t+4])
#                            for t in range(seg5.shape[1]-4)])
#     return torch.tensor(feats, dtype=torch.float32)
#
# # ── extract one centroid per pattern‑group ────────────────────────
# drive = pickle.load(PKL.open("rb"))
# group_embs, group_labels = [], []
#
# for g_id, trips in drive.splitted_ts_groups.items():
#     g_name  = drive.neigh_dict[g_id]                 # e.g. 'Hadar Yosef'
#     drv_lab = name2drv.get(g_name)
#     if drv_lab is None:
#         continue                                     # group not labelled
#
#     win_embs = []
#     for df in trips:
#         raw = df[["speed","d_speed","acceleration_est_1",
#                    "jerk","angular_acc"]].to_numpy().T
#         if raw.shape[1] < WIN_LEN:
#             continue
#         for s in range(0, raw.shape[1] - WIN_LEN + 1, STRIDE):
#             seg  = raw[:, s:s+WIN_LEN]               # 5×30
#             mat  = to_stats(seg)                     # 35×L
#             x    = mat.T.unsqueeze(0).to(device)     # 1×L×35
#             with torch.no_grad():
#                 _, proj, _, _ = model(x)             # 1×128
#             win_embs.append(proj.squeeze(0).cpu().numpy())
#
#     if win_embs:
#         centroid = np.mean(win_embs, axis=0)         # 128‑D
#         group_embs.append(centroid)
#         group_labels.append(drv_lab)
#
# E = np.vstack(group_embs)            # N_groups × 128
# y = np.array(group_labels)
#
# print("Pattern groups:", len(y))
#
# # ── cluster the centroids ─────────────────────────────────────────
# k = len(set(y))                      # should be 3
# pred = AgglomerativeClustering(
#             n_clusters=k, linkage='average').fit_predict(E)
#
# # ── metrics ───────────────────────────────────────────────────────
# def acc(true, pred):
#     D = max(true.max(), pred.max()) + 1
#     w = np.zeros((D, D), dtype=np.int64)
#     for t,p in zip(true, pred): w[p, t] += 1
#     r,c = linear_sum_assignment(w.max() - w)
#     return w[r,c].sum() / len(true)
#
# print(f"NMI = {NMI(y, pred):.3f}")
# print(f"AMI = {AMI(y, pred):.3f}")
# print(f"ACC = {acc(y, pred):.3f}")
# eval_cluster_hdbscan.py
# ---------------------------------------------------------------
import pickle, torch, numpy as np, pandas as pd, warnings
from pathlib import Path
from sklearn.metrics import normalized_mutual_info_score as NMI, \
                            adjusted_mutual_info_score  as AMI
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import SpectralClustering, AgglomerativeClustering
import hdbscan                                     #  <-- pip install hdbscan
from src.arnet_unsup.model   import ARNetUnsup
import src.arnet_unsup.dataset as dset             # STATS list only
from src.utils.Drives import drives

# ─── settings ────────────────────────────────────────────────────
CAR_ID   = 460631
PKL      = Path(fr"C:\Users\dorex\Desktop\Unsupervised Identification of Drivers Using only GPS Trajectory Data\Unsupervised-Identification-of-Drivers-Using-only-GPS-Trajectory-Data\data_backup\{CAR_ID}.pkl")
CHKPT    = Path("checkpoints/arnet_unsup_e.pth")
device = "cpu"

WIN_LEN, STRIDE = 10, 5         # 5-min windows  (same as training)

GT = {                           # pattern-group → driver ID
    1: ['Hadar Yosef', 'Neot Afeka A', 'Rom 2000', 'Herzliya B'],
    2: ['Tel Aviv University', 'HaOgen'],
    3: ['Naot Uzi']
}
name2drv = {n: d for d, names in GT.items() for n in names}

WIN_LEN, STRIDE = 10, 5

GT = {                      # pattern-group → driver
    1: ['Hadar Yosef', 'Neot Afeka A', 'Rom 2000', 'Herzliya B'],
    2: ['Tel Aviv University', 'HaOgen'],
    3: ['Naot Uzi']
}
name2drv = {n: d for d, names in GT.items() for n in names}

N_RUNS = 10                 # number of repeats for mean/std

# ─── helpers ──────────────────────────────────────────────────────
def to_stats(seg5):
    feats = []
    for fn in dset.STATS:
        for ch in range(seg5.shape[0]):
            feats.append([fn(seg5[ch, t:t+4])
                          for t in range(seg5.shape[1]-4)])
    return torch.tensor(feats, dtype=torch.float32)

def hungarian_acc(t, p):
    D = max(t.max(), p.max()) + 1
    W = np.zeros((D, D), dtype=np.int64)
    for ti, pi in zip(t, p):  W[pi, ti] += 1
    r, c = linear_sum_assignment(W.max() - W)
    return W[r, c].sum() / len(t)

# ─── load model ──────────────────────────────────────────────────
model = ARNetUnsup(latent_dim=32).to(device)
model.load_state_dict(torch.load(CHKPT, map_location=device))
model.eval()

drive = pickle.load(PKL.open("rb"))

# ─── extract pattern-group centroids ─────────────────────────────
embs, labs = [], []
for g_id, trips in drive.splitted_ts_groups.items():
    drv_lab = name2drv.get(drive.neigh_dict[g_id])
    if drv_lab is None:
        continue

    win = []
    for df in trips:
        raw = df[["speed","d_speed","acceleration_est_1",
                  "jerk","angular_acc"]].to_numpy().T
        if raw.shape[1] < WIN_LEN:
            continue
        for s in range(0, raw.shape[1]-WIN_LEN+1, STRIDE):
            seg = raw[:, s:s+WIN_LEN]
            mat = to_stats(seg).T.unsqueeze(0).to(device)
            with torch.no_grad():
                _, z, _, _ = model(mat)
            win.append(z.squeeze(0).cpu().numpy())
    if win:
        embs.append(np.mean(win, axis=0))
        labs.append(drv_lab)

X = np.vstack(embs)
y = np.array(labs)
k = len(np.unique(y))

print(f"Pattern groups: {len(y)} | Drivers: {k}")

# ─── evaluation loop ─────────────────────────────────────────────
def run_agglom():
    return AgglomerativeClustering(n_clusters=k, linkage='ward').fit_predict(X)

def run_spectral(seed):
    nn = min(10, len(X) - 1)
    return SpectralClustering(
        n_clusters=k, affinity="nearest_neighbors",
        n_neighbors=nn, assign_labels="kmeans",
        random_state=seed).fit_predict(X)

results = {"Algo": [], "Metric": [], "Mean": [], "Std": []}

for algo, runner in [
        ("Agglomerative", lambda _: run_agglom()),
        ("Spectral",      run_spectral)]:

    nmis, amis, accs = [], [], []
    for i in range(N_RUNS):
        pred = runner(i)
        nmis.append(NMI(y, pred))
        amis.append(AMI(y, pred))
        accs.append(hungarian_acc(y, pred))

    for name, vals in [("NMI", nmis), ("AMI", amis), ("ACC", accs)]:
        results["Algo"].append(algo)
        results["Metric"].append(name)
        results["Mean"].append(np.mean(vals))
        results["Std"].append(np.std(vals, ddof=1))

df = (pd.DataFrame(results)
        .set_index(["Algo", "Metric"])
        .sort_index())

print(df.to_markdown(floatfmt=".3f"))