# # active_learning_minirkt_xgb.py  ──────────────────────────────────────────
# import pickle, numpy as np, pandas as pd, matplotlib.pyplot as plt
# from pathlib import Path
# from collections import defaultdict
# from itertools import chain
# from matplotlib import animation
# from sklearn.decomposition import PCA
# from sklearn.metrics import pairwise_distances
# from sktime.transformations.panel.rocket import MiniRocketMultivariate
# from xgboost import XGBClassifier
# from statsmodels.stats.proportion import proportion_confint         # Wilson CI
# from scipy.optimize import linear_sum_assignment                   # for ACC (optional)
#
# # ─── 1. Load data & build pseudo-label dicts ──────────────────────────────
# CAR_ID = 460631
# PKL    = Path(r"C:\Users\dorex\Desktop\Unsupervised Identification of Drivers Using only GPS Trajectory Data\Unsupervised-Identification-of-Drivers-Using-only-GPS-Trajectory-Data\data_backup") / f"{CAR_ID}.pkl"
# drive  = pickle.load(PKL.open("rb"))
#
# drivers = {0: [], 1: []}                    # 1 = Irad, 0 = Hili
# for key, trips in drive.groups_splitted_normlized_ts.items():
#     if key in (1, 4):                       # Irad groups
#         drivers[1].extend(trips)
#     elif key != 3:                          # skip “ignore” group
#         drivers[0].extend(trips)
#
# print("Irad trips:", len(drivers[1]), " | Hili trips:", len(drivers[0]))
#
# # ─── 2. Collect segment tensors (array 3×L) + meta ------------------------
# seg_arrs, seg_tid, seg_y = [], [], []
#
# for label, trips in drivers.items():
#     for df in trips:
#         tid = int(df["drive_id"].iloc[0])
#         a   = df[["speed", "acceleration_est_1", "angular_acc"]].to_numpy().T
#         seg_arrs.append(a)          # variable length
#         seg_tid.append(tid)
#         seg_y.append(label)
#
# seg_tid = np.array(seg_tid)
# seg_y   = np.array(seg_y)
#
# # ─── 3. MiniROCKET per length, concat features ---------------------------
# def concat_feat(arrs, kernels=4000, pad=-10.):
#     Ls = sorted({a.shape[1] for a in arrs})
#     rockets, blocks = {}, []
#
#     # fit one rocket per window length ≥9
#     for L in [L for L in Ls if L >= 9]:
#         X_L = np.stack([a for a in arrs if a.shape[1]==L])
#         rockets[L] = MiniRocketMultivariate(num_kernels=kernels,
#                                             random_state=0).fit(X_L)
#
#     # transform every segment with every rocket
#     for L, rkt in rockets.items():
#         X_all = []
#         for a in arrs:
#             if a.shape[1]==L:
#                 X_all.append(a)
#             elif a.shape[1]<L:          # pad
#                 pad_arr = np.full((a.shape[0], L-a.shape[1]), pad)
#                 X_all.append(np.hstack([a, pad_arr]))
#             else:                       # crop centre
#                 s = (a.shape[1]-L)//2
#                 X_all.append(a[:, s:s+L])
#         blocks.append(rkt.transform(np.stack(X_all)))
#     return np.hstack(blocks)
#
# seg_feat = concat_feat(seg_arrs, kernels=4000)
# print("Feature matrix:", seg_feat.shape)
#
# # ─── 4. build trip centroids in PCA-10 ------------------------------------
# trip_cent = (pd.DataFrame(seg_feat)
#                .assign(tid=seg_tid, lbl=seg_y)
#                .groupby("tid").mean())
# # pca = PCA(10, random_state=0).fit(trip_cent)
# proj = PCA(n_components=10, random_state=0).fit_transform(trip_cent.values)
#
# # map tid → label
# trip_lbl = dict(zip(trip_cent.index,
#                     pd.Series(seg_y, index=seg_tid).groupby(level=0).first()))
#
# # ─── 5. seeded anchors: top-20 % far-from-driver-centroid -----------------
# anchor_tids = set()
# for driver_lbl in (0,1):
#     # subset of trips for this driver
#     idx = [i for i,t in enumerate(trip_cent.index)
#            if trip_lbl[t]==driver_lbl]
#     sub = proj[idx]
#     driver_cent = sub.mean(axis=0, keepdims=True)
#     dists = np.linalg.norm(sub - driver_cent, axis=1)
#     n_seed = max(1, int(0.50*len(dists)))        # 20 %
#     seed_ids = [trip_cent.index[idx[i]] for i in np.argsort(dists)[-n_seed:]]
#     anchor_tids.update(seed_ids)
#
# print(f"Seed anchors per driver: {anchor_tids}")
#
#
# def ordered_trip_list(prob_dict):
#     d0 = [(tid,p) for tid,p in prob_dict.items() if trip_lbl[tid]==0]
#     d1 = [(tid,p) for tid,p in prob_dict.items() if trip_lbl[tid]==1]
#     d0 = sorted(d0, key=lambda kv: kv[1])          # ascending
#     d1 = sorted(d1, key=lambda kv: kv[1], reverse=True)
#     return d0 + d1
# # ─── 6. active-learning loop ---------------------------------------------
# def wilson(p,n,a=0.05): return proportion_confint(int(p*n), n, method="wilson")
#
# def new_xgb():
#     return XGBClassifier(n_estimators=600,
#                          learning_rate=0.05,
#                          max_depth=10,
#                          subsample=0.8,
#                          colsample_bytree=0.8,
#                          eval_metric="logloss",
#                          n_jobs=-1)
#
# max_iter=30; δ=0.1
# seg_prob = np.zeros(len(seg_y))
# history = []
# for it in range(1, max_iter+1):
#     mask = np.isin(seg_tid, list(anchor_tids))
#     clf  = new_xgb().fit(seg_feat[mask], seg_y[mask])
#     seg_prob = clf.predict_proba(seg_feat)[:,1]
#
#     dfp = (pd.DataFrame({"tid": seg_tid, "p": seg_prob})
#              .groupby("tid").agg(m=("p","mean"), n=("p","size")))
#     changed=False
#     for tid,row in dfp.iterrows():
#         if tid in anchor_tids: continue
#         lo,hi = wilson(row.m,row.n)
#         if hi < 0.5-δ or lo > 0.5+δ:
#             anchor_tids.add(tid); changed=True
#         history.append(dfp["m"].to_dict())
#     print(f"iter {it:<2} | anchors {len(anchor_tids):>3} / {len(dfp)}")
#     if not changed: break
#
# print("loop finished.")
#
# # ─── 7. trip-prob plot ----------------------------------------------------
# prob = dfp["m"].to_dict()
# ordered = sorted(prob.items(), key=lambda kv: kv[1], reverse=True)
# color = ["tab:orange" if trip_lbl[tid]==1 else "tab:blue"
#          for tid,_ in ordered]
#
# plt.figure(figsize=(12,4))
# plt.bar(range(len(ordered)), [p for _,p in ordered], color=color)
# plt.axhline(0.5, ls="--", lw=1)
# plt.title("Trip-level probability for driver ‘1’ after AL")
# plt.ylabel("Mean P(driver=1)")
# plt.xlabel("Trips (sorted)")
# plt.tight_layout()
# plt.show()
#
# # 2️⃣  build animation ------------------------------------------------------
# fig, ax = plt.subplots(figsize=(12,4))
# ax.axhline(0.5, ls="--", lw=1, color="k")
# bars = ax.bar([], [])
#
# ax.set_ylim(0,1)
# ax.set_ylabel("Mean P(driver=1)")
# ax.set_xlabel("Trips (sorted, d0→d1)")
# ax.set_title("Active-learning progress")
# legend_handles = [plt.Rectangle((0,0),1,1,color="tab:blue"),
#                   plt.Rectangle((0,0),1,1,color="tab:orange")]
# ax.legend(legend_handles, ["Driver 0", "Driver 1"], loc="upper right")
#
# def init():
#     ax.clear()
#     return bars
#
# def update(frame):
#     prob_now = history[frame]
#     ord_now  = ordered_trip_list(prob_now)
#     colors   = ["tab:blue"  if trip_lbl[tid]==0 else "tab:orange"
#                 for tid,_ in ord_now]
#
#     ax.clear()
#     ax.bar(range(len(ord_now)), [p for _,p in ord_now], color=colors)
#     ax.axhline(0.5, ls="--", lw=1, color="k")
#     ax.set_ylim(0,1)
#     ax.set_title(f"Iteration {frame+1}/{len(history)}")
#     ax.set_ylabel("Mean P(driver=1)")
#     ax.set_xlabel("Trips (sorted, d0→d1)")
#     ax.legend(legend_handles, ["Driver 0", "Driver 1"], loc="upper right")
#     return bars
#
# ani = animation.FuncAnimation(fig, update,
#                               frames=len(history),
#                               init_func=init,
#                               blit=False, repeat=False)
#
# ani.save("progress.gif", writer="pillow", fps=2)
# plt.close(fig)   # close animation figure
#
# # 3️⃣  final static plot ----------------------------------------------------
# final_ord = ordered_trip_list(history[-1])
# final_cols = ["tab:blue" if trip_lbl[tid]==0 else "tab:orange"
#               for tid,_ in final_ord]
#
# plt.figure(figsize=(12,4))
# plt.bar(range(len(final_ord)), [p for _,p in final_ord], color=final_cols)
# plt.axhline(0.5, ls="--", lw=1)
# plt.ylim(0,1)
# plt.title("Trip-level probability after active learning (sorted)")
# plt.ylabel("Mean P(driver=1)")
# plt.xlabel("Trips (Driver 0 → Driver 1)")
# plt.legend(legend_handles, ["Driver 0", "Driver 1"], loc="upper right")
# plt.tight_layout()
# plt.show()
#
# print("Animation saved as progress.gif")


# semi_selftraining_minirkt_xgb.py  ────────────────────────────────────────
# import pickle, numpy as np, pandas as pd, matplotlib.pyplot as plt
# from pathlib import Path
# # import geopandas as gpd, contextily as cx
# from collections import defaultdict
# from matplotlib import animation, colors
# from sklearn.decomposition import PCA
# from sktime.transformations.panel.rocket import MiniRocketMultivariate
# from xgboost import XGBClassifier
# from src.utils.Drives import drives
# # ─── 1. load trips & pseudo-labels ────────────────────────────────────────
# CAR_ID = 460631
# PKL    = Path(r"C:\Users\dorex\Desktop\Unsupervised Identification of Drivers Using only GPS Trajectory Data\Unsupervised-Identification-of-Drivers-Using-only-GPS-Trajectory-Data\data_backup") / f"{CAR_ID}.pkl"
# drive  = pickle.load(PKL.open("rb"))
# #
#
# drivers = {0: [], 1: []}                 # 0 = Hili, 1 = Irad
# for g, trips in drive.groups_splitted_normlized_ts.items():
#     if g in (1, 4):
#         drivers[1].extend(trips)
#     elif g != 3:
#         drivers[0].extend(trips)
#
# # ─── 2. build segment feature matrix with MiniROCKET concat  ──────────────
# seg_arr, seg_tid, seg_y = [], [], []
# for lbl, trips in drivers.items():
#     for df in trips:
#         seg_arr.append(df[["speed", "acceleration_est_1",
#                            "angular_acc"]].to_numpy().T)
#         seg_tid.append(int(df["drive_id"].iloc[0]))
#         seg_y.append(lbl)
# seg_tid, seg_y = map(np.array, [seg_tid, seg_y])
#
# def concat_feat(arrs, kernels=3000, pad=-10):
#     uniqL = sorted({a.shape[1] for a in arrs if a.shape[1] >= 9})
#     blocks = []
#     for L in uniqL:
#         rocket = MiniRocketMultivariate(num_kernels=kernels,
#                                         random_state=0).fit(
#             np.stack([a for a in arrs if a.shape[1]==L]))
#         padded = []
#         for a in arrs:
#             if a.shape[1]==L:               padded.append(a)
#             elif a.shape[1]<L:              # pad
#                 pad_arr = np.full((a.shape[0], L-a.shape[1]), pad)
#                 padded.append(np.hstack([a, pad_arr]))
#             else:                           # centre crop
#                 s = (a.shape[1]-L)//2
#                 padded.append(a[:, s:s+L])
#         blocks.append(rocket.transform(np.stack(padded)))
#     return np.hstack(blocks)
#
# seg_feat = concat_feat(seg_arr, kernels=3000)
#
# # mapping trip_id -> majority label
# trip_lbl = dict(pd.Series(seg_y, index=seg_tid).groupby(level=0).first())
#
# # ─── 3. iterative confident-set self-training ─────────────────────────────
# def new_xgb():
#     return XGBClassifier(
#         n_estimators=300, learning_rate=0.05, max_depth=10,
#         subsample=0.8, colsample_bytree=0.8,
#         eval_metric="logloss", n_jobs=-1)
#
# δ = 0.20                # confidence margin
# max_iter = 5
# history   = []          # list of {trip:prob}
# conf_mask = np.ones(len(seg_y), dtype=bool)   # start with ALL segments
#
# for it in range(max_iter):
#     clf = new_xgb().fit(seg_feat[conf_mask], seg_y[conf_mask])
#     seg_prob = clf.predict_proba(seg_feat)[:, 1]
#
#     # trip-level probability = mean of segment probs
#     dfp = (pd.DataFrame({"tid": seg_tid, "p": seg_prob})
#              .groupby("tid").p.mean())
#     history.append(dfp.to_dict())
#
#     # decide next confident set
#     good_tids = dfp[(dfp < 0.5-δ) | (dfp > 0.5+δ)].index
#     next_mask = np.isin(seg_tid, good_tids)
#
#     print(f"iter {it:02d} | confident trips {len(good_tids)}/{len(dfp)}")
#     if next_mask.sum() == conf_mask.sum():   # no change
#         break
#     conf_mask = next_mask
#
# print("Finished iterations:", len(history))
#
#
#
# # ─── 4. helpers for plotting ──────────────────────────────────────────────
# def sorted_trips(prob_dict):
#     d0 = [(t, p) for t, p in prob_dict.items() if trip_lbl[t] == 0]
#     d1 = [(t, p) for t, p in prob_dict.items() if trip_lbl[t] == 1]
#     return (sorted(d0, key=lambda kv: kv[1]) +
#             sorted(d1, key=lambda kv: kv[1], reverse=True))
#
# n_trips   = len(np.unique(seg_tid))
# legend_el = [plt.Rectangle((0, 0), 1, 1, color="tab:blue"),
#              plt.Rectangle((0, 0), 1, 1, color="tab:orange")]
#
# # ─── 5. animation with cumulative averages ────────────────────────────────
# fig, ax = plt.subplots(figsize=(12, 4))
#
# def animate(frame):
#     ax.clear()
#     ax.axhline(0.5, ls="--", lw=1, color="k")
#     probs = cum_hist[frame]
#     ord_trips = sorted_trips(probs)
#
#     cols = ["tab:blue" if trip_lbl[t] == 0 else "tab:orange"
#             for t, _ in ord_trips]
#     ax.bar(range(len(ord_trips)), [p for _, p in ord_trips], color=cols)
#     ax.set_ylim(0, 1)
#     ax.set_title(f"Iteration {frame+1}/{N_ITER}  –  "
#                  f"trips evaluated so far: {len(probs)} / {n_trips}")
#     ax.set_ylabel("Cumulative mean P(driver = 1)")
#     ax.set_xlabel("Trips (Driver-0 → Driver-1)")
#     ax.legend(legend_el, ["Driver 0", "Driver 1"], loc="upper right")
#
# ani = animation.FuncAnimation(fig, animate,
#                               frames=len(cum_hist),
#                               repeat=False,
#                               interval=50)      # interval = #trips (ms)
# ani.save("prob_progress.gif", writer="pillow")
# plt.close(fig)
#
# # ─── 6. final static cumulative plot ──────────────────────────────────────
# final_prob = cum_hist[-1]
# ord_final  = sorted_trips(final_prob)
# cols_final = ["tab:blue" if trip_lbl[t] == 0 else "tab:orange"
#               for t, _ in ord_final]
#
# plt.figure(figsize=(12, 4))
# plt.bar(range(len(ord_final)), [p for _, p in ord_final], color=cols_final)
# plt.axhline(0.5, ls="--", lw=1)
# plt.ylim(0, 1)
# plt.title("Trip probabilities – cumulative mean after all iterations")
# plt.ylabel("Cumulative mean P(driver = 1)")
# plt.xlabel("Trips (Driver-0 → Driver-1)")
# plt.legend(legend_el, ["Driver 0", "Driver 1"], loc="upper right")
# plt.tight_layout()
# plt.show()
#
# print("GIF saved as prob_progress.gif")

import pickle, pandas as pd, matplotlib.pyplot as plt
from pathlib import Path
from matplotlib import animation
from sktime.transformations.panel.rocket import MiniRocketMultivariate
from xgboost import XGBClassifier
from src.modeling.dataset_an import *
# ─── 1. LOAD & PREPARE ───────────────────────────────────────────────────
CAR_ID = 460631
PKL    = Path(r"C:\Users\dorex\Desktop\Unsupervised Identification of Drivers Using only GPS Trajectory Data\Unsupervised-Identification-of-Drivers-Using-only-GPS-Trajectory-Data\data_backup") / f"{CAR_ID}.pkl"
drive  = pickle.load(PKL.open("rb"))



drivers = {0: [], 1: []}
for g, trips in drive.groups_splitted_normlized_ts.items():
    if g in (1, 4):  drivers[1].extend(trips)
    elif g != 3:     drivers[0].extend(trips)

# build lists
seg_arr, seg_tid, seg_y, seg_time = [], [], [], []
for lbl, trips in drivers.items():
    for df in trips:
        seg_arr.append(df[["speed","acceleration_est_1","angular_acc"]].to_numpy().T)
        seg_tid.append(int(df["drive_id"].iloc[0]))
        seg_y.append(lbl)
        seg_time.append(pd.to_datetime(df["orig_time"].iloc[0]))   # start timestamp
seg_tid, seg_y, seg_time = map(np.array, [seg_tid, seg_y, seg_time])


# arrays_all = your list of np.ndarray (3 × T)
ds = TripRnnDataset(arrays_all)
dl = DataLoader(ds, batch_size=32, shuffle=True, collate_fn=collate)

device = "cuda" if torch.cuda.is_available() else "cpu"
ae = RnnAutoEncoder(n_features=3, hid=128, latent=128).to(device)
opt = torch.optim.Adam(ae.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

EPOCH = 30
for ep in range(EPOCH):
    ae.train()
    total = 0
    for x, lens in dl:
        x, lens = x.to(device), lens.to(device)
        recon, _ = ae(x, lens)
        # mask padding
        mask = torch.arange(x.size(1), device=device)[None, :] < lens[:, None]
        loss = loss_fn(recon[mask], x[mask])
        opt.zero_grad(); loss.backward(); opt.step()
        total += loss.item()*x.size(0)
    print(f"epoch {ep:02d} | recon MSE = {total/len(ds):.4f}")

# ─ MiniROCKET features (one vector per trip) ─────────────────────────────
def concat_feat(arrs, kernels=5000, pad=-10):
    uniqL, blocks = sorted({a.shape[1] for a in arrs if a.shape[1] >= 9}), []
    for L in uniqL:
        rocket = MiniRocketMultivariate(num_kernels=kernels, random_state=0).fit(
            np.stack([a for a in arrs if a.shape[1] == L]))
        pad_stack = []
        for a in arrs:
            if a.shape[1] == L:                              pad_stack.append(a)
            elif a.shape[1] < L:                             # pad
                pad_arr = np.full((a.shape[0], L - a.shape[1]), pad)
                pad_stack.append(np.hstack([a, pad_arr]))
            else:                                            # centre-crop
                s = (a.shape[1] - L)//2
                pad_stack.append(a[:, s:s+L])
        blocks.append(rocket.transform(np.stack(pad_stack)))
    return np.hstack(blocks)

seg_feat = concat_feat(seg_arr, kernels=10000)

# helper maps
trip_start = pd.Series(seg_time, index=seg_tid).groupby(level=0).first()
trip_lbl   = pd.Series(seg_y,    index=seg_tid).groupby(level=0).first()

# ─── 2. INITIAL TRAIN SET = first year ───────────────────────────────────
t0          = trip_start.min()
cutoff      = t0 + pd.Timedelta(days=365)
init_train  = trip_start[trip_start <= cutoff].index            # trip_ids
stream_trips = trip_start[trip_start >  cutoff].sort_values().index.tolist()

train_mask = np.isin(seg_tid, init_train)

# ─── 2c. PCA-10 for trip centroids (optional) ─────────────────────────────
      # Rocket features are sparse-ish
from sklearn.decomposition import PCA
# 2a. keep a fixed number of components
pca   = PCA(n_components=12, random_state=0)     # ← e.g. 100 comps
seg_feat = pca.fit_transform(seg_feat)

# 2b. --OR-- keep enough comps to explain ≥ 95 % variance
# pca   = PCA(n_components=0.95, svd_solver="full", random_state=0)
# X_pca = pca.fit_transform(X_scaled)

# ─── 3. Active learning, chronological, one trip per iter ───────────────
def new_xgb(seed):
    return XGBClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=10,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="logloss", n_jobs=-1, random_state=seed)

δ = 0.1
history, added_hist = [], []        # for animation

# initialise “unknown” probs to 0.5
base_probs = {tid: 0.5 for tid in trip_start.index}

for it, tid in enumerate(stream_trips, start=1):
    clf = new_xgb(it)
    clf.fit(seg_feat[train_mask], seg_y[train_mask])

    # predict for **all** trips seen so far (train + current)
    seg_prob  = clf.predict_proba(seg_feat)[:, 1]
    trip_prob = (pd.DataFrame({"tid": seg_tid, "p": seg_prob})
                   .groupby("tid").p.mean()).to_dict()

    # override: future trips (still unseen) stay at 0.5
    for future_tid in stream_trips[it:]:
        trip_prob[future_tid] = 0.5

    # store snapshot
    history.append(trip_prob)

    # decision on the current trip
    added_this = set()
    p_curr = trip_prob[tid]
    if p_curr < 0.5-δ or p_curr > 0.5+δ:
        train_mask |= (seg_tid == tid)
        added_this.add(tid)
    added_hist.append(added_this)

    print(f"iter {it:03d} | current trip {tid} | p={p_curr:.3f} | "
          f"{'added' if added_this else 'kept 50/50'}")

# add an initial frame (before any streaming)
history.insert(0, base_probs)
added_hist.insert(0, set())

# ─── 4. fixed chronological order for bars ───────────────────────────────
trip_order = trip_start.sort_values().index.tolist()

legend_el = [plt.Rectangle((0, 0), 1, 1, color="tab:blue"),
             plt.Rectangle((0, 0), 1, 1, color="tab:orange"),
             plt.Rectangle((0, 0), 1, 1, facecolor="none",
                           edgecolor="k", hatch="///", lw=0)]

# ─── 5. animation ────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 4))

def animate(frame):
    ax.clear()
    ax.axhline(0.5, ls="--", lw=1, color="k")

    probs_now = history[frame]
    added_now = added_hist[frame]

    y_vals = [probs_now[t] for t in trip_order]
    base_c = ["tab:blue" if trip_lbl[t]==0 else "tab:orange" for t in trip_order]

    for i, (y, col, tid) in enumerate(zip(y_vals, base_c, trip_order)):
        hatch = "///" if tid in added_now else None
        ax.bar(i, y, color=col, hatch=hatch,
               edgecolor="k" if hatch else None)

    ax.set_ylim(0, 1)
    title = ("Initial model" if frame==0
             else f"Iter {frame}/{len(history)-1}  –  trip {list(added_now)[0] if added_now else 'no-add'}")
    ax.set_title(title)
    ax.set_ylabel("P(driver = 1)")
    ax.set_xlabel("Trips (chronological)")
    ax.legend(legend_el, ["Driver 0", "Driver 1", "Added to train"],
              loc="upper right")

ani = animation.FuncAnimation(fig, animate,
                              frames=len(history),
                              repeat=False,
                              interval=400)
ani.save("prob_progress_chrono.gif", writer="pillow")
plt.close(fig)
print("GIF saved to prob_progress_chrono.gif")
