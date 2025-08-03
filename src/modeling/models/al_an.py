# full_pipeline_autoencoder.py
import pickle, numpy as np, pandas as pd, matplotlib.pyplot as plt
from pathlib import Path
from matplotlib import animation

import torch, torch.nn as nn
from torch.utils.data import DataLoader

from xgboost import XGBClassifier

# ----------------------------------------------------------------------
# 0.  IMPORT THE CLASSES YOU ALREADY HAVE
#     (RnnAutoEncoder, TripRnnDataset, collate)
# ----------------------------------------------------------------------
from src.modeling.dataset_an import RnnAutoEncoder, TripRnnDataset, collate

# ----------------------------------------------------------------------
# 1. LOAD TRIPS  →  arrays, labels, ids, timestamps
# ----------------------------------------------------------------------
CAR_ID = 460631
PKL = Path(
    r"/data_backup"
) / f"{CAR_ID}.pkl"
drive = pickle.load(PKL.open("rb"))

drivers = {0: [], 1: []}  # 0 = Hili, 1 = Irad
for g, trips in drive.groups_splitted_normlized_ts.items():
    if g in (1, 4):
        drivers[1].extend(trips)
    elif g != 3:
        drivers[0].extend(trips)

seg_arr, seg_tid, seg_y, seg_time = [], [], [], []
for lbl, trips in drivers.items():
    for df in trips:
        seg_arr.append(df[["speed", "acceleration_est_1", "angular_acc"]].to_numpy().T)
        seg_tid.append(int(df["drive_id"].iloc[0]))
        seg_y.append(lbl)
        seg_time.append(pd.to_datetime(df["orig_time"].iloc[0]))

seg_tid, seg_y, seg_time = map(np.array, [seg_tid, seg_y, seg_time])

# convenience maps (for plotting / chronology)
trip_start = pd.Series(seg_time, index=seg_tid).groupby(level=0).first()
trip_lbl   = pd.Series(seg_y,    index=seg_tid).groupby(level=0).first()

# ----------------------------------------------------------------------
# 2. TRAIN UNSUPERVISED RNN AUTO-ENCODER ON *ALL* TRIPS
# ----------------------------------------------------------------------
arrays_all = seg_arr                                   # shorthand

ds = TripRnnDataset(arrays_all)
dl = DataLoader(ds, batch_size=32, shuffle=True, collate_fn=collate)

device = "cuda" if torch.cuda.is_available() else "cpu"
ae      = RnnAutoEncoder(n_features=3, hid=128, latent=32).to(device)
opt     = torch.optim.Adam(ae.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()
l1_lambda = 0.05
EPOCH = 10
for ep in range(EPOCH):
    ae.train()
    total = 0
    for x, lens in dl:
        x, lens = x.to(device), lens.to(device)
        recon, z = ae(x, lens)

        # Build a (B, T) boolean mask for the valid timesteps
        mask = torch.arange(x.size(1), device=device)[None, :] < lens[:, None]

        # 1️⃣ Reconstruction loss
        recon_loss = loss_fn(recon[mask], x[mask])

        # 2️⃣ L1 penalty on the latent representation
        l1_loss = l1_lambda * z.abs().mean()

        # 3️⃣ Total loss
        loss = recon_loss + l1_loss

        # Back-prop
        opt.zero_grad()
        loss.backward()
        opt.step()

        total += loss.item() * x.size(0)
    print(f"[AE] epoch {ep:02d} | recon MSE = {total/len(ds):.4f}")

# helper – encode every trip once
def encode_trips(ae_model, arrays, batch=64):
    ae_model.eval()
    Z = np.zeros((len(arrays), ae_model.latent), dtype=np.float32)
    enc_dl = DataLoader(TripRnnDataset(arrays),
                        batch_size=batch, shuffle=False, collate_fn=collate)
    i = 0
    with torch.no_grad():
        for x, lens in enc_dl:
            z = ae_model(x.to(device), lens.to(device))[1]
            Z[i:i+len(z)] = z.cpu().numpy()
            i += len(z)
    return Z

embeddings = encode_trips(ae, arrays_all)              # (n_trips × 128)
# ----------------------------------------------------------------------
# 2.5  OUTLIER DETECTION  (DBSCAN per-driver, ⍺-tune by silhouette)
# ----------------------------------------------------------------------
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
import warnings

def eps_candidates(X, k=40,
                   qs=np.linspace(0.70, 0.99, 6)):        # 70–95 % k-distance
    """Heuristic grid for `eps` based on the k-distance curve."""
    nbrs  = NearestNeighbors(n_neighbors=k + 1).fit(X)
    dists = nbrs.kneighbors(X)[0][:, k]                  # distance to the k-th NN
    dists.sort()
    return np.quantile(dists, qs)

def best_dbscan(X, eps_grid, ms_grid=(3, 4, 5)):
    """Return labels from the (eps, min_samples) combo with max silhouette."""
    best_labels, best_score, best_params = None, -1, None
    for eps in eps_grid:
        for ms in ms_grid:
            labels = DBSCAN(eps=float(eps), min_samples=ms).fit_predict(X)
            core = labels >= 0
            # Need ≥2 clusters (ignoring noise) for silhouette
            if core.sum() == 0 or len(np.unique(labels[core])) < 2:
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # ignore degenerate cases
                score = silhouette_score(X[core], labels[core])
            if score > best_score:
                best_labels, best_score, best_params = labels, score, (eps, ms)
    return best_labels, best_score, best_params

noise_trip_ids = set()
for drv in (0, 1):
    idx = np.where(seg_y == drv)[0]          # rows for this driver
    Xd  = embeddings[idx]

    labels, sil, params = best_dbscan(
        Xd,
        eps_candidates(Xd)                   # data-dependent eps grid
    )

    if labels is None:                       # fell back to “one big cluster”
        print(f"[DBSCAN] driver {drv}: no valid split – keeping all trips.")
        continue

    eps, ms = params
    keep    = labels >= 0
    noise   = seg_tid[idx][~keep]
    noise_trip_ids.update(noise)

    print(f"[DBSCAN] driver {drv}: eps={eps:.3f}, min_s={ms}, "
          f"sil={sil:.3f}, removed {len(noise)} outliers")

# ----------  drop the detected outliers everywhere downstream ----------
keep_mask = ~np.isin(seg_tid, list(noise_trip_ids))

embeddings = embeddings[keep_mask]
seg_tid    = seg_tid[keep_mask]
seg_y      = seg_y[keep_mask]
seg_time   = seg_time[keep_mask]

# rebuild the Series mappings (trip-level)
trip_start = pd.Series(seg_time, index=seg_tid).groupby(level=0).first()
trip_lbl   = pd.Series(seg_y,    index=seg_tid).groupby(level=0).first()

print(f"Remaining trips after outlier removal: {len(trip_start)}")

# ----------------------------------------------------------------------
# 3. CHRONOLOGICAL ACTIVE-LEARNING ON THE EMBEDDINGS
# ----------------------------------------------------------------------
t0     = trip_start.min()
cutoff = t0 + pd.Timedelta(days=450)
init_train   = trip_start[trip_start <= cutoff].index
stream_trips = trip_start[trip_start >  cutoff].sort_values().index.tolist()

train_mask = np.isin(seg_tid, init_train)

def new_xgb(seed):
    return XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=15,
        subsample=0.85,
        colsample_bytree=0.95,
        eval_metric="logloss",
        n_jobs=-1,
        random_state=42,
    )

δ = 0.010
history, added_hist = [], []

base_probs = {tid: 0.5 for tid in trip_start.index}  # unseen = 0.5

for it, tid in enumerate(stream_trips, start=1):
    clf = new_xgb(it)
    clf.fit(embeddings[train_mask], seg_y[train_mask])

    seg_prob  = clf.predict_proba(embeddings)[:, 1]
    trip_prob = (
        pd.DataFrame({"tid": seg_tid, "p": seg_prob})
        .groupby("tid")
        .p.mean()
    ).to_dict()

    for future_tid in stream_trips[it:]:
        trip_prob[future_tid] = 0.5    # keep future trips at 0.5

    history.append(trip_prob)

    added_this = set()
    p_curr = trip_prob[tid]
    if p_curr < 0.5 - δ or p_curr > 0.5 + δ:
        train_mask |= seg_tid == tid
        added_this.add(tid)
    added_hist.append(added_this)

    true_lbl = trip_lbl[tid]  # 0 = Hili, 1 = Irad
    # -----------------------------------------------

    print(
        f"iter {it:03d} | trip {tid} | true={true_lbl} | "
        f"p={p_curr:.3f} | {'added' if added_this else 'kept 50/50'}"
    )

history.insert(0, base_probs)
added_hist.insert(0, set())

# ----------------------------------------------------------------------
# 4.  ANIMATION (bars fixed, hatched if added)
# ----------------------------------------------------------------------
trip_order = trip_start.sort_values().index.tolist()

legend_el = [
    plt.Rectangle((0, 0), 1, 1, color="tab:blue"),
    plt.Rectangle((0, 0), 1, 1, color="tab:orange"),
    plt.Rectangle((0, 0), 1, 1, facecolor="none", edgecolor="k", hatch="///", lw=0),
]

fig, ax = plt.subplots(figsize=(13, 4))


def animate(frame):
    ax.clear()
    ax.axhline(0.5, ls="--", lw=1, color="k")

    probs = history[frame]
    added = added_hist[frame]

    heights = [probs[t] for t in trip_order]
    colors  = ["tab:blue" if trip_lbl[t] == 0 else "tab:orange" for t in trip_order]

    for i, (h, c, tid) in enumerate(zip(heights, colors, trip_order)):
        hatch = "///" if tid in added else None
        ax.bar(i, h, color=c, hatch=hatch, edgecolor="k" if hatch else None)

    ax.set_ylim(0, 1)
    title = (
        "Initial model"
        if frame == 0
        else f"Iter {frame}/{len(history)-1} – trip {list(added)[0] if added else 'no-add'}"
    )
    ax.set_title(title)
    ax.set_ylabel("P(driver = 1)")
    ax.set_xlabel("Trips (chronological)")
    ax.legend(legend_el, ["Driver 0", "Driver 1", "Added to train"], loc="upper right")


ani = animation.FuncAnimation(fig, animate, frames=len(history),
                              repeat=False, interval=400)
ani.save("prob_progress_chrono_autoenc.gif", writer="pillow")
plt.close(fig)
print("GIF saved to prob_progress_chrono_autoenc.gif")
