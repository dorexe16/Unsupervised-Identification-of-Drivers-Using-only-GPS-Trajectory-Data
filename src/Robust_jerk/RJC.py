# import pickle, math, numpy as np, pandas as pd
# from collections import Counter
# from sklearn.metrics import silhouette_score
# from sklearn.cluster import AgglomerativeClustering, KMeans, SpectralClustering
# from sklearn.mixture import GaussianMixture
# from sklearn_extra.cluster import KMedoids             # pip install scikit-learn-extra
# from sklearn.metrics import normalized_mutual_info_score as nmi
# from src.utils.Drives import drives   # your helper
# # ----------------------------------------------------
# CAR_ID   = 460631
# DATA_DIR = r"C:\Users\dorex\Desktop\Unsupervised Identification of Drivers Using only GPS Trajectory Data\Unsupervised-Identification-of-Drivers-Using-only-GPS-Trajectory-Data\data"
# drive_inst = pickle.load(open(f"{DATA_DIR}\\{CAR_ID}.pkl", "rb"))
# group_label = {1:1, 5:1, 6:1, 7:1, 2:2, 4:2, 3:-1}
# # ----------------------------------------------------
# # 1. Pre-compute jerk vectors once
# precomputed = []
# for key, trips in drive_inst.groups_splitted_normlized_ts.items():
#     lab = group_label.get(key, -1)
#     if lab < 0:          # skip “ignore” groups
#         continue
#     for df in trips:
#         dt   = df.sort_values("orig_time")["orig_time"].diff().dt.total_seconds()
#         jerk = (df["acceleration_est_1"].diff() / dt).dropna().values
#         if jerk.size:
#             precomputed.append((jerk, lab))
# # ----------------------------------------------------
# def build_feature(alpha, min_len):
#     """Return X (n×1) and ground-truth labels after filtering by min_len."""
#     X, y = [], []
#     for jerk, lab in precomputed:
#         if len(jerk) < min_len:
#             continue
#         sigma, q = np.std(jerk), len(jerk)
#         X.append([math.exp(-alpha * sigma / q)])
#         y.append(lab)
#     return np.asarray(X), np.asarray(y)
#
# # 2. Hyper-param grids
# min_lengths = [5, 10, 20]
# alphas      = [0.1, 0.5, 1.0, 2.0]
# cluster_ks  = [2, 3]           # nb. you already know there are ≤3 drivers here
# linkages    = ["ward", "average", "complete"]
#
# results = []                   # collect one row per (alg, params)
#
# for min_len in min_lengths:
#     for alpha in alphas:
#         X, y_true = build_feature(alpha, min_len)
#         if len(np.unique(y_true)) < 2:          # need ≥2 true classes to score accuracy
#             continue
#
#         for k in cluster_ks:
#             # ---------- Agglomerative ----------
#             for link in linkages:
#                 if link == "ward" and X.shape[1] != 1:
#                     continue                    # ward requires Euclidean 1-D works fine
#                 model = AgglomerativeClustering(n_clusters=k, linkage=link)
#                 preds = model.fit_predict(X)
#                 if len(set(preds)) >= 2:
#                     nmi_val = nmi(y_true, preds, average_method='arithmetic')
#                     sil = silhouette_score(X, preds)
#                     acc = (preds == y_true).mean()
#                     results.append(("agg", {"linkage": link}, k,
#                                     min_len, alpha, nmi_val, acc))
#                 else:
#                     nmi_val = 0
#             # ---------- K-means ----------
#             km = KMeans(n_clusters=k, n_init=20, random_state=0)
#             preds = km.fit_predict(X)
#             nmi_val = nmi(y_true, preds, average_method='arithmetic')
#             acc = (preds == y_true).mean()
#             results.append(("kmeans", {}, k, min_len, alpha, nmi_val, acc))
#
#             # ---------- Spectral ----------
#             sp = SpectralClustering(n_clusters=k, gamma=0.5, assign_labels="kmeans",
#                                     random_state=0)
#             preds = sp.fit_predict(X)
#             nmi_val = nmi(y_true, preds, average_method='arithmetic')
#             acc = (preds == y_true).mean()
#             results.append(("spectral", {}, k, min_len, alpha, nmi_val, acc))
#
#             # ---------- Gaussian Mixture ----------
#             gm = GaussianMixture(n_components=k, covariance_type="full",
#                                  init_params="kmeans", random_state=0)
#             preds = gm.fit_predict(X)
#             nmi_val = nmi(y_true, preds, average_method='arithmetic')
#             acc = (preds == y_true).mean()
#             results.append(("gmm", {}, k, min_len, alpha, nmi_val, acc))
#
#             # ---------- K-medoids ----------
#             med = KMedoids(n_clusters=k, method="pam", metric="euclidean",
#                            init="k-medoids++", random_state=0)
#             preds = med.fit_predict(X)
#             nmi_val = nmi(y_true, preds, average_method='arithmetic')
#             acc = (preds == y_true).mean()
#             results.append(("kmedoids", {}, k, min_len, alpha, nmi_val, acc))
#
# # 3. Wrap results in DataFrame & pick best for each algorithm
# cols = ["algorithm", "extra_param", "k", "min_len", "alpha",
#         "NMI", "accuracy"]
# df = (pd.DataFrame(results, columns=cols)
#         .sort_values(["algorithm", "accuracy"], ascending=[True, False]))
#
# # OPTIONAL: collapse to “best by algorithm”:
# best_df = (df.loc[df.groupby("algorithm")["accuracy"].idxmax()]
#              .reset_index(drop=True)
#              .sort_values("accuracy", ascending=False))
#
# print("\n=== All runs ===")
# print(df.head(15))
# print("\n=== Best config per algorithm ===")
# print(best_df.to_string(index=False))


# eval_jerk_clustering_std.py  ─────────────────────────────────────
import pickle, math, numpy as np, pandas as pd
from pathlib import Path
from sklearn.cluster import (AgglomerativeClustering, KMeans,
                             SpectralClustering)
from sklearn.mixture import GaussianMixture
from sklearn_extra.cluster import KMedoids            # pip install scikit-learn-extra
from sklearn.metrics import (normalized_mutual_info_score as nmi,
                             adjusted_mutual_info_score   as ami)
from scipy.optimize import linear_sum_assignment

from src.utils.Drives import drives                   # your helper

# ─── constants ──────────────────────────────────────────────────
CAR_ID   = 460631
DATA_DIR = Path(r"/data")
drive_inst = pickle.load(open(DATA_DIR / f"{CAR_ID}.pkl", "rb"))
group_label = {1:1, 5:1, 6:1, 7:1, 2:2, 4:2, 3:-1}

N_RUNS = 30
SEEDS  = range(N_RUNS)

# ─── pre-compute jerk vectors once ──────────────────────────────
precomputed = []
for key, trips in drive_inst.groups_splitted_normlized_ts.items():
    lab = group_label.get(key, -1)
    if lab < 0:
        continue
    for df in trips:
        dt   = df.sort_values("orig_time")["orig_time"].diff().dt.total_seconds()
        jerk = (df["acceleration_est_1"].diff() / dt).dropna().values
        if jerk.size:
            precomputed.append((jerk, lab))

# ─── helpers ────────────────────────────────────────────────────
def build_feature(alpha, min_len):
    X, y = [], []
    for jerk, lab in precomputed:
        if len(jerk) < min_len:
            continue
        sigma, q = np.std(jerk), len(jerk)
        X.append([math.exp(-alpha * sigma / q)])
        y.append(lab)
    return np.asarray(X), np.asarray(y)

def hungarian_acc(true, pred):
    D = max(true.max(), pred.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for t, p in zip(true, pred):
        w[p, t] += 1
    r, c = linear_sum_assignment(w.max() - w)
    return w[r, c].sum() / len(true)

# ─── grids ──────────────────────────────────────────────────────
min_lengths = [5, 10, 20]
alphas      = [0.1, 0.5, 1.0, 2.0]
cluster_ks  = [2, 3]
linkages    = ["ward", "average", "complete"]

records = []   # one row per (algo, seed, hyper-params)

for min_len in min_lengths:
    for alpha in alphas:
        X, y_true = build_feature(alpha, min_len)
        if len(np.unique(y_true)) < 2:
            continue

        for k in cluster_ks:
            # ---------- Agglomerative (deterministic) ----------
            for link in linkages:
                if link == "ward" and X.shape[1] != 1:
                    continue
                preds = AgglomerativeClustering(n_clusters=k,
                                                linkage=link).fit_predict(X)
                records.append(("agg", link, k, min_len, alpha,
                                nmi(y_true, preds),
                                ami(y_true, preds),
                                hungarian_acc(y_true, preds)))

            # ---------- Stochastic clusterers ----------
            for seed in SEEDS:
                # K-means
                km_preds = KMeans(n_clusters=k, n_init=20,
                                  random_state=seed).fit_predict(X)
                records.append(("kmeans", None, k, min_len, alpha,
                                nmi(y_true, km_preds),
                                ami(y_true, km_preds),
                                hungarian_acc(y_true, km_preds)))

                # Spectral
                sp_preds = SpectralClustering(n_clusters=k, gamma=0.5,
                                              assign_labels="kmeans",
                                              random_state=seed).fit_predict(X)
                records.append(("spectral", None, k, min_len, alpha,
                                nmi(y_true, sp_preds),
                                ami(y_true, sp_preds),
                                hungarian_acc(y_true, sp_preds)))

                # Gaussian Mixture
                gm_preds = GaussianMixture(n_components=k,
                                           init_params="kmeans",
                                           random_state=seed).fit_predict(X)
                records.append(("gmm", None, k, min_len, alpha,
                                nmi(y_true, gm_preds),
                                ami(y_true, gm_preds),
                                hungarian_acc(y_true, gm_preds)))

                # K-medoids
                med_preds = KMedoids(n_clusters=k, method="pam",
                                     init="k-medoids++",
                                     random_state=seed).fit_predict(X)
                records.append(("kmedoids", None, k, min_len, alpha,
                                nmi(y_true, med_preds),
                                ami(y_true, med_preds),
                                hungarian_acc(y_true, med_preds)))

# ─── aggregate mean ± std ───────────────────────────────────────
df = pd.DataFrame(records,
                  columns=["algo", "linkage", "k", "min_len", "alpha",
                           "NMI", "AMI", "ACC"])

summary = (df.groupby("algo")
             .agg(NMI_mean=("NMI", "mean"), NMI_std=("NMI", "std"),
                  AMI_mean=("AMI", "mean"), AMI_std=("AMI", "std"),
                  ACC_mean=("ACC", "mean"), ACC_std=("ACC", "std"))
             .sort_values("ACC_mean", ascending=False))

pd.options.display.float_format = "{:.3f}".format

print("\n=== Mean ± Std across seeds & hyper-params ===\n")
print(summary.to_string())
