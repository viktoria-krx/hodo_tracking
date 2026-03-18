%load_ext autoreload
%autoreload 2

from read_file import *
from find_tracks import *
from fit_tracks import *
from fit_vertex import *
from draw_tracks import *
from get_event_features import *
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
import numpy as np
from sklearn.linear_model import RANSACRegressor
from sklearn.cluster import DBSCAN, HDBSCAN
import cProfile, pstats
import time
from itertools import combinations
from mpl_toolkits.axes_grid1 import make_axes_locatable


from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score, precision_score, recall_score, f1_score, roc_curve, auc, precision_recall_curve
import xgboost as xgb

# plt.rcParams.update({"figure.dpi": 72})

import warnings
from sklearn.exceptions import UndefinedMetricWarning

warnings.filterwarnings("ignore", message="R\^2 score is not well-defined")
plt.style.use("asacusa.mplstyle")

plot=True

root_path = "~/Documents/Hodoscope/cern_data/2025_Data/output_000636.root" # cosmics
root_path = "~/Documents/Hodoscope/cern_data/2025_Data/output_001825_6567.root" # cosmics

# BGdf = build_hits_df_fast(root_path)
# BGdf["Hbar_BG"] = "BG"
# hits_df["z_used"] = np.where(np.isnan(hits_df["z_reco"]), hits_df["z"], hits_df["z_reco"])

# max_ev = BGdf["event"].max()

rand_list = pd.read_csv("../cern_data/2025_Data/random_run_list.csv", sep=", ")
rand_list.sort_values("cusp", inplace=True)
rand_list.reset_index(drop=True, inplace=True)
print(rand_list)

run_list = rand_list.hodo.values

# run_list = np.concatenate([np.arange(1705, 1721), np.arange(1724, 1730)])
Hbardf = build_hits_df_from_runs(run_list, version="cusp_run")
Hbardf["Hbar_BG"] = ["Hbar" if m == True else "BG" for m in Hbardf["mixGate"]]
# Hbardf.loc[:, "event"] = Hbardf["event"] + max_ev

# hits_df = pd.concat([BGdf, Hbardf], ignore_index=True)
# hits_df = pd.concat([BGdf[BGdf.event <= 5000], Hbardf], ignore_index=True)
hits_df = Hbardf

hits_df["z_used"] = np.where(np.isnan(hits_df["z_reco"]), hits_df["z"], hits_df["z_reco"])
hits_df["dz_used"] = np.where(np.isnan(hits_df["dz_reco"]), hits_df["dz"], hits_df["dz_reco"])
hits_df["bgoToT"] = hits_df[(hits_df["detector"] == "bgo") & hits_df["LE"].notna() & hits_df["TE"].notna()& (hits_df["LE"] < hits_df["TE"])]["TE"] - hits_df[(hits_df["detector"] == "bgo") & hits_df["LE"].notna() & hits_df["TE"].notna() & (hits_df["LE"] < hits_df["TE"])]["LE"]


params = {
    "eps": [0.8],                            
    "zweight_same": [1], 
    "zweight_diff": [0],  
    "weight_power": [0.5], #[0.25, 0.5, 0.75],
    "weight_power_z": [0.85], #[0.7, 0.8, 0.85, 0.9], #[0.75, 0.8, 0.9, 1],
    "dist_bgo": [300],
    "vertex_cluster": [True],
    "vertex_eps": [200], #, 300, 400], #[50, 100, 150, 200, 250, 300, 350, 400],
    "vertex_alpha": [100] #[1, 10, 50, 100]
}

import json
with open('vertex_params.json', 'w') as fp:
    json.dump(params, fp)


eps = params["eps"][0]
z_w_same = params["zweight_same"][0]
z_w_diff = params["zweight_diff"][0]
w_pow = params["weight_power"][0]
w_pow_z = params["weight_power_z"][0]
dist_bgo = params["dist_bgo"][0]
vertex_cluster = params["vertex_cluster"][0]
vertex_eps = params["vertex_eps"][0]
vertex_alpha = params["vertex_alpha"][0]

clustered_list = []
for event_id, ev in hits_df[hits_df.LE < 0].groupby("event"):
    ev["det_key"] = ev["detector"] + "_" + ev["channel"].astype(str)

    # select only hodo/tile hits for clustering
    ev_ht = ev[ev.detector.isin(["hodoO","hodoI","tileO","tileI"])].copy()
    if ev_ht.empty:
        ev["track_id"] = -1
        clustered_list.append(ev)
        continue

    labels = cluster_by_phi_layer_uncertainty(ev_ht,
            base_eps=eps,
            min_samples=2,
            z_weight_same=z_w_same,
            z_weight_diff=z_w_diff,
            sigma_floor_deg=0.5,
            coords="cylindrical")
    
    # labels = cluster_by_phi_uncertainty(ev_ht, base_eps=eps, min_samples=2, theta_weight=th_w, coords="cylindrical")
    ev_ht["track_id"] = labels

    # merge labels back into the full event (bgo keep -1)
    ev = ev.merge(ev_ht[["det_key","track_id"]], on="det_key", how="left")
    ev["track_id"] = ev["track_id"].fillna(-1).astype(int)

    clustered_list.append(ev)

print("clustering done")

clustered_hits = pd.concat(clustered_list, ignore_index=True)

print(f"{clustered_hits[(clustered_hits.track_id > -1)].groupby('event').ngroups} of {hits_df.groupby('event').ngroups} events have at least one cluster\t {clustered_hits[(clustered_hits.track_id > -1)].groupby('event').ngroups/hits_df.groupby('event').ngroups*100:.2f}%")

print(f"{clustered_hits[(clustered_hits.z_used != 0)].groupby(['event']).ngroups} of {hits_df.groupby('event').ngroups} events have at least one track\t {clustered_hits[(clustered_hits.z_used != 0)].groupby(['event']).ngroups/hits_df.groupby('event').ngroups*100:.2f}%")


_lines_df = fit_lines_from_clusters_svd(clustered_hits, include_bgo=False, 
                                    use_xyz_errors=True, xyz_error_cols=["dx", "dy", "dz_used"], weight_power=w_pow, weight_power_z=w_pow_z, prefilter_ransac=False, ransac_thresh=15.0, weighted=False, weight_col="dz_used")

print("line fitting done")


vertices_df = reconstruct_vertex_from_midpoints(clustered_hits, _lines_df,
                                        bgo_radius=45.0, 
                                        max_dist_to_bgo=dist_bgo, 
                                        cluster_mids=vertex_cluster, cluster_eps=vertex_eps, cluster_alpha=vertex_alpha)#25.0)

print("vertex reconstruction done")

print(f"{vertices_df.groupby('event').ngroups} of {hits_df.groupby('event').ngroups} events have a reconstructed vertex\t {vertices_df.groupby('event').ngroups/hits_df.groupby('event').ngroups*100:.2f}%")

print(f"{vertices_df[(vertices_df.Vz != 0)].groupby(['event']).ngroups} of {hits_df.groupby('event').ngroups} events have a reconstructed non-zero vertex\t {vertices_df[(vertices_df.Vz != 0)].groupby(['event']).ngroups/hits_df.groupby('event').ngroups*100:.2f}%")

# vertices_df = find_vertices_from_tracks(lines_df, eps=5.0)
lines_df = _lines_df.merge(vertices_df, on="event", how="left")

clustered_hits = clustered_hits.merge(lines_df, on=["event", "track_id"], how="left")

plot_events(clustered_hits, lines_df, clustered_hits[clustered_hits.Hbar_BG == "pbar"].event.unique()[:30])

event_features_df = compute_event_features_from_clustered_hits(
    clustered_hits,
    bgo_center=(0, 0, 0)
)

print("event features done")



plt.hist(event_features_df[(event_features_df.mix)].bgoEdep, 100, range=(0, 100), histtype="step", label="Mixing")
plt.hist(event_features_df[(~event_features_df.mix)].bgoEdep, 100, range=(0, 100), histtype="step", label="BG")
plt.legend()
plt.xlabel("BGO E dep in MeV")
plt.show()



plt.hist(event_features_df[(event_features_df.mix)].bgoEdep, 100, range=(0, 20), histtype="step", label="Mixing")
plt.hist(event_features_df[(~event_features_df.mix)].bgoEdep, 100, range=(0, 20), histtype="step", label="BG")
plt.legend()
plt.xlabel("BGO E dep in MeV")
plt.show()



plt.hist2d(event_features_df.time, event_features_df.bgoEdep, bins=(35, 120), range=((0, 350e9), (0, 120)), norm="log")
plt.xlabel("Time in ns")
plt.ylabel("BGO E dep in MeV")
plt.colorbar()
plt.show()


plt.hist2d(event_features_df.n_bgo, event_features_df.bgoEdep, bins=(20, 120), range=((0, 20), (0, 120)), norm="log")
plt.xlabel("# BGO hits")
plt.ylabel("BGO E dep in MeV")
plt.colorbar()
plt.show()


plt.hist2d(event_features_df.time, event_features_df.n_bgo, bins=(35, 30), range=((0, 350e9), (0, 30)), norm="log")
plt.xlabel("Time in ns")
plt.ylabel("# BGO hits")
plt.colorbar()
plt.show()




plt.hist2d(event_features_df[(event_features_df.n_bgo > 2)].time, event_features_df[(event_features_df.n_bgo > 2)].bgoEdep, bins=(35, 120), range=((0, 350e9), (0, 120)), norm="log")
plt.xlabel("Time in ns")
plt.ylabel("BGO E dep in MeV")
plt.colorbar()
plt.show()


first_mix = event_features_df[(event_features_df.mix)].groupby("cusp").time.min()
last_mix = event_features_df[(event_features_df.mix)].groupby("cusp").time.max()
total_time = event_features_df.groupby("cusp").time.max()

mix_time = np.sum(last_mix - first_mix)
bg_time = np.sum(total_time) - mix_time




cum_E_BG = plt.hist(event_features_df[(~event_features_df.mix)].bgoEdep, histtype="step", cumulative=True, bins=100, range=(0, 100), density=True)
for i, cut in enumerate([0.8, 0.9, 0.95, 0.99]):
    plt.axvline(np.where(cum_E_BG[0] > cut)[0][0], label=f"{cut*100}% BG = {int(cum_E_BG[1][np.where(cum_E_BG[0] > cut)[0][0]])} MeV", color=f"C{i+1}")
plt.xlabel("BGO E dep in MeV")
plt.ylabel("Cumulative fraction")
plt.legend()
plt.title("Background")
plt.show()

cum_E_Mix = plt.hist(event_features_df[(event_features_df.mix)].bgoEdep, histtype="step", cumulative=True, bins=100, range=(0, 100), density=True)
for i, cut in enumerate([0.8, 0.9, 0.95, 0.99]):
    plt.axvline(np.where(cum_E_BG[0] > cut)[0][0], label=f"{int(cum_E_BG[1][np.where(cum_E_BG[0] > cut)[0][0]])} MeV = {cum_E_Mix[0][np.where(cum_E_Mix[1] > np.where(cum_E_BG[0] > cut)[0][0])[0][0]]*100:4.1f}% Mix", color=f"C{i+1}")
plt.xlabel("BGO E dep in MeV")
plt.ylabel("Cumulative fraction")
plt.legend()
plt.title("Mixing")
plt.show()


cum_nbgo_BG = plt.hist(event_features_df[(~event_features_df.mix)].n_bgo, histtype="step", cumulative=True, bins=30, range=(0, 30), density=True)

for i, cut in enumerate([0.8, 0.9, 0.95, 0.99]):
    plt.axvline(np.where(cum_nbgo_BG[0] > cut)[0][0], label=f"{cut*100}% BG = {int(cum_nbgo_BG[1][np.where(cum_nbgo_BG[0] > cut)[0][0]])}", color=f"C{i+1}")
plt.xlabel("# BGO Hits")
plt.ylabel("Cumulative fraction")
plt.legend()
plt.title("Background")
plt.show()

cum_nbgo_Mix = plt.hist(event_features_df[(event_features_df.mix)].n_bgo, histtype="step", cumulative=True, bins=30, range=(0, 30), density=True)
for i, cut in enumerate([0.8, 0.9, 0.95, 0.99]):
    plt.axvline(np.where(cum_nbgo_BG[0] > cut)[0][0], label=f"{int(cum_nbgo_BG[1][np.where(cum_nbgo_BG[0] > cut)[0][0]])} MeV = {cum_nbgo_Mix[0][np.where(cum_nbgo_Mix[1] > np.where(cum_nbgo_BG[0] > cut)[0][0])[0][0]]*100:4.1f}% Mix", color=f"C{i+1}")
plt.xlabel("# BGO Hits")
plt.ylabel("Cumulative fraction")
plt.legend()
plt.title("Mixing")
plt.show()



hist_E_nbgo_BG, xedges, yedges = np.histogram2d(event_features_df[(~event_features_df.mix)].n_bgo, event_features_df[(~event_features_df.mix)].bgoEdep, bins=(30, 120), range=((0, 30), (0, 120)), density=True)

dx = xedges[1] - xedges[0]
dy = yedges[1] - yedges[0]

cdf_BG = hist_E_nbgo_BG.cumsum(axis=0).cumsum(axis=1)*dx*dy

plt.pcolormesh(xedges, yedges, cdf_BG.T)
for i, cut in enumerate([0.8, 0.9, 0.95, 0.99]):
    plt.axvline(np.where(cdf_BG > cut)[0][0], color=f"C{i+1}")
    plt.axhline(np.where(cdf_BG > cut)[1][0], color=f"C{i+1}")
plt.xlabel("# BGO Hits")
plt.ylabel("BGO E dep in MeV")
plt.colorbar()
plt.title("Background")
plt.show()


percentiles = [0.8, 0.9, 0.95, 0.99]

cuts = {}
for p in percentiles:
    mask = cdf_BG >= p
    idx = np.argwhere(mask)

    # choose the cut with minimal "area" (tightest cut)
    best = idx[np.argmin(idx[:,0] * idx[:,1])]

    i, j = best
    cuts[p] = (xedges[i+1], yedges[j+1])

    
plt.pcolormesh(xedges, yedges, cdf_BG.T, shading="auto")
for p, (xcut, ycut) in cuts.items():
    plt.axvline(xcut, linestyle="--", label=f"{int(p*100)}% BG")
    plt.axhline(ycut, linestyle="--")

plt.xlabel("# BGO Hits")
plt.ylabel("BGO E dep [MeV]")
plt.colorbar(label="CDF")
plt.legend()
plt.title("Background 2D CDF with percentile cuts")
plt.show()


for p, (ncut, ecut) in cuts.items():
    mix_frac = np.mean(
        (event_features_df.mix) &
        (event_features_df.n_bgo <= ncut) &
        (event_features_df.bgoEdep <= ecut)
    )
    print(f"{int(p*100)}% BG cut → {mix_frac*100:.2f}% Mix")



hist_E_nbgo_Mix, xedges, yedges = np.histogram2d(event_features_df[(event_features_df.mix)].n_bgo, event_features_df[(event_features_df.mix)].bgoEdep, bins=(30, 120), range=((0, 30), (0, 120)), density=True)

dx = xedges[1] - xedges[0]
dy = yedges[1] - yedges[0]

cdf_Mix = hist_E_nbgo_Mix.cumsum(axis=0).cumsum(axis=1)*dx*dy

plt.pcolormesh(xedges, yedges, cdf_Mix.T)
for i, cut in enumerate([0.8, 0.9, 0.95, 0.99]):
    plt.axvline(np.where(cdf_BG > cut)[0][0], color=f"C{i+1}")
    plt.axhline(np.where(cdf_BG > cut)[1][0], color=f"C{i+1}")
plt.xlabel("# BGO Hits")
plt.ylabel("BGO E dep in MeV")
plt.colorbar()
plt.title("Mixing")
plt.show()





plt.hist2d(event_features_df[(event_features_df.n_bgo > 2)].time, event_features_df[(event_features_df.n_bgo > 2)].n_tracks, bins=(35, 12), range=((0, 350e9), (0, 12)), norm="log")
plt.xlabel("Time in ns")
plt.ylabel("n tracks")
plt.colorbar()
plt.show()


plt.hist2d(event_features_df.n_bgo, event_features_df.n_tracks, bins=(30, 10), range=((0, 30), (0, 10)), norm="log")
plt.xlabel("n BGO")
plt.ylabel("n tracks")
plt.colorbar()
plt.show()




hist_BG, xedges, yedges = np.histogram2d(
    event_features_df[(~event_features_df.mix)].n_bgo,
    event_features_df[(~event_features_df.mix)].bgoEdep,
    bins=(26, 120),
    range=((0, 26), (0, 120)),
    density=True
)


# plt.hist2d(event_features_df[(~event_features_df.mix)].n_bgo,
#     event_features_df[(~event_features_df.mix)].bgoEdep,
#     bins=(26, 480),
#     range=((0, 26), (0, 120)),
#     density=True)


# p_vals = np.concatenate([np.linspace(0, 0.9, 10), [0.95, 0.99]]) #[0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]

# hist_BG, xedges, yedges = np.histogram2d(
#     event_features_df[(~event_features_df.mix)].n_bgo,
#     event_features_df[(~event_features_df.mix)].bgoEdep,
#     bins=(26, 480),
#     range=((0, 26), (0, 120)),
#     density=True
# )


# hist_Mix, xedges, yedges = np.histogram2d(
#     event_features_df[(event_features_df.mix)].n_bgo,
#     event_features_df[(event_features_df.mix)].bgoEdep,
#     bins=(26, 480),
#     range=((0, 26), (0, 120)),
#     density=True
# )

# dx = xedges[1] - xedges[0]
# dy = yedges[1] - yedges[0]

# pdf = hist_BG.flatten()
# bin_areas = dx * dy
# prob_per_bin = pdf * bin_areas

# order = np.argsort(pdf)[::-1]

# cumprob = np.cumsum(prob_per_bin[order])

# levels = {}
# for p in p_vals:
#     idx = np.searchsorted(cumprob, p)
#     levels[p] = pdf[order][idx]

# X, Y = np.meshgrid(xedges[:-1], yedges[:-1], indexing="ij")
# plt.pcolormesh(xedges, yedges, hist_BG.T, shading="auto", norm="log")
# for i, p_lvl in enumerate(levels.items()):
#     p, lvl = p_lvl
#     plt.contour(
#         X, Y, hist_BG,
#         levels=[lvl],
#         linewidths=1,
#         colors=f"C{i+1}",
#         # label=f"{int(p*100)}%"
#     )
# plt.xlabel("# BGO Hits")
# plt.ylabel("BGO E dep [MeV]")
# #plt.colorbar(label="PDF")
# plt.title("Background density with HPD contours")
# plt.show()

# hpd_masks = {}

# for p in p_vals:
#     mask_flat = np.zeros_like(pdf, dtype=bool)

#     idx = np.searchsorted(cumprob, p)
#     mask_flat[order[:idx+1]] = True

#     hpd_masks[p] = mask_flat.reshape(hist_BG.shape)

# N_BG  = np.sum((~event_features_df.mix))
# N_Mix = np.sum((event_features_df.mix))
# BG_counts  = hist_BG  * bin_areas * N_BG
# Mix_counts = hist_Mix * bin_areas * N_Mix

# snr = {}

# # Loop over HPD percentiles for BG envelope
# for p in levels:
#     mask = hpd_masks[p]  # True = inside BG envelope
#     # Invert mask: outside BG envelope = signal
#     mask_signal = ~mask

#     B = np.sum(BG_counts[mask])          # background inside envelope
#     S = np.sum(Mix_counts[mask_signal])  # signal outside envelope

#     snr[p] = S / np.sqrt(B) if B > 0 else 0

#     print(
#         f"{int(p*100):>3}% BG envelope: "
#         f"B = {B:8.1f},  "
#         f"S = {S:8.1f},  "
#         f"S/√B = {snr[p]:.3f}"
#     )

# # Optional: print actual BG efficiency inside envelope
# # for p in levels:
# #     mask = hpd_masks[p]
# #     eff = np.sum(BG_counts[mask]) / N_BG
# #     print(f"{int(p*100)}% target → actual BG inside envelope = {eff:.4f}")

# # Scan all density levels for optimal S/√B
# best = {"snr": 0}

# for lvl in np.unique(hist_BG[hist_BG > 0]):
#     mask = hist_BG >= lvl          # inside envelope
#     mask_signal = ~mask            # outside envelope = signal

#     B = np.sum(BG_counts[mask])
#     S = np.sum(Mix_counts[mask_signal])

#     if B > 0:
#         val = S / np.sqrt(B)
#         if val > best["snr"]:
#             best = {"snr": val, "lvl": lvl, "S": S, "B": B}

# bg_eff = best["B"] / N_BG
# print(f"Optimal BG envelope (max S/√B): BG efficiency = {bg_eff:.3f}, S/√B = {best['snr']:.3f}")

# # Plot S/√B vs BG percentile
# plt.plot([int(p*100) for p in levels], list(snr.values()), "o-")
# plt.xlabel("% BG envelope")
# plt.ylabel("S/√B (signal outside envelope)")
# plt.title("Signal-to-noise vs background envelope fraction")
# plt.show()




# mask = hpd_masks[0.80]

# # Map event values to bin indices
# ix = np.searchsorted(xedges, event_features_df.n_bgo.values) - 1
# iy = np.searchsorted(yedges, event_features_df.bgoEdep.values) - 1

# # Initialize all events as passing (signal)
# passes = np.ones(len(event_features_df), dtype=bool)

# # Events inside histogram edges
# inside = (ix >= 0) & (ix < mask.shape[0]) & (iy >= 0) & (iy < mask.shape[1])

# # For events inside histogram, pass if outside BG envelope
# passes[inside] = ~mask[ix[inside], iy[inside]]

# event_features_df["pass_HPD"] = passes






# mask = hpd_masks[0.80]

# mix_ratio = []
# tot_ratio = []
# bg_ratio = []


# for cut, mask in hpd_masks.items():

#     # Map event values to bin indices
#     ix = np.searchsorted(xedges, event_features_df.n_bgo.values) - 1
#     iy = np.searchsorted(yedges, event_features_df.bgoEdep.values) - 1

#     # Initialize all events as passing (signal)
#     passes = np.ones(len(event_features_df), dtype=bool)

#     # Events inside histogram edges
#     inside = (ix >= 0) & (ix < mask.shape[0]) & (iy >= 0) & (iy < mask.shape[1])

#     # For events inside histogram, pass if outside BG envelope
#     passes[inside] = ~mask[ix[inside], iy[inside]]

#     event_features_df["pass_HPD"] = passes

#     mix_ratio.append(event_features_df[event_features_df.pass_HPD & event_features_df.mix].shape[0]/event_features_df[event_features_df.mix].shape[0])

#     tot_ratio.append(event_features_df[event_features_df.pass_HPD].shape[0]/event_features_df.shape[0])

#     bg_ratio.append(event_features_df[event_features_df.pass_HPD & (~event_features_df.mix)].shape[0]/event_features_df[(~event_features_df.mix)].shape[0])

#     print(f"{cut:.2f} Cut: {event_features_df[event_features_df.pass_HPD].shape[0]/event_features_df.shape[0]*100:.2f}", "% events pass -", f"{event_features_df[event_features_df.pass_HPD & event_features_df.mix].shape[0]/event_features_df[event_features_df.mix].shape[0]*100:.2f}", "% mixing events pass")


#     plt.hist(event_features_df[(event_features_df.mix) & event_features_df.pass_HPD].bgoEdep, 100, range=(0, 100), histtype="step", label="Mixing")
#     plt.hist(event_features_df[(~event_features_df.mix) & event_features_df.pass_HPD].bgoEdep, 100, range=(0, 100), histtype="step", label="BG")
#     plt.legend()
#     plt.xlabel("BGO E dep in MeV")
#     plt.title(f"Pass HPD {cut:.2f} Cut")
#     plt.show()


#     plt.hist(event_features_df[(event_features_df.mix) & ~event_features_df.pass_HPD].bgoEdep, 100, range=(0, 100), histtype="step", label=f"Mixing: {event_features_df[(event_features_df.mix) & ~event_features_df.pass_HPD].shape[0]}")
#     plt.hist(event_features_df[(~event_features_df.mix) & ~event_features_df.pass_HPD].bgoEdep, 100, range=(0, 100), histtype="step", label=f"BG: {event_features_df[(~event_features_df.mix) & ~event_features_df.pass_HPD].shape[0]}")
#     plt.legend()
#     plt.xlabel("BGO E dep in MeV")
#     plt.title(f"Don't pass HPD {cut:.2f} Cut")
#     plt.show()

# plt.plot(hpd_masks.keys(), mix_ratio, "o", label="Mixing Events kept")
# plt.plot(hpd_masks.keys(), tot_ratio, "o", label="Events kept")
# plt.plot(hpd_masks.keys(), bg_ratio, "o", label="BG Events kept")
# plt.xlabel("HPD Cut")
# plt.ylabel("% Events kept")
# plt.legend(loc="lower left")
# plt.show()





# plt.hist2d(event_features_df.n_bgo, event_features_df.bgoEdep, bins=(25, 120), range=((0, 25), (0, 120)), norm="log")
# plt.xlabel("# BGO hits")
# plt.ylabel("BGO E dep in MeV")
# plt.colorbar()
# # plt.title("Pass HPD Cut")
# plt.show()


# plt.hist2d(event_features_df[event_features_df.pass_HPD].n_bgo, event_features_df[event_features_df.pass_HPD].bgoEdep, bins=(25, 120), range=((0, 25), (0, 120)), norm="log")
# plt.xlabel("# BGO hits")
# plt.ylabel("BGO E dep in MeV")
# plt.colorbar()
# plt.title("Pass HPD Cut")
# plt.show()


# plt.hist2d(event_features_df[~event_features_df.pass_HPD].n_bgo, event_features_df[~event_features_df.pass_HPD].bgoEdep, bins=(25, 120), range=((0, 25), (0, 120)), norm="log")
# plt.xlabel("# BGO hits")
# plt.ylabel("BGO E dep in MeV")
# plt.colorbar()
# plt.title("Don't pass HPD Cut")
# plt.show()












from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KernelDensity

# -----------------------------
# 1️⃣ Select background and mixing events
# -----------------------------
bg_events = event_features_df[(~event_features_df.mix)]
mix_events = event_features_df[(event_features_df.mix)]

X_bg = np.vstack([bg_events.n_bgo.values, bg_events.bgoEdep.values]).T
X_mix = np.vstack([mix_events.n_bgo.values, mix_events.bgoEdep.values]).T

# -----------------------------
# 2️⃣ Scale features for KDE
# -----------------------------
scaler = StandardScaler()
X_bg_scaled = scaler.fit_transform(X_bg)
X_mix_scaled = scaler.transform(X_mix)

# -----------------------------
# 3️⃣ Fit 2D KDE on background
# -----------------------------
kde = KernelDensity(bandwidth=0.5, kernel='gaussian')  # tweak bandwidth if needed
kde.fit(X_bg_scaled)

# Evaluate KDE density at background events for HPD calculation
log_dens_bg = kde.score_samples(X_bg_scaled)
dens_bg = np.exp(log_dens_bg)

# -----------------------------
# 4️⃣ Compute HPD masks for multiple percentiles
# -----------------------------
# p_vals = [0.5, 0.8, 0.9, 0.95, 0.99]
p_vals = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99] # np.linspace(0, 0.999, 40) #np.concatenate([np.linspace(0, 0.9, 10), [0.95, 0.99, 0.999]])

order = np.argsort(dens_bg)[::-1]               # highest density first
cumprob = np.cumsum(dens_bg[order]) / np.sum(dens_bg)

hpd_masks = {}
thresholds = {}

for p in p_vals:
    idx = np.searchsorted(cumprob, p)
    mask = np.zeros(len(dens_bg), dtype=bool)
    mask[order[:idx+1]] = True                  # True = inside HPD envelope
    hpd_masks[p] = mask
    thresholds[p] = dens_bg[order[idx]]         # density threshold for this percentile

# -----------------------------
# 5️⃣ Apply HPD cut to any events (signal = outside envelope)
# -----------------------------
X_all = np.vstack([event_features_df.n_bgo.values, event_features_df.bgoEdep.values]).T
X_all_scaled = scaler.transform(X_all)
dens_all = np.exp(kde.score_samples(X_all_scaled))

for p in p_vals:
    event_features_df[f"pass_HPD_{int(p*100)}"] = dens_all < thresholds[p]

# -----------------------------
# 6️⃣ Compute S/√B for each percentile
# -----------------------------
N_BG = len(bg_events)
N_Mix = len(mix_events)

hp_stats = {}
for p in p_vals:
#    mask_signal = event_features_df[f"pass_HPD_{int(p*100)}"]

    outside = event_features_df[f"pass_HPD_{int(p*100)}"]
    inside  = ~outside

    B_inside  = np.sum((~event_features_df.Hbar) & inside)
    B_outside = np.sum((~event_features_df.Hbar) & outside)

    S_inside  = np.sum(event_features_df.Hbar & inside)
    S_outside = np.sum(event_features_df.Hbar & outside)

    assert B_inside + B_outside == N_BG
    assert S_inside + S_outside == N_Mix

    # B = np.sum((~event_features_df.Hbar) & (~mask_signal))  # BG inside envelope
    # S = np.sum(event_features_df.Hbar & mask_signal)                # Signal outside envelope
    snr = S_outside / np.sqrt(B_outside) if B_outside>0 else 0

    bg_eff = B_inside / N_BG

    print(
        f"{p*100:.1f}% HPD: "
        f"B_in={B_inside}, B_out={B_outside}, "
        f"S_out={S_outside}, S_in={S_inside}, S/√B={snr:.2f}, "
        f"BG inside={bg_eff:.3f}"
    )

    hp_stats[p] = {
        "B_inside": B_inside,
        "B_outside": B_outside,
        "S_inside": S_inside,
        "S_outside": S_outside,
        "SNR": snr,
        "BG_fraction_inside": bg_eff
    }

percentiles = [int(p*1000)/10 for p in hp_stats.keys()]  # 0.0→99.9%
B_vals = [hp_stats[p]['B_inside'] for p in hp_stats.keys()]
S_vals = [hp_stats[p]['S_outside'] for p in hp_stats.keys()]
S_in_vals = [hp_stats[p]['S_inside'] for p in hp_stats.keys()]
B_out_vals = [hp_stats[p]['B_outside'] for p in hp_stats.keys()]
SNR_vals = [hp_stats[p]['SNR'] for p in hp_stats.keys()]
BG_frac = [hp_stats[p]['BG_fraction_inside'] for p in hp_stats.keys()]

plt.figure()
plt.plot(percentiles, B_vals, 'o-', label='B inside envelope')
plt.plot(percentiles, S_vals, 's-', label='S outside envelope')
plt.plot(percentiles, SNR_vals, '^-', label='S/√B')
plt.xlabel("HPD percentile")
plt.ylabel("Counts / SNR")
plt.title("B, S, S/√B vs HPD percentile")
plt.legend()
plt.show()

plt.figure()
plt.plot(percentiles, BG_frac, 'o-')
plt.xlabel("HPD percentile")
plt.ylabel("BG fraction inside envelope")
plt.title("BG fraction vs HPD percentile")
plt.show()

plt.figure()
plt.plot(percentiles, SNR_vals, 'o-')
plt.xlabel("HPD percentile")
plt.ylabel("SNR")
plt.title("Signal to Noise Ratio")
plt.show()

plt.figure()
plt.plot(percentiles, np.array(S_in_vals)/np.array(S_vals), 'o-')
plt.xlabel("HPD percentile")
plt.ylabel("Signal inside/Signal outside")
# plt.title("Signal to Noise Ratio")
plt.show()


# -----------------------------
# 7️⃣ Optional: plot 2D KDE with HPD contours
# -----------------------------
# Create a grid
n_bgo_min, n_bgo_max = 0, 25
bgoE_min, bgoE_max = 0, 120
nbins_x, nbins_y = 25, 240
x_grid = np.linspace(n_bgo_min, n_bgo_max, nbins_x)
y_grid = np.linspace(bgoE_min, bgoE_max, nbins_y)
X, Y = np.meshgrid(x_grid, y_grid)
grid_points = np.vstack([X.ravel(), Y.ravel()]).T
grid_points_scaled = scaler.transform(grid_points)

# Evaluate KDE
dens_grid = np.exp(kde.score_samples(grid_points_scaled))
dens_grid = dens_grid.reshape(X.shape)

plt.figure(figsize=(8,6))
plt.pcolormesh(X, Y, dens_grid, shading="auto", cmap="viridis")
plt.colorbar(label="KDE density")

# Overlay HPD contours
for i, p in enumerate(p_vals):
    plt.contour(X, Y, dens_grid, levels=[thresholds[p]], colors=f"C{i+1}", linewidths=1.5)

plt.xlabel("# BGO Hits")
plt.ylabel("BGO E dep [MeV]")
plt.title("2D KDE of Background with HPD contours")
plt.show()


plt.figure(figsize=(8,6))
# Plot KDE density (background)
plt.pcolormesh(X, Y, dens_grid, shading="auto", cmap="viridis")
plt.colorbar(label="BG KDE density")

# Overlay HPD contours (same as before)
for i, p in enumerate(p_vals):
    plt.contour(X, Y, dens_grid, levels=[thresholds[p]], colors=f"C{i+1}", linewidths=1.5)

# Overlay mixing events
mask_signal = event_features_df[f"pass_HPD_{int(p*100)}"].values
plt.scatter(event_features_df.n_bgo.values[mask_signal & event_features_df.mix],
            event_features_df.bgoEdep.values[mask_signal & event_features_df.mix],
            s=10, c='red', label='Signal (outside HPD)')

plt.scatter(event_features_df.n_bgo.values[~mask_signal & event_features_df.mix],
            event_features_df.bgoEdep.values[~mask_signal & event_features_df.mix],
            s=10, c='blue', label='Inside HPD (background-like)')

plt.xlabel("# BGO Hits")
plt.ylabel("BGO E dep [MeV]")
plt.title("Mixing events over BG KDE with HPD contours")
plt.legend()
plt.show()



plt.figure(figsize=(8,6))

plt.hist2d(event_features_df.n_bgo.values, event_features_df.bgoEdep.values, bins=(nbins_x, nbins_y), cmap="viridis", density=True, norm="log")

# Overlay HPD contours (same as before)
for i, p in enumerate(p_vals):
    CS = plt.contour(X, Y, dens_grid, levels=[thresholds[p]], colors=f"C{i+1}", linewidths=1.5)
    # plt.clabel(CS, inline=1, fontsize=5)
plt.xlim(0,25)
plt.ylim(0,120)
plt.xlabel("# BGO Hits")
plt.ylabel("BGO E dep [MeV]")
# plt.title("Mixing events over BG KDE with HPD contours")
plt.legend()
plt.show()



plt.figure(figsize=(8,6))
plt.hist2d(event_features_df[event_features_df.mix].n_bgo.values, event_features_df[event_features_df.mix].bgoEdep.values, bins=(nbins_x, nbins_y), cmap="viridis", density=True, norm="log")

# Overlay HPD contours (same as before)
for i, p in enumerate(p_vals):
    CS = plt.contour(X, Y, dens_grid, levels=[thresholds[p]], colors=f"C{i+1}", linewidths=1.5)
    # plt.clabel(CS, inline=1, fontsize=5)
plt.xlim(0,25)
plt.ylim(0,120)
plt.xlabel("# BGO Hits")
plt.ylabel("BGO E dep [MeV]")
# plt.title("Mixing events over BG KDE with HPD contours")
plt.legend()
plt.show()


plt.figure(figsize=(8,6))
plt.hist2d(event_features_df[~event_features_df.mix].n_bgo.values, event_features_df[~event_features_df.mix].bgoEdep.values, bins=(nbins_x, nbins_y), cmap="viridis", density=True, norm="log")

# Overlay HPD contours (same as before)
for i, p in enumerate(p_vals):
    CS = plt.contour(X, Y, dens_grid, levels=[thresholds[p]], colors=f"C{i+1}", linewidths=1.5)
    # plt.clabel(CS, inline=1, fontsize=5)
plt.xlim(0,25)
plt.ylim(0,120)
plt.xlabel("# BGO Hits")
plt.ylabel("BGO E dep [MeV]")
# plt.title("Mixing events over BG KDE with HPD contours")
plt.legend()
plt.show()


plt.hist([event_features_df[event_features_df.pass_HPD_99].time*1e-9, event_features_df[~event_features_df.pass_HPD_99].time*1e-9], bins=100, stacked=True, label=["Signal", "Background"])
plt.legend()
plt.show()

plt.scatter(event_features_df.n_bgo.values[event_features_df.pass_HPD_95], event_features_df.bgoEdep.values[event_features_df.pass_HPD_95], s=10, label='Signal')
plt.scatter(event_features_df.n_bgo.values[~event_features_df.pass_HPD_95], event_features_df.bgoEdep.values[~event_features_df.pass_HPD_95], s=10, label='Background')






p_choice = 0.95
mask_name = f"pass_HPD_{int(p_choice*100)}"

# Background
bg = event_features_df[~event_features_df.mix]
mix = event_features_df[event_features_df.mix]

plt.figure(figsize=(7, 6))

# Background density
plt.hist2d(
    bg.n_bgo,
    bg.bgoEdep,
    bins=(np.arange(0, 26)-0.5, 240),
    range=((0, 26), (0, 120)),
    norm=plt.matplotlib.colors.LogNorm(),

    #cmap="Greys"
)

# HPD contour (from your KDE grid)
cnt = plt.contour(
    X, Y, dens_grid,
    levels=[thresholds[p_choice]],
    colors="C3",
    linewidths=2,
    # label=f"{int(p_choice*100)}% HPD"
)
artists, labels = cnt.legend_elements()

# # Mixing events
# plt.hist2d(
#     mix.n_bgo + 1/3,
#     mix.bgoEdep,
#     bins=(26*3, 80*3),
#     range=((0, 26), (0, 120)),
#     norm=plt.matplotlib.colors.LogNorm(),
#     cmap="plasma"
# )

scs = plt.scatter(
    mix.n_bgo,
    mix.bgoEdep,
    s=5,
    c="C1",
    alpha=0.5,
    label="Mixing events"
)

handles, labels = scs.legend_elements(prop="sizes", alpha=0.6)

# artists.append(handles)


plt.xlabel("# BGO hits")
plt.ylabel("BGO deposited energy [MeV]")
plt.title("Background density with HPD envelope and mixing events")
plt.legend([artists[0], scs], ["95 % HPD envelope", "Mixing events"])
plt.tight_layout()
plt.show()



fig, axs = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
axs[0].hist2d(
    bg.n_bgo,
    bg.bgoEdep,
    bins=(np.arange(0, 26)-0.5, 240),
    range=((0, 26), (0, 120)),
    norm=plt.matplotlib.colors.LogNorm(),
)
cnt = axs[0].contour(
    X, Y, dens_grid,
    levels=[thresholds[p_choice]],
    colors="C1",
    linewidths=3,
)
artists, labels = cnt.legend_elements()
axs[1].hist2d(
    mix.n_bgo,
    mix.bgoEdep,
    bins=(np.arange(0, 26)-0.5, 240),
    range=((0, 26), (0, 120)),
    norm=plt.matplotlib.colors.LogNorm(),
)
axs[1].contour(
    X, Y, dens_grid,
    levels=[thresholds[p_choice]],
    colors="C1",
    linewidths=3,
)
axs[0].set_xlabel("# BGO hits")
axs[0].set_xlim(0, 25)
axs[1].set_xlabel("# BGO hits")
axs[1].set_xlim(0, 25)
axs[0].set_ylabel("BGO Energy Deposit in MeV")
axs[0].legend(artists, [f"{int(p_choice*100)} % BG envelope"], loc="upper left")
axs[1].legend(artists, [f"{int(p_choice*100)} % BG envelope"], loc="upper left")
axs[0].set_title("Events outside mixing window")
axs[1].set_title("Events inside mixing window")
plt.subplots_adjust(wspace=0.05, hspace=0.1)
plt.savefig("2dhists_KDE_envelope.png", dpi=300)
plt.show()




p_sorted = np.array(sorted(hp_stats.keys()))
snr_vals = np.array([hp_stats[p]["SNR"] for p in p_sorted])

plt.figure(figsize=(6, 4))
plt.plot(p_sorted * 100, snr_vals, "o-", lw=2)

plt.axvline(p_choice * 100, color="C1", ls="--", label="Chosen cut")

plt.xlabel("HPD percentile [%]")
plt.ylabel(r"$S / \sqrt{B}$")
plt.title("Signal significance vs HPD cut")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("snr_vs_hpd.png", dpi=300)
plt.show()



S_total = hp_stats[0]["S_outside"] + hp_stats[0]["S_inside"]
B_total = hp_stats[0]["B_outside"] + hp_stats[0]["B_inside"]

sig_eff = np.array([hp_stats[p]["S_outside"] / S_total for p in p_sorted])
bg_rej  = np.array([1 - hp_stats[p]["B_outside"] / B_total for p in p_sorted])

plt.figure(figsize=(6, 4))

plt.plot(p_sorted * 100, sig_eff, "o-", label="Signal efficiency")
plt.plot(p_sorted * 100, bg_rej,  "o-", label="Background rejection")

plt.axvline(p_choice * 100, color="k", ls="--")

plt.xlabel("HPD percentile [%]")
plt.ylabel("Fraction")
plt.title("Signal efficiency vs background rejection")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("eff_vs_rej.png", dpi=300)
plt.show()



cut_def = {
    "xedges": xedges,
    "yedges": yedges,
    "density_level": thresholds[p_choice],
    "percentile": p_choice,
    "mask_BG": hist_BG >= thresholds[p_choice]
}

cut_def["notes"] = {
    "variables": ["n_bgo", "bgoEdep"],
    "bg_sample": "event > border_event, mix == False",
    "method": "2D HPD envelope from histogram",
    "signal_definition": "outside envelope"
}


np.save("bg_hpd_cut.npy", cut_def, allow_pickle=True)


# cut = np.load("bg_hpd_cut.npy", allow_pickle=True).item()

# xedges = cut["xedges"]
# yedges = cut["yedges"]
# lvl    = cut["density_level"]

# ix = np.searchsorted(xedges, new_df.n_bgo.values) - 1
# iy = np.searchsorted(yedges, new_df.bgoEdep.values) - 1
# valid = (
#     (ix >= 0) & (ix < len(xedges) - 1) &
#     (iy >= 0) & (iy < len(yedges) - 1)
# )
# passes = np.ones(len(new_df), dtype=bool)  # default = signal

# # inside histogram range → apply envelope
# passes[valid] = hist_BG[ix[valid], iy[valid]] < lvl

# new_df["pass_HPD"] = passes




import joblib

cut_def = {
    "scaler": scaler,           # StandardScaler object
    "kde": kde,                 # fitted KernelDensity object
    "thresholds": thresholds,   # dict {percentile: density_threshold}
    "percentiles": p_vals
}

joblib.dump(cut_def, "HPD_cut_kde.pkl")


# cut_def = joblib.load("HPD_cut_kde.pkl")
# scaler = cut_def["scaler"]
# kde    = cut_def["kde"]
# thresholds = cut_def["thresholds"]
# p_vals = cut_def["percentiles"]

# # Prepare new events
# X_new = np.vstack([new_df.n_bgo.values, new_df.bgoEdep.values]).T
# X_new_scaled = scaler.transform(X_new)

# dens_new = np.exp(kde.score_samples(X_new_scaled))

# # Apply HPD mask for each percentile
# for p in p_vals:
#     new_df[f"pass_HPD_{int(p*100)}"] = dens_new < thresholds[p]