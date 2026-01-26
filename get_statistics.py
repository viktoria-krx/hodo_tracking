%load_ext autoreload
%autoreload 2

from read_file import *
from find_tracks import *
from fit_tracks import *
from fit_vertex import *
from draw_tracks import *
from get_event_features import *
from ML_functions import *
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


run_list = np.concatenate([np.arange(1705, 1721), np.arange(1724, 1730)])


rand_list = pd.read_csv("../cern_data/2025_Data/random_run_list.csv", sep=", ")
rand_list.sort_values("cusp", inplace=True)
rand_list.reset_index(drop=True, inplace=True)
print(rand_list)

run_list = rand_list.hodo.values



n_hits_arr = np.zeros(len(run_list))


hits_df = build_hits_df_from_runs(run_list, version="cusp_run")
hits_df["Hbar_BG"] = ["Hbar" if m == True else "BG" for m in hits_df["mixGate"]]

hits_df["z_used"] = np.where(np.isnan(hits_df["z_reco"]), hits_df["z"], hits_df["z_reco"])
hits_df["dz_used"] = np.where(np.isnan(hits_df["z_reco"]), hits_df["dz"], hits_df["dz_reco"])
hits_df["bgoToT"] = hits_df[(hits_df["detector"] == "bgo") & hits_df["LE"].notna() & hits_df["TE"].notna()& (hits_df["LE"] < hits_df["TE"])]["TE"] - hits_df[(hits_df["detector"] == "bgo") & hits_df["LE"].notna() & hits_df["TE"].notna() & (hits_df["LE"] < hits_df["TE"])]["LE"]

for i, run in enumerate(hits_df.cuspRunNumber.unique()):
    n_hits_arr[i] = hits_df[hits_df.cuspRunNumber == run].shape[0]

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

print(f"{vertices_df.groupby('event').ngroups} of {hits_df.groupby('event').ngroups} events have at least one track\t {vertices_df.groupby('event').ngroups/hits_df.groupby('event').ngroups*100:.2f}%")

print(f"{vertices_df[(vertices_df.Vz != 0)].groupby(['event']).ngroups} of {hits_df.groupby('event').ngroups} events have at least one track\t {vertices_df[(vertices_df.Vz != 0)].groupby(['event']).ngroups/hits_df.groupby('event').ngroups*100:.2f}%")

# vertices_df = find_vertices_from_tracks(lines_df, eps=5.0)
lines_df = _lines_df.merge(vertices_df, on="event", how="left")

clustered_hits = clustered_hits.merge(lines_df, on=["event", "track_id"], how="left")

plot_events(clustered_hits, lines_df, clustered_hits[clustered_hits.Hbar_BG == "pbar"].event.unique()[:30])

event_features_df = compute_event_features_from_clustered_hits(
    clustered_hits,
    bgo_center=(0, 0, 0)
)

print("event features done")


print(f"{len(event_features_df[(event_features_df.vertex_x.abs() < 5) & (event_features_df.vertex_y.abs() < 5) & (event_features_df.vertex_z.abs() < 5)])} of {hits_df.groupby('event').ngroups} events have at least one track\t {len(event_features_df[(event_features_df.vertex_x.abs() < 5) & (event_features_df.vertex_y.abs() < 5) & (event_features_df.vertex_z.abs() < 5)])/hits_df.groupby('event').ngroups*100:.2f}%")




fig, axs = plt.subplots(1, 3, figsize=(18, 4))
axs[0].hist(event_features_df.vertex_x, bins=50, range=(-200, 200), density=True)
axs[0].set_xlabel("Vertex x in mm")
axs[1].hist(event_features_df.vertex_y, bins=50, range=(-200, 200), density=True)
axs[1].set_xlabel("Vertex y in mm")
axs[2].hist(event_features_df.vertex_z, bins=50, range=(-200, 200), density=True)
axs[2].set_xlabel("Vertex z in mm")
plt.savefig("Hbar_vertex_rawtree.png", bbox_inches="tight", pad_inches=0.1 , dpi=300)
plt.show()

fig, axs = plt.subplots(1, 3, figsize=(18, 4))
axs[0].hist(event_features_df.vertex_x, bins=50, range=(-20, 20), density=True)
axs[0].set_xlabel("Vertex x in mm")
axs[1].hist(event_features_df.vertex_y, bins=50, range=(-20, 20), density=True)
axs[1].set_xlabel("Vertex y in mm")
axs[2].hist(event_features_df.vertex_z, bins=50, range=(-10, 10), density=True)
axs[2].set_xlabel("Vertex z in mm")
plt.savefig("Hbar_vertex_zoom_rawtree.png", bbox_inches="tight", pad_inches=0.1, dpi=300)
plt.show()








hits_df = build_hits_df_from_runs(run_list, version="cusp_run", tree="EventTree")
hits_df["Hbar_BG"] = ["Hbar" if m == True else "BG" for m in hits_df["mixGate"]]

hits_df["z_used"] = np.where(np.isnan(hits_df["z_reco"]), hits_df["z"], hits_df["z_reco"])
hits_df["dz_used"] = np.where(np.isnan(hits_df["z_reco"]), hits_df["dz"], hits_df["dz_reco"])
hits_df["bgoToT"] = hits_df[(hits_df["detector"] == "bgo") & hits_df["LE"].notna() & hits_df["TE"].notna()& (hits_df["LE"] < hits_df["TE"])]["TE"] - hits_df[(hits_df["detector"] == "bgo") & hits_df["LE"].notna() & hits_df["TE"].notna() & (hits_df["LE"] < hits_df["TE"])]["LE"]

n_hits_arr_EventTree = np.zeros(len(run_list))

for i, run in enumerate(hits_df.cuspRunNumber.unique()):
    n_hits_arr_EventTree[i] = hits_df[hits_df.cuspRunNumber == run].shape[0]

print(n_hits_arr_EventTree/n_hits_arr*100)

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

print(f"{vertices_df.groupby('event').ngroups} of {hits_df.groupby('event').ngroups} events have at least one track\t {vertices_df.groupby('event').ngroups/hits_df.groupby('event').ngroups*100:.2f}%")

print(f"{vertices_df[(vertices_df.Vz != 0)].groupby(['event']).ngroups} of {hits_df.groupby('event').ngroups} events have at least one track\t {vertices_df[(vertices_df.Vz != 0)].groupby(['event']).ngroups/hits_df.groupby('event').ngroups*100:.2f}%")

# vertices_df = find_vertices_from_tracks(lines_df, eps=5.0)
lines_df = _lines_df.merge(vertices_df, on="event", how="left")

clustered_hits = clustered_hits.merge(lines_df, on=["event", "track_id"], how="left")

plot_events(clustered_hits, lines_df, clustered_hits[clustered_hits.Hbar_BG == "pbar"].event.unique()[:30])

event_features_df = compute_event_features_from_clustered_hits(
    clustered_hits,
    bgo_center=(0, 0, 0)
)

print("event features done")


print(f"{len(event_features_df[(event_features_df.vertex_x.abs() < 5) & (event_features_df.vertex_y.abs() < 5) & (event_features_df.vertex_z.abs() < 5)])} of {hits_df.groupby('event').ngroups} events have at least one track\t {len(event_features_df[(event_features_df.vertex_x.abs() < 5) & (event_features_df.vertex_y.abs() < 5) & (event_features_df.vertex_z.abs() < 5)])/hits_df.groupby('event').ngroups*100:.2f}%")




fig, axs = plt.subplots(1, 3, figsize=(18, 4))
axs[0].hist(event_features_df.vertex_x, bins=50, range=(-200, 200), density=True)
axs[0].set_xlabel("Vertex x in mm")
axs[1].hist(event_features_df.vertex_y, bins=50, range=(-200, 200), density=True)
axs[1].set_xlabel("Vertex y in mm")
axs[2].hist(event_features_df.vertex_z, bins=50, range=(-200, 200), density=True)
axs[2].set_xlabel("Vertex z in mm")
plt.savefig("Hbar_vertex.png", bbox_inches="tight", pad_inches=0.1 , dpi=300)
plt.show()

fig, axs = plt.subplots(1, 3, figsize=(18, 4))
axs[0].hist(event_features_df.vertex_x, bins=50, range=(-20, 20), density=True)
axs[0].set_xlabel("Vertex x in mm")
axs[1].hist(event_features_df.vertex_y, bins=50, range=(-20, 20), density=True)
axs[1].set_xlabel("Vertex y in mm")
axs[2].hist(event_features_df.vertex_z, bins=50, range=(-10, 10), density=True)
axs[2].set_xlabel("Vertex z in mm")
plt.savefig("Hbar_vertex_zoom.png", bbox_inches="tight", pad_inches=0.1, dpi=300)
plt.show()






# event_features_df[event_features_df.n_tracks > 5].bgoEdep.hist(bins=50, density=True, histtype="step")
# event_features_df[event_features_df.n_tracks <= 5].bgoEdep.hist(bins=50, density=True, histtype="step")


# event_features_df[event_features_df.n_tracks > 5].Annihilation.hist(bins=2, density=True, histtype="step")
# event_features_df[event_features_df.n_tracks <= 5].Annihilation.hist(bins=2, density=True, histtype="step")





# event_features_df[event_features_df.mix].event.nunique()


event_features_df[~event_features_df.mix].event.nunique()/event_features_df[event_features_df.mix].event.nunique()

event_features_df.event.nunique()/(event_features_df.time.max()*1e-9)

np.max(event_features_df.time*1e-9)

bg_rates = []
mix_durs = []
mix_cts = []

for run in event_features_df.cusp.unique():
    print(run)

    mix_start = event_features_df[(event_features_df.cusp == run) & (event_features_df.mix)].time.min()*1e-9

    mix_end = event_features_df[(event_features_df.cusp == run) & (event_features_df.mix)].time.max()*1e-9

    mix_cts.append(event_features_df[(event_features_df.cusp == run) & (event_features_df.mix)].event.nunique())

    mix_dur = mix_end - mix_start

    bg_time = event_features_df[(event_features_df.cusp == run)].time.max()*1e-9 - mix_dur

    print(mix_start, mix_end)
    print(bg_time,  event_features_df[(event_features_df.cusp == run) & (~event_features_df.mix)].event.nunique()/bg_time)

    bg_rates.append(event_features_df[(event_features_df.cusp == run) & (~event_features_df.mix)].event.nunique()/bg_time)
    mix_durs.append(mix_dur)

    plt.hist([event_features_df[(event_features_df.cusp == run) & (event_features_df.mix)].bgoEdep, event_features_df[(event_features_df.cusp == run) & (~event_features_df.mix)].bgoEdep], histtype="step", label=["Mixing Gate", "BG"], bins=20)
    plt.legend()
    plt.xlabel("BGO Edep in MeV")
    plt.show()

    plt.hist([event_features_df[(event_features_df.cusp == run) & (event_features_df.mix)].time*1e-9, event_features_df[(event_features_df.cusp == run) & (~event_features_df.mix)].time*1e-9], bins=35, range=(0, 350), stacked=True, label=["Mixing Gate", "BG"])
    plt.show()

print("BG Rate", np.mean(np.array(bg_rates)))
print("BG Events", np.mean(np.array(bg_rates)*np.array(mix_durs)))
print("BG %", np.mean(np.array(bg_rates)*np.array(mix_durs)/(np.array(mix_cts))))

plt.hist([event_features_df[(event_features_df.mix)].bgoEdep, event_features_df[(~event_features_df.mix)].bgoEdep], histtype="step", label=["Mixing", "BG"], bins=60, range=(0, 150), linewidth=1.5)
plt.legend()
plt.xlabel("BGO Edep in MeV")
plt.savefig("bgoedep_mixing.png", bbox_inches="tight", pad_inches=0.1, dpi=300)
plt.show()

event_features_df[(event_features_df.mix) & (event_features_df.bgoEdep > 20)].event.nunique()

event_features_df[(event_features_df.mix) & (event_features_df.bgoEdep > 45)].event.nunique()

event_features_df[(event_features_df.mix) & (event_features_df.bgoEdep > 0)].event.nunique()



# def E(ToT):
#     return 0.0581 + 4.699 * ToT - 11.85 *np.exp(- 0.02459 *ToT)

# plt.plot(np.linspace(0, 12000, 120), E(np.linspace(0, 12000, 120)))

# plt.plot(E(np.linspace(0, 120, 120)))


