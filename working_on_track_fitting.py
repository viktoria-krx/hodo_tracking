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

from itertools import product

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

root_path = "~/Documents/Hodoscope/cern_data/2025_Data/output_001825_6567.root" # cosmics

BGdf = build_hits_df_fast(root_path)
BGdf["Hbar_BG"] = "BG"

max_ev = BGdf["event"].max()

run_list = np.concatenate([np.arange(1705, 1721), np.arange(1724, 1730)])
Hbardf = build_hits_df_from_runs(run_list, version="cusp_run")
Hbardf["Hbar_BG"] = ["Hbar" if m == True else "BG" for m in Hbardf["mixGate"]]
Hbardf.loc[:, "event"] = Hbardf["event"] + max_ev

# hits_df = pd.concat([BGdf, Hbardf], ignore_index=True)
hits_df = pd.concat([BGdf[BGdf.event <= 5000], Hbardf], ignore_index=True)
hits_df = Hbardf

hits_df["z_used"] = np.where(np.isnan(hits_df["z_reco"]), hits_df["z"], hits_df["z_reco"])
hits_df["dz_used"] = np.where(np.isnan(hits_df["dz_reco"]), hits_df["dz"], hits_df["dz_reco"])
hits_df["bgoToT"] = hits_df[(hits_df["detector"] == "bgo") & hits_df["LE"].notna() & hits_df["TE"].notna()& (hits_df["LE"] < hits_df["TE"])]["TE"] - hits_df[(hits_df["detector"] == "bgo") & hits_df["LE"].notna() & hits_df["TE"].notna() & (hits_df["LE"] < hits_df["TE"])]["LE"]



file1 = open("param_scan.csv", "w")
file1.write("eps, theta_weight, weight_power, dist_bgo, rec_vs, z0_vert \n")
file1.close()


last_cluster_params = [np.nan, np.nan, np.nan]
last_linefit_params = [np.nan, np.nan]

scan_params = {
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

results_list = []

# Get the parameter names
param_names = list(scan_params.keys())

# Get list of lists
param_values = [scan_params[key] for key in param_names]

# Cartesian product of all parameter lists
all_combinations = product(*param_values)

for combo in all_combinations:
    # combo is a tuple of parameter values in the same order as param_names
    params = dict(zip(param_names, combo))

    print(params)
    eps = params["eps"]
    z_w_same = params["zweight_same"]
    z_w_diff = params["zweight_diff"]
    w_pow = params["weight_power"]
    w_pow_z = params["weight_power_z"]
    dist_bgo = params["dist_bgo"]
    vertex_cluster = params["vertex_cluster"]
    vertex_eps = params["vertex_eps"]
    vertex_alpha = params["vertex_alpha"]

    if w_pow > w_pow_z:
        continue

    # automatic param string
    param_string = "_".join(f"{k}-{v}" for k, v in params.items())
    # print(param_string)

    print("\n".join(f"{k}: {v}" for k, v in params.items()))

    if [eps, z_w_same, z_w_diff] != last_cluster_params:

        clustered_list = []
        for event_id, ev in hits_df[hits_df.LE < 0].groupby("event"):
            ev["det_key"] = ev["detector"] + "_" + ev["channel"].astype(str)

            # select only hodo/tile hits for clustering
            ev_ht = ev[ev.detector.isin(["hodoO","hodoI","tileO","tileI"])].copy()
            if ev_ht.empty:
                ev["track_id"] = -1
                clustered_list.append(ev)
                continue

            # labels = cluster_by_sin_cos_phi_uncertainty(ev_ht, base_eps=eps, min_samples=2, z_weight=0.2)
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

    

    last_cluster_params = [eps, z_w_same, z_w_diff]

    clustered_hits = pd.concat(clustered_list, ignore_index=True)
    
    if [w_pow, w_pow_z] != last_linefit_params:

        _lines_df = fit_lines_from_clusters_svd(clustered_hits, include_bgo=False, 
                                            use_xyz_errors=True, xyz_error_cols=["dx", "dy", "dz_used"], weight_power=w_pow, weight_power_z=w_pow_z, prefilter_ransac=False, ransac_thresh=15.0, weighted=False, weight_col="dz_used")

        print("line fitting done")

    vertices_df = reconstruct_vertex_from_midpoints(clustered_hits, _lines_df,
                                                bgo_radius=45.0, 
                                                max_dist_to_bgo=dist_bgo, 
                                                cluster_mids=vertex_cluster, cluster_eps=vertex_eps, cluster_alpha=vertex_alpha)#25.0)

    print("vertex reconstruction done")

    # file1 = open("param_scan.csv", "a")
    # file1.write(f"{eps}, {th_w}, {w_pow}, {dist_bgo}, {vertices_df.Vx.notna().sum()}, {(vertices_df.Vz == 0).sum()} \n")
    # file1.close()

    # vertices_df = find_vertices_from_tracks(lines_df, eps=5.0)
    lines_df = _lines_df.merge(vertices_df, on="event", how="left")

    clustered_hits = clustered_hits.merge(lines_df, on=["event", "track_id"], how="left")

    interesting_evs = [19627, 19629, 19632, 19639, 19640]

    plot_events(clustered_hits, lines_df, interesting_evs, save=True, title=param_string)

    # plot_events(clustered_hits, lines_df, clustered_hits.event.unique()[50:80])

    # plt.hist(vertices_df[vertices_df.Vz != 0].Vz, range=(-100, 100), bins=100, label=f"{(vertices_df.Vz == 0).sum()} Vz == 0")
    # plt.legend()
    # plt.xlabel("Vz in mm")
    # # plt.title(f"eps = {eps} weight power = {w_pow}, {vertices_df.Vx.notna().sum()} vertices")
    # plt.savefig(param_string + "_Vz.png")
    # plt.show()

    plt.hist2d(vertices_df.Vx, vertices_df.Vy, range=[(-200, 200), (-200, 200)], bins=100, norm="log")
    plt.gca().set_aspect("equal")
    plt.colorbar()
    plt.xlabel("Vx in mm")
    plt.ylabel("Vy in mm")
    # plt.title(f"eps = {eps} weight power = {w_pow}, {vertices_df.Vx.notna().sum()} vertices")
    # plt.savefig(param_string + "_Vxy.png")
    plt.show()

    plt.hist2d(vertices_df.Vx, vertices_df.Vz, range=[(-200, 200), (-200, 200)], bins=100, norm="log")
    plt.gca().set_aspect("equal")
    plt.colorbar()
    plt.xlabel("Vx in mm")
    plt.ylabel("Vz in mm")
    # plt.title(f"eps = {eps} weight power = {w_pow}, {vertices_df.Vx.notna().sum()} vertices")
    # plt.savefig(param_string + "_Vxz.png")
    plt.show()

    plt.hist2d(vertices_df.Vz, vertices_df.Vy, range=[(-200, 200), (-200, 200)], bins=100, norm="log")
    plt.gca().set_aspect("equal")
    plt.colorbar()
    plt.xlabel("Vz in mm")
    plt.ylabel("Vy in mm")
    # plt.title(f"eps = {eps} weight power = {w_pow}, {vertices_df.Vx.notna().sum()} vertices")
    # plt.savefig(param_string + "_Vzy.png")
    plt.show()

    # Store results and parameters in a dictionary
    result = {
        **params,         # unpack all parameter values
        "n_vertices": vertices_df.Vx.notna().sum(),
        "n_z0": (vertices_df.Vz == 0).sum(),
        "Vx_sig": vertices_df.Vx.std(),
        "Vy_sig": vertices_df.Vy.std(),
        "Vz_sig": vertices_df.Vz.std(),
        "Vx_mean": vertices_df.Vx.mean(),
        "Vy_mean": vertices_df.Vy.mean(),
        "Vz_mean": vertices_df.Vz.mean(),
        "n_mid_mean": vertices_df.n_midpoints.mean(),
        "n_mid_std": vertices_df.n_midpoints.std(),
    }
    results_list.append(result)

        # plot_tracks_3d(clustered_hits, lines_df, interesting_evs, plot_whole_event=True)


results_df = pd.DataFrame(results_list)

plt.scatter(results_df.vertex_eps, results_df.vertex_alpha, c=results_df.n_z0)
plt.xlabel("vertex eps")
plt.ylabel("vertex alpha")
plt.colorbar(label="vertices with z == 0")
plt.show()

plt.scatter(results_df.vertex_eps, results_df.vertex_alpha, c=results_df.Vx_sig)
plt.xlabel("vertex eps")
plt.ylabel("vertex alpha")
plt.colorbar(label="Vx std")
plt.show()

plt.scatter(results_df.vertex_eps, results_df.vertex_alpha, c=results_df.Vz_sig)
plt.xlabel("vertex eps")
plt.ylabel("vertex alpha")
plt.colorbar(label="Vz std")
plt.show()


plt.scatter(results_df.vertex_eps, results_df.vertex_alpha, c=results_df.Vx_mean, s=results_df.Vx_sig**3/300)
plt.xlabel("vertex eps")
plt.ylabel("vertex alpha")
plt.colorbar(label="Vx mean")
plt.show()

plt.scatter(results_df.vertex_eps, results_df.vertex_alpha, c=results_df.Vy_mean, s=results_df.Vy_sig**3/300)
plt.xlabel("vertex eps")
plt.ylabel("vertex alpha")
plt.colorbar(label="Vy mean")
plt.show()

plt.scatter(results_df.vertex_eps, results_df.vertex_alpha, c=results_df.Vz_mean)
plt.xlabel("vertex eps")
plt.ylabel("vertex alpha")
plt.colorbar(label="Vz mean")
plt.show()


plt.scatter(results_df.vertex_eps, results_df.vertex_alpha, c=results_df.n_z0/results_df.n_vertices)
plt.xlabel("vertex eps")
plt.ylabel("vertex alpha")
plt.colorbar(label="ratio z=0/rec. vertices")
plt.show()


plt.scatter(results_df.vertex_eps, results_df.vertex_alpha, c=results_df.n_vertices)
plt.xlabel("vertex eps")
plt.ylabel("vertex alpha")
plt.colorbar(label="n vertices")
plt.show()




plt.scatter(results_df.weight_power, results_df.weight_power_z, c=results_df.n_z0)
plt.xlabel("weight power")
plt.ylabel("weight power z")
plt.colorbar(label="vertices with z == 0")
plt.show()



plt.scatter(results_df.weight_power, results_df.weight_power_z, c=np.array(results_df.n_z0)/np.array(results_df.n_vertices))
plt.xlabel("weight power")
plt.ylabel("weight power z")
plt.colorbar(label="ratio z=0/rec. vertices")
plt.show()


plt.scatter(results_df.weight_power, results_df.weight_power_z, c=results_df.Vz_sig)
plt.xlabel("weight power")
plt.ylabel("weight power z")
plt.colorbar(label="Vz std")
plt.show()


plt.scatter(results_df.weight_power, results_df.weight_power_z, c=results_df.Vx_sig)
plt.xlabel("weight power")
plt.ylabel("weight power z")
plt.colorbar(label="Vx std")
plt.show()

plt.scatter(results_df.weight_power, results_df.weight_power_z, c=results_df.Vy_sig)
plt.xlabel("weight power")
plt.ylabel("weight power z")
plt.colorbar(label="Vy std")
plt.show()




plt.scatter(x_eps, y_th_w, c=rec_vs)
plt.xlabel("eps")
plt.ylabel("th_w")
plt.colorbar(label="reconstructed vertices")
plt.show()

plt.scatter(x_eps, y_th_w, c=z0_vert)
plt.xlabel("eps")
plt.ylabel("th_w")
plt.colorbar(label="vertices with z == 0")
plt.show()

plt.scatter(x_eps, y_th_w, c=np.array(z0_vert)/np.array(rec_vs), s=rec_vs)
plt.xlabel("eps")
plt.ylabel("th_w")
plt.colorbar(label="ratio z=0/rec. vertices")
plt.show()

plot_events(clustered_hits, lines_df, clustered_hits[clustered_hits.mixGate].event.unique()[30:50])

plot_tracks_3d(clustered_hits, lines_df, clustered_hits[clustered_hits.mixGate].event.unique()[30:50], plot_whole_event=True)

param_strings = ["eps", "th_w", "w_pow", "d_bgo"]
for i, param in enumerate([x_eps, y_th_w, y_w_pow, d_bgo]):
    plt.scatter(rec_vs, (np.array(z0_vert)/np.array(rec_vs)), c=param)
    plt.xlabel("reconstructed vertices")
    plt.ylabel("ratio z=0/rec. vertices")
    plt.colorbar(label=param_strings[i])
    plt.show()



    plt.scatter(rec_vs, z0_vert, c=param)
    plt.xlabel("reconstructed vertices")
    plt.ylabel("vertices with z == 0")
    plt.colorbar(label=param_strings[i])
    plt.show()

plt.scatter(rec_vs, z0_vert, c=y_th_w, s=np.array(y_th_w)*100)
plt.xlabel("reconstructed vertices")
plt.ylabel("vertices with z == 0")
plt.colorbar(label=param_strings[i])
plt.show()


# d_bgo = 100
# w_pow < 0.5
# eps > 0.8
# th_w < 0.5



param_strings = ["eps", "z_w_same", "z_w_diff", "w_pow", "d_bgo"]
for i, param in enumerate([x_eps, z_ws_same, z_ws_diff, y_w_pow, d_bgo]):
    plt.scatter(rec_vs, (np.array(z0_vert)/np.array(rec_vs)), c=param)
    plt.xlabel("reconstructed vertices")
    plt.ylabel("ratio z=0/rec. vertices")
    plt.colorbar(label=param_strings[i])
    plt.show()



    plt.scatter(rec_vs, z0_vert, c=param)
    plt.xlabel("reconstructed vertices")
    plt.ylabel("vertices with z == 0")
    plt.colorbar(label=param_strings[i])
    plt.show()
