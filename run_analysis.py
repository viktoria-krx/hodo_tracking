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
import pickle
import json



def load_hits(run, version="cusp_run", tree="RawEventTree"):
    if type(run) == list:
        # print("List of run(s)")
        hits_df = build_hits_df_from_runs(run_list, version=version, tree=tree)
    elif not hasattr(run, "__len__"):
        # print("Single run converted to list")
        hits_df = build_hits_df_from_runs([run], version=version, tree=tree)
    else:
        # print("Multiple runs converted to list")
        hits_df = build_hits_df_fast(list(run), version=version, tree=tree)

    hits_df["z_used"] = np.where(np.isnan(hits_df["z_reco"]), hits_df["z"], hits_df["z_reco"])
    hits_df["dz_used"] = np.where(np.isnan(hits_df["dz_reco"]), hits_df["dz"], hits_df["dz_reco"])
    hits_df["bgoToT"] = hits_df[(hits_df["detector"] == "bgo") & hits_df["LE"].notna() & hits_df["TE"].notna()& (hits_df["LE"] < hits_df["TE"])]["TE"] - hits_df[(hits_df["detector"] == "bgo") & hits_df["LE"].notna() & hits_df["TE"].notna() & (hits_df["LE"] < hits_df["TE"])]["LE"]

    return hits_df



def get_clustered_hits(hits_df, params):

    eps = params["eps"][0]
    z_w_same = params["zweight_same"][0]
    z_w_diff = params["zweight_diff"][0]

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

    # print("clustering done")

    return pd.concat(clustered_list, ignore_index=True)


def get_lines_and_vertices_df(clustered_hits, params):
    w_pow = params["weight_power"][0]
    w_pow_z = params["weight_power_z"][0]

    _lines_df = fit_lines_from_clusters_svd(clustered_hits, include_bgo=False, 
                                        use_xyz_errors=True, xyz_error_cols=["dx", "dy", "dz_used"], weight_power=w_pow, weight_power_z=w_pow_z, prefilter_ransac=False, ransac_thresh=15.0, weighted=False, weight_col="dz_used")

    # print("line fitting done")
    # return _lines_df

# _lines_df = get_lines_df(clustered_hits, params)

# def get_vertices_df(clustered_hits, _lines_df, params):
    dist_bgo = params["dist_bgo"][0]
    vertex_cluster = params["vertex_cluster"][0]
    vertex_eps = params["vertex_eps"][0]
    vertex_alpha = params["vertex_alpha"][0]

    vertices_df = reconstruct_vertex_from_midpoints(clustered_hits, _lines_df,
                                            bgo_radius=45.0, 
                                            max_dist_to_bgo=dist_bgo, 
                                            cluster_mids=vertex_cluster, cluster_eps=vertex_eps, cluster_alpha=vertex_alpha)#25.0)

    # print("vertex reconstruction done")
    # return vertices_df

# vertices_df = get_vertices_df(clustered_hits, _lines_df, params)

    # print(f"{vertices_df.groupby('event').ngroups} of {hits_df.groupby('event').ngroups} events have a reconstructed vertex\t {vertices_df.groupby('event').ngroups/hits_df.groupby('event').ngroups*100:.2f}%")

    # print(f"{vertices_df[(vertices_df.Vz != 0)].groupby(['event']).ngroups} of {hits_df.groupby('event').ngroups} events have a reconstructed non-zero vertex\t {vertices_df[(vertices_df.Vz != 0)].groupby(['event']).ngroups/hits_df.groupby('event').ngroups*100:.2f}%")

    # vertices_df = find_vertices_from_tracks(lines_df, eps=5.0)
    lines_df = _lines_df.merge(vertices_df, on="event", how="left")

    clustered_hits = clustered_hits.merge(lines_df, on=["event", "track_id"], how="left")

    return lines_df, vertices_df, clustered_hits




run_list = 2150

for run in  np.arange(2150, 2170): #[2078]:
    
    hits_df = load_hits(run, tree="EventTree")

    params = json.load(open("vertex_params.json"))

    clustered_hits = get_clustered_hits(hits_df, params)

    # print(f"{clustered_hits[(clustered_hits.track_id > -1)].groupby('event').ngroups} of {hits_df.groupby('event').ngroups} events have at least one cluster\t {clustered_hits[(clustered_hits.track_id > -1)].groupby('event').ngroups/hits_df.groupby('event').ngroups*100:.2f}%")

    # print(f"{clustered_hits[(clustered_hits.z_used != 0)].groupby(['event']).ngroups} of {hits_df.groupby('event').ngroups} events have at least one track\t {clustered_hits[(clustered_hits.z_used != 0)].groupby(['event']).ngroups/hits_df.groupby('event').ngroups*100:.2f}%")

    lines_df, vertices_df, clustered_hits = get_lines_and_vertices_df(clustered_hits, params)

    event_features_df = compute_event_features_from_clustered_hits_data(
        clustered_hits,
        bgo_center=(0, 0, 0)
    )

    # print("event features done")

    loaded_model = pickle.load(open("best_xgb_nan.pkl", 'rb'))

    run_feats = set(["event", "cusp", "time", "mix", "Hbar"])
    all_feats = [ele for ele in event_features_df.columns if ele not in run_feats]
    ML_feats = [ele for ele in event_features_df.columns if ele not in set(["vertex", "trigger", "mix", "time", "cusp", "event", "Hbar", "Annihilation"])]

    event_features_df["proba"] = loaded_model.predict_proba(event_features_df[ML_feats].values)[:, 1]

    print(f"Run {run}")
    print(f"{event_features_df[event_features_df.proba > 0.5].shape[0]} events with proba > 0.5")
    print(f"{event_features_df[event_features_df.proba > 0.5].shape[0]/event_features_df.shape[0]*100:.2f}% of events with proba > 0.5")



    plt.hist([event_features_df.time[(event_features_df.Hbar)]*1e-9, event_features_df.time[(~event_features_df.Hbar)]*1e-9], bins=35, stacked=True, label=["Mixing", "BG"])
    plt.hist(event_features_df.time[(event_features_df.proba > 0.5)]*1e-9, bins=35, histtype="step", label="Hbar XGB NaN prob > 0.5", lw=2)
    plt.legend()
    plt.xlabel("Time in s")
    # plt.savefig("signal_timing_proba.png", dpi=300, bbox_inches="tight", pad_inches=0.1)
    # plt.yscale("log")
    plt.title(f"Run {run}")
    plt.show()


    thresholds = np.linspace(0.1, 0.9, 9)
    for thresh in thresholds:

        # print(f"Threshold = {thresh}")
        print(f"Events with proba > {thresh}: {event_features_df[event_features_df.proba > thresh].shape[0]}")

        plt.hist([event_features_df[event_features_df.proba > thresh].time*1e-9, event_features_df[event_features_df.proba <= thresh].time*1e-9], bins=100, stacked=True, label=[f"proba > {thresh}", f"proba <= {thresh}"])
        plt.legend()
        plt.xlabel("Time in s")
        plt.title(f"Run {run} Threshold = {thresh}")
        plt.show()


# plot_events(clustered_hits, lines_df, [91, 185, 226, 293, 299, 308, 312], title=f"{run}_", save=True)

# plot_events(clustered_hits, lines_df, [249], title=f"{run}_", save=True)


# plt.hist(event_features_df.time[event_features_df.vertex_z < -100]*1e-9)

# plt.hist(event_features_df.vertex_z, 100)



plt.hist([event_features_df.time[(event_features_df.Hbar)]*1e-9, event_features_df.time[(~event_features_df.Hbar)]*1e-9], bins=35, stacked=True, label=["Mixing", "BG"])
plt.hist(event_features_df.time[(event_features_df.proba > 0.5)]*1e-9, bins=35, histtype="step", label="Hbar XGB NaN prob > 0.5", lw=2)
plt.legend()
plt.xlabel("Time in s")
# plt.savefig("signal_timing_proba.png", dpi=300, bbox_inches="tight", pad_inches=0.1)
# plt.yscale("log")
plt.show()