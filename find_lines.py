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

import warnings
from sklearn.exceptions import UndefinedMetricWarning

warnings.filterwarnings("ignore", message="R\^2 score is not well-defined")
plt.style.use("asacusa.mplstyle")


root_path = "~/Documents/Hodoscope/cern_data/2025_Data/output_001392.root" # Hbar

root_path = "~/Documents/Hodoscope/cern_data/2025_Data/output_000636.root" # cosmics

root_path = "~/Documents/Hodoscope/cern_data/2025_Data/output_001791_6535_noBug.root" # test_file


# hits_df = build_hits_df_fast(root_path)
# hits_df
# hits_df["z_used"] = np.where(np.isnan(hits_df["z_reco"]), hits_df["z"], hits_df["z_reco"])

# run_list = np.arange(1392, 1412)
# hits_df = build_hits_df_from_runs(run_list, version="simple")
run_list = np.concatenate([np.arange(1705, 1721), np.arange(1724, 1730)])
hits_df = build_hits_df_from_runs(run_list, version="cusp_run")
# hits_df
hits_df["z_used"] = np.where(np.isnan(hits_df["z_reco"]), hits_df["z"], hits_df["z_reco"])
hits_df["bgoToT"] = hits_df[(hits_df["detector"] == "bgo") & hits_df["LE"].notna() & hits_df["TE"].notna()& (hits_df["LE"] < hits_df["TE"])]["TE"] - hits_df[(hits_df["detector"] == "bgo") & hits_df["LE"].notna() & hits_df["TE"].notna() & (hits_df["LE"] < hits_df["TE"])]["LE"]

for eps in [1]:

    clustered_list = []
    for event_id, ev in hits_df[hits_df.LE < 0].groupby("event"):
        ev["det_key"] = ev["detector"] + "_" + ev["channel"].astype(str)
        # select only hodo/tile hits for clustering
        ev_ht = ev[ev.detector.isin(["hodoO","hodoI","tileO","tileI"])].copy()
        if ev_ht.empty:
            ev["track_id"] = -1
            clustered_list.append(ev)
            continue

        labels = cluster_by_phi_uncertainty(ev_ht, base_eps=eps, min_samples=2, theta_weight=0.2, coords="cylindrical")
        # labels = cluster_by_phi_hdbscan(ev_ht, min_samples=2, theta_weight=0, coords="cylindrical")
        ev_ht["track_id"] = labels

        # merge labels back into the full event (bgo keep -1)
        ev = ev.merge(ev_ht[["det_key","track_id"]], on="det_key", how="left")
        ev["track_id"] = ev["track_id"].fillna(-1).astype(int)

        clustered_list.append(ev)

    clustered_hits = pd.concat(clustered_list, ignore_index=True)


    lines_df = fit_lines_from_clusters_svd(clustered_hits, include_bgo=True,
                                            prefilter_ransac=True, ransac_thresh=15.0, weighted=True, weight_col="dz")

    vertices_df = reconstruct_vertex_from_midpoints(clustered_hits, lines_df,
                                              bgo_radius=45.0, 
                                              max_dist_to_bgo=25.0)


    # vertices_df = find_vertices_from_tracks(lines_df, eps=5.0)
    lines_df = lines_df.merge(vertices_df, on="event", how="left")

    clustered_hits = clustered_hits.merge(lines_df, on=["event", "track_id"], how="left")

    print(lines_df.groupby("event").Vx.mean().notna().sum(), "reconstructed vertices")
    vertices_checked = check_vertex_bgo_proximity(vertices_df, clustered_hits, sigma_level=2.0)
    close_vertices = vertices_checked[vertices_checked["within_sigma"]]
    print(len(close_vertices), "reconstructed vertices on BGO")


    plot_events(clustered_hits, lines_df, clustered_hits.event.unique()[60:80])





plt.plot(clustered_hits.groupby("event").fpgaTimeTag.mean()*1e-9, clustered_hits.groupby("event").Vx.mean(), "o")
plt.show()


plt.hist(clustered_hits[clustered_hits.Vx.notna()].groupby("event").fpgaTimeTag.mean()*1e-9)
plt.xlabel("Time in s")
plt.ylabel("Reconstructed Vertices")
plt.show()


plt.hist(clustered_hits[clustered_hits.Vx.notna() & clustered_hits.near_bgo].groupby("event").fpgaTimeTag.mean()*1e-9, bins=35)
plt.xlabel("Time in s")
plt.ylabel("Reconstructed Vertices")
plt.show()


plt.hist2d(clustered_hits.groupby("event").fpgaTimeTag.mean()*1e-9, clustered_hits.groupby("event").det_key.nunique(), range=((0, 400), (0, 50)), bins=(40, 20))
plt.xlabel("Time in s")
plt.ylabel("# Hits")
plt.show()



events = clustered_hits.groupby("event").ngroups
tile_events = clustered_hits[clustered_hits.detector.isin(["tileO", "tileI"])].groupby("event").ngroups
rec_events = clustered_hits[clustered_hits.Vx.notna() & clustered_hits.near_bgo].groupby("event").ngroups

triggercondition = clustered_hits["event"].isin(
    np.intersect1d(
        clustered_hits.loc[clustered_hits.detector == "bgo", "event"].unique(),
        clustered_hits.groupby("event")
        .filter(lambda g: 
            ({"hodoO", "hodoI"} <= set(g.loc[
                g.detector.isin(["hodoO", "hodoI"]) &
                g["LE_Us"].notna() &
                g["LE_Ds"].notna(),
                "detector"
            ]))
        )["event"].unique()
    )
)


full_events = clustered_hits[triggercondition].groupby("event").ngroups

full_rec_events = clustered_hits[triggercondition & (clustered_hits.Vx.notna() & clustered_hits.near_bgo)].groupby("event").ngroups

rec_mix_events = clustered_hits[(clustered_hits.Vx.notna() & clustered_hits.near_bgo) & clustered_hits.mixGate].groupby("event").ngroups

print(f"{tile_events}/{events} events have tile hits")
print(f"{rec_events}/{events} events have a reconstructed vertex")
print(f"{full_events}/{events} events have trigger condition (only LEs)")

print(f"{full_rec_events}/{full_events} events reconstructed from full events")


clustered_hits[(clustered_hits.Vx.notna() & clustered_hits.near_bgo) & clustered_hits.mixGate].groupby("event").event.first()


event_features_df = compute_event_features_from_clustered_hits(
    clustered_hits,
    bgo_center=(0, 0, 0)
)


plt.scatter(event_features_df.mean_angle, event_features_df.dist_to_bgo, c=event_features_df.n_tracks)
plt.xlabel("Mean inter-track angle [°]")
plt.ylabel("Distance to BGO center [mm]")
plt.colorbar(label="Track count")
plt.show()

plt.scatter(event_features_df.mean_angle, event_features_df.n_tracks, c=event_features_df.n_tracks)
plt.xlabel("Mean inter-track angle [°]")
plt.ylabel("Track count")
plt.colorbar(label="Track count")
plt.show()


plt.scatter(event_features_df.time*1e-9, event_features_df.mean_angle, c=event_features_df.n_tracks)
plt.xlabel("Time [s]")
plt.ylabel("Mean inter-track angle [°]")
plt.colorbar(label="Number of tracks")
plt.show()


plt.scatter(event_features_df.max_angle, event_features_df.min_angle, c=event_features_df.time*1e-9)
plt.xlabel("Max inter-track angle [°]")
plt.ylabel("Min inter-track angle [°]")
plt.colorbar(label="Time [s]")
plt.show()


plt.scatter(event_features_df.time*1e-9, event_features_df.max_angle, c=event_features_df.n_tracks)
plt.xlabel("Time [s]")
plt.ylabel("Max inter-track angle [°]")
plt.colorbar(label="Min inter-track angle [°]")
plt.show()


plt.scatter(event_features_df.time*1e-9, event_features_df.dt_min, c=event_features_df.dt_mean)
plt.xlabel("Time [s]")
# plt.ylabel("Mean inter-track angle [°]")
plt.colorbar(label="Distance to BGO center [mm]")
plt.show()


plt.scatter(event_features_df.dt_max, event_features_df.mean_angle, c=event_features_df.time*1e-9)
plt.xlabel("Max time difference [ns]")
plt.ylabel("Mean inter-track angle [°]")
plt.colorbar(label="Time [s]")
plt.show()

plt.hist2d(event_features_df.dt_max, event_features_df.mean_angle, range=((0, 360), (0, 180)), bins=(50, 18))
plt.xlabel("Max time difference [ns]")
plt.ylabel("Mean inter-track angle [°]")
plt.colorbar()
plt.show()



plt.hist2d(event_features_df.time*1e-9, event_features_df.mean_angle, range=((0, 380), (0, 180)), bins=(95, 18))
plt.xlabel("Time [s]")
plt.ylabel("Mean inter-track angle [°]")
# plt.colorbar(label="Min inter-track angle [°]")
plt.show()

y_bins = np.logspace(np.log10(10), np.log10(8000), 40)  # from 1 ns to 8000 ns
x_bins = np.linspace(0, 380, 95)
plt.hist2d(event_features_df.time*1e-9, event_features_df.bgoToTSum, bins=(x_bins, y_bins))
plt.xlabel("Time [s]")
plt.ylabel("BGO ToT sum [ns]")
plt.yscale("log")
plt.colorbar()
# plt.colorbar(label="Min inter-track angle [°]")
plt.show()

y_bins = np.logspace(np.log10(10), np.log10(8000), 40)  # from 1 ns to 8000 ns
x_bins = np.linspace(0, 180, 18)
plt.hist2d(event_features_df.mean_angle, event_features_df.bgoToTSum, bins=(x_bins, y_bins))
plt.xlabel("Mean inter-track angle [°]")
plt.ylabel("BGO ToT sum [ns]")
plt.yscale("log")
plt.colorbar()
# plt.colorbar(label="Min inter-track angle [°]")
plt.show()

y_bins = np.logspace(np.log10(10), np.log10(8000), 40)  # from 1 ns to 8000 ns
x_bins = np.linspace(0, 180, 18)
plt.hist2d(event_features_df[event_features_df.mix].mean_angle, event_features_df[event_features_df.mix].bgoToTSum, bins=(x_bins, y_bins))
plt.xlabel("Mean inter-track angle [°]")
plt.ylabel("BGO ToT sum [ns]")
plt.yscale("log")
plt.colorbar()
plt.title("Mixing")
# plt.colorbar(label="Min inter-track angle [°]")
plt.show()

y_bins = np.logspace(np.log10(10), np.log10(8000), 40)  # from 1 ns to 8000 ns
x_bins = np.linspace(0, 180, 18)
plt.hist2d(event_features_df[~event_features_df.mix].mean_angle, event_features_df[~event_features_df.mix].bgoToTSum, bins=(x_bins, y_bins))
plt.xlabel("Mean inter-track angle [°]")
plt.ylabel("BGO ToT sum [ns]")
plt.yscale("log")
plt.colorbar()
plt.title("Not Mixing")
# plt.colorbar(label="Min inter-track angle [°]")
plt.show()


plt.hist2d(event_features_df.time*1e-9, event_features_df.dt_max, range=((0, 380), (0, 1400)), bins=(95, 14))
plt.show()

plt.hist2d(event_features_df.time*1e-9, event_features_df.dt_min, range=((0, 380), (0, 1400)), bins=(95, 14))
plt.show()

plt.hist2d(event_features_df.time*1e-9, event_features_df.dt_mean, range=((0, 380), (0, 1400)), bins=(95, 14))
plt.show()



plt.hist([event_features_df[event_features_df.mean_angle > 150].time*1e-9, event_features_df[event_features_df.mean_angle <= 150].time*1e-9], range=(0, 350), bins=35, label=["> 150°", "<= 150°"], stacked=True, color=["C0", "C2"])
plt.xlabel("Time [s]")
plt.ylabel("Number of events (stacked)")
plt.legend(title="Mean inter-track angle")
plt.show()


plt.hist(event_features_df.mean_angle, 18, range=(0, 180))
plt.show()

plt.hist2d(event_features_df.mean_angle, event_features_df.n_tracks, range=((0, 180), (0, 10)), bins=(18, 10), norm="log")
plt.show()

plt.hist(event_features_df.n_tracks, 20, range=(0, 20))
plt.show()


plt.hist(clustered_hits[clustered_hits.detector.isin(["hodoO", "hodoI"])].groupby(["event", "det_key"]).LE.diff(), 100)
plt.show()


plt.hist(clustered_hits[clustered_hits.detector.isin(["hodoO"]) & (clustered_hits.track_id > -1)].groupby(["event", "track_id"]).LE.diff(), 100, range=(0, 1000))
plt.show()

plt.hist(clustered_hits.groupby(["event", "track_id", "layer"]).LE.mean(), 100)
plt.show()




# plt.scatter(vertices_checked["sigma_radius"], vertices_checked["min_bgo_dist"])
# plt.xlabel("Vertex 1σ radius [mm]")
# plt.ylabel("Min distance to BGO [mm]")
# plt.axline((0,0), slope=1, color='r', linestyle='--', label="1σ boundary")
# plt.legend()
# plt.show()



# angs = compute_angles_and_uncertainties(clustered_hits)

# plt.hist(angs["theta"]/np.pi, 100)
# plt.xlabel("θ in pi")
# plt.show()

# plt.hist(angs["phi"]/np.pi, 31)
# plt.xlabel("φ in pi")
# plt.show()



# plt.hist(clustered_hits[clustered_hits.detector.isin(["tileO"])].channel, 120)

# plt.hist(clustered_hits[clustered_hits.detector.isin(["tileI"])].channel, 120)