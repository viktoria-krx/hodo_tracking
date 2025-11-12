import numpy as np
import pandas as pd
from itertools import combinations
from get_z_from_LEs import compute_z_positions_from_dLE
from sklearn.cluster import DBSCAN

def find_vertices_from_tracks(lines_df, eps=5.0, min_samples=2):

    results = []

    for event_id, ev_lines in lines_df.groupby("event"):
        if len(ev_lines) < 2:
            continue

        # Extract origins and directions
        origins = np.stack(ev_lines["origin"].to_numpy())
        dirs = np.stack(ev_lines["direction"].to_numpy())

        midpoints = []
        for (i, j) in combinations(range(len(ev_lines)), 2):
            C, D = origins[i], origins[j]
            e, f = dirs[i], dirs[j]

            # Compute intersection midpoints
            n = np.cross(e, f)
            n_norm = np.linalg.norm(n)
            if n_norm < 1e-6:
                continue  # nearly parallel, skip

            n1 = np.cross(e, n)
            n2 = np.cross(f, n)

            denom1 = np.dot(e, n2)
            denom2 = np.dot(f, n1)
            if np.abs(denom1) < 1e-9 or np.abs(denom2) < 1e-9:
                continue

            c1 = C + np.dot(D - C, n2) / denom1 * e
            c2 = D + np.dot(C - D, n1) / denom2 * f

            midpoints.append((c1 + c2) / 2)

        if not midpoints:
            continue

        midpoints = np.array(midpoints)

        # cluster intersection points
        db = DBSCAN(eps=eps, min_samples=min_samples)
        labels = db.fit_predict(midpoints)

        # choose the densest cluster (if any)
        if len(midpoints < min_samples):
            vertex = np.mean(midpoints, axis=0)
            sigma = np.std(midpoints, axis=0)

        else:
            valid = labels >= 0
            if not np.any(valid):
                continue

            biggest = np.bincount(labels[valid]).argmax()
            # biggest = np.bincount(labels).argmax()
            main_cluster = midpoints[labels == biggest]

            vertex = np.mean(main_cluster, axis=0)
            sigma = np.std(main_cluster, axis=0)

        results.append({
            "event": event_id,
            "n_tracks": len(ev_lines),
            "n_pairs": len(midpoints),
            "Vx": vertex[0],
            "Vy": vertex[1],
            "Vz": vertex[2],
            "Vx_sig": sigma[0],
            "Vy_sig": sigma[1],
            "Vz_sig": sigma[2],
        })

    return pd.DataFrame(results)


def reconstruct_vertex_from_midpoints(clustered_hits, lines_df, 
                                      bgo_radius=45.0, 
                                      max_dist_to_bgo=25.0,
                                      min_midpoints=1):
    """
    Compute event-by-event vertex from line intersections near the central BGO.
    
    Parameters
    ----------
    clustered_hits : DataFrame
        With columns [event, detector, x, y, z]
    lines_df : DataFrame
        With columns [event, track_id, origin, direction]
    bgo_radius : float
        Spatial extent (mm) of the central BGO region (around origin)
    max_dist_to_bgo : float
        Maximum distance [mm] from midpoint to any BGO hit to consider it "on BGO"
    min_midpoints : int
        Minimum number of valid midpoints to define a vertex
    
    Returns
    -------
    vertex_df : DataFrame with columns
        [event, Vx, Vy, Vz, n_midpoints, near_bgo, flagged_tracks]
    """
    results = []

    for ev, ev_lines in lines_df.groupby("event"):
        if len(ev_lines) < 2:
            continue

        # BGO hits near the center
        bgo_hits = clustered_hits[
            (clustered_hits.event == ev) &
            (clustered_hits.detector == "bgo") &
            (np.hypot(clustered_hits.x, clustered_hits.y) < bgo_radius)
        ]
        if bgo_hits.empty:
            continue
        bgo_pos = bgo_hits[["x", "y", "z"]].to_numpy()

        midpoints = []
        for (iA, rowA), (iB, rowB) in combinations(ev_lines.iterrows(), 2):
            C = np.array(rowA.origin)
            D = np.array(rowB.origin)
            e = np.array(rowA.direction)
            f = np.array(rowB.direction)

            # intersection midpoint between two skew lines
            n = np.cross(e, f)
            if np.linalg.norm(n) < 1e-6:
                continue  # nearly parallel
            n1 = np.cross(e, n)
            n2 = np.cross(f, n)
            denom1 = np.dot(e, n2)
            denom2 = np.dot(f, n1)
            if np.abs(denom1) < 1e-6 or np.abs(denom2) < 1e-6:
                continue
            c1 = C + np.dot(D - C, n2) / denom1 * e
            c2 = D + np.dot(C - D, n1) / denom2 * f
            M = (c1 + c2) / 2.0
            midpoints.append(M)

        if not midpoints:
            continue
        midpoints = np.vstack(midpoints)

        # Compute distance from each midpoint to all BGO hits
        dists = np.linalg.norm(midpoints[:, None, :] - bgo_pos[None, :, :], axis=2)
        min_dist = np.min(dists, axis=1)
        near_bgo_mask = min_dist < max_dist_to_bgo

        # Midpoints near BGO
        near_midpoints = midpoints[near_bgo_mask]
        n_mid = len(near_midpoints)

        if n_mid >= min_midpoints:
            V = np.mean(near_midpoints, axis=0)
            Vsig = np.std(near_midpoints, axis=0)
            near_bgo = True
        else:
            V = np.array([np.nan, np.nan, np.nan])
            Vsig = np.array([np.nan, np.nan, np.nan])
            near_bgo = False

        # Optional: flag tracks that don't go near the vertex
        flagged_tracks = []
        for _, line in ev_lines.iterrows():
            origin = np.array(line.origin)
            direction = np.array(line.direction)
            t = np.dot(V - origin, direction)
            closest_point = origin + t * direction
            dist_to_vertex = np.linalg.norm(closest_point - V)
            if dist_to_vertex > max_dist_to_bgo:
                flagged_tracks.append(line.track_id)

        results.append({
            "event": ev,
            "Vx": V[0], "Vy": V[1], "Vz": V[2],
            "Vx_sig": Vsig[0], "Vy_sig": Vsig[1], "Vz_sig": Vsig[2],
            "n_midpoints": n_mid,
            "near_bgo": near_bgo,
            "flagged_tracks": flagged_tracks
        })

    return pd.DataFrame(results)



def check_vertex_bgo_proximity(vertices_df, clustered_hits, sigma_level=2.0):
    """
    Check if reconstructed vertex lies within N-sigma of any BGO hit.

    Parameters
    ----------
    vertices_df : pd.DataFrame
        Must contain columns [event, Vx, Vy, Vz, Vx_sig, Vy_sig, Vz_sig]
    clustered_hits : pd.DataFrame
        Must contain columns [event, x, y, z, detector]
    sigma_level : float
        How many sigmas to consider as "close" (e.g. 1.0 or 2.0)

    Returns
    -------
    result_df : pd.DataFrame
        vertices_df with an added column `vertex_close_to_bgo` (bool)
        and min distance to BGO (`min_bgo_dist`).
    """
    results = []
    for _, v in vertices_df.iterrows():
        ev_id = v["event"]
        vx, vy, vz = v["Vx"], v["Vy"], v["Vz"]

        # uncertainty radius (combined sigma)
        sigma_r = np.sqrt(v["Vx_sig"]**2 + v["Vy_sig"]**2 + v["Vz_sig"]**2)

        # get BGO hits from same event
        bgo_hits = clustered_hits[
            (clustered_hits["event"] == ev_id)
            & (clustered_hits["detector"] == "bgo")
        ]

        if bgo_hits.empty:
            min_dist = np.nan
            close = False
        else:
            bx = bgo_hits["x"].to_numpy()
            by = bgo_hits["y"].to_numpy()
            bz = bgo_hits["z"].to_numpy()
            dists = np.sqrt((bx - vx)**2 + (by - vy)**2 + (bz - vz)**2)
            min_dist = np.min(dists)
            close = min_dist <= sigma_level * sigma_r

        results.append({
            "event": ev_id,
            "min_bgo_dist": min_dist,
            "sigma_radius": sigma_r,
            "within_sigma": close
        })

    return vertices_df.merge(pd.DataFrame(results), on="event", how="left")

