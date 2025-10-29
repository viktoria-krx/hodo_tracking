from read_file import *
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

root_path = "~/Documents/Hodoscope/cern_data/2025_Data/output_000990.root"

root_path = "~/Documents/Hodoscope/cern_data/2025_Data/output_000636.root"

hits_df = build_hits_df_fast(root_path)
hits_df["z_used"] = np.where(np.isnan(hits_df["z_reco"]), hits_df["z"], hits_df["z_reco"])

# def cartesian_to_spherical(df):
#     """
#     Convert (x, y, z) coordinates to spherical (r, θ, φ),
#     using z_reco if available (and not NaN).
#     Also computes angular uncertainties σθ and σφ.
#     """
#     # Use z_reco if present, otherwise fallback to z
#     if "z_reco" in df.columns:
#         z = np.where(~np.isnan(df["z_reco"]), df["z_reco"], df["z"])
#     else:
#         z = df["z"].to_numpy()

#     x = df["x"].to_numpy()
#     y = df["y"].to_numpy()

#     # Position uncertainties
#     dx = df.get("dx", pd.Series(np.zeros(len(df)))).to_numpy()
#     dy = df.get("dy", pd.Series(np.zeros(len(df)))).to_numpy()
#     dz = df.get("dz", pd.Series(np.zeros(len(df)))).to_numpy()

#     # Compute spherical coordinates
#     r = np.sqrt(x**2 + y**2 + z**2)
#     with np.errstate(divide="ignore", invalid="ignore"):
#         theta = np.arccos(np.clip(z / r, -1.0, 1.0))  # polar angle
#         phi = np.arctan2(y, x)                        # azimuthal angle

#     # Uncertainty propagation (approximate)
#     # Avoid division by zero
#     r_safe = np.where(r == 0, np.nan, r)
#     sin_theta = np.sin(theta)
#     sin_theta_safe = np.where(sin_theta == 0, np.nan, sin_theta)

#     # Approximate angular uncertainties
#     sigma_theta = np.sqrt(dx**2 + dy**2) / r_safe
#     sigma_phi = np.sqrt(dx**2 + dy**2) / (r_safe * sin_theta_safe)

#     # Replace any NaNs due to geometry with zeros
#     sigma_theta = np.nan_to_num(sigma_theta)
#     sigma_phi = np.nan_to_num(sigma_phi)

#     # Return DataFrame
#     return pd.DataFrame({
#         "r": r,
#         "theta": theta,
#         "phi": phi,
#         "dtheta": sigma_theta,
#         "dphi": sigma_phi
#     })


# def cluster_by_angles(event_hits, eps_theta=0.01, eps_phi=0.01, min_samples=2):
#     angles = event_hits[["theta", "phi"]].to_numpy()
    
#     # Normalize angular distances (you can tune eps to ~1 degree = 0.017 rad)
#     db = DBSCAN(eps=np.mean([eps_theta, eps_phi]), min_samples=min_samples, metric='euclidean')
#     labels = db.fit_predict(angles)
#     event_hits["track_id"] = labels
#     return event_hits


# sph = cartesian_to_spherical(hits_df)
# hits_df = pd.concat([hits_df, sph], axis=1)


# clustered_hits = hits_df.groupby("event", group_keys=False).apply(cluster_by_angles)



# def fit_lines_ransac(points, min_points=3, residual_threshold=5.0):
#     """Fit multiple lines to 3D points using iterative RANSAC."""
#     lines = []
#     remaining = points.copy()

#     while len(remaining) >= min_points:
#         # Fit z as function of x and y (or vice versa)
#         X = remaining[:, :2]
#         y = remaining[:, 2]

#         ransac = RANSACRegressor(residual_threshold=residual_threshold)
#         ransac.fit(X, y)
#         inlier_mask = ransac.inlier_mask_

#         inliers = remaining[inlier_mask]
#         outliers = remaining[~inlier_mask]

#         if len(inliers) < min_points:
#             break

#         # Fit direction vector (PCA)
#         direction = np.linalg.svd(inliers - inliers.mean(axis=0))[2][0]
#         origin = inliers.mean(axis=0)
#         lines.append((origin, direction, len(inliers)))

#         remaining = outliers

#     return lines

# def fit_lines_ransac(points, min_points=3, residual_threshold=5.0):
#     """
#     Fit multiple 3D lines using RANSAC, falling back to direct line fitting for small clusters.
#     """
#     lines = []
#     remaining = points.copy()

#     while len(remaining) >= 2:  # run until fewer than 2 points remain
#         if len(remaining) == 2:
#             # Directly define a line through the two points
#             origin = remaining.mean(axis=0)
#             direction = remaining[1] - remaining[0]
#             direction = direction / np.linalg.norm(direction)
#             lines.append((origin, direction, 2))
#             break  # nothing left to fit

#         # Fit z as a function of x,y using RANSAC
#         X = remaining[:, :2]
#         y = remaining[:, 2]

#         ransac = RANSACRegressor(residual_threshold=residual_threshold, min_samples=min(3, len(remaining)))

#         try:
#             ransac.fit(X, y)
#         except ValueError:
#             # Fallback: points are degenerate, use PCA
#             mean = remaining.mean(axis=0)
#             direction = np.linalg.svd(remaining - mean)[2][0]
#             lines.append((mean, direction, len(remaining)))
#             break

#         inlier_mask = getattr(ransac, "inlier_mask_", np.ones(len(remaining), dtype=bool))
#         inliers = remaining[inlier_mask]
#         outliers = remaining[~inlier_mask]

#         if len(inliers) < 2:
#             break

#         # Refine the line using PCA on inliers
#         origin = inliers.mean(axis=0)
#         direction = np.linalg.svd(inliers - origin)[2][0]
#         lines.append((origin, direction, len(inliers)))

#         remaining = outliers

#     return lines


# results = []
# for event_id, event_hits in hits_df[hits_df.event < 20].groupby("event"):
#     points = event_hits[["x", "y", "z"]].dropna().to_numpy()
#     if len(points) < 2:
#         continue
#     lines = fit_lines_ransac(points, 2)
#     results.append({
#         "event": event_id,
#         "n_hits": len(points),
#         "n_lines": len(lines),
#         "lines": lines,
#     })

# lines_df = pd.DataFrame(results)

# for ev in lines_df[lines_df.n_lines >0].event[:20]:
#     fig, ax = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
#     ax[0].scatter(hits_df[hits_df.event == ev].x, hits_df[hits_df.event == ev].y)
#     for line in lines_df[lines_df.event == ev].lines.values[0]:
#         ax[0].quiver(line[0][0], line[0][1], line[1][0], line[1][1], units='xy', scale=0.001)
#     ax[0].set_aspect('equal')
#     ax[0].set_xlim(-150, 150)
#     ax[0].set_ylim(-150, 150)
#     ax[0].set_xlabel("x in mm")
#     ax[0].set_ylabel("y in mm")
#     ax[1].scatter(hits_df[hits_df.event == ev].z, hits_df[hits_df.event == ev].y)
#     for line in lines_df[lines_df.event == ev].lines.values[0]:
#         ax[1].quiver(line[0][2], line[0][1], line[1][2], line[1][1], units='xy', scale=0.001)
#     ax[1].set_aspect("equal")
#     ax[1].set_xlim(-250, 250)
#     ax[1].set_ylim(-150, 150)
#     ax[1].set_xlabel("z in mm")
#     ax[1].set_ylabel("y in mm")




# clustered_hits["z_used"] = np.where(np.isnan(clustered_hits["z_reco"]), clustered_hits["z"], clustered_hits["z_reco"])

# def fit_line(points):
#     """
#     Fit a line (direction + origin) to 3D points via PCA.
#     """
#     centroid = points.mean(axis=0)
#     # The first principal component gives the main direction
#     direction = np.linalg.svd(points - centroid)[2][0]
#     return centroid, direction


# def analyze_event_tracks(clustered_hits, min_points=3):
#     """
#     Fit lines to each track cluster within an event.
#     """
#     lines = []
#     for track_id, track_hits in clustered_hits.groupby("track_id"):
#         if track_id == -1 or len(track_hits) < min_points:
#             continue  # skip noise or tiny clusters
#         pts = track_hits[["x", "y", "z_used"]].dropna().to_numpy()
#         if len(pts) < 2:
#             continue
#         origin, direction = fit_line(pts)
#         lines.append((origin, direction, len(pts)))
#     return lines


# # === Example processing for first few events ===
# results = []
# for event_id, event_hits in clustered_hits.groupby("event"):
#     lines = analyze_event_tracks(event_hits)
#     results.append({
#         "event": event_id,
#         "n_hits": len(event_hits),
#         "n_lines": len(lines),
#         "lines": lines
#     })

# lines_df = pd.DataFrame(results)





# def cluster_event_hits(event_hits, eps_theta=0.01, eps_phi=0.01, min_samples=2):
#     """Cluster hodo/tile hits by direction in spherical angles (no BGO)."""
#     hodo_tile = event_hits[event_hits.detector.isin(["hodoO", "hodoI", "tileO", "tileI"])].copy()
    
#     if len(hodo_tile) < min_samples:
#         event_hits["track_id"] = -1
#         return event_hits

#     # Convert to spherical coordinates (you already have z_reco vs z logic)
#     x = hodo_tile["x"].to_numpy()
#     y = hodo_tile["y"].to_numpy()
#     z = np.where(np.isnan(hodo_tile["z_reco"]), hodo_tile["z"], hodo_tile["z_reco"])

#     r = np.sqrt(x**2 + y**2 + z**2)
#     theta = np.arccos(z / r)
#     phi = np.arctan2(y, x)

#     hodo_tile["theta"] = theta
#     hodo_tile["phi"] = phi

#     # Cluster by direction
#     db = DBSCAN(eps=np.mean([eps_theta, eps_phi]), min_samples=min_samples)
#     hodo_tile["track_id"] = db.fit_predict(hodo_tile[["theta", "phi"]])

#     # Merge back into original hits (unclustered detectors keep -1)
#     event_hits = event_hits.merge(
#         hodo_tile[["event", "channel", "track_id"]],
#         on=["event", "channel"],
#         how="left"
#     )
#     event_hits["track_id"] = event_hits["track_id"].fillna(-1).astype(int)
#     return event_hits

def fit_tracks_with_bgo(event_hits, residual_threshold=5.0):
    tracks = []
    bgo_hits = event_hits[event_hits.detector == "bgo"][["x", "y", "z"]].to_numpy()

    for track_id, cluster in event_hits[event_hits.track_id >= 0].groupby("track_id"):
        track_points = cluster[["x", "y", "z"]].to_numpy()
        if len(track_points) < 2:
            continue

        # Combine with BGO (anchor points)
        all_points = np.vstack([track_points, bgo_hits]) if len(bgo_hits) > 0 else track_points

        # Fit direction using PCA (fast, robust)
        origin = np.mean(all_points, axis=0)
        direction = np.linalg.svd(all_points - origin)[2][0]
        tracks.append({
            "track_id": track_id,
            "origin": origin,
            "direction": direction,
            "n_hits": len(track_points)
        })
    return tracks

# clustered_hits = []
# tracks = []

# for event_id, ev_hits in hits_df.groupby("event"):
#     clustered = cluster_event_hits(ev_hits)
#     clustered_hits.append(clustered)

#     track_info = fit_tracks_with_bgo(clustered)
#     for t in track_info:
#         t["event"] = event_id
#     tracks.extend(track_info)

# clustered_hits = pd.concat(clustered_hits, ignore_index=True)
# tracks_df = pd.DataFrame(tracks)



from matplotlib.colors import ListedColormap
def plot_event_with_tracks(event_id, hits_df, tracks_df):
    """Plot event hits and fitted tracks in XY and ZY projections."""
    event_hits = hits_df[hits_df.event == event_id]
    event_tracks = tracks_df[tracks_df.event == event_id]
    
    if event_hits.empty:
        print(f"No hits found for event {event_id}")
        return

    # --- setup ---
    fig, ax = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    cmap = plt.cm.get_cmap("viridis", len(event_hits.track_id.unique()))
    
    # --- left plot: XY plane ---
    for det, style in zip(
        ["hodoO", "hodoI", "tileO", "tileI", "bgo"],
        ["o", "s", "^", "v", "x"]
    ):
        subset = event_hits[event_hits.detector == det]
        if subset.empty:
            continue
        ax[0].scatter(
            subset.x,
            subset.y,
            c=[cmap(i) for i in subset.track_id],
            label=det,
            marker=style,
            s=20,
            alpha=0.8
        )

    # --- right plot: ZY plane ---
    for det, style in zip(
        ["hodoO", "hodoI", "tileO", "tileI", "bgo"],
        ["o", "s", "^", "v", "x"]
    ):
        subset = event_hits[event_hits.detector == det]
        if subset.empty:
            continue
        ax[1].scatter(
            subset.z,
            subset.y,
            c=[cmap(i) for i in subset.track_id],
            label=det,
            marker=style,
            s=20,
            alpha=0.8
        )

    # --- overlay tracks (lines) ---
    for _, row in event_tracks.iterrows():
        p0 = row.origin
        v = row.direction / np.linalg.norm(row.direction)
        t = np.linspace(-300, 300, 50)
        line_pts = p0[None, :] + t[:, None] * v[None, :]

        # XY projection
        ax[0].plot(line_pts[:, 0], line_pts[:, 1], "--", color="k", lw=1.2)
        # ZY projection
        ax[1].plot(line_pts[:, 2], line_pts[:, 1], "--", color="k", lw=1.2)

    # --- formatting ---
    for a in ax:
        a.set_aspect("equal")
        a.set_ylim(-200, 200)
        a.legend(fontsize=8)
    
    ax[0].set_xlim(-200, 200)
    ax[0].set_xlabel("x [mm]")
    ax[0].set_ylabel("y [mm]")
    ax[0].set_title(f"Event {event_id} — XY plane")

    ax[1].set_xlim(-250, 250)
    ax[1].set_xlabel("z [mm]")
    ax[1].set_title("ZY plane (side view)")

    plt.tight_layout()
    plt.show()

# for ev in range(10, 20):
#     plot_event_with_tracks(ev, clustered_hits, tracks_df)





# helper: wrap phi difference into [-pi, pi]
def delta_phi(phi1, phi2):
    d = phi1 - phi2
    d = (d + np.pi) % (2 * np.pi) - np.pi
    return d

def delta_phi_bidirectional(phi1, phi2):
    dphi = np.mod(phi1 - phi2 + np.pi, 2*np.pi) - np.pi
    dphi_flipped = np.mod(phi1 - phi2 + np.pi/2, 2*np.pi) - np.pi
    return np.minimum(np.abs(dphi), np.abs(dphi - np.pi))


def compute_angles_and_uncertainties(df, coords="cylindrical"):
    """
    Given a DataFrame with x,y,z, z_reco (optional), dx,dy,dz,
    compute r, theta, phi, sigma_theta, sigma_phi arrays.
    """

    x = df["x"].to_numpy()
    y = df["y"].to_numpy()
    z = df["z_used"].to_numpy()

    dx = df.get("dx", pd.Series(0, index=df.index)).to_numpy()
    dy = df.get("dy", pd.Series(0, index=df.index)).to_numpy()
    dz = df.get("dz", pd.Series(0, index=df.index)).to_numpy()

    if coords == "spherical":
        r = np.sqrt(x**2 + y**2 + z**2)
        # avoid division by zero
        with np.errstate(divide="ignore", invalid="ignore"):
            theta = np.arccos(np.clip(z / r, -1.0, 1.0))
        phi = np.arctan2(y, x)

        # First-order propagation (small-angle approx)
        r_safe = np.where(r == 0, np.nan, r)
        sin_theta = np.sin(theta)
        sin_theta_safe = np.where(sin_theta == 0, 1e-12, sin_theta)  # avoid blow-ups

        # radial transverse uncertainty ~ sqrt(dx^2 + dy^2)
        r_trans = np.sqrt(dx**2 + dy**2)

        sigma_theta = r_trans / r_safe            # ≈ sqrt(dx^2+dy^2)/r
        sigma_phi = r_trans / (r_safe * sin_theta_safe)  # ≈ sqrt(dx^2+dy^2)/(r sinθ)

        # for cases where sinθ is extremely small, sigma_phi may be huge; clip if you want
        sigma_phi = np.nan_to_num(sigma_phi, nan=np.inf, posinf=np.inf)

        # replace NaNs with big uncertainties so they don't form clusters erroneously
        sigma_theta = np.nan_to_num(sigma_theta, nan=np.inf, posinf=np.inf)
        sigma_phi = np.nan_to_num(sigma_phi, nan=np.inf, posinf=np.inf)

        return {
            "r": r, "theta": theta, "phi": phi,
            "sigma_theta": sigma_theta, "sigma_phi": sigma_phi
        }

    if coords == "cylindrical":
        r = np.sqrt(x**2 + y**2)
        phi = np.arctan2(y, x)

        # transverse uncertainty
        r_trans = np.sqrt(dx**2 + dy**2)

        # sigma_phi: avoid divide-by-zero by flooring r
        r_safe = np.where(r == 0, np.nan, r)
        sigma_phi = r_trans / r_safe
        # replace NaN/inf with a large uncertainty (so these hits are loose)
        sigma_phi = np.nan_to_num(sigma_phi, nan=np.inf, posinf=np.inf)

        return {
            "r": r,
            "phi": phi,
            "sigma_phi": sigma_phi,
        }





def cluster_by_angles_uncertainty(event_hits, base_eps=2.0, min_samples=2, coords="cylindrical"):
    """
    Uncertainty-aware clustering in (theta,phi) for a single event.

    Parameters
    ----------
    event_hits : pd.DataFrame
        Must contain columns x,y,z (and optionally z_reco), dx,dy,dz.
    base_eps : float
        Clustering threshold in units of combined sigma (e.g. 2 -> 2σ).
    min_samples : int
        DBSCAN min samples.

    Returns
    -------
    labels : np.ndarray
        track_id labels in same order as event_hits.index (DBSCAN labels, -1 = noise).
    """
    n = len(event_hits)
    if n == 0:
        return np.array([], dtype=int)

    if coords == "spherical":
        # compute angles + uncertainties
        ang = compute_angles_and_uncertainties(event_hits, coords="spherical")
        theta = ang["theta"]
        phi = ang["phi"]
        s_theta = ang["sigma_theta"]
        s_phi = ang["sigma_phi"]

        # if very few points fallback to trivial labeling
        if n < min_samples:
            return np.full(n, -1, dtype=int)

        # build pairwise distance matrix (weighted by uncertainties)
        # d_ij = sqrt( (dtheta^2)/(s_th_i^2 + s_th_j^2) + (dphi^2)/(s_ph_i^2 + s_ph_j^2) )
        # handle phi wrap-around via delta_phi
        dists = np.zeros((n, n), dtype=float)
        for i in range(n):
            dtheta2 = (theta[i] - theta)**2
            dphi_raw = delta_phi(phi[i], phi)  # shape (n,)
            dphi2 = dphi_raw**2

            denom_theta = s_theta[i]**2 + s_theta**2
            denom_phi = s_phi[i]**2 + s_phi**2

            # avoid division by zero: if denom == 0 set huge value (no info)
            denom_theta = np.where(denom_theta == 0, np.inf, denom_theta)
            denom_phi = np.where(denom_phi == 0, np.inf, denom_phi)

            dists[i, :] = np.sqrt(dtheta2 / denom_theta + dphi2 / denom_phi)

    elif coords == "cylindrical":
        # compute angles + uncertainties
        ang = compute_angles_and_uncertainties(event_hits)
        phi = ang["phi"]
        s_phi = ang["sigma_phi"]

        # Optionally include z for separation along barrel
        z = event_hits["z_used"].to_numpy() if "z_used" in event_hits.columns else event_hits["z"].to_numpy()
        s_z = event_hits.get("dz", np.zeros_like(z)).to_numpy()

        if n < min_samples:
            return np.full(n, -1, dtype=int)

        dists = np.zeros((n, n), dtype=float)
        for i in range(n):
            # φ difference (circular)
            dphi_raw = delta_phi(phi[i], phi)
            dphi2 = dphi_raw**2
            # z difference (linear)
            dz2 = (z[i] - z)**2

            denom_phi = s_phi[i]**2 + s_phi**2
            denom_z = s_z[i]**2 + s_z**2

            denom_phi = np.where(denom_phi == 0, np.inf, denom_phi)
            denom_z = np.where(denom_z == 0, np.inf, denom_z)

            # weighted metric: φ dominates; scale z contribution if desired
            dists[i, :] = np.sqrt(dphi2 / denom_phi + dz2 / denom_z)

    else:
        raise ValueError(f"Unknown coords mode: {coords}")


    # symmetric matrix; use DBSCAN with precomputed metric
    db = DBSCAN(eps=base_eps, min_samples=min_samples, metric="precomputed")
    labels = db.fit_predict(dists)

    return labels

def cluster_by_phi_uncertainty(event_hits, base_eps=1.0, min_samples=2, theta_weight=0.1, sigma_floor_deg=0.5, coords="cylindrical"):
    """
    Cluster hits mainly by φ (azimuth) with uncertainty weighting.
    θ differences are included with reduced weight.
    """
    if len(event_hits) == 0:
        return np.array([], dtype=int)
    
    n = len(event_hits)
    dists = np.zeros((n, n))

    if coords == "spherical":
        # compute angles and uncertainties
        ang = compute_angles_and_uncertainties(event_hits, coords="spherical")
        theta = ang["theta"]
        phi = ang["phi"]
        s_theta = ang["sigma_theta"]
        s_phi = ang["sigma_phi"]

        # unwrap phi to avoid 2π discontinuity
        phi = np.unwrap(phi)

        # apply sigma floor to avoid zero denominators
        sigma_floor = np.deg2rad(sigma_floor_deg)
        s_phi = np.maximum(s_phi, sigma_floor)
        s_theta = np.maximum(s_theta, sigma_floor)

    
        for i in range(n):
            dphi_raw = phi[i] - phi
            dtheta = theta[i] - theta

            denom_phi = s_phi[i]**2 + s_phi**2
            denom_theta = s_theta[i]**2 + s_theta**2

            dists[i, :] = np.sqrt(
                (dphi_raw**2 / denom_phi) +
                theta_weight * (dtheta**2 / denom_theta)
            )

    elif coords == "cylindrical":
        # compute angles + uncertainties
        ang = compute_angles_and_uncertainties(event_hits)
        phi = ang["phi"]
        s_phi = ang["sigma_phi"]

        # Optionally include z for separation along barrel
        z = event_hits["z_used"].to_numpy() if "z_used" in event_hits.columns else event_hits["z"].to_numpy()
        s_z = event_hits.get("dz", np.zeros_like(z)).to_numpy()

        dists = np.zeros((n, n), dtype=float)
        for i in range(n):
            # φ difference (circular)
            dphi_raw = delta_phi(phi[i], phi)
            # dphi_raw = delta_phi_bidirectional(phi[i], phi)
            dphi2 = dphi_raw**2
            # z difference (linear)
            dz2 = (z[i] - z)**2

            denom_phi = s_phi[i]**2 + s_phi**2
            denom_z = s_z[i]**2 + s_z**2

            denom_phi = np.where(denom_phi == 0, np.inf, denom_phi)
            denom_z = np.where(denom_z == 0, np.inf, denom_z)

            # weighted metric: φ dominates; scale z contribution if desired
            dists[i, :] = np.sqrt(dphi2 / denom_phi + theta_weight * dz2 / denom_z)

    else:
        raise ValueError(f"Unknown coords mode: {coords}")

    # DBSCAN on precomputed angular distance
    db = DBSCAN(eps=base_eps, min_samples=min_samples, metric="precomputed")
    labels = db.fit_predict(dists)
    return labels

def cluster_by_phi_hdbscan(event_hits, min_cluster_size=2, min_samples=1, theta_weight=0.1, sigma_floor_deg=0.5, coords="cylindrical"):
    if len(event_hits) == 0:
        return np.array([], dtype=int)

    n = len(event_hits)
    dists = np.zeros((n, n))

    if coords == "spherical":
        # compute angles and uncertainties
        ang = compute_angles_and_uncertainties(event_hits, coords="spherical")
        theta = ang["theta"]
        phi = ang["phi"]
        s_theta = ang["sigma_theta"]
        s_phi = ang["sigma_phi"]

        # unwrap phi to avoid 2π discontinuity
        phi = np.unwrap(phi)

        # apply sigma floor to avoid zero denominators
        sigma_floor = np.deg2rad(sigma_floor_deg)
        s_phi = np.maximum(s_phi, sigma_floor)
        s_theta = np.maximum(s_theta, sigma_floor)

    
        for i in range(n):
            dphi_raw = phi[i] - phi
            dtheta = theta[i] - theta

            denom_phi = s_phi[i]**2 + s_phi**2
            denom_theta = s_theta[i]**2 + s_theta**2

            dists[i, :] = np.sqrt(
                (dphi_raw**2 / denom_phi) +
                theta_weight * (dtheta**2 / denom_theta)
            )

    elif coords == "cylindrical":
        # compute angles + uncertainties
        ang = compute_angles_and_uncertainties(event_hits)
        phi = ang["phi"]
        s_phi = ang["sigma_phi"]

        # Optionally include z for separation along barrel
        z = event_hits["z_used"].to_numpy() if "z_used" in event_hits.columns else event_hits["z"].to_numpy()
        s_z = event_hits.get("dz", np.zeros_like(z)).to_numpy()

        dists = np.zeros((n, n), dtype=float)
        for i in range(n):
            # φ difference (circular)
            dphi_raw = delta_phi(phi[i], phi)
            dphi2 = dphi_raw**2
            # z difference (linear)
            dz2 = (z[i] - z)**2

            denom_phi = s_phi[i]**2 + s_phi**2
            denom_z = s_z[i]**2 + s_z**2

            denom_phi = np.where(denom_phi == 0, np.inf, denom_phi)
            denom_z = np.where(denom_z == 0, np.inf, denom_z)

            # weighted metric: φ dominates; scale z contribution if desired
            dists[i, :] = np.sqrt(dphi2 / denom_phi + theta_weight * dz2 / denom_z)

    else:
        raise ValueError(f"Unknown coords mode: {coords}")

    clusterer = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, metric="precomputed")
    labels = clusterer.fit_predict(dists)
    return labels

# def fit_lines_from_clusters(clustered_hits, residual_threshold=5.0, min_points=2):
#     """
#     Fit straight lines to each (event_id, track_id) cluster.
#     Uses your previous RANSAC-based line fitting.
#     """
#     results = []

#     for (event_id, track_id), group in clustered_hits.groupby(["event", "track_id"]):
#         # skip noise and too few hits
#         if track_id == -1 or len(group) < min_points:
#             continue

#         points = np.column_stack((group["x"], group["y"], group["z_used"]))
#         lines = fit_lines_ransac(points, min_points=min_points, residual_threshold=residual_threshold)

#         for origin, direction, n_inliers in lines:
#             results.append({
#                 "event": event_id,
#                 "track_id": track_id,
#                 "n_hits": len(points),
#                 "n_inliers": n_inliers,
#                 "origin": origin,
#                 "direction": direction,
#             })

#     return pd.DataFrame(results)


# with cProfile.Profile() as pr:

# stats = pstats.Stats(pr)
# stats.sort_stats(pstats.SortKey.CUMULATIVE).print_stats(20)


# toc = time.perf_counter()
# print(f"{toc - tic:0.4f} seconds")



# ---------- Basic SVD (PCA) line fit ----------
def fit_line_svd(points):
    """
    Fit a 3D line to points (Nx3) using SVD.
    Returns (origin, direction) where origin is centroid and direction is unit vector.
    """
    pts = np.asarray(points)
    if pts.shape[0] == 0:
        return None, None
    centroid = np.mean(pts, axis=0)
    if pts.shape[0] == 1:
        # direction undefined; choose arbitrary
        return centroid, np.array([0.0, 0.0, 1.0])
    # SVD on centered data
    U, S, Vt = np.linalg.svd(pts - centroid, full_matrices=False)
    direction = Vt[0]
    # ensure unit norm
    norm = np.linalg.norm(direction)
    if norm == 0 or not np.isfinite(norm):
        direction = np.array([0.0, 0.0, 1.0])
    else:
        direction = direction / norm
    return centroid, direction


# ---------- Weighted SVD (weighted PCA) ----------
def fit_line_weighted_svd(points, weights):
    """
    Weighted PCA line fit. points: Nx3, weights: N (positive).
    Returns (origin, direction).
    Uses weighted centroid and weighted covariance.
    """
    pts = np.asarray(points)
    w = np.asarray(weights).astype(float)
    if pts.shape[0] == 0:
        return None, None
    wsum = np.sum(w)
    if wsum == 0:
        return fit_line_svd(pts)
    centroid = np.sum(pts * w[:, None], axis=0) / wsum
    X = pts - centroid
    # Weighted covariance: C = (X^T * diag(w) * X) / sum(w)
    # We'll compute weighted scatter matrix
    WX = X * np.sqrt(w)[:, None]
    # SVD on WX
    U, S, Vt = np.linalg.svd(WX, full_matrices=False)
    direction = Vt[0]
    direction = direction / np.linalg.norm(direction)
    return centroid, direction


# ---------- RANSAC prefilter (on z vs x,y) then SVD ----------
def fit_line_ransac_then_svd(points, residual_threshold=10.0, min_samples=3, max_trials=100):
    """
    Use RANSAC to remove gross outliers by fitting z = f(x,y) with RANSAC,
    then perform SVD on the inliers to obtain a robust 3D line.
    Points shape Nx3.
    Returns (origin, direction, n_inliers)
    """
    pts = np.asarray(points)
    if pts.shape[0] == 0:
        return None, None, 0
    if pts.shape[0] <= 2:
        origin, direction = fit_line_svd(pts)
        return origin, direction, pts.shape[0]
    # Fit z = a*x + b*y + c with RANSAC to remove big outliers
    X = pts[:, :2]
    y = pts[:, 2]
    ransac = RANSACRegressor(residual_threshold=residual_threshold,
                             min_samples=min_samples, max_trials=max_trials)
    try:
        ransac.fit(X, y)
        inlier_mask = ransac.inlier_mask_
    except Exception:
        # fallback: use all points as inliers
        inlier_mask = np.ones(len(pts), dtype=bool)
    inliers = pts[inlier_mask]
    if inliers.shape[0] == 0:
        # nothing survived; return PCA on all
        origin, direction = fit_line_svd(pts)
        return origin, direction, 0
    origin, direction = fit_line_svd(inliers)
    return origin, direction, inliers.shape[0]


# # ---------- Utility: perpendicular residuals (distance point->line) ----------
# def point_line_distance(points, origin, direction):
#     """
#     Compute perpendicular distances from points (Nx3) to the line (origin, direction).
#     direction should be unit vector.
#     Returns distances (N,)
#     """
#     pts = np.asarray(points)
#     v = pts - origin
#     # projection length along direction
#     proj = np.dot(v, direction)
#     perp = v - np.outer(proj, direction)
#     dist = np.linalg.norm(perp, axis=1)
#     return dist


# ---------- High-level: fit per cluster, with options ----------
def fit_lines_from_clusters_svd(clustered_hits, include_bgo=True,
                                prefilter_ransac=False, ransac_thresh=10.0,
                                weighted=False, weight_col=None, min_points=2):
    """
    For each (event, track_id) cluster, fit a 3D line using SVD/PCA.
    Options:
      - include_bgo: if True, include BGO hits from same event in the fit (anchor)
      - prefilter_ransac: run RANSAC to remove gross outliers before SVD
      - weighted: if True, compute weighted PCA using weight_col in the group
    Returns list of fitted lines (dicts) and DataFrame.
    """
    results = []
    for (event_id, track_id), group in clustered_hits.groupby(["event", "track_id"]):
        if track_id == -1:
            continue
        # collect cluster points (x,y,z_used)
        pts_cluster = group[["x", "y", "z_used"]].dropna().to_numpy()
        # optionally append BGO hits from same event
        if include_bgo:
            bgo = clustered_hits[(clustered_hits.event == event_id) & (clustered_hits.detector == "bgo")]
            if not bgo.empty:
                bgo_pts = bgo[["x", "y", "z_used"]].dropna().to_numpy()
                # Option: you might want bgo to contribute less weight; here appended equally
                pts_all = np.vstack([pts_cluster, bgo_pts]) if pts_cluster.size else bgo_pts
            else:
                pts_all = pts_cluster
        else:
            pts_all = pts_cluster

        if pts_all.shape[0] == 0:
            continue
        # small-N special case
        if pts_all.shape[0] == 1:
            origin = pts_all[0]
            direction = np.array([0.0, 0.0, 1.0])
            n_inliers = 1
        elif pts_all.shape[0] == 2:
            origin = pts_all.mean(axis=0)
            vec = pts_all[1] - pts_all[0]
            nrm = np.linalg.norm(vec)
            if nrm == 0 or not np.isfinite(nrm):
                direction = np.array([0.0, 0.0, 1.0])
            else:
                direction = vec / nrm
            n_inliers = 2
        else:
            # optionally use weighted PCA
            if weighted and (weight_col is not None) and (weight_col in group.columns):
                # define weights: higher weight = more trust; default weight = 1/sigma_z^2 type
                # w = group[weight_col].to_numpy()
                w = 1.0 / np.clip(group[weight_col].to_numpy(), 1e-6, np.inf)**2
                w /= np.nansum(w)
                # if BGO appended, need to create matching w_all; simple approach: duplicate mean weight for BGO
                
                if include_bgo and not bgo.empty:
                    # assume bgo weight same as cluster mean (or tune)
                    w_bgo = np.full(len(bgo_pts), np.nanmean(w) if len(w)>0 else 1.0)
                    w_all = np.concatenate([w, w_bgo])
                else:
                    w_all = w
                origin, direction = fit_line_weighted_svd(pts_all, w_all)
                n_inliers = pts_all.shape[0]
            else:
                if prefilter_ransac:
                    origin, direction, n_inliers = fit_line_ransac_then_svd(pts_all, residual_threshold=ransac_thresh)
                else:
                    origin, direction = fit_line_svd(pts_all)
                    n_inliers = pts_all.shape[0]

        results.append({
            "event": event_id,
            "track_id": track_id,
            "n_hits": pts_cluster.shape[0],
            "n_used": pts_all.shape[0],
            "n_inliers": n_inliers,
            "origin": origin,
            "direction": direction
        })
    return pd.DataFrame(results)


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


for eps in [1]:

    clustered_list = []
    for event_id, ev in hits_df.groupby("event"):
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


    lines_df = fit_lines_from_clusters_svd(clustered_hits, include_bgo=False,
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

    for ev in clustered_hits[(clustered_hits.Vx.notna() & clustered_hits.near_bgo) & ~clustered_hits.mixGate].groupby("event").event.first(): #clustered_hits.event.unique()[20:60]:
        event_hits = clustered_hits[clustered_hits.event == ev]
        event_lines = lines_df[lines_df.event == ev]

        # Normalize track_id to [0, 1] for colormap
        track_ids = event_hits.track_id.values
        norm = Normalize(vmin=np.nanmin(track_ids), vmax=np.nanmax(track_ids))
        cmap = cm.viridis

        fig, ax = plt.subplots(figsize=(8, 8))

        # --- Left: XY plane ---
        ax.scatter(event_hits.x, event_hits.y, c=cmap(norm(track_ids)), s=20, zorder=3)
        ax.errorbar(event_hits.x, event_hits.y, xerr=event_hits.dx, yerr=event_hits.dy, fmt="none", ecolor="gray", elinewidth=1, zorder=1)

        ax.set_aspect('equal')
        ax.set_xlim(-200, 200)
        ax.set_ylim(-200, 200)
        # ax.set_xlabel("x [mm]")
        ax.set_ylabel("y [mm]")

        divider = make_axes_locatable(ax)
        # below height and pad are in inches
        ax_xz = divider.append_axes("bottom", 4, pad=0.1, sharex=ax)
        ax_zy = divider.append_axes("right", 4, pad=0.1, sharey=ax)


        ax.xaxis.set_tick_params(labelbottom=False)
        ax_zy.yaxis.set_tick_params(labelleft=False)

        # --- Right: ZY plane ---
        ax_zy.scatter(event_hits.z_used, event_hits.y, c=cmap(norm(track_ids)), s=20, zorder=3)
        ax_zy.errorbar(event_hits.z_used, event_hits.y, xerr=event_hits.dz, yerr=event_hits.dy, fmt="none", ecolor="gray", elinewidth=1, zorder=1)

        # --- Bottom: XZ plane ---
        ax_xz.scatter(event_hits.x, event_hits.z_used, c=cmap(norm(track_ids)), s=20, zorder=3)
        ax_xz.errorbar(event_hits.x, event_hits.z_used, xerr=event_hits.dx, yerr=event_hits.dz, fmt="none", ecolor="gray", elinewidth=1, zorder=1)

        if "Vx" in event_lines.columns and event_lines.Vx.notna().any():
            ax.scatter(event_lines.Vx, event_lines.Vy, c="C1", s=100, marker="x")
            ax.errorbar(event_lines.Vx, event_lines.Vy, xerr=event_lines.Vx_sig, yerr=event_lines.Vy_sig, fmt="C1", capsize=5)
            ax_zy.scatter(event_lines.Vz, event_lines.Vy, c="C1", s=100, marker="x")
            ax_zy.errorbar(event_lines.Vz, event_lines.Vy, xerr=event_lines.Vz_sig, yerr=event_lines.Vy_sig, fmt="C1", capsize=5)
            ax_xz.scatter(event_lines.Vx, event_lines.Vz, c="C1", s=100, marker="x")
            ax_xz.errorbar(event_lines.Vx, event_lines.Vz, xerr=event_lines.Vx_sig, yerr=event_lines.Vz_sig, fmt="C1", capsize=5)

        for _, line in event_lines.iterrows():

            origin = line.origin
            direction = line.direction / np.linalg.norm(line.direction)
            color = cmap(norm(line.track_id))
            
            L = 150  # length to extend both sides
            p1 = origin - direction * L
            p2 = origin + direction * L

            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=color, lw=2)
            ax_zy.plot([p1[2], p2[2]], [p1[1], p2[1]], color=color, lw=2)
            ax_xz.plot([p1[0], p2[0]], [p1[2], p2[2]], color=color, lw=2)



        # --- Right: ZY plane ---
        # ax_zy.set_aspect('equal')
        ax_zy.set_xlim(-250, 250)
        ax_zy.set_ylim(-200, 200)
        ax_zy.set_xlabel("z [mm]")
        # ax_zy.set_ylabel("y [mm]")

        # --- Bottom: XZ plane ---
        # ax_xz.set_aspect('equal')
        ax_xz.set_xlim(-200, 200)
        ax_xz.set_ylim(-250, 250)
        ax_xz.set_xlabel("x [mm]")
        ax_xz.set_ylabel("z [mm]")

        # for ax_ in ax.flatten()[:-1]:
        #     ax_.grid()

        fig.suptitle(f"Event {ev} - eps {eps}", fontsize=14)
        plt.tight_layout()
        plt.show()




plt.plot(clustered_hits.groupby("event").fpgaTimeTag.mean()*1e-9, clustered_hits.groupby("event").Vx.mean(), "o")

plt.hist(clustered_hits[clustered_hits.Vx.notna()].groupby("event").fpgaTimeTag.mean()*1e-9)
plt.xlabel("Time in s")
plt.ylabel("Reconstructed Vertices")

plt.hist(clustered_hits[clustered_hits.Vx.notna() & clustered_hits.near_bgo].groupby("event").fpgaTimeTag.mean()*1e-9, bins=35)
plt.xlabel("Time in s")
plt.ylabel("Reconstructed Vertices")

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



def compute_event_features_from_clustered_hits(clustered_hits, bgo_center=(0, 0, 0)):
    """
    Compute event-level geometry and timing features from clustered hits.
    Requires that clustered_hits includes merged track and vertex info.

    Parameters
    ----------
    clustered_hits : pd.DataFrame
        Must contain at least columns:
        ['event', 'track_id', 'origin', 'direction', 'Vx', 'Vy', 'Vz', 'fpgaTimeTag']
    bgo_center : tuple
        (x, y, z) center of the BGO for distance calculation.

    Returns
    -------
    pd.DataFrame : event_features_df
        One row per event with summary metrics useful for classifying
        cosmics vs annihilation events.
    """

    features = []

    for event_id, ev in clustered_hits.groupby("event"):
        # --- select valid reconstructed tracks ---
        lines = ev.drop_duplicates(subset=["track_id"])
        lines = lines[np.isfinite(lines.direction.map(lambda v: v[0] if isinstance(v, (list, np.ndarray)) else np.nan))]

        if len(lines) == 0:
            continue

        dirs = np.stack(lines.direction.to_numpy())
        origins = np.stack(lines.origin.to_numpy())

        # 1. Number of tracks
        n_tracks = len(dirs)

        # 2. Pairwise angular distribution
        angles = []
        for i in range(n_tracks):
            for j in range(i + 1, n_tracks):
                cosang = np.clip(np.dot(dirs[i], dirs[j]) / (np.linalg.norm(dirs[i]) * np.linalg.norm(dirs[j])), -1, 1)
                angles.append(np.degrees(np.arccos(cosang)))
        mean_angle = np.nanmean(angles) if len(angles) else np.nan
        min_angle = np.nanmin(angles) if len(angles) else np.nan
        max_angle = np.nanmax(angles) if len(angles) else np.nan

        # 3. Mean |dz| component (verticality)
        mean_abs_dz = np.mean(np.abs(dirs[:, 2]))

        # 4. Vertex position and distance to BGO
        if "Vx" in ev.columns and ev["Vx"].notna().any():
            vertex = np.nanmean(ev[["Vx", "Vy", "Vz"]].dropna().to_numpy(), axis=0)
            vertex_rms = np.nanstd(ev[["Vx", "Vy", "Vz"]].dropna().to_numpy(), axis=0).mean()
            dist_to_bgo = np.linalg.norm(vertex - np.array(bgo_center))
        else:
            vertex = np.array([np.nan, np.nan, np.nan])
            vertex_rms = np.nan
            dist_to_bgo = np.nan
        
        # 5. Timing spread based on LE times across all detectors
        if any(col in ev.columns for col in ["LE_Us", "LE_Ds", "LE"]):
            le_all = []

            # --- Bars (both LE_Us and LE_Ds) ---
            if {"LE_Us", "LE_Ds"}.issubset(ev.columns):
                bar_mask = ev[["LE_Us", "LE_Ds"]].notna().any(axis=1)
                bar_mask = bar_mask & ((ev.LE_Us > 0) | (ev.LE_Ds > 0))

                bar_mean = ev.loc[bar_mask, ["LE_Us", "LE_Ds"]].mean(axis=1, skipna=True)
                le_all.append(bar_mean)

            # --- Single-ended detectors (tiles, BGO) ---
            if "LE" in ev.columns:
                single_le = ev.loc[ev["LE"].notna() & (ev["LE"] > 0), "LE"]
                le_all.append(single_le)

            # --- Combine all hit times ---
            if len(le_all) > 0:
                le_combined = pd.concat(le_all, axis=0)

                dt_min = le_combined.min()
                dt_max = le_combined.max()
                dt = dt_max - dt_min
                dt_mean = le_combined.mean()
            else:
                dt = dt_min = dt_max = dt_mean = np.nan
        else:
            dt = dt_min = dt_max = dt_mean = np.nan


        features.append({
            "event": event_id,
            "n_tracks": n_tracks,
            "mean_angle": mean_angle,
            "min_angle": min_angle,
            "max_angle": max_angle,
            "mean_abs_dz": mean_abs_dz,
            "vertex_x": vertex[0],
            "vertex_y": vertex[1],
            "vertex_z": vertex[2],
            "vertex_rms": vertex_rms,
            "dist_to_bgo": dist_to_bgo,
            "dt": dt,
            "dt_max": dt_max,
            "dt_min": dt_min,
            "dt_mean": dt_mean,
            "time": ev.fpgaTimeTag.min()
        })

    event_features_df = pd.DataFrame(features)
    return event_features_df


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

plt.scatter(event_features_df.max_angle, event_features_df.min_angle, c=event_features_df.time*1e-9)
plt.xlabel("Max inter-track angle [°]")
plt.ylabel("Min inter-track angle [°]")
plt.colorbar(label="Time [s]")

plt.scatter(event_features_df.time*1e-9, event_features_df.max_angle, c=event_features_df.n_tracks)
plt.xlabel("Time [s]")
plt.ylabel("Max inter-track angle [°]")
plt.colorbar(label="Min inter-track angle [°]")


plt.scatter(event_features_df.time*1e-9, event_features_df.dt_min, c=event_features_df.dt_mean)
plt.xlabel("Time [s]")
# plt.ylabel("Mean inter-track angle [°]")
plt.colorbar(label="Distance to BGO center [mm]")


plt.scatter(event_features_df.dt_max, event_features_df.mean_angle, c=event_features_df.time*1e-9)
plt.xlabel("Mean time difference [ns]")
plt.ylabel("Mean inter-track angle [°]")
plt.colorbar(label="Time [s]")


plt.hist2d(event_features_df.time*1e-9, event_features_df.dt_max, range=((0, 380), (0, 1400)), bins=(95, 14))

plt.hist2d(event_features_df.time*1e-9, event_features_df.dt_min, range=((0, 380), (0, 1400)), bins=(95, 14))

plt.hist2d(event_features_df.time*1e-9, event_features_df.dt_mean, range=((0, 380), (0, 1400)), bins=(95, 14))



plt.hist([event_features_df[event_features_df.mean_angle > 150].time*1e-9, event_features_df[event_features_df.mean_angle <= 150].time*1e-9], range=(0, 350), bins=35, label=["> 150°", "<= 150°"], stacked=True, color=["C0", "C2"])
plt.xlabel("Time [s]")
plt.ylabel("Number of events (stacked)")
plt.legend(title="Mean inter-track angle")
plt.show()


plt.hist(event_features_df.mean_angle, 18, range=(0, 180))

plt.hist2d(event_features_df.mean_angle, event_features_df.n_tracks, range=((0, 180), (0, 10)), bins=(18, 10), norm="log")

plt.hist(event_features_df.n_tracks, 20, range=(0, 20))



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