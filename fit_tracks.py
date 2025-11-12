import numpy as np
import pandas as pd
from sklearn.linear_model import RANSACRegressor


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

