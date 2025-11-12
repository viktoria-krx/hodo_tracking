import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, HDBSCAN

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
