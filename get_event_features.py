import numpy as np
import pandas as pd

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

            # # --- Bars (both LE_Us and LE_Ds) ---
            # if {"LE_Us", "LE_Ds"}.issubset(ev.columns):
            #     bar_mask = ev[["LE_Us", "LE_Ds"]].notna().any(axis=1)
            #     bar_mask = bar_mask & ((ev.LE_Us > 0) | (ev.LE_Ds > 0))

            #     bar_mean = ev.loc[bar_mask, ["LE_Us", "LE_Ds"]].mean(axis=1, skipna=True)
            #     le_all.append(bar_mean)

            # --- Single-ended detectors (tiles, BGO) ---
            if "LE" in ev.columns:
                single_le = ev.loc[ev["LE"].notna(), "LE"]
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
            "time": ev.fpgaTimeTag.min(),
            "bgoToTSum": ev.bgoToT.sum(),
            "mix": ev.mixGate.unique()[0]
        })

    event_features_df = pd.DataFrame(features)
    return event_features_df
