import numpy as np
import pandas as pd

def ToT_to_E(ToT, params= None, fit_param_path = "/home/viktoria/Documents/Hodoscope/cern_data/Calibrations/EnergyCalibrationBGO/fit_params_exp_2025.txt"):
    # E = a0 + a1*ToT + a2*exp(-a3*ToT)

    if params is not None:
        _, a0, a1, a2, a3, _ = params.values[0]
        return (a0 + a1 * ToT + a2 * np.exp(-a3 * ToT))
    else:
        params = pd.read_csv(fit_param_path, delimiter = ", ")
        _, a0, a1, a2, a3, _ = params.values[0]
        return (a0 + a1 * ToT + a2 * np.exp(-a3 * ToT))


def hodo_valid_hits(ev, detector_name):
    if detector_name not in ev.detector.values:
        return False
    # For hodoscopes, check both LE_Us and LE_Ds
    le_us_ok = "LE_Us" in ev.columns and ev.loc[ev.detector == detector_name, "LE_Us"].notna().any()
    le_ds_ok = "LE_Ds" in ev.columns and ev.loc[ev.detector == detector_name, "LE_Ds"].notna().any()
    te_ok = "TE" in ev.columns and ev.loc[ev.detector == detector_name, "TE"].notna().any()
    # Require LE_Us, LE_Ds, and TE to be valid
    return le_us_ok and le_ds_ok and te_ok


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

    params = pd.read_csv("/home/viktoria/Documents/Hodoscope/cern_data/Calibrations/EnergyCalibrationBGO/fit_params_exp_2025.txt", delimiter = ", ")
    # print(params)
    # print(params.values[0])

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

        # # 2. Pairwise angular distribution
        # angles = []
        # for i in range(n_tracks):
        #     for j in range(i + 1, n_tracks):
        #         cosang = np.clip(np.dot(dirs[i], dirs[j]) / (np.linalg.norm(dirs[i]) * np.linalg.norm(dirs[j])), -1, 1)
        #         angles.append(np.degrees(np.arccos(cosang)))
        # mean_angle = np.nanmean(angles) if len(angles) else np.nan
        # min_angle = np.nanmin(angles) if len(angles) else np.nan
        # max_angle = np.nanmax(angles) if len(angles) else np.nan

        # 2. Mean |dz| component (verticality)
        mean_abs_dz = np.mean(np.abs(dirs[:, 2]))

        # 3. Vertex position and distance to BGO
        if "Vx" in ev.columns and ev["Vx"].notna().any():
            vertex = np.nanmean(ev[["Vx", "Vy", "Vz"]].dropna().to_numpy(), axis=0)
            vertex_rms = np.nanstd(ev[["Vx", "Vy", "Vz"]].dropna().to_numpy(), axis=0).mean()
            dist_to_bgo = np.linalg.norm(vertex - np.array(bgo_center))
        else:
            vertex = np.array([np.nan, np.nan, np.nan])
            vertex_rms = np.nan
            dist_to_bgo = np.nan
        
        # 4. Timing spread based on LE times across all detectors
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

                dt_min = le_combined[le_combined.notna()].diff().min()
                dt_max = le_combined[le_combined.notna()].diff().max()
                dt = dt_max - dt_min
                dt_mean = le_combined[le_combined.notna()].diff().mean()
            else:
                dt = dt_min = dt_max = dt_mean = np.nan
        else:
            dt = dt_min = dt_max = dt_mean = np.nan

        # 5. Pairwise angular distribution and 
        # hodoO Δt for opposite track pairs (angle ~180°)
        opp_track_dt = np.nan
        opp_threshold = 160  # degrees

        # Need track_ids aligned with dirs array
        track_ids = lines.track_id.to_numpy()

        opp_dts = []
        angles = []
        # Loop over all pairs again, but now track which ones are opposite
        for i in range(n_tracks):
            for j in range(i + 1, n_tracks):

                # Recompute angle
                cosang = np.clip(
                    np.dot(dirs[i], dirs[j]) /
                    (np.linalg.norm(dirs[i]) * np.linalg.norm(dirs[j])),
                    -1, 1
                )
                angle_ij = np.degrees(np.arccos(cosang))
                angles.append(np.degrees(np.arccos(cosang)))

                if angle_ij > opp_threshold:

                    # The two specific track IDs
                    ti = track_ids[i]
                    tj = track_ids[j]

                    # Select hodoO hits for only these two tracks
                    pair_hits = ev[
                        (ev.detector == "hodoO") &
                        (ev.track_id.isin([ti, tj])) &
                        (ev.LE.notna())
                    ]

                    if len(pair_hits) >= 2:
                        times = np.sort(pair_hits["LE"].to_numpy())
                        opp_dts.append(times[-1] - times[0])

        mean_angle = np.nanmean(angles) if len(angles) else np.nan
        min_angle = np.nanmin(angles) if len(angles) else np.nan
        max_angle = np.nanmax(angles) if len(angles) else np.nan

        if len(opp_dts) > 0:
            opp_track_dt = np.min(opp_dts)  # or max/mean depending on your choice

        # 6. Check if trigger condition is fulfilled
        has_bgo = ev.loc[ev.detector == "bgo", "LE"].notna().any()
        bgo_count = len(ev.loc[ev.detector == "bgo", "LE"].notna())
        trigger_condition = has_bgo and hodo_valid_hits(ev, "hodoO") and hodo_valid_hits(ev, "hodoI")

        if "Hbar_BG" in ev.columns:
            what = ev.Hbar_BG.unique()[0]
            Hbar = what == "Hbar"
            
            Anni = [0.9 if what == "Hbar" else 1 if what == "pbar" else 0]
        else:
            Hbar = ev.mixGate.unique()[0]

        features.append({
            "event": event_id,
            "cusp": ev.cuspRunNumber.unique()[0],
            "n_tracks": n_tracks,
            "n_bgo": bgo_count,
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
            "opp_track_dt": opp_track_dt,
            "time": ev.fpgaTimeTag.min(),
            "bgoToTSum": ev.bgoToT.sum(),
            "bgoEdep": ToT_to_E(ev.bgoToT.sum(), params=params),
            "mix": ev.mixGate.unique()[0],
            "Hbar": Hbar,
            "Annihilation": Anni[0],
            "trigger": trigger_condition,
            "vertex": ~np.any(np.isnan(vertex))
        })

    event_features_df = pd.DataFrame(features)
    return event_features_df
