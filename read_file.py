import uproot
import awkward as ak
import pandas as pd
from typing import List
import numpy as np
import time
from pathlib import Path
import json
import cProfile, pstats
import glob

# --- 1. Preload geometry into a fast lookup ---
_GEOMETRY_CACHE = {}
_CI_Z_STATS = {}

def load_geometry(geometry_path="./geometry_files/geometry.json"):
    global _GEOMETRY_CACHE
    if not _GEOMETRY_CACHE:
        with open(Path(geometry_path)) as f:
            geom = json.load(f)
        _GEOMETRY_CACHE = {
            det_name: {int(ch["channel_id"]): ch for ch in det_list}
            for det_name, det_list in geom.items()
        }
    return _GEOMETRY_CACHE

def load_ci_z(ci_path="./geometry_files/CI_z.json"):
    global _CI_Z_STATS
    if not _CI_Z_STATS:
        with open(Path(ci_path)) as f:
            _CI_Z_STATS = json.load(f)
    return _CI_Z_STATS

def get_geom(det, ch):
    det_map = {
        "hodoO": "outer_bars", "hodoI": "inner_bars",
        "tileO": "outer_tiles", "tileI": "inner_tiles", "bgo": "bgo"
    }
    g = _GEOMETRY_CACHE[det_map[det]]
    return g.get(int(ch), None)

# def reconstruct_bar_z(df, detector, outer_or_inner):
#     """
#     Compute reconstructed z-positions for bar detectors based on ΔLE.
#     Works vectorized over a dataframe.
#     """
#     ci_stats = load_ci_z()
#     stats = ci_stats[f"{outer_or_inner} bars"]
#     z_min, z_max = (-225, 225) if outer_or_inner == "outer" else (-150, 150)

#     # Compute ΔLE = LE_Us - LE_Ds
#     dLE = df[f"{detector}UsLE"] - df[f"{detector}DsLE"]

#     # Get channel-specific calibration info
#     ci0 = np.array([stats["ci"][int(ch)][0] for ch in df["channel"]])
#     ci1 = np.array([stats["ci"][int(ch)][1] for ch in df["channel"]])

#     # Compute z_reco (vectorized)
#     z = (dLE - ci0) / (ci1 - ci0) * (z_max - z_min) + z_min

#     # Mask unreasonable z values
#     z[np.abs(z) > (z_max + 5)] = np.nan

#     return z


def reconstruct_bar_z(df, detector, outer_or_inner):
    """
    Compute reconstructed z-positions for bar detectors based on ΔLE.
    Vectorized and safe against missing channels.
    """
    ci_stats = load_ci_z()
    stats = ci_stats[f"{outer_or_inner} bars"]
    ci = np.array(stats["ci"])  # shape: (N_channels, 2)
    z_min, z_max = (-225, 225) if outer_or_inner == "outer" else (-150, 150)

    sigma = np.array(stats["sigma"])
    width = np.array(stats["width"])

    # ΔLE = LE_Us - LE_Ds
    dLE = df[f"{detector}UsLE"] - df[f"{detector}DsLE"]
    channels = df["channel"]
    # print(channels)
    # print(dLE)

    # Build arrays of ci0, ci1, handling invalid channels
    ci0 = np.full(len(channels), np.nan)
    ci1 = np.full(len(channels), np.nan)
    valid_mask = (channels >= 0) & (channels < len(ci))
    ci0 = ci[channels, 0]
    ci1 = ci[channels, 1]

    # Compute dz
    dz = sigma[channels]*5 + width[channels]

    # Compute z
    z = (dLE - ci0) / (ci1 - ci0) * (z_max - z_min) + z_min
    cutoff = np.abs(z) > (z_max + dz)
    z[cutoff] = np.nan
    dz[cutoff] = np.nan  

    return z, dz

# tic = time.perf_counter()

# # --- 2. Load ROOT file efficiently ---
# tree = uproot.open("~/Documents/Hodoscope/cern_data/2025_Data/output_000269.root")["EventTree"]
# branches = ["eventID", "fpgaTimeTag", "hodoODsLE", "hodoODsTE", "hodoOUsLE", "hodoOUsTE", "hodoIDsLE", "hodoIDsTE", "hodoIUsLE", "hodoIUsTE", "tileOLE", "tileOTE", "tileILE", "tileOLE", "bgoLE", "bgoTE"]
# df = ak.to_dataframe(tree.arrays(branches, library="ak"), how="outer")
# df["channel"] = df.index.get_level_values("subentry")
# df["event"] = df.index.get_level_values("entry")

# # --- 3. Preload geometry cache ---
# load_geometry()

# # --- 4. Build physics DataFrame ---
# rows = []
# for det in ["hodoO", "hodoI", "tileO", "tileI", "bgo"]:
#     le_col = f"{det}LE" if f"{det}LE" in df else None
#     # tot_col = f"{det}ToT" if f"{det}ToT" in df else None
#     df_det = df[[c for c in [le_col, "channel", "event"] if c in df]]

#     for _, row in df_det.iterrows():
#         geom = get_geom(det, row["channel"])
#         if geom is None:
#             continue
#         x, y, z = geom["position"]
#         dx, dy, dz = geom["width"]/2, geom["thickness"]/2, geom["length"]/2
#         rows.append({
#             "event": row["event"],
#             "detector": det,
#             "channel": row["channel"],
#             "x": x, "y": y, "z": z,
#             "dx": dx, "dy": dy, "dz": dz,
#             # "ToT": row.get(tot_col, np.nan),
#             "LE": row.get(le_col, np.nan),
#         })

# hits_df = pd.DataFrame(rows)

# toc = time.perf_counter()
# print(f"{toc - tic:0.4f} seconds")


def build_hits_df(root_path, geometry_path="./geometry_files/geometry.json", ci_path="./geometry_files/CI_z.json"):
    tree = uproot.open(root_path)["EventTree"]
    branches = ["eventID", "fpgaTimeTag", "hodoODsLE", "hodoODsTE", "hodoOUsLE", "hodoOUsTE", "hodoIDsLE", "hodoIDsTE", "hodoIUsLE", "hodoIUsTE", "tileOLE", "tileOTE", "tileILE", "tileOLE", "bgoLE", "bgoTE", "bgoToTSum"]
    df = ak.to_dataframe(tree.arrays(branches, library="ak"), how="outer")
    df["channel"] = df.index.get_level_values("subentry")
    df["event"] = df.index.get_level_values("entry")

    load_geometry(geometry_path)
    load_ci_z(ci_path)

    rows = []
    for det in ["hodoO", "hodoI", "tileO", "tileI", "bgo"]:
        if det.startswith("hodo"):  # double-sided
            le_us_col = f"{det}UsLE"
            le_ds_col = f"{det}DsLE"
            # tot_us_col = f"{det}UsToT"
            # tot_ds_col = f"{det}DsToT"

            # if not all(c in df for c in [le_us_col, le_ds_col]):
            #     continue

            outer_or_inner = "outer" if det == "hodoO" else "inner"
            df_det = df[[le_us_col, le_ds_col, "channel", "event"]][df.channel < 32].dropna(how="all")
            # df_det = df[[le_us_col, le_ds_col, tot_us_col, tot_ds_col, "channel", "event"]].dropna(how="all")

            z_reco, dz_reco = reconstruct_bar_z(df_det.copy(), det, outer_or_inner)

            for i, row in df_det.iterrows():
                geom = _GEOMETRY_CACHE["outer_bars" if outer_or_inner == "outer" else "inner_bars"].get(int(row["channel"]))
                if geom is None:
                    continue

                x, y, z_geom = geom["position"]
                dx, dy, dz = geom["width"]/2, geom["thickness"]/2, geom["length"]/2

                rows.append({
                    "event": row["event"],
                    "detector": det,
                    "layer": "outer" if det == "hodoO" else "inner",
                    "channel": int(row["channel"]),
                    "x": x, "y": y, "z": float(z_reco[i]) if not np.isnan(z_reco[i]) else z_geom,
                    "dx": dx, "dy": dy, "dz": dz,
                    # "ToT_Us": row.get(tot_us_col, np.nan),
                    # "ToT_Ds": row.get(tot_ds_col, np.nan),
                    "LE_Us": row.get(le_us_col, np.nan),
                    "LE_Ds": row.get(le_ds_col, np.nan),
                    "time": row.get("fpgaTimeTag", np.nan)
                })

        else:  # single-sided detectors
            le_col = f"{det}LE"
            # tot_col = f"{det}ToT"

            # if not all(c in df for c in [le_col, tot_col]):
            #     continue

            df_det = df[[le_col, "channel", "event"]].dropna(how="all")
            # df_det = df[[le_col, tot_col, "channel", "event"]].dropna(how="all")

            det_map = {
                "tileO": "outer_tiles",
                "tileI": "inner_tiles",
                "bgo": "bgo"
            }

            for _, row in df_det.iterrows():
                geom = _GEOMETRY_CACHE[det_map[det]].get(int(row["channel"]))
                if geom is None:
                    continue

                x, y, z = geom["position"]
                dx, dy, dz = geom["width"]/2, geom["thickness"]/2, geom["length"]/2

                rows.append({
                    "event": row["event"],
                    "detector": det,
                    "layer": "outer" if "O" in det else "inner" if "I" in det else "central",
                    "channel": int(row["channel"]),
                    "x": x, "y": y, "z": z,
                    "dx": dx, "dy": dy, "dz": dz,
                    # "ToT": row.get(tot_col, np.nan),
                    "LE": row.get(le_col, np.nan),
                    "time": row.get("fpgaTimeTag", np.nan),
                    "bgoToTSum": row.get("bgoToTSum", np.nan)
                })
        

    return pd.DataFrame(rows)



def build_hits_df_fast(root_path, geometry_path="./geometry_files/geometry.json", ci_path="./geometry_files/CI_z.json"):
    tree = uproot.open(root_path)["RawEventTree"]
    branches = [
        "eventID", "fpgaTimeTag", "mixGate",
        "hodoODsLE", "hodoODsTE", "hodoOUsLE", "hodoOUsTE",
        "hodoIDsLE", "hodoIDsTE", "hodoIUsLE", "hodoIUsTE",
        "tileOLE", "tileOTE", "tileILE", "tileITE",
        "bgoLE", "bgoTE", "trgLE", "cuspRunNumber"
    ]
    df = ak.to_dataframe(tree.arrays(branches, library="ak"), how="outer")
    df["channel"] = df.index.get_level_values("subentry")
    df["event"] = df.index.get_level_values("entry")

    # print(df)
    # print(df.columns

    if "fpgaTimeTag" in df.columns:
        df["fpgaTimeTag"] = df.groupby("event")["fpgaTimeTag"].transform("first")

    if "mixGate" in df.columns:
        df["mixGate"] = df.groupby("event")["mixGate"].transform("first")

    print(df[df.channel < 4].trgLE)

    def get_trg(row, tdc):
        try:
            return row["trgLE"][tdc] if isinstance(row["trgLE"], (list, np.ndarray)) and len(row["trgLE"]) > tdc else np.nan
        except Exception:
            return np.nan

    print(df)

    # Preload geometry & calibration
    load_geometry(geometry_path)
    load_ci_z(ci_path)

    # to do trigger subtraction I need to get which TDC each channel belongs to
    def get_tdc(det, ch):
        if det in ("hodoO", "hodoI"):
            if 4 <= ch <= 19:
                return 0
            elif (20 <= ch <= 31) or (0 <= ch <= 3):
                return 1
        elif det == "tileO":
            if 0 <= ch <= 59:
                return 0
            elif 60 <= ch <= 119:
                return 1
        elif det == "tileI":
            if 0 <= ch <= 119:
                return 2
        elif det == "bgo":
            if 0 <= ch <= 63:
                return 3
        return None

    trg_map = {}
    for event, group in df.groupby("event"):
        valid_trg = group[group.channel < 4].dropna(subset=["trgLE"])
        if not valid_trg.empty:
            trg_map[event] = valid_trg.set_index("channel")["trgLE"].to_dict()
        else:
            trg_map[event] = {}  # no triggers for this event
    print(trg_map)

    def get_trg_val(row, det):
        tdc = get_tdc(det, row["channel"])
        if tdc is None:
            return np.nan
        # look up trigger for this event and TDC
        return trg_map.get(row["event"], {}).get(tdc, np.nan)
    
    hits_list = []

    # ---------- Double-sided detectors (bars) ----------
    for det, outer_or_inner in [("hodoO", "outer"), ("hodoI", "inner")]:
        le_us_col = f"{det}UsLE"
        le_ds_col = f"{det}DsLE"
        te_us_col = f"{det}UsTE"
        te_ds_col = f"{det}DsTE"
        # df_det = df.loc[df.channel < 32, ["channel", "event", le_us_col, le_ds_col]].dropna(how="all")
        df_det = df[[le_us_col, le_ds_col, te_us_col, te_ds_col, "trgLE", "channel", "event", "fpgaTimeTag", "mixGate", "cuspRunNumber"]][df.channel < 32].dropna(subset=[le_us_col, le_ds_col, te_us_col, te_ds_col], how="all").reset_index(drop=True)

        # df_det = (
        #     df[[le_us_col, le_ds_col, "channel", "event"]]
        #     [df.channel < 32]
        #     .dropna(subset=[le_us_col, le_ds_col])
        # )


        if df_det.empty:
            continue

        # Compute trigger value per hit
        df_det["trg_val"] = df_det.apply(lambda r: get_trg_val(r, det), axis=1)

        # Subtract trigger from LE/TE times
        for col in df_det.columns:
            if col.endswith("LE") or col.endswith("TE"):
                df_det[col] = df_det[col] - df_det["trg_val"]

        # Vectorized z reconstruction
        z_reco, dz_reco = reconstruct_bar_z(df_det.copy(), det, outer_or_inner)

        # Get geometry dataframe for this detector
        geom_key = "outer_bars" if outer_or_inner == "outer" else "inner_bars"
        geom_data = pd.DataFrame.from_dict(_GEOMETRY_CACHE[geom_key], orient="index")
        geom_data["channel"] = geom_data.index.astype(int)
        geom_data[["x", "y", "z"]] = geom_data["position"].tolist()
        geom_data = geom_data.drop(columns=["position"])
        # geom_data = geom_data.explode("position")  # ensure correct shape if nested


        # Merge df_det + geometry by channel (vectorized join)
        df_det = df_det.merge(geom_data, on="channel", how="left")
        df_det["rot_rad"] = np.deg2rad(df_det["rotation"])
        # df_det["z_geom"] = df_det["position"].apply(lambda p: p[2] if isinstance(p, (list, tuple)) else np.nan)
        # df_det["x"] = df_det["position"].apply(lambda p: p[0] if isinstance(p, (list, tuple)) else np.nan)
        # df_det["y"] = df_det["position"].apply(lambda p: p[1] if isinstance(p, (list, tuple)) else np.nan)

        # print(det, len(df_det), len(z_reco))
        df_det["z_reco"] = z_reco
        df_det["dz_reco"] = dz_reco
        # df_det["z"] = np.where(np.isnan(z_reco), df_det["z_geom"], z_reco)
        df_det["layer"] = outer_or_inner
        df_det["detector"] = det
        df_det["dx_local"] = df_det["thickness"] / 2.0
        df_det["dy_local"] = df_det["width"] / 2.0

        df_det["dx"] = np.sqrt(
            (df_det["dx_local"] * np.cos(df_det["rot_rad"]))**2 +
            (df_det["dy_local"] * np.sin(df_det["rot_rad"]))**2
        )
        df_det["dy"] = np.sqrt(
            (df_det["dx_local"] * np.sin(df_det["rot_rad"]))**2 +
            (df_det["dy_local"] * np.cos(df_det["rot_rad"]))**2
        )

        df_det["LE"] = df_det[[le_us_col, le_ds_col]].mean(axis=1)
        df_det["TE"] = df_det[[te_us_col, te_ds_col]].mean(axis=1)

        df_det["dz"] = df_det["length"] / 2.0

        df_det.rename(columns={le_us_col: "LE_Us", le_ds_col: "LE_Ds", te_us_col: "TE_Us", te_ds_col: "TE_Ds"}, inplace=True)
        hits_list.append(df_det[[
            "event", "detector", "layer", "channel", "fpgaTimeTag", "mixGate", "cuspRunNumber",
            "x", "y", "z", "z_reco", "dx", "dy", "dz", "dz_reco",
            "LE_Us", "LE_Ds", "TE_Us", "TE_Ds", "LE", "TE", "trg_val"
        ]])

    # ---------- Single-sided detectors ----------
    for det in ["tileO", "tileI", "bgo"]:
        le_col = f"{det}LE"
        te_col = f"{det}TE"
        df_det = df.loc[:, ["channel", "event", le_col, te_col, "trgLE", "fpgaTimeTag", "mixGate", "cuspRunNumber"]].dropna(subset=[le_col], how="all")
        if df_det.empty:
            continue

        # Assign TDC
        df_det["TDC"] = df_det["channel"].apply(lambda ch: get_tdc(det, ch))

        # Compute trigger value per hit
        df_det["trg_val"] = df_det.apply(lambda r: get_trg_val(r, det), axis=1)

        # Subtract trigger from LE/TE times
        for col in df_det.columns:
            if col.endswith("LE") or col.endswith("TE"):
                df_det[col] = df_det[col] - df_det["trg_val"]

        det_map = {
            "tileO": "outer_tiles",
            "tileI": "inner_tiles",
            "bgo": "bgo"
        }

        geom_key = det_map[det]
        geom_data = pd.DataFrame.from_dict(_GEOMETRY_CACHE[geom_key], orient="index")
        geom_data["channel"] = geom_data.index.astype(int)
        geom_data[["x", "y", "z"]] = geom_data["position"].tolist()
        geom_data = geom_data.drop(columns=["position"])

        df_det = df_det.merge(geom_data, on="channel", how="left")
        df_det["rot_rad"] = np.deg2rad(df_det["rotation"])
        # df_det["x"] = df_det["position"].apply(lambda p: p[0] if isinstance(p, (list, tuple)) else np.nan)
        # df_det["y"] = df_det["position"].apply(lambda p: p[1] if isinstance(p, (list, tuple)) else np.nan)
        # df_det["z"] = df_det["position"].apply(lambda p: p[2] if isinstance(p, (list, tuple)) else np.nan)

        df_det["layer"] = (
            "outer" if "O" in det else
            "inner" if "I" in det else
            "central"
        )
        df_det["detector"] = det
        # df_det["dx"] = df_det["width"] / 2.0
        # df_det["dy"] = df_det["thickness"] / 2.0

        df_det["dx_local"] = df_det["thickness"] / 2.0
        df_det["dy_local"] = df_det["width"] / 2.0

        df_det["dx"] = np.sqrt(
            (df_det["dx_local"] * np.cos(df_det["rot_rad"]))**2 +
            (df_det["dy_local"] * np.sin(df_det["rot_rad"]))**2
        )
        df_det["dy"] = np.sqrt(
            (df_det["dx_local"] * np.sin(df_det["rot_rad"]))**2 +
            (df_det["dy_local"] * np.cos(df_det["rot_rad"]))**2
        )

        df_det["dz"] = df_det["length"] / 2.0

        df_det.rename(columns={le_col: "LE", te_col: "TE"}, inplace=True)
        hits_list.append(df_det[[
            "event", "detector", "layer", "channel", "fpgaTimeTag", "mixGate", "cuspRunNumber",
            "x", "y", "z", "dx", "dy", "dz", "LE", "TE", "trg_val"
        ]])

    hits_df = pd.concat(hits_list, ignore_index=True)
    return hits_df

# tic = time.perf_counter()

# root_path = "~/Documents/Hodoscope/cern_data/2025_Data/output_000269.root"

# with cProfile.Profile() as pr:
#     hits_df = build_hits_df_fast("~/Documents/Hodoscope/cern_data/2025_Data/output_000269.root")

# toc = time.perf_counter()
# print(f"{toc - tic:0.4f} seconds")

# stats = pstats.Stats(pr)
# stats.sort_stats(pstats.SortKey.CUMULATIVE).print_stats(20)



def build_hits_df_from_runs(run_list, base_path="~/Documents/Hodoscope/cern_data/2025_Data/",
                            geometry_path="./geometry_files/geometry.json",
                            ci_path="./geometry_files/CI_z.json", 
                            version="cusp_run"):
    """
    Loads multiple ROOT files sequentially and builds one combined hits_df.
    Ensures event numbering continues across files (no resets to 0).
    """

    base_path = Path(base_path).expanduser()
    all_hits = []
    cumulative_event_offset = 0

    for i, run in enumerate(run_list):
        if version == "simple":
            root_path = base_path / f"output_00{run}.root"
            print(f"Loading run {run} from {root_path} ...")
        else: 
            pattern = str(base_path / f"output_00{run}_*.root")
            matches = sorted(glob.glob(pattern))
            root_path = matches[0]
            print(f"Loading run {run} from {root_path} ...")

        # build_hits_df_fast returns a dataframe with an 'event' column
        hits_df = build_hits_df_fast(str(root_path),
                                     geometry_path=geometry_path,
                                     ci_path=ci_path)

        # Apply the event offset
        hits_df["event"] = hits_df["event"] + cumulative_event_offset

        # Update cumulative offset for next file
        cumulative_event_offset = hits_df["event"].max() + 1

        all_hits.append(hits_df)

    combined_hits_df = pd.concat(all_hits, ignore_index=True)
    return combined_hits_df