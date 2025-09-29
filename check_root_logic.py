import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import transforms
import pandas as pd
import json
import uproot
import awkward as ak

tree = uproot.open("output_000264.root")["RawEventTree"]

rdf = ak.to_dataframe(tree.arrays(filter_name="*", library="ak"), how="outer")

bgoLE_tCut = np.where((rdf["bgoLE"] < 400) & (rdf["bgoLE"] > 0), rdf["bgoLE"], np.nan)
rdf["bgoToT"] = np.where((~np.isnan(bgoLE_tCut)) & (rdf["bgoTE"] > bgoLE_tCut), rdf["bgoTE"] - bgoLE_tCut, np.nan)

for det in ["hodoODs", "hodoOUs", "hodoIDs", "hodoIUs", "tileO", "tileI"]:
    le = rdf[det+"LE"]
    te = rdf[det+"TE"]

    rdf[det+"ToT"] = np.where((le > 0) & (te > le), te - le, np.nan)

layers = ["hodoODsToT", "hodoOUsToT", "hodoIDsToT", "hodoIUsToT", "bgoToT"]

valid_tot_counts = rdf[layers].notna().groupby(level="entry").sum()
event_mask = (valid_tot_counts >= 1).all(axis=1)
rdf_filtered = rdf.loc[rdf.index.get_level_values("entry").isin(event_mask[event_mask].index)]

def bar_coincidence_tot(le_a, le_b):
    return np.where(~np.isnan(le_a) & ~np.isnan(le_b), le_a, np.nan)

# Apply bar coincidence logic to each hodo layer pair
for det_a, det_b, name in [
    ("hodoODsToT", "hodoOUsToT", "barODsToT"),
    ("hodoOUsToT", "hodoODsToT", "barOUsToT"),
    ("hodoIDsToT", "hodoIUsToT", "barIDsToT"),
    ("hodoIUsToT", "hodoIDsToT", "barIUsToT"),
]:
    rdf_filtered[name] = bar_coincidence_tot(rdf_filtered[det_a], rdf_filtered[det_b])

# List of the bar coincidence columns to check
layers = ["barODsToT", "barOUsToT", "barIDsToT", "barIUsToT"]

# Step 1: Count non-NaNs per event (across subentries/channels)
valid_counts = rdf_filtered[layers].notna().groupby(level="entry").sum()

# Step 2: Identify entries (events) where all four layers have at least one valid ToT
entries_with_valid = valid_counts[(valid_counts > 0).all(axis=1)].index
valid_counts = valid_counts[(valid_counts > 0).all(axis=1)]


# Step 3: Filter the dataframe to keep only events with all bar layers firing
rdf_filtered = rdf_filtered.loc[entries_with_valid]


rdf_filtered["barODsCts"]  = rdf_filtered.index.get_level_values("entry").map(valid_counts.barODsToT)

rdf_filtered["barOUsCts"]  = rdf_filtered.index.get_level_values("entry").map(valid_counts.barOUsToT)
rdf_filtered["barIDsCts"]  = rdf_filtered.index.get_level_values("entry").map(valid_counts.barIDsToT)
rdf_filtered["barIUsCts"]  = rdf_filtered.index.get_level_values("entry").map(valid_counts.barIUsToT)
rdf_filtered["bgoCts"] = rdf_filtered.index.get_level_values("entry").map(rdf_filtered["bgoToT"].notna().groupby(level="entry").sum())

rdf_filtered["channel"] = rdf_filtered.index.get_level_values("subentry")
rdf_filtered["event"] = rdf_filtered.index.get_level_values("entry")

df_filtered = rdf_filtered[rdf_filtered["bgoCts"] > 1]

