import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import transforms
import pandas as pd
import json
import uproot
import awkward as ak
from scipy.optimize import curve_fit
from scipy.stats import norm

hodo_colors = {0: "#33638d", 45: "#482677", 90: "#bc4174", 135: "#d35040", 180: "#fba100", 225: "#fde725", 270: "#95d840", 315: "#3cbb75"}

def gauss(x, a, x0, sigma):
    return a * np.exp(-(x - x0)**2 / (2 * sigma**2))

tree = uproot.open("~/Documents/Hodoscope/cern_data/2025_Data/output_000265.root")["EventTree"]

# tree = uproot.open("~/Documents/Hodoscope/cern_data/2025_Data/output_000525.root")["EventTree"]
df = ak.to_dataframe(tree.arrays(filter_name="*", library="ak"), how="outer")

df["channel"] = df.index.get_level_values("subentry")
df["event"] = df.index.get_level_values("entry")

# df.reset_index(level=("subentry",), inplace=True, names=["channel", "channel"])
# df["event"] = df.index.get_level_values("entry")

with open("geometry.json", "r") as f:
        geometry = json.load(f)


def plot_event(event_id, df, geometry, hodo_colors=None):
 
    # Collect active channels
    df_event = df[df.event == event_id]
    active_outer_bars = df_event[df_event["hodoODsToT"].notna()].channel.to_list()
    active_inner_bars = df_event[df_event["hodoIDsToT"].notna()].channel.to_list()
    active_outer_tiles = df_event[df_event["tileOToT"].notna()].channel.to_list()
    active_inner_tiles = df_event[df_event["tileIToT"].notna()].channel.to_list()
    active_bgo = df_event[df_event["bgoToT"].notna()].channel.to_list()

    # Combine all active hits per layer for quick lookup
    active_channels = {
        "hodo_outer_bars": set(active_outer_bars),
        "hodo_inner_bars": set(active_inner_bars),
        "tile_outer": set(active_outer_tiles),
        "tile_inner": set(active_inner_tiles),
        "bgo": set(active_bgo),
    }

    print(active_channels)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect('equal')
    ax.set_title(f"Hodoscope Geometry - Event {event_id}")
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")

    for layer in geometry:
        for scint in geometry[layer]:
            x, y, _ = scint["position"]
            angle = scint["rotation"]
            channel_id = scint["channel_id"]
            alph = 1

            if layer == "bgo":
                is_active = channel_id in active_channels["bgo"]
                col = "dimgray" if is_active else "lightgray"
            elif layer == "outer_bars":
                is_active = channel_id in active_channels["hodo_outer_bars"]
                col = hodo_colors.get(angle, "lightgray") if is_active else "lightgray"
            elif layer == "outer_tiles":
                is_active = channel_id in active_channels["tile_outer"]
                col = hodo_colors.get(angle, "lightgray") if is_active else "lightgray"
                if not is_active:
                    alph = 0
            elif layer == "inner_bars":
                is_active = channel_id in active_channels["hodo_inner_bars"]
                col = hodo_colors.get(angle, "lightgray") if is_active else "lightgray"
            elif layer == "inner_tiles":
                is_active = channel_id in active_channels["tile_inner"]
                col = hodo_colors.get(angle, "lightgray") if is_active else "lightgray"
                if not is_active:
                    alph = 0
            else:
                col = "gray"

            rect = patches.Rectangle(
                (-scint["thickness"]/2, -scint["width"]/2),
                scint["thickness"], scint["width"],
                angle=0.0,
                linewidth=0.1,
                alpha=alph,
                edgecolor="dimgray",
                facecolor=col
            )

            # Transform: rotate and translate
            t = transforms.Affine2D().rotate_deg(angle).translate(x, y) + ax.transData
            rect.set_transform(t)
            ax.add_patch(rect)

            # Optional: draw channel number
            ax.text(x, y, str(channel_id), ha='center', va='center', fontsize=5)

    ax.autoscale()
    plt.grid(True)
    plt.show()


for ev in df.event.unique()[:30]:
    plot_event(ev, df, geometry, hodo_colors)


df["hodoOdLE"] = df.hodoOUsLE - df.hodoODsLE
df["hodoIdLE"] = df.hodoIUsLE - df.hodoIDsLE

# sigO = np.zeros(32)*np.nan
# meanO = np.zeros(32)*np.nan
# ciO = np.zeros([32,2])*np.nan
# sigI = np.zeros(32)*np.nan
# meanI = np.zeros(32)*np.nan
# ciI = np.zeros([32,2])*np.nan

# for ch in np.arange(32):
#     dat = df.hodoOdLE[(df.channel == ch) & (df.hodoOdLE > -20) & (df.hodoOdLE < 20)]
#     dLE_hist = plt.hist(dat, bins=80, range=(-20, 20), label=rf"$\Delta$LE ch {ch}", density=True)
#     fit_hist = norm.fit(dat)
#     sigO[ch] = fit_hist[1]
#     meanO[ch] = fit_hist[0]
#     ciO[ch,:] = norm.interval(0.95, *fit_hist)
#     # print(fit_hist)
#     plt.plot(dLE_hist[1], norm.pdf(dLE_hist[1], *fit_hist))
#     plt.legend()
#     plt.close()

# for ch in np.arange(32):
#     dat = df.hodoIdLE[(df.channel == ch) & (df.hodoIdLE > -20) & (df.hodoIdLE < 20)]
#     dLE_hist = plt.hist(dat, bins=80, range=(-20, 20), label=rf"$\Delta$LE ch {ch}", density=True)
#     fit_hist = norm.fit(dat)
#     sigI[ch] = fit_hist[1]
#     meanI[ch] = fit_hist[0]
#     ciI[ch,:] = norm.interval(0.95, *fit_hist)
#     # print(fit_hist)
#     plt.plot(dLE_hist[1], norm.pdf(dLE_hist[1], *fit_hist))
#     plt.legend()
#     plt.close()


# def map_dLE_to_z(dLE, CI, min, max):
#     return (dLE - CI[0]) / (CI[1] - CI[0]) * (max-min) + min


# zdLE_O = np.full(len(df), np.nan)

# for ch in range(32):
#     mask = (df["channel"] == ch)
#     zdLE_O[mask] = map_dLE_to_z(
#         df.loc[mask, "hodoOdLE"],
#         ciO[ch],  # channel-specific CI
#         -225,
#         225 )





# def compute_z_positions_from_dLE(df, dLE_col, num_channels=32, fit_range=(-20, 20), z_min=-225, z_max=225, ci_level=0.95, plot=False):
#     """
#     Compute z-positions based on delta LE values (from bar ends) for each channel.

#     Parameters:
#         df (DataFrame): Input DataFrame with dLE values.
#         dLE_col (str): Name of the delta LE column (e.g. 'hodoOdLE' or 'hodoIdLE').
#         channel_col (str): Column indicating the bar/channel number.
#         fit_range (tuple): Range to include for Gaussian fit.
#         z_min (float): Minimum physical z coordinate for mapping.
#         z_max (float): Maximum physical z coordinate for mapping.
#         ci_level (float): Confidence interval level (e.g. 0.95).
#         plot (bool): Whether to generate and show plots for each channel.
    
#     Returns:
#         z_positions (np.array): Array of z values for each row in df (same order).
#         channel_stats (dict): Dictionary with mean, sigma, CI per channel.
#     """
#     z_positions = np.full(len(df), np.nan)
#     channel_stats = {
#         "mean": np.full(num_channels, np.nan),
#         "sigma": np.full(num_channels, np.nan),
#         "ci": np.full((num_channels, 2), np.nan),
#     }

#     for ch in range(num_channels):
#         mask = (df["channel"] == ch)
#         dat = df.loc[mask, dLE_col]
#         dat = dat[(dat > fit_range[0]) & (dat < fit_range[1])]

#         if len(dat) < 10:
#             continue  # Skip if not enough data to fit

#         mu, sigma = norm.fit(dat)
#         ci = norm.interval(ci_level, loc=mu, scale=sigma)

#         # Store fit stats
#         channel_stats["mean"][ch] = mu
#         channel_stats["sigma"][ch] = sigma
#         channel_stats["ci"][ch] = ci

#         # Map to z
#         dLE_vals = df.loc[mask, dLE_col]
#         z_vals = (dLE_vals - ci[0]) / (ci[1] - ci[0]) * (z_max - z_min) + z_min
#         z_positions[mask] = z_vals

#         if plot:
#             plt.hist(dat, bins=40, range=fit_range, density=True,  label=f"ch {ch}")
#             x = np.linspace(*fit_range, 500)
#             plt.plot(x, norm.pdf(x, mu, sigma), label=f"Fit ch {ch}")
#             plt.legend()
#             plt.title(f"ΔLE Distribution (Channel {ch})")
#             plt.xlabel("ΔLE")
#             plt.ylabel("Density")
#             plt.show()

#     return z_positions, channel_stats


# df["barOzLE"], statsO = compute_z_positions_from_dLE(df, "hodoOdLE", z_min=-225, z_max=225, plot=True)

# df["barIzLE"], statsI = compute_z_positions_from_dLE(df, "hodoIdLE", z_min=-150, z_max=150, plot=True)



