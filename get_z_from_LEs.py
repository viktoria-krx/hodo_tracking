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
plt.style.use("asacusa.mplstyle")



def compute_z_positions_from_dLE(df, detector, outer_or_inner = "outer", num_channels=32, fit_range=(-20, 20), plot=False):
    """
    Compute z-positions based on delta LE values (from bar ends) for each channel.

    Parameters:
        df (DataFrame): Input DataFrame with dLE values.
        dLE_col (str): Name of the delta LE column (e.g. 'hodoOdLE' or 'hodoIdLE').
        outer_or_inner (str): 'outer' or 'inner' bars
        num_channels (int): Amount of channels.
        fit_range (tuple): Range to include for Gaussian fit.
        plot (bool): Whether to generate and show plots for each channel.
    
    Returns:
        z_positions (np.array): Array of z values for each row in df (same order).
    """
    if outer_or_inner == "outer":
         z_min=-225
         z_max=225
    elif outer_or_inner == "inner":
         z_min=-150
         z_max=150
    else:
         plot("Error: outer_or_inner must be 'outer' or 'inner'")
         exit

    with open("geometry_files/CI_z.json", "r") as f:
            stats = json.load(f)

    z_positions = np.full(len(df), np.nan)

    dLE_col = f"{detector}dLE"
    df[dLE_col] = df[f"{detector}UsLE"] - df[f"{detector}DsLE"]

    

    for ch in range(num_channels):
        mask = (df["channel"] == ch)
        dat = df.loc[mask, dLE_col]

        mu = stats[outer_or_inner+" bars"]["mean"][ch]
        sigma = stats[outer_or_inner+" bars"]["sigma"][ch]
        ci = stats[outer_or_inner+" bars"]["ci"][ch]

        # Map to z
        dLE_vals = df.loc[mask, dLE_col]
        z_vals = (dLE_vals - ci[0]) / (ci[1] - ci[0]) * (z_max - z_min) + z_min
        z_positions[mask] = z_vals

        if plot:
            plt.hist(z_positions, bins=40, range=(z_min, z_max), density=True,  label=f"ch {ch}")
            # x = np.linspace(*fit_range, 500)
            # plt.plot(x, norm.pdf(x, mu, sigma), label=f"Fit ch {ch}")
            plt.legend()
            plt.title(f"z Distribution (Channel {ch})")
            plt.xlabel("z in mm")
            plt.ylabel("Density")
            plt.show()

    return z_positions

# Oz = compute_z_positions_from_dLE(rdf, "hodoOdLE", "outer", ci_level=0.99, plot=False)

# Iz = compute_z_positions_from_dLE(rdf, "hodoIdLE", "inner", ci_level=0.99, plot=False)