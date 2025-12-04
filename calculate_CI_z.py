import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import transforms
import pandas as pd
import json
import uproot
import awkward as ak
from scipy.stats import norm, uniform
from scipy.signal import fftconvolve
from scipy.optimize import curve_fit
plt.style.use("asacusa.mplstyle")

root_path = "~/Documents/Hodoscope/cern_data/2025_Data/output_000265.root"
root_path = "~/Documents/Hodoscope/cern_data/2025_Data/output_001825_6567.root" 

tree = uproot.open(root_path)["RawEventTree"]

branches = ["eventID", "hodoODsLE", "hodoOUsLE", "hodoIDsLE", "hodoIUsLE"]

rdf = pd.DataFrame()

for batch in tree.iterate(filter_name=branches, library="ak", step_size=50000):
    df = ak.to_dataframe(batch, how="outer")
    df["hodoOdLE"] = df.hodoOUsLE - df.hodoODsLE
    df["hodoIdLE"] = df.hodoIUsLE - df.hodoIDsLE
    df["channel"] = df.index.get_level_values("subentry")
    df = df[df.hodoOdLE.notna() | df.hodoIdLE.notna()]

    rdf = pd.concat([rdf, df[["eventID", "channel", "hodoOdLE", "hodoIdLE"]]], ignore_index=True)
    print(len(rdf))


# rdf = ak.to_dataframe(tree.arrays(filter_name=branches, library="ak"), how="outer")


# rdf["hodoOdLE"] = rdf.hodoOUsLE - rdf.hodoODsLE
# rdf["hodoIdLE"] = rdf.hodoIUsLE - rdf.hodoIDsLE

def gauss_offset(x, mu, width, sigma, A=1):
    return A+norm.pdf(x, loc=mu, scale=sigma)

def convolved_pdf(x, mu, width, sigma, A=1, B=0):
    # Define range for the convolution
    grid_min = mu - width/2 - 5*sigma
    grid_max = mu + width/2 + 5*sigma
    grid = np.linspace(grid_min, grid_max, 5000)
    dx = grid[1] - grid[0]

    # Define PDFs on grid
    uniform_pdf = uniform.pdf(grid, loc=mu - width/2, scale=width)
    normal_pdf = norm.pdf(grid, loc=mu, scale=sigma)

    # Convolve
    conv = fftconvolve(uniform_pdf, normal_pdf, mode='same') * dx
    conv /= np.trapz(conv, grid)

    # Interpolate to return values at x
    return B+A*np.interp(x, grid, conv)

def convolved_cdf(mu, width, sigma, grid=None):
    if grid is None:
        grid = np.linspace(mu - width/2 - 5*sigma, mu + width/2 + 5*sigma, 5000)
    dx = grid[1] - grid[0]

    # Compute PDF
    uniform_pdf = uniform.pdf(grid, loc=mu - width/2, scale=width)
    normal_pdf = norm.pdf(grid, loc=mu, scale=sigma)
    conv_pdf = fftconvolve(uniform_pdf, normal_pdf, mode='same') * dx
    conv_pdf /= np.trapz(conv_pdf, grid)

    # Compute CDF
    conv_cdf = np.cumsum(conv_pdf) * dx
    return grid, conv_cdf

def get_confidence_interval(mu, width, sigma, level=0.95):
    grid, cdf = convolved_cdf(mu, width, sigma)

    lower_p = (1 - level) / 2
    upper_p = 1 - lower_p

    lower = np.interp(lower_p, cdf, grid)
    upper = np.interp(upper_p, cdf, grid)

    return [lower, upper]


for ch in range(32):
    hist = plt.hist(rdf.hodoOdLE[(rdf.hodoOdLE > -20) & (rdf.hodoOdLE < 20) & (rdf.channel == ch)], 100, range=(-20,20), density=True)
    x = hist[1][:-1]+(hist[1][1]-hist[1][0])/2
    # Step 3: Fit the model to the data
    p0 = [0, 10, 0.5, 1, 0]  # initial guesses: mu, width, sigma
    popt, pcov = curve_fit(convolved_pdf, x, hist[0], p0=p0, maxfev=5000)
    # mu_fit, width_fit, sigma_fit = popt
    perr = np.sqrt(np.diag(pcov))  # standard errors
    plt.plot(hist[1], convolved_pdf(hist[1], *popt), 'r-')
    plt.show()
    print(popt)


def compute_z_positions_from_dLE(df, dLE_col, num_channels=32, fit_range=(-20, 20), z_min=-225, z_max=225, ci_level=0.95, plot=False):
    """
    Compute z-positions based on delta LE values (from bar ends) for each channel.

    Parameters:
        df (DataFrame): Input DataFrame with dLE values.
        dLE_col (str): Name of the delta LE column (e.g. 'hodoOdLE' or 'hodoIdLE').
        channel_col (str): Column indicating the bar/channel number.
        fit_range (tuple): Range to include for Gaussian fit.
        z_min (float): Minimum physical z coordinate for mapping.
        z_max (float): Maximum physical z coordinate for mapping.
        ci_level (float): Confidence interval level (e.g. 0.95).
        plot (bool): Whether to generate and show plots for each channel.
    
    Returns:
        z_positions (np.array): Array of z values for each row in df (same order).
        channel_stats (dict): Dictionary with mean, sigma, CI per channel.
    """
    z_positions = np.full(len(df), np.nan)
    channel_stats = {
        "mean": np.full(num_channels, np.nan),
        "sigma": np.full(num_channels, np.nan),
        "ci": np.full((num_channels, 2), np.nan),
    }

    for ch in range(num_channels):
        mask = (df["channel"] == ch)
        dat = df.loc[mask, dLE_col]
        dat = dat[(dat > fit_range[0]) & (dat < fit_range[1])]

        if len(dat) < 10:
            continue  # Skip if not enough data to fit

        mu, sigma = norm.fit(dat)
        ci = norm.interval(ci_level, loc=mu, scale=sigma)

        # Store fit stats
        channel_stats["mean"][ch] = mu
        channel_stats["sigma"][ch] = sigma
        channel_stats["ci"][ch] = ci

        # Map to z
        dLE_vals = df.loc[mask, dLE_col]
        z_vals = (dLE_vals - ci[0]) / (ci[1] - ci[0]) * (z_max - z_min) + z_min
        z_positions[mask] = z_vals

        if plot:
            plt.hist(dat, bins=80, range=fit_range, density=True,  label=f"ch {ch}")
            x = np.linspace(*fit_range, 500)
            plt.plot(x, norm.pdf(x, mu, sigma), label=f"Fit ch {ch}")
            plt.legend()
            plt.title(f"ΔLE Distribution (Channel {ch})")
            plt.xlabel("ΔLE")
            plt.ylabel("Density")
            plt.show()

    return z_positions, channel_stats


def compute_z_positions_from_dLE(df, dLE_col, num_channels=32, fit_range=(-20, 20), z_min=-225, z_max=225, ci_level=0.95, plot=False):
    """
    Compute z-positions based on delta LE values (from bar ends) for each channel.

    Parameters:
        df (DataFrame): Input DataFrame with dLE values.
        dLE_col (str): Name of the delta LE column (e.g. 'hodoOdLE' or 'hodoIdLE').
        channel_col (str): Column indicating the bar/channel number.
        fit_range (tuple): Range to include for Gaussian fit.
        z_min (float): Minimum physical z coordinate for mapping.
        z_max (float): Maximum physical z coordinate for mapping.
        ci_level (float): Confidence interval level (e.g. 0.95).
        plot (bool): Whether to generate and show plots for each channel.
    
    Returns:
        z_positions (np.array): Array of z values for each row in df (same order).
        channel_stats (dict): Dictionary with mean, sigma, CI per channel.
    """
    z_positions = np.full(len(df), np.nan)
    channel_stats = {
        "mean": np.full(num_channels, np.nan),
        "sigma": np.full(num_channels, np.nan),
        "width": np.full(num_channels, np.nan),
        "A": np.full(num_channels, np.nan),
        "B":np.full(num_channels, np.nan),
        "ci": np.full((num_channels, 2), np.nan),
    }

    for ch in range(num_channels):
        mask = (df["channel"] == ch)
        dat = df.loc[mask, dLE_col]
        dat = dat[(dat > fit_range[0]) & (dat < fit_range[1])]

        if len(dat) < 10:
            continue  # Skip if not enough data to fit

        hist_vals, bin_edges = np.histogram(dat, bins=80, range=fit_range, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2


        # Fit using the convolved PDF
        try:
            p0 = [np.mean(dat), 6, 0.5, 1, 0]  # initial guess: mu, width, sigma
            popt, pcov = curve_fit(
                convolved_pdf,
                bin_centers, hist_vals, p0=p0, maxfev=5000
            )
            mu, width, sigma, A, B = popt
        except RuntimeError:
            continue  # skip this channel if fit fails

        # Confidence interval using normal approx of convolved peak
        ci = get_confidence_interval(mu, width, sigma, level=ci_level) #norm.interval(ci_level, loc=mu, scale=sigma)

        # Store fit stats
        channel_stats["mean"][ch] = mu
        channel_stats["sigma"][ch] = sigma
        channel_stats["width"][ch] = width
        channel_stats["A"][ch] = A
        channel_stats["B"][ch] = B
        channel_stats["ci"][ch] = ci

        # Map to z
        dLE_vals = df.loc[mask, dLE_col]
        z_vals = (dLE_vals - ci[0]) / (ci[1] - ci[0]) * (z_max - z_min) + z_min
        z_positions[mask] = z_vals

        if plot:
            plt.hist(dat, bins=80, range=fit_range, density=True,  label=f"ch {ch}")
            x = np.linspace(*fit_range, 500)
            # plt.plot(x, norm.pdf(x, mu, sigma), label=f"Fit ch {ch}")
            plt.plot(x, convolved_pdf(x, mu, width, sigma, A, B), label=rf"Fit ch {ch} $\sigma$ = {sigma:.2f} mm")
            plt.axvline(ci[0], color="gray")
            plt.axvline(ci[1], color="gray")
            plt.legend()
            plt.title(f"ΔLE Distribution (Channel {ch})")
            plt.xlabel("ΔLE")
            plt.ylabel("Density")
            plt.show()

    return z_positions, channel_stats

compute_z_positions_from_dLE(rdf, "hodoOdLE", z_min=-225, z_max=225, ci_level=0.99, plot=True)

rdf["barOzLE"], statsO = compute_z_positions_from_dLE(rdf, "hodoOdLE", z_min=-225, z_max=225, ci_level=0.99, plot=True)

rdf["barIzLE"], statsI = compute_z_positions_from_dLE(rdf, "hodoIdLE", z_min=-150, z_max=150, ci_level=0.99, plot=True)


def make_json_serializable(stats_dict):
    return {
        key: value.tolist() if isinstance(value, np.ndarray) else value
        for key, value in stats_dict.items()
    }

channel_stats = {"outer bars": make_json_serializable(statsO), 
                 "inner bars": make_json_serializable(statsI)}


# Save to file
with open("geometry_files/CI_z.json", "w") as f:
    json.dump(channel_stats, f, indent=2)

