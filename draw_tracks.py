import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import cm

plt.style.use("asacusa.mplstyle")

def plot_events(clustered_hits, lines_df, events, eps=None, xlim=(-200, 200), ylim=(-200, 200), zlim=(-250, 250)):
    """
    Plot 3-view event displays (XY, XZ, ZY) for the given event IDs.

    Parameters
    ----------
    clustered_hits : pd.DataFrame
        DataFrame containing hit positions, errors, and track_id.
    lines_df : pd.DataFrame
        DataFrame containing fitted track lines and vertices per event.
    events : list or array-like
        List of event IDs to plot.
    eps : float, optional
        Epsilon (clustering) parameter to include in title.
    xlim, ylim, zlim : tuple, optional
        Axis limits for x, y, z.
    """

    for ev in events:
        event_hits = clustered_hits[clustered_hits.event == ev]
        event_lines = lines_df[lines_df.event == ev]

        if event_hits.empty:
            print(f"Skipping event {ev} (no hits).")
            continue

        # Normalize track_id to [0, 1] for colormap
        track_ids = event_hits.track_id.values
        norm = Normalize(vmin=np.nanmin(track_ids), vmax=np.nanmax(track_ids))
        cmap = cm.viridis

        fig, ax = plt.subplots(figsize=(8, 8))

        # --- XY plane ---
        ax.scatter(event_hits.x, event_hits.y, c=cmap(norm(track_ids)), s=20, zorder=3)
        ax.errorbar(event_hits.x, event_hits.y, xerr=event_hits.dx, yerr=event_hits.dy,
                    fmt="none", ecolor="gray", elinewidth=1, zorder=1)

        ax.set_aspect('equal')
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_ylabel("y [mm]")

        # --- Create shared axes for XZ and ZY ---
        divider = make_axes_locatable(ax)
        ax_xz = divider.append_axes("bottom", 4, pad=0.1, sharex=ax)
        ax_zy = divider.append_axes("right", 4, pad=0.1, sharey=ax)

        ax.xaxis.set_tick_params(labelbottom=False)
        ax_zy.yaxis.set_tick_params(labelleft=False)

        # --- ZY plane ---
        ax_zy.scatter(event_hits.z_used, event_hits.y, c=cmap(norm(track_ids)), s=20, zorder=3)
        ax_zy.errorbar(event_hits.z_used, event_hits.y, xerr=event_hits.dz, yerr=event_hits.dy,
                       fmt="none", ecolor="gray", elinewidth=1, zorder=1)
        ax_zy.set_xlim(*zlim)
        ax_zy.set_ylim(*ylim)
        ax_zy.set_xlabel("z [mm]")

        # --- XZ plane ---
        ax_xz.scatter(event_hits.x, event_hits.z_used, c=cmap(norm(track_ids)), s=20, zorder=3)
        ax_xz.errorbar(event_hits.x, event_hits.z_used, xerr=event_hits.dx, yerr=event_hits.dz,
                       fmt="none", ecolor="gray", elinewidth=1, zorder=1)
        ax_xz.set_xlim(*xlim)
        ax_xz.set_ylim(*zlim)
        ax_xz.set_xlabel("x [mm]")
        ax_xz.set_ylabel("z [mm]")

        # --- Vertices ---
        if "Vx" in event_lines.columns and event_lines.Vx.notna().any():
            ax.scatter(event_lines.Vx, event_lines.Vy, c="C1", s=100, marker="x")
            ax.errorbar(event_lines.Vx, event_lines.Vy, xerr=event_lines.Vx_sig, yerr=event_lines.Vy_sig, fmt="C1", capsize=5)
            ax_zy.scatter(event_lines.Vz, event_lines.Vy, c="C1", s=100, marker="x")
            ax_zy.errorbar(event_lines.Vz, event_lines.Vy, xerr=event_lines.Vz_sig, yerr=event_lines.Vy_sig, fmt="C1", capsize=5)
            ax_xz.scatter(event_lines.Vx, event_lines.Vz, c="C1", s=100, marker="x")
            ax_xz.errorbar(event_lines.Vx, event_lines.Vz, xerr=event_lines.Vx_sig, yerr=event_lines.Vz_sig, fmt="C1", capsize=5)

        # --- Track lines ---
        for _, line in event_lines.iterrows():
            origin = line.origin
            direction = line.direction / np.linalg.norm(line.direction)
            color = cmap(norm(line.track_id))

            L = 150  # extend both sides
            p1 = origin - direction * L
            p2 = origin + direction * L

            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=color, lw=2)
            ax_zy.plot([p1[2], p2[2]], [p1[1], p2[1]], color=color, lw=2)
            ax_xz.plot([p1[0], p2[0]], [p1[2], p2[2]], color=color, lw=2)

        # --- Titles ---
        fig.suptitle(f"Event {ev}" + (f" - eps {eps}" if eps is not None else ""), fontsize=14)
        plt.tight_layout()
        plt.show()