import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import cm

plt.style.use("asacusa.mplstyle")

def get_oct(width_mm):
    # Parameters
    width_mm = width_mm + 26
    n = 8
    # Circumscribed radius so bounding box = width_mm
    R = width_mm / 2.0
    # Base angles
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    # Rotate so top/bottom edges are horizontal (22.5° = pi/8)
    rotation = np.pi / 8
    angles_rot = angles + rotation
    # Vertices
    vertices = np.column_stack((R * np.cos(angles_rot),
                                R * np.sin(angles_rot)))
    # Close polygon
    return np.vstack([vertices, vertices[0]])

def get_rect(width_mm, length_mm, orient="horizontal"):

    width_mm = width_mm + 8

    vertices = np.array([[-width_mm/2, -length_mm/2], [width_mm/2, -length_mm/2], [width_mm/2, length_mm/2], [-width_mm/2, length_mm/2], [-width_mm/2, -length_mm/2]])

    return vertices

def get_circle(radius_mm):

    radius_mm = radius_mm - 4

    theta = np.linspace(0, 2*np.pi, 100)
    x = radius_mm * np.cos(theta)
    y = radius_mm * np.sin(theta)

    return np.column_stack((x, y))


def plot_events(clustered_hits, lines_df, events, eps=None, xlim=(-200, 200), ylim=(-200, 200), zlim=(-250, 250), save=False, title=""):
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

        cusp_nr = clustered_hits[clustered_hits.event == ev].cuspRunNumber.unique()[0]
        mix = ["mixing" if clustered_hits[clustered_hits.event == ev].mixGate.unique()[0] else ""]
        time = clustered_hits[clustered_hits.event == ev].fpgaTimeTag.unique()[0]*1e-9
        if np.isnan(time):
            time_str = ""
        else:
            time_str = f"at {time:.2f} s"

        if event_hits.empty:
            print(f"Skipping event {ev} (no hits).")
            continue

        # Normalize track_id to [0, 1] for colormap
        track_ids = event_hits.track_id.values
        norm = Normalize(vmin=np.nanmin(track_ids), vmax=np.nanmax(track_ids))
        cmap = cm.viridis

        outer_oct = get_oct(350.0)
        inner_oct = get_oct(200.0)
        inner_rect = get_rect(200.0, 300.0)
        outer_rect = get_rect(350.0, 450.0)
        circle = get_circle(45.0)

        fig, ax = plt.subplots(figsize=(8, 8))

        # --- XY plane ---
        ax.scatter(event_hits.x, event_hits.y, c=cmap(norm(track_ids)), s=20, zorder=3)
        ax.errorbar(event_hits.x, event_hits.y, xerr=event_hits.dx, yerr=event_hits.dy,
                    fmt="none", ecolor="gray", elinewidth=1, zorder=1)


        ax.plot(outer_oct[:,0], outer_oct[:,1], lw=5, color="gray", zorder=2, alpha=0.1)
        ax.plot(inner_oct[:,0], inner_oct[:,1], lw=5, color="gray", zorder=2, alpha=0.1)
        ax.plot(circle[:,0], circle[:,1], lw=5, color="gray", zorder=2, alpha=0.1)

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
        ax_zy.errorbar(event_hits.z_used, event_hits.y, xerr=event_hits.dz_used, yerr=event_hits.dy,
                       fmt="none", ecolor="gray", elinewidth=1, zorder=1)
        ax_zy.plot(outer_rect[:,1], outer_rect[:,0], lw=5, color="gray", zorder=2, alpha=0.1)
        ax_zy.plot(inner_rect[:,1], inner_rect[:,0], lw=5, color="gray", zorder=2, alpha=0.1)
        ax_zy.plot([0, 0], [-41, 41], lw=5, color="gray", zorder=2, alpha=0.1)
        ax_zy.set_xlim(*zlim)
        ax_zy.set_ylim(*ylim)
        ax_zy.set_xlabel("z [mm]")

        # --- XZ plane ---
        ax_xz.scatter(event_hits.x, event_hits.z_used, c=cmap(norm(track_ids)), s=20, zorder=3)
        ax_xz.errorbar(event_hits.x, event_hits.z_used, xerr=event_hits.dx, yerr=event_hits.dz_used,
                       fmt="none", ecolor="gray", elinewidth=1, zorder=1)
        
        ax_xz.plot(outer_rect[:,0], outer_rect[:,1], lw=5, color="gray", zorder=2, alpha=0.1)
        ax_xz.plot(inner_rect[:,0], inner_rect[:,1], lw=5, color="gray", zorder=2, alpha=0.1)
        ax_xz.plot([-41, 41], [0, 0], lw=5, color="gray", zorder=2, alpha=0.1)
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
        fig.suptitle(f"Event {ev} Run {cusp_nr} {mix[0]} {time_str}" + (f" - eps {eps}" if eps is not None else ""), fontsize=14)
        plt.tight_layout()
        if save:
            plt.savefig(title+f"_{ev}.png")
        plt.show()




from mpl_toolkits.mplot3d import Axes3D    # needed for older MPL
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# --- ellipse helper ---
def ellipse_3d(center, radii, normal, n_steps=60):
    center = np.asarray(center)
    normal = np.asarray(normal) / np.linalg.norm(normal)

    # Build orthonormal basis for plane perpendicular to normal
    if np.allclose(normal, [0, 0, 1]):
        u = np.array([1, 0, 0])
    else:
        u = np.cross(normal, [0, 0, 1])
        u /= np.linalg.norm(u)
    v = np.cross(normal, u)

    theta = np.linspace(0, 2*np.pi, n_steps)
    a, b = radii

    pts = [center + a*np.cos(t)*u + b*np.sin(t)*v for t in theta]
    return pts


# --- main function ---
def plot_tracks_3d(clustered_hits, lines_df, events, plot_whole_event=False,
                   eps=None,
                   xlim=(-200, 200),
                   ylim=(-200, 200),
                   zlim=(-250, 250),
                   cmap=plt.cm.tab10):
    """
    Plot one 3D track display per track for the selected event IDs.
    """

    for ev in events:
        event_hits = clustered_hits[clustered_hits.event == ev]
        event_lines = lines_df[lines_df.event == ev]

        if event_hits.empty:
            print(f"No hits for event {ev}")
            continue

        print(f"Plotting event {ev}: {event_hits.track_id.nunique()} tracks")

        if plot_whole_event:
            fig = plt.figure(figsize=(8, 8))
            ax = fig.add_subplot(111, projection="3d")

        # Loop through *tracks* inside the event
        for tid, hits in event_hits.groupby("track_id"):

            # Get the corresponding fitted line
            line = event_lines[event_lines.track_id == tid]
            if line.empty:
                continue
            line = line.iloc[0]

            # ------------- CREATE FIGURE --------------
            if not plot_whole_event:
                fig = plt.figure(figsize=(8, 8))
                ax = fig.add_subplot(111, projection="3d")

            normal = line.direction / np.linalg.norm(line.direction)

            # ------------- PLOT HITS + 3D ELLIPSES --------------
            for _, hit in hits.iterrows():
                # radii from uncertainties (customize as you wish)
                a = hit.dx
                b = hit.dy

                ell = ellipse_3d(
                    center=[hit.x, hit.y, hit.z_used],
                    radii=(a, b),
                    normal=normal
                )

                color = cmap(tid % 10)

                poly = Poly3DCollection([ell], alpha=0.25,
                                        facecolor=color,
                                        edgecolor='k',
                                        linewidth=0.3)
                ax.add_collection3d(poly)

                ax.scatter(hit.x, hit.y, hit.z_used,
                           color=color, s=16)

            # ------------- TRACK LINE --------------
            origin = line.origin
            direction = normal
            L = 250

            p1 = origin - L * direction
            p2 = origin + L * direction

            ax.plot(
                [p1[0], p2[0]],
                [p1[1], p2[1]],
                [p1[2], p2[2]],
                color=color,
                lw=2
            )

            # ------------- AXES LIMITS + LABELS --------------

            if not plot_whole_event:
                ax.set_xlim(*xlim)
                ax.set_ylim(*ylim)
                ax.set_zlim(*zlim)

                ax.set_xlabel("x [mm]")
                ax.set_ylabel("y [mm]")
                ax.set_zlabel("z [mm]")

                ax.set_title(
                    f"Event {ev} — Track {tid}" +
                    (f" (eps={eps})" if eps is not None else "")
                )

                ax.set_box_aspect([1, 1, 1])
                ax.view_init(azim=45, elev=30)

                plt.tight_layout()
                plt.show()
    
        if plot_whole_event:

            ax.set_title(
                f"Event {ev}" +
                (f" (eps={eps})" if eps is not None else "")
            )

            ax.set_box_aspect([1, 1, 1])
            ax.view_init(azim=90, elev=30, )

            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
            ax.set_zlim(*zlim)

            ax.set_xlabel("x [mm]")
            ax.set_ylabel("y [mm]")
            ax.set_zlabel("z [mm]")

            plt.tight_layout()
            plt.show()