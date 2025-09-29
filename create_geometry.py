import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import json

hodo_colors = ["#33638d", "#482677", "#bc4174", "#d35040", "#fba100", "#fde725", "#95d840", "#3cbb75"]

def rotate_vector(vec, angle_deg):
    """Rotate 2D vector by angle."""
    angle_rad = np.radians(angle_deg)
    rot_matrix = np.array([
        [np.cos(angle_rad), -np.sin(angle_rad)],
        [np.sin(angle_rad),  np.cos(angle_rad)]
    ])
    return rot_matrix @ vec

def generate_bars(bar_width, bar_thickness, bar_length, diameter):

    bars = []
    channel_id = 0
    # bar_width = 35.0  # mm
    # bar_thickness = 5.0
    # bar_length = 450.0  # mm (along Z)
    
    R = diameter/2  # Radius from center to center of side (adjust as needed)
    spacing = 40.0  # Distance between bar centers along each side

    base_side_centers = np.array([
        [R, -bar_width*1.5],  # base position (bottom of side 0)
        [R, -bar_width*0.5],
        [R, bar_width*0.5],
        [R, bar_width*1.5]
        ])

    for side in range(8):
        angle = side * 45  # degrees
        # Rotate each base position to the correct octagon side
        for local_idx, xy in enumerate(base_side_centers):
            xy_rot = rotate_vector(xy, angle)
            position = [round(xy_rot[0], 2), round(xy_rot[1], 2), 0.0]  # flat in Z for now
            bars.append({
                "channel_id": channel_id,
                "length": bar_length,
                "width": bar_width,
                "thickness": bar_thickness,
                "position": position,
                "rotation": angle
            })
            channel_id += 1

    return bars

def generate_tiles(tile_width, tile_thickness, tile_length, diameter):
    
    tiles = []
    channel_id = 0
    R = diameter / 2  # radius from center to face center

    # Central XY position of bar on side 0 (before rotation)
    base_xy = np.array([R, 0.0])

    # Z stacking parameters
    z_start = - 7.5*tile_width
    z_spacing = tile_width  # evenly fill bar_length

    for side in range(8):
        angle = side * 45  # degrees
        # Rotate base XY position to get location for this side
        xy_rot = rotate_vector(base_xy, angle)
        x, y = round(xy_rot[0], 2), round(xy_rot[1], 2)

        for i in range(15):
            z = round(z_start + i * tile_width + tile_width / 2, 2)  # center each bar
            tiles.append({
                "channel_id": channel_id,
                "length": tile_width,  # individual short bar
                "width": tile_length,
                "thickness": tile_thickness,
                "position": [x, y, z],
                "rotation": angle  # rotation around Z
            })
            channel_id += 1

    return tiles

def generate_bgo():
    bgo = []
    df = pd.read_csv("~/Documents/Hodoscope/blt_frontend/hodo_daq_control/config/bgo_geom.csv")
    for ch in df.Channel:
        bgo.append({
            "channel_id": int(ch),
            "length": 5,
            "width": 10,
            "thickness": 5,
            "position": [float(df.x[ch]), float(df.y[ch]), 0],
            "rotation": 90
        })
    return bgo

outer_diam = 350
inner_diam = 200

outer_bars = generate_bars(35.0, 5.0, 450.0, outer_diam+5)
inner_bars = generate_bars(20.0, 5.0, 300.0, inner_diam+5)
outer_tiles = generate_tiles(30.0, 5.0, 129.0, outer_diam-8)
inner_tiles = generate_tiles(20.0, 5.0, 84.0, inner_diam+5+13)
bgo = generate_bgo()

# Wrap into a structured JSON dictionary
geometry = {
    "outer_bars": outer_bars,
    "inner_bars": inner_bars,
    "outer_tiles": outer_tiles,
    "inner_tiles": inner_tiles,
    "bgo": bgo
}

# Save to file
with open("geometry.json", "w") as f:
    json.dump(geometry, f, indent=2)


with open("geometry.json", "r") as f:
        data = json.load(f)



fig, ax = plt.subplots(figsize=(8, 8))
ax.set_aspect('equal')
ax.set_title("Hodoscope XY Plane")
ax.set_xlabel("X (mm)")
ax.set_ylabel("Y (mm)")

for layer in data:
    for scint in data[layer]:
        x, y, _ = scint["position"]
        angle = scint["rotation"]
        if layer == "bgo":
            col = "gray"
        else:
            col = hodo_colors[angle//45]
        # Create a rectangle centered at (x, y) and rotated
        rect = patches.Rectangle(
            (-scint["thickness"]/2, -scint["width"]/2),
            scint["thickness"], scint["width"],
            angle=0.0,
            linewidth=0,
            alpha=1,
            color = col
        )

        # Transformation: rotate and shift
        t = plt.matplotlib.transforms.Affine2D().rotate_deg(angle).translate(x, y) + ax.transData
        rect.set_transform(t)

        ax.add_patch(rect)

        # Draw channel ID at center
        if layer[6:] != "tiles":
            ax.text(x, y, f'{scint["channel_id"]}', ha='center', va='center', fontsize=6)

# Auto-scale
ax.autoscale()
plt.grid(True)
plt.show()



