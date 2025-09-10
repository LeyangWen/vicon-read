import argparse
import os.path
from Skeleton import *
import matplotlib
matplotlib.use('Qt5Agg')

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse
from scipy import ndimage


"""
First in blender scan run blender_to_heightmap_scripty.py
Then run this script to visualize the height and walkable field numpy files.
Put in Linux tokenhsi/data/assets/terrain/#####
change _construction.yaml config to use
"""
def parse_args():
    """
    Parse command-line arguments to specify the input file paths.
    """
    parser = argparse.ArgumentParser(
        description="Visualize terrain using height and walkable field numpy files."
    )

    parser.add_argument(
        "--height_file",
        type=str,
        default="/Users/leyangwen/Documents/Isaac/terrain_model/scan/vicon_lab/height_map.npy",
        # default="/Users/leyangwen/Documents/Isaac/terrain_model/download/pipeline-construction-site/height_map.npy",
        help="Path to the height field numpy file (default: height_field_raw.npy)",
    )
    parser.add_argument(
        "--grid_resolution",
        type=float,
        default="0.01",
        # default="0.2",
        help="Grid resolution used in the blender height feild generation (default: 0.01m per pixel",
    )
    # parser.add_argument(
    #     "--walkable_file",
    #     type=str,
    #     default="/Users/leyangwen/Documents/Isaac/terrain_model/download/pipeline-construction-site/height_map_walkable.npy",
    #     # "/Users/leyangwen/Documents/Isaac/motion_output/terrain_default/terrain/walkable_field_raw.npy",
    #     help="Path to the walkable field numpy file (default: walkable_field_raw.npy)",
    # )
    return parser.parse_args()

def max_slope(height_map: np.ndarray, grid_resolution):
    """
    Return a (H, W) array where each cell is the maximum slope ratio
    (abs(rise) / run) toward its N, S, E, W neighbors.
    NaNs are preserved where undefined or missing data.
    """
    h = height_map.astype(np.float32, copy=False)

    # Allow anisotropic spacing
    if isinstance(grid_resolution, (tuple, list, np.ndarray)):
        dx, dy = float(grid_resolution[0]), float(grid_resolution[1])
    else:
        dx = dy = float(grid_resolution)

    # Neighbor heights via roll
    nbh_n = np.roll(h, -1, axis=0)  # +Y (north)
    nbh_s = np.roll(h,  1, axis=0)  # -Y (south)
    nbh_e = np.roll(h, -1, axis=1)  # +X (east)
    nbh_w = np.roll(h,  1, axis=1)  # -X (west)

    # Raw slope ratios
    s_n = np.abs(nbh_n - h) / dy
    s_s = np.abs(nbh_s - h) / dy
    s_e = np.abs(nbh_e - h) / dx
    s_w = np.abs(nbh_w - h) / dx

    # Invalidate wraparound edges
    s_n[-1, :] = np.nan
    s_s[ 0, :] = np.nan
    s_e[:, -1] = np.nan
    s_w[:,  0] = np.nan

    # Invalidate where source or neighbor is NaN
    nan_h = np.isnan(h)
    s_n[np.isnan(nbh_n) | nan_h] = np.nan
    s_s[np.isnan(nbh_s) | nan_h] = np.nan
    s_e[np.isnan(nbh_e) | nan_h] = np.nan
    s_w[np.isnan(nbh_w) | nan_h] = np.nan

    # Max across directions (ignore NaNs)
    stacked = np.stack([s_n, s_s, s_e, s_w], axis=0)
    max_slope_f = np.nanmax(stacked, axis=0)

    # If all four are NaN at a cell, keep NaN
    all_nan = np.isnan(s_n) & np.isnan(s_s) & np.isnan(s_e) & np.isnan(s_w)
    max_slope_f[all_nan] = np.nan

    return max_slope_f

if __name__ == "__main__":
    args = parse_args()
    # Load height field and walkable field arrays from files
    height_field = np.load(args.height_file)

    # flip by x axis
    height_field = height_field[:, ::-1]
    min_z = np.nanmin(height_field)
    height_field = height_field - min_z  # set lowest point to be 0


    # height_field = height_field   # height is in cm, convert to meters
    #
    grid_resolution = 0.01  # each pixel is 0.05m
    # height_field = height_field / grid_resolution  # Convert to meters

    max_slope_field = max_slope(height_field, grid_resolution=grid_resolution)
    walkable_field = (max_slope_field > np.tan(np.radians(70))).astype(int)  # walkable if slope < 15 degrees


    is_nan = max_slope_field == np.nan
    walkable_field[is_nan] = 1

    too_high = height_field > 0.6
    walkable_field[too_high] = 1

    way_too_high = height_field > 2.1
    height_field[way_too_high] = np.nan



    ## save the processed height and walkable field
    walkable_file_name = args.height_file.replace('.npy', '_walkable.npy')
    np.save(walkable_file_name, walkable_field)

    np.save(args.height_file.replace('.npy', f'_processed_{grid_resolution}.npy'), height_field)

    # Create a meshgrid for the height field
    rows, cols = height_field.shape
    x_vals = np.arange(0, (cols)*grid_resolution, grid_resolution)
    y_vals = np.arange(0, (rows)*grid_resolution, grid_resolution)
    X, Y = np.meshgrid(x_vals, y_vals)

    # Create a figure with two subplots: one for the height field and one for the walkable field
    fig = plt.figure(figsize=(14, 6))

    # 3D Surface Plot for Height Field
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    surf = ax1.plot_surface(X, Y, height_field, cmap='terrain', edgecolor='none')
    ax1.set_title("Height Field")
    ax1.set_xlabel("Y (m)")
    ax1.set_ylabel("X (m)")
    ax1.set_zlabel("Z (m)")

    # Calculate the range of x, y, and z
    x_range = X.max() - X.min()
    y_range = Y.max() - Y.min()
    z_range = height_field.max() - height_field.min()
    max_range = max(x_range, y_range, z_range)

    # Center the axes and set limits to ensure equal scaling
    x_mid = (X.max() + X.min()) / 2
    y_mid = (Y.max() + Y.min()) / 2
    z_mid = (height_field.max() + height_field.min()) / 2

    ax1.set_xlim(x_mid - max_range / 2, x_mid + max_range / 2)
    ax1.set_ylim(y_mid - max_range / 2, y_mid + max_range / 2)
    meter = 1
    ax1.set_zlim(0, 2 * meter)
    ax1.set_xlim(ax1.get_xlim()[::-1])

    # Set equal aspect ratio for x, y, z
    ax1.set_box_aspect([max_range, max_range, 2 * meter])  # Equal scaling for all axes
    ax1.set_zticks([0, 1 * meter, 2 * meter])
    fig.colorbar(surf, ax=ax1, shrink=0.5, aspect=10)


    # 2D Heatmap for Height Field

    walkable_field = ndimage.binary_dilation(walkable_field, iterations=3).astype(int) # this is done in TokenHSI when reading
    # Heatmap for Walkable Field
    ax2 = fig.add_subplot(1, 2, 2)
    heatmap = ax2.imshow(walkable_field, cmap='gray', origin='lower')  #, vmin=0, vmax=1)
    # flip plot by y axis
    ax2.set_xlim(ax2.get_xlim()[::-1])
    # verticle axis to right
    ax2.yaxis.tick_right()
    ax2.set_title("Walkable Field")
    ax2.set_xlabel("Y (m)")
    ax2.set_ylabel("X (m)")
    fig.colorbar(heatmap, ax=ax2, shrink=0.5, aspect=10)

    plt.tight_layout()
    plt.show()