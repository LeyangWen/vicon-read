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
        default="/Users/leyangwen/Documents/Isaac/terrain_model/download/pipeline-construction-site/height_map.npy",
                # "/Users/leyangwen/Documents/Isaac/motion_output/terrain_default/terrain/height_field_raw.npy",
        help="Path to the height field numpy file (default: height_field_raw.npy)",
    )
    parser.add_argument(
        "--walkable_file",
        type=str,
        default="/Users/leyangwen/Documents/Isaac/terrain_model/download/pipeline-construction-site/height_map_walkable.npy",
        # "/Users/leyangwen/Documents/Isaac/motion_output/terrain_default/terrain/walkable_field_raw.npy",
        help="Path to the walkable field numpy file (default: walkable_field_raw.npy)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    # Load height field and walkable field arrays from files
    height_field = np.load(args.height_file)
    walkable_field = np.load(args.walkable_file)

    # height_field = height_field   # height is in cm, convert to meters
    #
    # grid_resolution = 0.1  # each pixel is 0.05m
    # height_field = height_field / grid_resolution  # Convert to meters





    # height_field = height_field[500:-500, 500:-500]  # Crop the height field

    # Create a meshgrid for the height field
    rows, cols = height_field.shape
    X = np.arange(cols)
    Y = np.arange(rows)
    X, Y = np.meshgrid(X, Y)

    # Create a figure with two subplots: one for the height field and one for the walkable field
    fig = plt.figure(figsize=(14, 6))

    # 3D Surface Plot for Height Field
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    surf = ax1.plot_surface(X, Y, height_field, cmap='terrain', edgecolor='none')
    ax1.set_title("Height Field")
    ax1.set_xlabel("Y (dm)")
    ax1.set_ylabel("X (dm)")
    ax1.set_zlabel("Height (dm)")

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
    meter = 10
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
    ax2.set_xlabel("Y (dm)")
    ax2.set_ylabel("X (dm)")
    fig.colorbar(heatmap, ax=ax2, shrink=0.5, aspect=10)

    plt.tight_layout()
    plt.show()