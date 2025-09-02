import bpy
import bmesh
import numpy as np
import mathutils
from mathutils.bvhtree import BVHTree

# ========== Parameters ========== #
# Define the XY range (in world coordinates) and resolution of the height map.

# pipe
xmin, xmax = 0.0, 15.8   # Minimum and maximum X
ymin, ymax = 0.0, 24.0    # Minimum and maximum Y
grid_resolution = 0.1            # Number of samples along each axis

# /Users/leyangwen/Documents/Isaac/terrain_model/download/construction-site-rawscan/site.blend
xmin, xmax = -25.2861, 47.2317   # Minimum and maximum X
ymin, ymax = -19.5259, 7.5388    # Minimum and maximum Y
grid_resolution = 0.4      # Number of samples along each axis

# Define the Z value from which to cast rays downward.
ray_start_z = 100.0         # Starting Z (should be above your model)

# Output file name
output_filename = "height_map.npy"

# ========== Get the Active Mesh Object ==========
obj = bpy.context.active_object
if obj is None or obj.type != 'MESH':
    raise RuntimeError("Please select a mesh object.")

# Evaluate the object so that modifiers are applied.
depsgraph = bpy.context.evaluated_depsgraph_get()
obj_eval = obj.evaluated_get(depsgraph)
mesh = obj_eval.to_mesh()

# Create a BMesh from the evaluated mesh
bm = bmesh.new()
bm.from_mesh(mesh)
bm.verts.ensure_lookup_table()
bm.faces.ensure_lookup_table()

# Build the BVHTree from the BMesh.
tree = BVHTree.FromBMesh(bm)

# Get the world transformation matrix.
world_matrix = obj.matrix_world

# ========== Create the Grid and Compute the Height Map ==========
x_vals = np.arange(xmin, xmax + grid_resolution, grid_resolution)
y_vals = np.arange(ymin, ymax + grid_resolution, grid_resolution)
height_map = np.full((len(y_vals), len(x_vals)), np.nan, dtype=np.float32)

for i, x in enumerate(x_vals):
    for j, y in enumerate(y_vals):
        origin = mathutils.Vector((x, y, ray_start_z))
        direction = mathutils.Vector((0, 0, -1))
        # Transform the ray's origin from object space to world space.
        origin_world = world_matrix @ origin

        hit = tree.ray_cast(origin_world, direction)
        if hit[0] is not None:
            hit_location = hit[0]
            height_map[j, i] = hit_location.z
        else:
            height_map[j, i] = np.nan

# ========== Save the Height Map ==========
np.save(output_filename, height_map)
print(f"Height map saved to {output_filename}")

# Cleanup: free the BMesh and evaluated mesh
bm.free()
obj_eval.to_mesh_clear()