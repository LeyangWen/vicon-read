import numpy as np
import time
from sklearn.neighbors import KDTree
import pickle

def format_large_number(num):
    """Format large numbers as ##k for better readability."""
    if num >= 1000:
        return f"{num // 1000}k"
    return str(num)

def select_diverse_poses(poses, tolerance):
    """
    Selects a maximally diverse set of poses using KDTree.
    A pose is marked as covered only if ALL joints are within the tolerance.

    Parameters:
    - poses: np.ndarray of shape (N, J, 3), where N is the number of frames, J is the number of joints
    - tolerance: float, threshold for distinct poses in mm

    Returns:
    - List of indices representing the most diverse subset of poses.
    """
    start_time = time.time()
    N, J, _ = poses.shape  # (Frames, Joints, 3D)
    diverse_indices = []
    covered = np.zeros(N, dtype=bool)

    print(f"Starting selection for {format_large_number(N)} poses with {J} joints. Tolerance: {tolerance} mm")

    # Step 1: Build KDTree for each joint
    trees = []
    tree_build_start = time.time()
    for j in range(J):
        trees.append(KDTree(poses[:, j, :], metric="euclidean"))
    tree_build_time = time.time() - tree_build_start
    print(f"KDTree built for all joints in {tree_build_time:.2f} seconds.")

    # Step 2: Iterate through all poses
    last_checkpoint_time = time.time()
    avg_batch_time = 0
    batch_size = 10000

    for i in range(N):
        if not covered[i]:  # Select uncovered poses
            diverse_indices.append(i)

            # Step 3: Find neighbors for each joint and keep only those common in all joints
            neighbors_per_joint = [set(trees[j].query_radius([poses[i, j, :]], r=tolerance)[0]) for j in range(J)]
            common_neighbors = set.intersection(*neighbors_per_joint)  # Find common across all joints

            # Mark all common neighbors as covered
            for idx in common_neighbors:
                covered[idx] = True

        # Print progress every batch_size iterations
        if i % batch_size == 0 and i > 0:
            current_time = time.time()
            batch_time = current_time - last_checkpoint_time  # Time for last batch
            total_elapsed_time = current_time - start_time  # Total elapsed time

            # Estimate remaining time
            processed_batches = i // batch_size
            if processed_batches > 0:
                avg_batch_time = (total_elapsed_time / processed_batches)  # Average batch time
                remaining_batches = (N - i) // batch_size
                estimated_remaining_time = avg_batch_time * remaining_batches
            else:
                estimated_remaining_time = 0  # Default to 0 if no estimation available yet

            print(f"Processed {format_large_number(i)}/{format_large_number(N)} poses. "
                  f"Selected {format_large_number(len(diverse_indices))} diverse poses. "
                  f"Batch Time: {batch_time:.2f}s, Total Time: {total_elapsed_time:.2f}s, "
                  f"Est. Remaining: {estimated_remaining_time:.2f}s.")

            last_checkpoint_time = current_time  # Reset checkpoint timer

    total_time = time.time() - start_time
    print(f"\nSelection completed. Total diverse poses selected: {format_large_number(len(diverse_indices))}")
    print(f"Total execution time: {total_time:.2f} seconds.")

    return diverse_indices

def tests():
    ## Example test
    ## Test with a small set of poses with known answers
    poses = np.array([
        [[0, 0, 0], [1, 1, 1], [2, 2, 2]],  # Pose 0
        [[0, 0, 0], [1, 1.1, 1], [2, 2, 2.1]],  # Pose 1 (same as Pose 1)
        [[10, 10, 10], [11, 11, 11], [12, 12, 12]],  # Pose 2 (different)
        [[0, 2, 0], [1, 1, 1], [2, 2, 2.1]],  # Pose 3 (different)
        [[0, 2, 0], [1, 1, 1], [2, 2, 2.1]],  # Pose 4 (same as 3)
    ])
    diverse_pose_indices = select_diverse_poses(poses, tolerance=1)
    print(f"Selected diverse poses indices: {diverse_pose_indices}")
    print(f"correct answer: [0, 2, 3]")

    ## Speed test
    N, J = 1000, 17  # Simulating 100k poses, 17 joints
    poses = np.random.rand(N, J, 3) * 1800  # Assume 3D joint locations in mm

    # Compute maximally diverse poses for 100mm tolerance
    diverse_pose_indices = select_diverse_poses(poses, tolerance=100)

    print(f"\nFinal number of diverse poses: {format_large_number(len(diverse_pose_indices))}")

    ## Correctness test
    # Check if any two selected poses are within tolerance
    for i in range(len(diverse_pose_indices)):
        for j in range(i + 1, len(diverse_pose_indices)):
            if np.linalg.norm(poses[diverse_pose_indices[i]] - poses[diverse_pose_indices[j]]) < 100:
                print(f"Error: Poses at indices {diverse_pose_indices[i]} and {diverse_pose_indices[j]} are too similar.")
                break

if __name__ == "__main__":
    file = r'/Users/leyangwen/Documents/Pose/paper/VEHS_6D_downsample2_keep1.pkl'
    with open(file, "rb") as f:
        data = pickle.load(f)
    for key in data.keys():
       data
    poses = data['joint3d_image']
    tolerance = 100
