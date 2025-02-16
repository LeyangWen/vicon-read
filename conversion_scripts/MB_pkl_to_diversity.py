import numpy as np
import time
from sklearn.neighbors import KDTree
import pickle
import argparse

def format_large_number(num):
    """Format large numbers as ##k for better readability."""
    if num >= 1000:
        return f"{num // 1000}k"
    return str(num)

def max_joint_distance(pose1, pose2):
    """
    Compute the maximum joint-wise Euclidean distance between two poses.
    """
    # i = 3000
    # pose1, pose2 = poses[i], poses[i+5]
    return np.max(np.linalg.norm(pose1 - pose2, axis=1))  # Max over all joints

def preprocess_poses(poses, tolerance=100, sequential_skip=300):
    """
    Preprocesses the poses by removing sequential similar pose
    """
    N, J, _ = poses.shape  # (Frames, Joints, 3D)
    processed_indices = [0]
    i = 0

    while i < N:
        processed_indices.append(i)
        for plus_i in range(1, sequential_skip):
            if i + plus_i < N:
                joint_dist =  max_joint_distance(poses[i], poses[i + plus_i])
                if joint_dist > tolerance:
                    break
        i += plus_i
        # print(f"i: {i}, +i: {plus_i}, joint_dist: {joint_dist}")
    frame_length = len(processed_indices)
    print(f"Reduced from {N} to {frame_length} frames, {frame_length/N*100:.2f}%")
    preprocess_poses = poses[processed_indices]
    return preprocess_poses, processed_indices

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
    # print(f"KDTree built for all joints in {tree_build_time:.2f} seconds.")

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

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--GT_file", type=str, default=r'/Users/leyangwen/Documents/Pose/paper/VEHS_6D_downsample2_keep1_37_v1_diversity.pkl', help="Path to the GT file.")
    parser.add_argument("--tolerance", type=int, default=100, help="mm")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    with open(args.GT_file, "rb") as f:
        data = pickle.load(f)

    store_results = {}
    for key in data.keys():
        poses = data[key]['joints_2.5d_image']
        sources = data[key]['source']
        split_idx = []
        ## find the index of the first unique source
        subject_name = sources[0].split('\\')[4]
        for i, source in enumerate(sources):
            if subject_name not in source:
                split_idx.append(i)
                subject_name = source.split('\\')[4]

        split_pose = np.split(poses, split_idx)
        for i, pose in enumerate(split_pose):
            N = pose.shape[0]
            print(f"Processing {key}-sub{i}: {subject_name} with {pose.shape[0]} frames")
            small_pose, _ = preprocess_poses(pose, tolerance=args.tolerance, sequential_skip=300)
            diverse_pose_indices = select_diverse_poses(small_pose, args.tolerance)
            print(f"Selected {len(diverse_pose_indices)} diverse pose from {N} frames, {len(diverse_pose_indices)/N*100:.2f}%")
            print("#" * 20)