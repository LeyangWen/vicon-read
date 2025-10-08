# pip install cdflib matplotlib numpy
import cdflib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')
file_path = '/Users/leyangwen/Documents/Pose/H36M_info/Directions 1.54138969.cdf'

cdf = cdflib.CDF(file_path)

def all_vars(cdf):
    info = cdf.cdf_info()
    names = []
    # rVariables / zVariables may exist independently
    if getattr(info, "rVariables", None):
        names.extend(info.rVariables)
    if getattr(info, "zVariables", None):
        names.extend(info.zVariables)
    return names

def squeeze(a):
    return np.squeeze(np.array(a))

def to_T_J_3(A):
    """
    Convert any pose-like array (with one axis size==3) into (T, J, 3).
    Also handles 2D cases where one dimension is a multiple of 3.
    """
    A = squeeze(A)
    # 2D: (T, 3J) or (3J, T)
    if A.ndim == 2:
        h, w = A.shape
        if w % 3 == 0:        # (T, 3J)
            T, J = h, w // 3
            return A.reshape(T, J, 3)
        if h % 3 == 0:        # (3J, T)
            J, T = h // 3, w
            return A.reshape(J, 3, T).transpose(2, 0, 1)  # -> (T, J, 3)
        raise ValueError(f"2D but not 3-multiple: {A.shape}")
    # >=3D: find the axis with size==3, make it last dim
    if 3 in A.shape:
        ax3 = A.shape.index(3)
        A = np.moveaxis(A, ax3, -1)  # ... , 3
        # choose T as the largest remaining axis, J as the next largest
        rest = A.shape[:-1]
        if len(rest) == 1:
            # (T, 3) with J==1
            return A.reshape(rest[0], 1, 3)
        # sort remaining dims by size (desc)
        order = np.argsort(rest)[::-1]
        # bring the two largest to front as (T, J, 3), flatten others if exist
        A_ = A
        # permute so the two largest dims come first
        perm = list(order) + [len(rest)]  # last is the 3-axis
        A_ = np.transpose(A, axes=perm)
        T, J = A_.shape[0], A_.shape[1]
        # flatten any leftover dims into J
        if A_.ndim > 3:
            J = int(np.prod(A_.shape[1:-1]))
            A_ = A_.reshape(T, J, 3)
        return A_
    raise ValueError(f"No axis of size 3 in array of shape {A.shape}")

# Scan variables to find a pose-like tensor
pose_var = None
poses_TJ3 = None
candidates_info = []

for name in all_vars(cdf):
    try:
        arr = cdf.varget(name)
        A = squeeze(arr)
        candidates_info.append((name, A.shape))
        try:
            TJ3 = to_T_J_3(A)
            # Heuristic: need at least a few joints and frames
            if TJ3.shape[1] >= 10 and TJ3.shape[0] >= 1:
                pose_var = name
                poses_TJ3 = TJ3
                break
        except Exception:
            pass
    except Exception:
        pass


if poses_TJ3 is None:
    print("Could not auto-detect a pose variable. Variables and shapes I saw:")
    for n, s in candidates_info:
        print(f"  {n}: {s}")
    raise ValueError("No pose-like variable found (with an axis of size 3).")

print(f"Using variable: {pose_var}, shape (T, J, 3) = {poses_TJ3.shape}")

# First frame
first_pose = poses_TJ3[0]   # (J, 3)
J = first_pose.shape[0]

# Joint labels (index + name)
joint_names = [
    "0 Hips",
    "1 RightUpLeg",
    "2 RightLeg",
    "3 RightFoot",
    "4 RightToeBase",
    "5 Site",
    "6 LeftUpLeg",
    "7 LeftLeg",
    "8 LeftFoot",
    "9 LeftToeBase",
    "10 Site",
    "11 Spine",
    "12 Spine1",
    "13 Neck",
    "14 Head",
    "15 Site",
    "16 LeftShoulder",
    "17 LeftArm",
    "18 LeftForeArm",
    "19 LeftHand",
    "20 LeftHandThumb",
    "21 Site",
    "22 L_Wrist_End",
    "23 Site",
    "24 RightShoulder",
    "25 RightArm",
    "26 RightForeArm",
    "27 RightHand",
    "28 RightHandThumb",
    "29 Site",
    "30 R_Wrist_End",
    "31 Site",
]

# === Define bone connectivity (parent-child pairs) ===
# (chosen based on Human3.6M / standard BVH hierarchy)
bones = [
    (0, 1), (1, 2), (2, 3), (3, 4),         # right leg
    (0, 6), (6, 7), (7, 8), (8, 9),         # left leg
    (0, 11), (11, 12), (12, 13), (13, 14),  # spine to head
    (13, 24), (24, 25), (25, 26), (26, 27), # right arm
    (13, 16), (16, 17), (17, 18), (18, 19)  # left arm
]
# === Plot with equal XYZ scaling ===
fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(111, projection='3d')

# Draw joints
ax.scatter(first_pose[:, 0], first_pose[:, 1], first_pose[:, 2], c='r', s=30)

# Annotate each joint
for i, name in enumerate(joint_names):
    x, y, z = first_pose[i]
    ax.text(x, y, z, name, fontsize=7)

# Draw skeleton lines
for (i, j) in bones:
    xs = [first_pose[i, 0], first_pose[j, 0]]
    ys = [first_pose[i, 1], first_pose[j, 1]]
    zs = [first_pose[i, 2], first_pose[j, 2]]
    ax.plot(xs, ys, zs, 'k-', lw=1)

# === Equal axis scaling ===
X, Y, Z = first_pose[:, 0], first_pose[:, 1], first_pose[:, 2]
max_range = np.ptp([X.min(), X.max(), Y.min(), Y.max(), Z.min(), Z.max()]) / 2.0
mid_x, mid_y, mid_z = np.mean(X), np.mean(Y), np.mean(Z)
ax.set_xlim(mid_x - max_range, mid_x + max_range)
ax.set_ylim(mid_y - max_range, mid_y + max_range)
ax.set_zlim(mid_z - max_range, mid_z + max_range)

ax.set_box_aspect([1, 1, 1])  # ensures equal 3D aspect ratio

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Human3.6M Skeleton — Frame 0 (Equal XYZ scale)')
plt.tight_layout()
plt.show()