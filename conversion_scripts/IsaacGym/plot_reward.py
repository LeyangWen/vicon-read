import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')

def compute_back_ergo_reward(back_angle, knee_angle, weight=1.0):
    exp_k = -5.0
    upper_limit_angle = 1 / 9.0 * np.pi  # ~20 degrees
    back_angle_diff = max(back_angle - upper_limit_angle, 0.0)
    knee_discount = max((knee_angle - 0.5 * np.pi) / (0.5 * np.pi), 0.0)
    adjusted_back_angle_diff = back_angle_diff * (1.0 - knee_discount)
    return weight * np.exp(exp_k * adjusted_back_angle_diff)

back_angles = np.linspace(0, np.pi /180*150, 30)
knee_angles = np.linspace(0, np.pi/180*150, 30)

B_vals, K_vals, Z_vals = [], [], []
for ba in back_angles:
    for ka in knee_angles:
        B_vals.append(np.degrees(ba))
        K_vals.append(np.degrees(ka))
        Z_vals.append(compute_back_ergo_reward(ba, ka))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_trisurf(B_vals, K_vals, Z_vals, cmap='viridis')
ax.set_xlabel("Back Angle (deg)")
ax.set_ylabel("Mean Knee Angle (deg)")
ax.set_zlabel("Reward")
ax.set_title("Back Ergonomics Reward")
plt.tight_layout()
plt.show()



# Define the reward function
def compute_box_ergo_reward(box_pos_diff_xy, max_box_edge, weight=1.0):
    box_pos_diff = np.clip(box_pos_diff_xy - max_box_edge, 0.0, None)
    reward = weight * np.exp(-5.0 * box_pos_diff)
    return reward

# Generate a grid of values
box_pos_diffs = np.linspace(0.2, 1.2, 50)       # distance from body (m)
max_box_edges = np.linspace(0.2, 1.2, 50)     # max box edge * (1+threshold)

X, Y = np.meshgrid(box_pos_diffs, max_box_edges)
Z = compute_box_ergo_reward(X, Y)

# Plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis')
ax.set_xlabel("Box Distance from Body (m)")
ax.set_ylabel("Box Size (m)")
ax.set_zlabel("Reward")
ax.set_title("Box Ergonomics Reward")

plt.tight_layout()
plt.show()

# Define the reward function
def compute_elbow_ergo_reward(left_elbow_angle, right_elbow_angle, weight=1.0):
    exp_k = -5.0
    desired_angle = 5.0 / 9.0 * np.pi  # ~80 degrees
    angle_margin = 1.0 / 9.0 * np.pi   # ~20 degrees

    left_diff = np.maximum(np.abs(left_elbow_angle - desired_angle) - angle_margin, 0.0)
    right_diff = np.maximum(np.abs(right_elbow_angle - desired_angle) - angle_margin, 0.0)

    reward = (weight / 2) * np.exp(exp_k * left_diff) + (weight / 2) * np.exp(exp_k * right_diff)
    return reward

# Generate elbow angles
left_angles = np.linspace(0, np.pi, 50)
right_angles = np.linspace(0, np.pi, 50)

L, R = np.meshgrid(left_angles, right_angles)
Z = compute_elbow_ergo_reward(L, R)

# Plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(L * 180 / np.pi, R * 180 / np.pi, Z, cmap='viridis')
ax.set_xlabel("Left Elbow Angle (deg)")
ax.set_ylabel("Right Elbow Angle (deg)")
ax.set_zlabel("Reward")
ax.set_title("Elbow Ergonomics Reward")

plt.tight_layout()
plt.show()