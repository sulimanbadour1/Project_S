import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the ranges for the joint angles and link lengths
theta_1_range = np.linspace(-np.pi, np.pi, 9)
theta_2_range = np.linspace(-np.pi / 2, np.pi / 2, 9)
theta_3_range = np.linspace(-np.pi / 2, np.pi / 2, 9)
theta_4_range = np.linspace(-np.pi, np.pi, 9)
theta_5_range = np.linspace(-np.pi, np.pi, 9)
d_1_value = 0.1  # Example value for d_1
d_5_value = 0.1  # Example value for d_5
a_2_value = 0.5  # Example value for a_2
a_3_value = 0.5  # Example value for a_3


# Define the transformation matrices
def dh_matrix(theta, d, a, alpha):
    alpha_rad = np.radians(alpha)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    cos_alpha = np.cos(alpha_rad)
    sin_alpha = np.sin(alpha_rad)
    return np.array(
        [
            [cos_theta, -sin_theta * cos_alpha, sin_theta * sin_alpha, a * cos_theta],
            [sin_theta, cos_theta * cos_alpha, -cos_theta * sin_alpha, a * sin_theta],
            [0, sin_alpha, cos_alpha, d],
            [0, 0, 0, 1],
        ]
    )


# Precompute constant transformation matrices
A1_const = dh_matrix(0, d_1_value, 0, 90)
A4_const = dh_matrix(0, 0, 0, 90)
A5_const = dh_matrix(0, d_5_value, 0, 0)

# Initialize arrays to store end-effector positions
positions = np.zeros(
    (
        len(theta_1_range)
        * len(theta_2_range)
        * len(theta_3_range)
        * len(theta_4_range)
        * len(theta_5_range),
        3,
    )
)

# Index for storing positions
index = 0

# Iterate over all combinations of joint angles
for theta_1 in theta_1_range:
    A1 = np.copy(A1_const)
    A1[:3, :3] = dh_matrix(theta_1, 0, 0, 90)[:3, :3]
    for theta_2 in theta_2_range:
        A2 = dh_matrix(theta_2, 0, a_2_value, 0)
        for theta_3 in theta_3_range:
            A3 = dh_matrix(theta_3, 0, a_3_value, 0)
            for theta_4 in theta_4_range:
                A4 = np.copy(A4_const)
                A4[:3, :3] = dh_matrix(theta_4, 0, 0, 90)[:3, :3]
                for theta_5 in theta_5_range:
                    A5 = np.copy(A5_const)
                    A5[:3, :3] = dh_matrix(theta_5, 0, 0, 0)[:3, :3]
                    T = A1 @ A2 @ A3 @ A4 @ A5

                    # Extract end-effector position
                    positions[index, :] = T[:3, 3]
                    index += 1

# Plot the workspace
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.scatter(
    positions[:, 0], positions[:, 1], positions[:, 2], c="b", marker="o", s=1
)  # Set the point size to 10

# Set labels and title
ax.set_xlabel("X axis")
ax.set_ylabel("Y axis")
ax.set_zlabel("Z axis")
ax.set_title("Workspace of the Robot")

# Calculate and print the limits on each axis
x_min, x_max = positions[:, 0].min(), positions[:, 0].max()
y_min, y_max = positions[:, 1].min(), positions[:, 1].max()
z_min, z_max = positions[:, 2].min(), positions[:, 2].max()
print(f"X axis limits: {x_min} to {x_max}")
print(f"Y axis limits: {y_min} to {y_max}")
print(f"Z axis limits: {z_min} to {z_max}")

# Annotate the limits on the plot
ax.text2D(
    0.05, 0.95, f"X axis limits: {x_min:.2f} to {x_max:.2f}", transform=ax.transAxes
)
ax.text2D(
    0.05, 0.90, f"Y axis limits: {y_min:.2f} to {y_max:.2f}", transform=ax.transAxes
)
ax.text2D(
    0.05, 0.85, f"Z axis limits: {z_min:.2f} to {z_max:.2f}", transform=ax.transAxes
)

plt.show()
