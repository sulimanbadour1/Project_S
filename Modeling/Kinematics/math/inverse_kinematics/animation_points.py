import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define DH parameters
d1, a2, a3, d5 = 0.1, 0.5, 0.5, 0.1


def inverse_kinematics(Px, Py, Pz, d1, a2, a3, d5, omega):
    R = d5 * math.cos(math.radians(omega))
    theta1 = math.degrees(math.atan2(Py, Px))
    theta1_rad = math.radians(theta1)

    Pxw = Px - R * math.cos(theta1_rad)
    Pyw = Py - R * math.sin(theta1_rad)
    Pzw = Pz + d5 * math.sin(math.radians(omega))

    Rw = math.sqrt(Pxw**2 + Pyw**2)
    S = math.sqrt((Pzw - d1) ** 2 + Rw**2)

    alpha = math.degrees(math.atan2(Pzw - d1, Rw))
    beta = math.degrees(math.acos((a2**2 + S**2 - a3**2) / (2 * a2 * S)))

    theta2 = alpha + beta
    theta2_alt = alpha - beta

    theta3 = math.degrees(math.acos((S**2 - a2**2 - a3**2) / (2 * a2 * a3)))
    theta3 = -theta3

    theta234 = 90 - omega
    theta4 = theta234 - theta2 - theta3

    return theta1, theta2, theta3, theta4


def forward_kinematics(d1, a2, a3, d5, theta1, theta2, theta3, theta4):
    theta1 = math.radians(theta1)
    theta2 = math.radians(theta2)
    theta3 = math.radians(theta3)
    theta4 = math.radians(theta4)

    omega = 90 - (theta2 + theta3 + theta4)

    x0, y0, z0 = 0, 0, 0
    x1, y1, z1 = 0, 0, d1
    x2 = a2 * math.cos(theta1) * math.cos(theta2)
    y2 = a2 * math.sin(theta1) * math.cos(theta2)
    z2 = d1 + a2 * math.sin(theta2)
    x3 = x2 + a3 * math.cos(theta1) * math.cos(theta2 + theta3)
    y3 = y2 + a3 * math.sin(theta1) * math.cos(theta2 + theta3)
    z3 = z2 + a3 * math.sin(theta2 + theta3)
    x4 = x3 + d5 * math.cos(math.radians(omega)) * math.cos(theta1) * math.cos(
        theta2 + theta3
    )
    y4 = y3 + d5 * math.cos(math.radians(omega)) * math.sin(theta1) * math.cos(
        theta2 + theta3
    )
    z4 = z3 + d5

    return [(x0, y0, z0), (x1, y1, z1), (x2, y2, z2), (x3, y3, z3), (x4, y4, z4)]


def plot_robot_and_trajectory(
    joint_positions,
    end_effector_positions,
    path_x,
    path_y,
    path_z,
    title="Robot and Trajectory",
):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Plot the trajectory of the end-effector
    ax.plot(path_x, path_y, path_z, "r--", label="Trajectory")

    # Plot the robot arm configuration at the first sampled point
    positions = joint_positions[0]
    x, y, z = zip(*positions)
    ax.plot(x, y, z, marker="o", label="Robot Configuration at Point 1")
    ax.scatter(
        x[-1], y[-1], z[-1], color="blue", s=100, label="End-Effector at Point 1"
    )

    # Formatting the plot
    ax.set_xlabel("X axis")
    ax.set_ylabel("Y axis")
    ax.set_zlabel("Z axis")
    ax.set_title(title)
    ax.legend()

    plt.show()


# Define the circular path parameters
radius = 0.8
num_points = 1000
z_level = 0.4

# Generate points on the circular path
angles = np.linspace(0, 2 * np.pi, num_points)
Px = radius * np.cos(angles)
Py = radius * np.sin(angles)
Pz = np.full_like(Px, z_level)

# Compute inverse kinematics for each point on the path
joint_angles = [
    inverse_kinematics(px, py, pz, d1, a2, a3, d5, omega=-90)
    for px, py, pz in zip(Px, Py, Pz)
]

# Sample indices to extract ten points evenly
sample_indices = np.linspace(0, num_points - 1, 10, dtype=int)

# Extract ten points from the circular path
sampled_Px = Px[sample_indices]
sampled_Py = Py[sample_indices]
sampled_Pz = Pz[sample_indices]

# Compute joint angles for these sampled points
sampled_joint_angles = [
    inverse_kinematics(px, py, pz, d1, a2, a3, d5, omega=-90)
    for px, py, pz in zip(sampled_Px, sampled_Py, sampled_Pz)
]

# Compute end-effector positions for these sampled angles
sampled_joint_positions = [
    forward_kinematics(d1, a2, a3, d5, *angles) for angles in sampled_joint_angles
]
sampled_end_effector_positions = [pos[-1] for pos in sampled_joint_positions]

# List to store end effector positions for the entire path
path_x = []
path_y = []
path_z = []

# Collect path coordinates
for pos in sampled_end_effector_positions:
    path_x.append(pos[0])
    path_y.append(pos[1])
    path_z.append(pos[2])

# Plot robot configuration and trajectory
plot_robot_and_trajectory(
    sampled_joint_positions, sampled_end_effector_positions, path_x, path_y, path_z
)
