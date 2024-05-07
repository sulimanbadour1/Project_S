import numpy as np
import matplotlib.pyplot as plt


# Function to compute the transformation matrix
def compute_transformation(a, alpha, d, theta):
    """Compute the homogeneous transformation matrix."""
    theta = np.deg2rad(theta)  # Convert degrees to radians for computation
    alpha = np.deg2rad(alpha)  # Convert degrees to radians for computation
    T = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0, a],
            [
                np.sin(theta) * np.cos(alpha),
                np.cos(theta) * np.cos(alpha),
                -np.sin(alpha),
                -d * np.sin(alpha),
            ],
            [
                np.sin(theta) * np.sin(alpha),
                np.cos(theta) * np.sin(alpha),
                np.cos(alpha),
                d * np.cos(alpha),
            ],
            [0, 0, 0, 1],
        ]
    )
    return T


# Joint angles and DH parameters
joint_angles = [30, -45, 45, -30, 15]  # degrees
dh_params = [
    (0, 0, 0.05, joint_angles[0]),
    (0.03, 90, 0, joint_angles[1]),
    (0.25, 0, 0, joint_angles[2]),
    (0.28, 0, 0, joint_angles[3]),
    (0.28, 0, 0, joint_angles[4]),
]

# Base frame
frames = [np.eye(4)]  # Start with the identity matrix for the base frame

# Compute transformations for each joint and accumulate
for a, alpha, d, theta in dh_params:
    T = compute_transformation(a, alpha, d, theta)
    frames.append(frames[-1] @ T)  # Matrix multiplication to accumulate transformations

# Extract positions of each joint
positions = np.array(
    [frame[:3, 3] for frame in frames]
).T  # Extract translation components

# Plotting
plt.figure(figsize=(10, 5))
plt.plot(positions[0], positions[2], "o-", label="Robot Arm")  # Plot x and z positions
plt.scatter(0, 0, color="red", label="Base")
plt.title("2D Side View of Robot Configuration")
plt.xlabel("X Position (m)")
plt.ylabel("Z Position (m)")
plt.grid(True)
plt.legend()
plt.axis("equal")
plt.show()


# plotting in 3d space
from mpl_toolkits.mplot3d import Axes3D


# Function to compute the transformation matrix for 3D visualization
def compute_3D_transformation(a, alpha, d, theta):
    """Compute the homogeneous transformation matrix for 3D visualization."""
    theta = np.deg2rad(theta)  # Convert degrees to radians for computation
    alpha = np.deg2rad(alpha)  # Convert degrees to radians for computation
    T = np.array(
        [
            [
                np.cos(theta),
                -np.sin(theta) * np.cos(alpha),
                np.sin(theta) * np.sin(alpha),
                a * np.cos(theta),
            ],
            [
                np.sin(theta),
                np.cos(theta) * np.cos(alpha),
                -np.cos(theta) * np.sin(alpha),
                a * np.sin(theta),
            ],
            [0, np.sin(alpha), np.cos(alpha), d],
            [0, 0, 0, 1],
        ]
    )
    return T


# Compute transformations for each joint and accumulate for 3D
frames_3D = [np.eye(4)]  # Start with the identity matrix for the base frame

# Compute transformations for each joint and accumulate
for a, alpha, d, theta in dh_params:
    T_3D = compute_3D_transformation(a, alpha, d, theta)
    frames_3D.append(
        frames_3D[-1] @ T_3D
    )  # Matrix multiplication to accumulate transformations

# Extract positions of each joint for 3D visualization
positions_3D = np.array(
    [frame[:3, 3] for frame in frames_3D]
).T  # Extract translation components

# Plotting in 3D
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")
ax.plot(positions_3D[0], positions_3D[1], positions_3D[2], "o-", label="Robot Arm")
ax.scatter(0, 0, 0, color="red", s=100, label="Base")
ax.set_title("3D View of Robot Configuration")
ax.set_xlabel("X Position (m)")
ax.set_ylabel("Y Position (m)")
ax.set_zlabel("Z Position (m)")
ax.legend()
ax.grid(True)
ax.view_init(elev=25, azim=45)  # Adjust the viewing angle for better perception
plt.show()
