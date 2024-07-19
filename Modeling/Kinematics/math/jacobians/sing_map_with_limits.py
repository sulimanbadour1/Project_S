import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random

# Define DH parameters
d1 = 0.1
d5 = 0.1
a2 = 0.5
a3 = 0.5
alpha = np.deg2rad([90, 0, 0, 90, 0])

# Define the joint limits (in radians)
joint_limits = [
    (-np.pi, np.pi),  # theta1
    (-np.pi, np.pi),  # theta3
    (-np.pi, np.pi),  # theta3
    (-np.pi, np.pi),  # theta4
    (-np.pi, np.pi),  # theta5
]

# Number of random samples to generate
num_samples = 5000


# Helper function to create a transformation matrix from DH parameters
def dh_matrix(theta, d, a, alpha):
    return np.array(
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


# Function to compute the Jacobian matrix
def compute_jacobian(thetas):
    theta1, theta2, theta3, theta4, theta5 = thetas
    A1 = dh_matrix(theta1, d1, 0, alpha[0])
    A2 = dh_matrix(theta2, 0, a2, alpha[1])
    A3 = dh_matrix(theta3, 0, a3, alpha[2])
    A4 = dh_matrix(theta4, 0, 0, alpha[3])
    A5 = dh_matrix(theta5, d5, 0, alpha[4])

    T = A1 @ A2 @ A3 @ A4 @ A5

    position = T[:3, 3]
    z0 = np.array([0, 0, 1])
    z1 = A1[:3, :3] @ z0
    z2 = (A1 @ A2)[:3, :3] @ z0
    z3 = (A1 @ A2 @ A3)[:3, :3] @ z0
    z4 = (A1 @ A2 @ A3 @ A4)[:3, :3] @ z0

    Jv = np.column_stack(
        [
            np.cross(z0, position),
            np.cross(z1, position - A1[:3, 3]),
            np.cross(z2, position - (A1 @ A2)[:3, 3]),
            np.cross(z3, position - (A1 @ A2 @ A3)[:3, 3]),
            np.cross(z4, position - (A1 @ A2 @ A3 @ A4)[:3, 3]),
        ]
    )

    Jw = np.column_stack([z0, z1, z2, z3, z4])

    J = np.vstack([Jv, Jw])

    return J


# Compute the transformation matrices for visualization
def compute_transforms(thetas):
    theta1, theta2, theta3, theta4, theta5 = thetas
    A1 = dh_matrix(theta1, d1, 0, alpha[0])
    A2 = dh_matrix(theta2, 0, a2, alpha[1])
    A3 = dh_matrix(theta3, 0, a3, alpha[2])
    A4 = dh_matrix(theta4, 0, 0, alpha[3])
    A5 = dh_matrix(theta5, d5, 0, alpha[4])

    T1 = A1
    T2 = A1 @ A2
    T3 = A1 @ A2 @ A3
    T4 = A1 @ A2 @ A3 @ A4
    T5 = A1 @ A2 @ A3 @ A4 @ A5

    return [np.eye(4), T1, T2, T3, T4, T5]


# Analyze singularities and workspace using random sampling
def analyze_singularities_workspace(joint_limits, num_samples):
    singularities = []
    workspace_points = []

    for _ in range(num_samples):
        thetas = [random.uniform(*joint_limits[i]) for i in range(5)]
        J = compute_jacobian(thetas)
        rank = np.linalg.matrix_rank(J)
        transforms = compute_transforms(thetas)
        end_effector_pos = transforms[-1][:3, 3]
        if rank < 5:
            singularities.append((end_effector_pos, rank))
        else:
            workspace_points.append(end_effector_pos)

    return singularities, workspace_points


# Analyze the singularities and workspace
singularities, workspace_points = analyze_singularities_workspace(
    joint_limits, num_samples
)


# Plot the singularity map and workspace
def plot_singularity_workspace_map(singularities, workspace_points):
    fig = plt.figure(figsize=(18, 6))

    # Define the views
    views = [
        ("Top View", 0, 90),
        ("Front View", 0, 0),
        ("Side View", 90, 0),
    ]

    # Plot each view
    for i, (view_name, elev, azim) in enumerate(views, 1):
        ax = fig.add_subplot(1, 3, i, projection="3d")

        # Plot workspace points
        workspace_points_np = np.array(workspace_points)
        ax.scatter(
            workspace_points_np[:, 0],
            workspace_points_np[:, 1],
            workspace_points_np[:, 2],
            c="b",
            marker="^",
            label="Workspace",
            alpha=0.3,
        )

        # Plot singularities in x, y, z workspace
        for pos, rank in singularities:
            ax.scatter(*pos, c="r", marker="o", label="Singularity Point")

        # Set view
        ax.view_init(elev=elev, azim=azim)

        # Add labels and title
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.set_title(view_name)

        # Add legend
        handles, labels = ax.get_legend_handles_labels()
        unique_labels = dict(zip(labels, handles))
        ax.legend(unique_labels.values(), unique_labels.keys())

    plt.tight_layout()
    plt.show()


# Plot the singularity map and workspace
plot_singularity_workspace_map(singularities, workspace_points)
