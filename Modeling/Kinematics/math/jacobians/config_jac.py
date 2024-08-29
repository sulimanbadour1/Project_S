import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random

# Define DH parameters with units in meters (m) and radians
d1 = 0.1  # meters
d5 = 0.1  # meters
a2 = 0.5  # meters
a3 = 0.5  # meters
alpha = np.deg2rad([90, 0, 0, 90, 0])

# Define the range of joint angles (in radians)
theta_range = np.deg2rad(np.arange(-180, 181, 45))


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


# Analyze singularities for the range of joint angles
def analyze_singularities(theta_range):
    singularities = []
    all_ranks = []

    for theta1 in theta_range:
        for theta2 in theta_range:
            for theta3 in theta_range:
                for theta4 in theta_range:
                    for theta5 in theta_range:
                        thetas = [theta1, theta2, theta3, theta4, theta5]
                        J = compute_jacobian(thetas)
                        rank = np.linalg.matrix_rank(J)
                        all_ranks.append(rank)
                        if rank < 5:
                            singularities.append((thetas, rank))

    return singularities, all_ranks


# Analyze the singularities
singularities, all_ranks = analyze_singularities(theta_range)

# Print the singular configurations and their ranks
print("\nSingular configurations and their ranks:")
for config, rank in singularities:
    print(f"Configuration (degrees): {np.rad2deg(config)}, Rank: {rank}")

# Enhanced Visualization of the distribution of ranks using a histogram
plt.figure(figsize=(10, 6))
plt.hist(
    all_ranks,
    bins=np.arange(2.5, 6.5, 1),
    edgecolor="black",
    align="left",
    color="skyblue",
)
plt.xlabel("Rank of Jacobian", fontsize=14)
plt.ylabel("Frequency (Number of Configs)", fontsize=14)
plt.title("Distribution of Jacobian Ranks", fontsize=16)
plt.xticks(ticks=np.arange(3, 6), labels=["Rank 3", "Rank 4", "Rank 5"], fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True, linestyle="--", alpha=0.7)
for i in range(3, 6):
    plt.text(
        i,
        all_ranks.count(i),
        str(all_ranks.count(i)),
        ha="center",
        va="bottom",
        fontsize=12,
    )
plt.tight_layout()
plt.show()


# Visualize the robot configurations that are singular
def plot_singular_robots(singularities):
    fig = plt.figure(figsize=(12, 4))  # Adjusted figure size for thinner charts

    for idx, (thetas, rank) in enumerate(singularities):
        ax = fig.add_subplot(1, 3, idx + 1, projection="3d")
        transforms = compute_transforms(thetas)
        xs, ys, zs = [], [], []
        for T in transforms:
            xs.append(T[0, 3])
            ys.append(T[1, 3])
            zs.append(T[2, 3])

        ax.plot(
            xs, ys, zs, "o-", label=f"Rank {rank}\nAngles {np.rad2deg(thetas)} degrees"
        )
        ax.set_xlabel("X (m)", fontsize=10, fontweight="bold")
        ax.set_ylabel("Y (m)", fontsize=10, fontweight="bold")
        ax.set_zlabel("Z (m)", fontsize=10, fontweight="bold")
        ax.set_title(f"Singular Configuration {idx + 1}")
        ax.legend(fontsize=10)  # Adjust the fontsize for better readability
        for i, (x, y, z) in enumerate(zip(xs, ys, zs)):
            ax.text(
                x,
                y,
                z,
                f"P{i}",
                fontsize=8,
                fontweight="bold",
                color="red",
            )

    plt.tight_layout()
    plt.show()


# Select three random singular configurations
random_singularities = random.sample(singularities, 3)

# Plot the selected singular configurations
plot_singular_robots(random_singularities)
