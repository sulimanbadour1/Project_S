import numpy as np
import matplotlib.pyplot as plt
from itertools import product
import concurrent.futures


# DH parameter helper function
def dh_matrix(theta, d, a, alpha):
    alpha_rad = np.radians(alpha)
    return np.array(
        [
            [
                np.cos(theta),
                -np.sin(theta) * np.cos(alpha_rad),
                np.sin(theta) * np.sin(alpha_rad),
                a * np.cos(theta),
            ],
            [
                np.sin(theta),
                np.cos(theta) * np.cos(alpha_rad),
                -np.cos(theta) * np.sin(alpha_rad),
                a * np.sin(theta),
            ],
            [0, np.sin(alpha_rad), np.cos(alpha_rad), d],
            [0, 0, 0, 1],
        ]
    )


# DH parameters
d_1, d_5 = 0.1, 0.1
a_2, a_3 = 0.5, 0.5
alpha = [90, 0, 0, 90, 0]
masses = [
    1.0,
    1.0,
    1.0,
    1.0,
    1.0 + 0.5 + 0.5,
]  # including camera and lights masses in the last link
g = np.array([0, 0, -9.81])


# Precompute transformation matrices for given theta values
def compute_transforms(thetas):
    A1 = dh_matrix(thetas[0], d_1, 0, alpha[0])
    A2 = dh_matrix(thetas[1], 0, a_2, alpha[1])
    A3 = dh_matrix(thetas[2], 0, a_3, alpha[2])
    A4 = dh_matrix(thetas[3], 0, 0, alpha[3])
    A5 = dh_matrix(thetas[4], d_5, 0, alpha[4])

    T1 = A1
    T2 = T1 @ A2
    T3 = T2 @ A3
    T4 = T3 @ A4
    T5 = T4 @ A5

    return [T1, T2, T3, T4, T5]


# Compute Jacobians for given transformation matrices
def compute_jacobians(transforms):
    Jvs = []
    for T in transforms:
        p = T[:3, 3] / 2
        Jv = np.zeros((3, 5))
        for i in range(5):
            if i == 0:
                Jv[:, i] = np.cross([0, 0, 1], p)
            else:
                Jv[:, i] = np.cross(
                    transforms[i - 1][:3, 2], p - transforms[i - 1][:3, 3]
                )
        Jvs.append(Jv)
    return Jvs


# Precompute gravity torques
def compute_gravity_torques(Jvs, masses):
    gravity_torques = np.zeros(5)
    for Jv, mass in zip(Jvs, masses):
        gravity_torques += Jv.T @ (mass * g)
    return gravity_torques


# Precompute Inertia Matrix (M) and Coriolis Matrix (C)
def compute_inertia_and_coriolis(Jvs, masses, dq):
    M = np.zeros((5, 5))
    for Jv, mass in zip(Jvs, masses):
        M += Jv.T @ Jv * mass

    C = np.zeros((5, 5))
    for i in range(5):
        for j in range(5):
            C[i, j] = 0.5 * sum(
                (M[i, k] * dq[k] + M[k, j] * dq[k] - M[i, j] * dq[k]) for k in range(5)
            )
    return M, C


# Compute dynamic torques
def compute_dynamic_torques(thetas, dqs, ddqs, Jvs, gravity_torques, M, C):
    tau_dynamic = M @ ddqs + C @ dqs + gravity_torques
    return tau_dynamic


# Define numerical values for testing
angle_range = np.linspace(-np.pi, np.pi, 10)
velocity_extremes = [-1, 0, 1]
acceleration_extremes = [-1, 0, 1]


# Function to compute torques for a given set of angles, velocities, and accelerations
def compute_torques_for_combination(combination):
    angles, velocities, accelerations = combination
    transforms = compute_transforms(angles)
    Jvs = compute_jacobians(transforms)
    gravity_torques = compute_gravity_torques(Jvs, masses)
    M, C = compute_inertia_and_coriolis(Jvs, masses, velocities)
    torques = compute_dynamic_torques(
        angles, velocities, accelerations, Jvs, gravity_torques, M, C
    )
    return np.abs(torques)


# Create a list of all combinations of angles, velocities, and accelerations
angle_combinations = list(product(angle_range, repeat=5))
velocity_combinations = list(product(velocity_extremes, repeat=5))
acceleration_combinations = list(product(acceleration_extremes, repeat=5))

all_combinations = list(
    product(angle_combinations, velocity_combinations, acceleration_combinations)
)

# Initialize the maximum torque tracker
max_torque_per_joint = np.zeros(5)

# Use concurrent.futures to parallelize the computation
with concurrent.futures.ProcessPoolExecutor() as executor:
    results = list(executor.map(compute_torques_for_combination, all_combinations))

# Find the maximum torque for each joint
for result in results:
    max_torque_per_joint = np.maximum(max_torque_per_joint, result)

# Plot the maximum torques
joints = ["Joint 1", "Joint 2", "Joint 3", "Joint 4", "Joint 5"]

plt.figure(figsize=(10, 6))
bars = plt.bar(joints, max_torque_per_joint.tolist(), color="blue")
plt.xlabel("Joints")
plt.ylabel("Maximum Torque (Nm)")
plt.title("Maximum Dynamic Torque on Each Joint Across All Configurations")
plt.grid(True, linestyle="--", alpha=0.6)

# Annotate bars with their values
for bar in bars:
    yval = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        yval,
        f"{yval:.2f}",
        va="bottom" if yval < 0 else "top",
        ha="center",
        color="black",
    )

plt.show()

# Print the maximum torques
print(f"Maximum Dynamic Torques for given values: {max_torque_per_joint.tolist()}")
