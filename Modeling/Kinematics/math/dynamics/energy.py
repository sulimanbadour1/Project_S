import numpy as np
import matplotlib.pyplot as plt

# Define the DH parameters
d1 = 0.1
d5 = 0.1
a2 = 0.5
a3 = 0.5

# Define the masses and inertia values
masses = [1, 1, 1, 1, 1]
inertias = [0.1, 0.1, 0.1, 0.1, 0.1]

# Define gravity
g = 9.81

# Define joint angles for starting, elbow down, and elbow up positions
theta_start = np.deg2rad([0, 0, 0, 0, 0])
theta_target1 = np.deg2rad([-10, -10, -90, 180, 0])
theta_target2 = np.deg2rad([-45, 30, -15, 90, 0])


# Function to interpolate angles between two positions
def interpolate_angles(theta_start, theta_end, time):
    return np.array(
        [
            np.linspace(start, end, len(time))
            for start, end in zip(theta_start, theta_end)
        ]
    )


# Function to compute transformation matrix from DH parameters
def dh_matrix(theta, d, a, alpha):
    alpha_rad = np.deg2rad(alpha)
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


# Define the function to compute transformation matrices more efficiently
def compute_transforms_efficient(theta):
    A1 = dh_matrix(theta[0], d1, 0, 90)
    A2 = dh_matrix(theta[1], 0, a2, 0)
    A3 = dh_matrix(theta[2], 0, a3, 0)
    A4 = dh_matrix(theta[3], 0, 0, 90)
    A5 = dh_matrix(theta[4], d5, 0, 0)
    return A1, A2, A3, A4, A5


# Function to compute the Jacobian for each joint
def compute_jacobian(T_matrices):
    J = []
    O_n = T_matrices[-1][:3, 3]
    for T in T_matrices:
        O_i = T[:3, 3]
        Z_i = T[:3, 2]
        J.append(np.hstack((np.cross(Z_i, O_n - O_i), Z_i)))
    return np.array(J).T[:3, :]  # Only consider linear velocities


# Optimized function to compute energies and torques
def compute_energies_and_torques(theta):
    A1, A2, A3, A4, A5 = compute_transforms_efficient(theta)

    T1 = A1
    T2 = T1 @ A2
    T3 = T2 @ A3
    T4 = T3 @ A4
    T5 = T4 @ A5

    T_matrices = [T1, T2, T3, T4, T5]
    J = compute_jacobian(T_matrices)

    P1 = T1[:3, 3] / 2
    P2 = T2[:3, 3] / 2
    P3 = T3[:3, 3] / 2
    P4 = T4[:3, 3] / 2
    P5 = T5[:3, 3] / 2

    V1 = masses[0] * g * P1[2]
    V2 = masses[1] * g * P2[2]
    V3 = masses[2] * g * P3[2]
    V4 = masses[3] * g * P4[2]
    V5 = masses[4] * g * P5[2]
    V = V1 + V2 + V3 + V4 + V5

    T1_kin = 0.5 * masses[0] * (np.linalg.norm(P1)) ** 2 + 0.5 * inertias[0] * (0) ** 2
    T2_kin = 0.5 * masses[1] * (np.linalg.norm(P2)) ** 2 + 0.5 * inertias[1] * (0) ** 2
    T3_kin = 0.5 * masses[2] * (np.linalg.norm(P3)) ** 2 + 0.5 * inertias[2] * (0) ** 2
    T4_kin = 0.5 * masses[3] * (np.linalg.norm(P4)) ** 2 + 0.5 * inertias[3] * (0) ** 2
    T5_kin = 0.5 * masses[4] * (np.linalg.norm(P5)) ** 2 + 0.5 * inertias[4] * (0) ** 2
    T = T1_kin + T2_kin + T3_kin + T4_kin + T5_kin

    torques = np.dot(J.T, np.array([T1_kin, T2_kin, T3_kin]))

    return T, V, torques


# Define the simulation time for transitions to target positions (5 seconds each) and back
total_time = np.linspace(0, 20, 800)

# Interpolated angles for transitions to target positions and back
theta_interp_target1 = interpolate_angles(theta_start, theta_target1, total_time[:200])
theta_interp_target2 = interpolate_angles(
    theta_target1, theta_target2, total_time[200:400]
)
theta_interp_back_target1 = interpolate_angles(
    theta_target2, theta_target1, total_time[400:600]
)
theta_interp_back_start = interpolate_angles(
    theta_target1, theta_start, total_time[600:]
)

# Combine all transitions for a complete cycle
theta_interp_full = np.hstack(
    (
        theta_interp_target1,
        theta_interp_target2,
        theta_interp_back_target1,
        theta_interp_back_start,
    )
)

# Compute energies and torques for each time step in the 20-second simulation using the optimized function
kinetic_energies_full = []
potential_energies_full = []
torques_full = []

for angles in theta_interp_full.T:
    T, V, torques = compute_energies_and_torques(angles)
    kinetic_energies_full.append(T)
    potential_energies_full.append(V)
    torques_full.append(torques)

kinetic_energies_full = np.array(kinetic_energies_full)
potential_energies_full = np.array(potential_energies_full)
torques_full = np.array(torques_full)

# Plotting the energies over time for the 20-second simulation
fig, ax = plt.subplots(2, 1, figsize=(12, 10))

# Kinetic Energy plot
ax[0].plot(total_time, kinetic_energies_full, "b-", label="Kinetic Energy", linewidth=2)
ax[0].set_title("Kinetic Energy over Time (20 seconds)", fontsize=16)
ax[0].set_xlabel("Time (s)", fontsize=14)
ax[0].set_ylabel("Kinetic Energy (J)", fontsize=14)
ax[0].legend(fontsize=12)
ax[0].grid(True)

# Potential Energy plot
ax[1].plot(
    total_time, potential_energies_full, "r-", label="Potential Energy", linewidth=2
)
ax[1].set_title("Potential Energy over Time (20 seconds)", fontsize=16)
ax[1].set_xlabel("Time (s)", fontsize=14)
ax[1].set_ylabel("Potential Energy (J)", fontsize=14)
ax[1].legend(fontsize=12)
ax[1].grid(True)

plt.tight_layout()
plt.show()

# Plotting the torques over time for each joint in the 20-second simulation
fig, ax = plt.subplots(5, 1, figsize=(12, 15))

joint_labels = ["Joint 1", "Joint 2", "Joint 3", "Joint 4", "Joint 5"]
for i in range(5):
    ax[i].plot(
        total_time, torques_full[:, i], label=f"Torque {joint_labels[i]}", linewidth=2
    )
    ax[i].set_title(f"Torque {joint_labels[i]} over Time (20 seconds)", fontsize=16)
    ax[i].set_xlabel("Time (s)", fontsize=14)
    ax[i].set_ylabel("Torque (Nm)", fontsize=14)
    ax[i].legend(fontsize=12)
    ax[i].grid(True)

plt.tight_layout()
plt.show()
