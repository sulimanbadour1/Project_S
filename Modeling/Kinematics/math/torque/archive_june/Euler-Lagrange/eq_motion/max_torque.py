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

# Define joint angle ranges (in radians)
theta_ranges = [
    np.deg2rad(np.linspace(-180, 180, 10)),  # Joint 1
    np.deg2rad(np.linspace(-90, 90, 10)),  # Joint 2
    np.deg2rad(np.linspace(-180, 180, 10)),  # Joint 3
    np.deg2rad(np.linspace(-180, 180, 10)),  # Joint 4
    np.deg2rad(np.linspace(-180, 180, 10)),  # Joint 5
]


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


# Function to compute transformation matrices
def compute_transforms(theta):
    A1 = dh_matrix(theta[0], d1, 0, 90)
    A2 = dh_matrix(theta[1], 0, a2, 0)
    A3 = dh_matrix(theta[2], 0, a3, 0)
    A4 = dh_matrix(theta[3], 0, 0, 90)
    A5 = dh_matrix(theta[4], d5, 0, 0)
    T1 = A1
    T2 = T1 @ A2
    T3 = T2 @ A3
    T4 = T3 @ A4
    T5 = T4 @ A5
    return [T1, T2, T3, T4, T5]


# Function to compute the Jacobian for each joint
def compute_jacobian(T_matrices):
    J = []
    O_n = T_matrices[-1][:3, 3]
    for T in T_matrices:
        O_i = T[:3, 3]
        Z_i = T[:3, 2]
        J.append(np.hstack((np.cross(Z_i, O_n - O_i), Z_i)))
    return np.array(J).T[:3, :]  # Only consider linear velocities


# Function to compute energies and torques
def compute_energies_and_torques(theta):
    T_matrices = compute_transforms(theta)
    J = compute_jacobian(T_matrices)

    # Potential energy
    P = [T[:3, 3] / 2 for T in T_matrices]
    V = sum(m * g * p[2] for m, p in zip(masses, P))

    # Kinetic energy
    T_kin = sum(0.5 * m * np.linalg.norm(p) ** 2 for m, p in zip(masses, P))

    # Torques
    torques = np.dot(J.T, np.array([T_kin] * 3))

    return T_kin, V, torques, T_matrices


# Function to interpolate angles between two positions
def interpolate_angles(theta_start, theta_end, time):
    return np.array(
        [
            np.linspace(start, end, len(time))
            for start, end in zip(theta_start, theta_end)
        ]
    )


# Grid search to find maximum torques
max_torques = np.zeros(5)
max_theta = None

for theta1 in theta_ranges[0]:
    for theta2 in theta_ranges[1]:
        for theta3 in theta_ranges[2]:
            for theta4 in theta_ranges[3]:
                for theta5 in theta_ranges[4]:
                    theta = [theta1, theta2, theta3, theta4, theta5]
                    _, _, torques, _ = compute_energies_and_torques(theta)
                    if np.max(np.abs(torques)) > np.max(max_torques):
                        max_torques = np.maximum(max_torques, np.abs(torques))
                        max_theta = theta

print("Maximum torques for each joint (Nm):")
for i, torque in enumerate(max_torques):
    print(f"Joint {i + 1}: {torque:.2f} Nm")

# Plot the robot in the maximum torque configuration
_, _, _, T_matrices_max = compute_energies_and_torques(max_theta)

# Extract positions for plotting
positions = [T[:3, 3] for T in T_matrices_max]
positions = np.array(positions).T

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")
ax.plot(positions[0], positions[1], positions[2], "bo-", linewidth=2, markersize=5)
ax.set_title("Robot Configuration at Maximum Torque", fontsize=16)
ax.set_xlabel("X (m)", fontsize=14)
ax.set_ylabel("Y (m)", fontsize=14)
ax.set_zlabel("Z (m)", fontsize=14)
plt.show()

# Plot the energies and torques for each joint in the maximum torque configuration
theta_start = np.deg2rad([0, 0, 0, 0, 0])  # Define a starting position
total_time = np.linspace(0, 20, 800)
theta_interp_full = interpolate_angles(theta_start, max_theta, total_time)

kinetic_energies_full = []
potential_energies_full = []
torques_full = []

for angles in theta_interp_full.T:
    T, V, torques, _ = compute_energies_and_torques(angles)
    kinetic_energies_full.append(T)
    potential_energies_full.append(V)
    torques_full.append(torques)

kinetic_energies_full = np.array(kinetic_energies_full)
potential_energies_full = np.array(potential_energies_full)
torques_full = np.array(torques_full)

# Plotting the energies over time
fig, ax = plt.subplots(2, 1, figsize=(14, 10))

# Kinetic Energy plot
ax[0].plot(total_time, kinetic_energies_full, "b-", label="Kinetic Energy", linewidth=2)
ax[0].set_title("Kinetic Energy over Time", fontsize=16)
ax[0].set_xlabel("Time (s)", fontsize=14)
ax[0].set_ylabel("Kinetic Energy (J)", fontsize=14)
ax[0].legend(fontsize=12)
ax[0].grid(True)

# Potential Energy plot
ax[1].plot(
    total_time, potential_energies_full, "r-", label="Potential Energy", linewidth=2
)
ax[1].set_title("Potential Energy over Time", fontsize=16)
ax[1].set_xlabel("Time (s)", fontsize=14)
ax[1].set_ylabel("Potential Energy (J)", fontsize=14)
ax[1].legend(fontsize=12)
ax[1].grid(True)

plt.tight_layout()
plt.show()

# Plotting the torques over time for each joint in the same graph
plt.figure(figsize=(14, 8))
joint_labels = ["Joint 1", "Joint 2", "Joint 3", "Joint 4", "Joint 5"]
colors = ["b", "g", "r", "c", "m"]
for i in range(5):
    plt.plot(
        total_time,
        torques_full[:, i],
        label=f"Torque {joint_labels[i]}",
        color=colors[i],
        linewidth=2,
    )

plt.title("Torques for Each Joint over Time", fontsize=16)
plt.xlabel("Time (s)", fontsize=14)
plt.ylabel("Torque (Nm)", fontsize=14)
plt.legend(fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.show()
