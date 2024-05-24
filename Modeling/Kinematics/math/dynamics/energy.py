import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

# Define symbolic variables for joint angles, velocities, accelerations
theta_1, theta_2, theta_3, theta_4, theta_5 = sp.symbols(
    "theta_1 theta_2 theta_3 theta_4 theta_5"
)
dtheta_1, dtheta_2, dtheta_3, dtheta_4, dtheta_5 = sp.symbols(
    "dtheta_1 dtheta_2 dtheta_3 dtheta_4 dtheta_5"
)
ddtheta_1, ddtheta_2, ddtheta_3, ddtheta_4, ddtheta_5 = sp.symbols(
    "ddtheta_1 ddtheta_2 ddtheta_3 ddtheta_4 ddtheta_5"
)

# DH parameters
d_1, d_5 = 0.1, 0.1
a_2, a_3 = 0.5, 0.5
alpha = [90, 0, 0, 90, 0]

# Masses and inertia tensors of the links
m1 = m2 = m3 = m4 = m5 = 1
I1 = I2 = I3 = I4 = I5 = 0.1
g = 9.81


# Helper function to create a transformation matrix from DH parameters
def dh_matrix(theta, d, a, alpha):
    alpha_rad = sp.rad(alpha)  # Convert alpha from degrees to radians
    return sp.Matrix(
        [
            [
                sp.cos(theta),
                -sp.sin(theta) * sp.cos(alpha_rad),
                sp.sin(theta) * sp.sin(alpha_rad),
                a * sp.cos(theta),
            ],
            [
                sp.sin(theta),
                sp.cos(theta) * sp.cos(alpha_rad),
                -sp.cos(theta) * sp.sin(alpha_rad),
                a * sp.sin(theta),
            ],
            [0, sp.sin(alpha_rad), sp.cos(alpha_rad), d],
            [0, 0, 0, 1],
        ]
    )


# Transformation matrices for each joint
A1 = dh_matrix(theta_1, d_1, 0, alpha[0])
A2 = dh_matrix(theta_2, 0, a_2, alpha[1])
A3 = dh_matrix(theta_3, 0, a_3, alpha[2])
A4 = dh_matrix(theta_4, 0, 0, alpha[3])
A5 = dh_matrix(theta_5, d_5, 0, alpha[4])

# Compute the overall transformation matrix by multiplying individual matrices
T1 = A1
T2 = T1 * A2
T3 = T2 * A3
T4 = T3 * A4
T5 = T4 * A5

# Position vectors of the center of mass of each link (assuming center of mass is at the middle of each link)
P1 = T1[:3, 3] / 2
P2 = T2[:3, 3] / 2
P3 = T3[:3, 3] / 2
P4 = T4[:3, 3] / 2
P5 = T5[:3, 3] / 2

# Potential energy of each link
V1 = m1 * g * P1[2]
V2 = m2 * g * P2[2]
V3 = m3 * g * P3[2]
V4 = m4 * g * P4[2]
V5 = m5 * g * P5[2]

# Total potential energy
V = V1 + V2 + V3 + V4 + V5

# Velocity vectors of the center of mass of each link
V1_vec = (
    P1.diff(theta_1) * dtheta_1
    + P1.diff(theta_2) * dtheta_2
    + P1.diff(theta_3) * dtheta_3
    + P1.diff(theta_4) * dtheta_4
    + P1.diff(theta_5) * dtheta_5
)
V2_vec = (
    P2.diff(theta_1) * dtheta_1
    + P2.diff(theta_2) * dtheta_2
    + P2.diff(theta_3) * dtheta_3
    + P2.diff(theta_4) * dtheta_4
    + P2.diff(theta_5) * dtheta_5
)
V3_vec = (
    P3.diff(theta_1) * dtheta_1
    + P3.diff(theta_2) * dtheta_2
    + P3.diff(theta_3) * dtheta_3
    + P3.diff(theta_4) * dtheta_4
    + P3.diff(theta_5) * dtheta_5
)
V4_vec = (
    P4.diff(theta_1) * dtheta_1
    + P4.diff(theta_2) * dtheta_2
    + P4.diff(theta_3) * dtheta_3
    + P4.diff(theta_4) * dtheta_4
    + P4.diff(theta_5) * dtheta_5
)
V5_vec = (
    P5.diff(theta_1) * dtheta_1
    + P5.diff(theta_2) * dtheta_2
    + P5.diff(theta_3) * dtheta_3
    + P5.diff(theta_4) * dtheta_4
    + P5.diff(theta_5) * dtheta_5
)

# Kinetic energy of each link
T1_kin = 0.5 * m1 * (V1_vec.dot(V1_vec)) + 0.5 * I1 * dtheta_1**2
T2_kin = 0.5 * m2 * (V2_vec.dot(V2_vec)) + 0.5 * I2 * dtheta_2**2
T3_kin = 0.5 * m3 * (V3_vec.dot(V3_vec)) + 0.5 * I3 * dtheta_3**2
T4_kin = 0.5 * m4 * (V4_vec.dot(V4_vec)) + 0.5 * I4 * dtheta_4**2
T5_kin = 0.5 * m5 * (V5_vec.dot(V5_vec)) + 0.5 * I5 * dtheta_5**2

# Total kinetic energy
T_kin = T1_kin + T2_kin + T3_kin + T4_kin + T5_kin

# Calculate torque due to gravity
tau1 = sp.diff(V, theta_1)
tau2 = sp.diff(V, theta_2)
tau3 = sp.diff(V, theta_3)
tau4 = sp.diff(V, theta_4)
tau5 = sp.diff(V, theta_5)

# Define joint angles for different configurations (elbow up and down)
configurations = {
    "Elbow Up": {
        "theta_1": 0,
        "theta_2": 45,
        "theta_3": -45,
        "theta_4": 0,
        "theta_5": 0,
    },
    "Elbow Down": {
        "theta_1": 0,
        "theta_2": -45,
        "theta_3": 45,
        "theta_4": 0,
        "theta_5": 0,
    },
    "Straight Up": {
        "theta_1": 0,
        "theta_2": 0,
        "theta_3": 0,
        "theta_4": 0,
        "theta_5": 0,
    },
    "Straight Forward": {
        "theta_1": 0,
        "theta_2": 0,
        "theta_3": 0,
        "theta_4": 90,
        "theta_5": 0,
    },
}

# Initialize pretty printing for better output readability
sp.init_printing(use_unicode=True)

# Simulation parameters
time_steps = np.linspace(0, 10, 100)
joint_angles = {
    "theta_1": np.deg2rad(30) * np.sin(0.1 * time_steps),
    "theta_2": np.deg2rad(30) * np.sin(0.1 * time_steps),
    "theta_3": np.deg2rad(30) * np.sin(0.1 * time_steps),
    "theta_4": np.deg2rad(30) * np.sin(0.1 * time_steps),
    "theta_5": np.deg2rad(30) * np.sin(0.1 * time_steps),
}

# Compute energies and torques for each configuration over time
energies = {}
kinetic_energies = {}
torques = {}

for name, angles in configurations.items():
    energy_vals = []
    kinetic_energy_vals = []
    torque_vals = []

    for t in time_steps:
        angle_subs = {
            theta_1: np.deg2rad(angles["theta_1"]) + joint_angles["theta_1"][int(t)],
            theta_2: np.deg2rad(angles["theta_2"]) + joint_angles["theta_2"][int(t)],
            theta_3: np.deg2rad(angles["theta_3"]) + joint_angles["theta_3"][int(t)],
            theta_4: np.deg2rad(angles["theta_4"]) + joint_angles["theta_4"][int(t)],
            theta_5: np.deg2rad(angles["theta_5"]) + joint_angles["theta_5"][int(t)],
            dtheta_1: 0,
            dtheta_2: 0,
            dtheta_3: 0,
            dtheta_4: 0,
            dtheta_5: 0,
        }

        energy_vals.append(V.subs(angle_subs))
        kinetic_energy_vals.append(T_kin.subs(angle_subs))

        torque_values = [
            tau1.subs(angle_subs),
            tau2.subs(angle_subs),
            tau3.subs(angle_subs),
            tau4.subs(angle_subs),
            tau5.subs(angle_subs),
        ]
        torque_vals.append(torque_values)

    energies[name] = energy_vals
    kinetic_energies[name] = kinetic_energy_vals
    torques[name] = torque_vals

# Plotting
fig, axes = plt.subplots(3, 1, figsize=(12, 15))

# Plot potential energies
for name in configurations.keys():
    axes[0].plot(time_steps, [float(energy) for energy in energies[name]], label=name)
axes[0].set_ylabel("Potential Energy (J)")
axes[0].set_title("Potential Energy over Time for Different Robot Configurations")
axes[0].legend()

# Plot kinetic energies
for name in configurations.keys():
    axes[1].plot(
        time_steps, [float(energy) for energy in kinetic_energies[name]], label=name
    )
axes[1].set_ylabel("Kinetic Energy (J)")
axes[1].set_title("Kinetic Energy over Time for Different Robot Configurations")
axes[1].legend()

# Plot torques
torque_labels = [f"Tau_{i+1}" for i in range(5)]
for i in range(5):
    for name in configurations.keys():
        torque_array = np.array([torques[name][t][i] for t in range(len(time_steps))])
        axes[2].plot(time_steps, torque_array, label=f"{name} {torque_labels[i]}")
axes[2].set_ylabel("Torque due to Gravity (Nm)")
axes[2].set_title("Torques due to Gravity over Time for Different Robot Configurations")
axes[2].legend()

plt.tight_layout()
plt.show()
