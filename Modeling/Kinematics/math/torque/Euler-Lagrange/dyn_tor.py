import numpy as np
import matplotlib.pyplot as plt
from itertools import product


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


def compute_torques_lagrangian_dynamic(
    d_1_val,
    d_5_val,
    a_2_val,
    a_3_val,
    masses,
    inertias,
    mass_camera,
    mass_lights,
):
    g = 9.81  # Gravity acceleration
    alpha = [np.pi / 2, 0, 0, np.pi / 2, 0]

    # Precompute transformation matrices
    def compute_transforms(angles):
        theta_1, theta_2, theta_3, theta_4, theta_5 = angles
        A1 = dh_matrix(theta_1, d_1_val, 0, alpha[0])
        A2 = dh_matrix(theta_2, 0, a_2_val, alpha[1])
        A3 = dh_matrix(theta_3, 0, a_3_val, alpha[2])
        A4 = dh_matrix(theta_4, 0, 0, alpha[3])
        A5 = dh_matrix(theta_5, d_5_val, 0, alpha[4])

        T1 = A1
        T2 = T1 @ A2
        T3 = T2 @ A3
        T4 = T3 @ A4
        T5 = T4 @ A5

        return [T1, T2, T3, T4, T5]

    # Compute the center of mass positions for each link
    def compute_com_positions(transforms):
        p1 = transforms[0][:3, 3] / 2
        p2 = transforms[1][:3, 3] / 2
        p3 = transforms[2][:3, 3] / 2
        p4 = transforms[3][:3, 3] / 2
        p5 = transforms[4][:3, 3] / 2
        return [p1, p2, p3, p4, p5]

    # Compute the angular velocities for each link
    def compute_angular_velocities(angle_dots):
        theta_1_dot, theta_2_dot, theta_3_dot, theta_4_dot, theta_5_dot = angle_dots
        omega_1 = np.array([0, 0, theta_1_dot])
        omega_2 = np.array([0, theta_2_dot, 0]) + omega_1
        omega_3 = np.array([0, 0, theta_3_dot]) + omega_2
        omega_4 = np.array([theta_4_dot, 0, 0]) + omega_3
        omega_5 = np.array([0, theta_5_dot, 0]) + omega_4
        return [omega_1, omega_2, omega_3, omega_4, omega_5]

    # Compute the linear velocities of the center of mass for each link using finite differences
    def compute_linear_velocities(com_positions, angle_dots):
        epsilon = 1e-5
        linear_velocities = []
        for i in range(5):
            perturbation = np.zeros(5)
            perturbation[i] = epsilon
            angles_plus = np.array(angle_dots) + perturbation
            angles_minus = np.array(angle_dots) - perturbation
            transforms_plus = compute_transforms(angles_plus)
            transforms_minus = compute_transforms(angles_minus)
            com_positions_plus = compute_com_positions(transforms_plus)
            com_positions_minus = compute_com_positions(transforms_minus)
            v = (np.array(com_positions_plus) - np.array(com_positions_minus)) / (
                2 * epsilon
            )
            linear_velocities.append(v[i])
        return linear_velocities

    # Compute the kinetic energy for each link
    def compute_kinetic_energy(angles, angle_dots, masses, inertias):
        transforms = compute_transforms(angles)
        com_positions = compute_com_positions(transforms)
        omega = compute_angular_velocities(angle_dots)
        p_dot = compute_linear_velocities(com_positions, angle_dots)

        T_kinetic = 0
        for i in range(5):
            T_kinetic += 0.5 * masses[i] * np.dot(p_dot[i], p_dot[i]) + 0.5 * np.dot(
                omega[i], inertias[i] @ omega[i]
            )
        return T_kinetic

    # Compute the potential energy for each link
    def compute_potential_energy(com_positions, masses, mass_camera, mass_lights):
        P_potential = 0
        for i in range(5):
            P_potential += masses[i] * g * com_positions[i][2]
        P_potential += (mass_camera + mass_lights) * g * com_positions[-1][2]
        return P_potential

    # Compute the Lagrangian
    def compute_lagrangian(
        angles, angle_dots, masses, inertias, mass_camera, mass_lights
    ):
        transforms = compute_transforms(angles)
        com_positions = compute_com_positions(transforms)
        T_kinetic = compute_kinetic_energy(angles, angle_dots, masses, inertias)
        P_potential = compute_potential_energy(
            com_positions, masses, mass_camera, mass_lights
        )
        L = T_kinetic - P_potential
        return L

    # Numerical values for testing
    angle_range = np.linspace(-np.pi, np.pi, 10)
    angle_dots = [0, 0, 0, 0, 0]  # Assuming initial velocities are zero
    max_torque_per_joint = np.zeros(5)

    # Iterate over all angle combinations and compute torques
    for angles in product(angle_range, repeat=5):
        L = compute_lagrangian(
            angles, angle_dots, masses, inertias, mass_camera, mass_lights
        )

        # Compute torques using finite differences for numerical derivatives
        torques = []
        for i in range(5):
            perturbation = np.zeros(5)
            perturbation[i] = 1e-5
            angle_dots_plus = np.array(angle_dots) + perturbation
            angle_dots_minus = np.array(angle_dots) - perturbation
            L_dot_plus = compute_lagrangian(
                angles, angle_dots_plus, masses, inertias, mass_camera, mass_lights
            )
            L_dot_minus = compute_lagrangian(
                angles, angle_dots_minus, masses, inertias, mass_camera, mass_lights
            )
            torque = (L_dot_plus - L_dot_minus) / (2 * 1e-5)
            torques.append(torque)

        max_torque_per_joint = np.maximum(max_torque_per_joint, np.abs(torques))

    # Plot the maximum torques
    joints = ["Joint 1", "Joint 2", "Joint 3", "Joint 4", "Joint 5"]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(joints, max_torque_per_joint.tolist(), color="blue")
    plt.xlabel("Joints")
    plt.ylabel("Maximum Torque (Nm)")
    plt.title(
        "Maximum Dynamic Torque on Each Joint Across All Configurations Using Lagrangian Formulation"
    )
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
    return max_torque_per_joint.tolist()


# Experiment with different values
d_1_val = 0.4
d_5_val = 0.2
a_2_val = 0.5
a_3_val = 0.3
masses = [5.0, 5.0, 5.0, 5.0, 5.0]
inertias = [
    np.diag([0.1, 0.1, 0.1]),
    np.diag([0.2, 0.2, 0.2]),
    np.diag([0.3, 0.3, 0.3]),
    np.diag([0.4, 0.4, 0.4]),
    np.diag([0.5, 0.5, 0.5]),
]
mass_camera = 1.0
mass_lights = 1.0

# Note: This is a dynamic analysis
print(
    "Performing dynamic torque analysis across all configurations using Lagrangian formulation..."
)
max_torque_per_joint = compute_torques_lagrangian_dynamic(
    d_1_val,
    d_5_val,
    a_2_val,
    a_3_val,
    masses,
    inertias,
    mass_camera,
    mass_lights,
)

print(f"Maximum Dynamic Torques for given values: {max_torque_per_joint}")

# Explanation for negative torques:
# The torques can be negative because they depend on the direction of the force/movement.
# Negative torque values indicate that the force/movement is in the opposite direction to the positive torque direction.
