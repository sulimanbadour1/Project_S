import sympy as sp
import numpy as np
import matplotlib.pyplot as plt


def compute_torques_lagrangian(
    d_1_val,
    d_5_val,
    a_2_val,
    a_3_val,
    masses,
    inertias,
    angles,
    mass_camera,
    mass_lights,
    external_forces,
    external_torques,
):
    # Define symbolic variables for joint angles and their derivatives
    theta_1, theta_2, theta_3, theta_4, theta_5 = sp.symbols(
        "theta_1 theta_2 theta_3 theta_4 theta_5"
    )
    dtheta_1, dtheta_2, dtheta_3, dtheta_4, dtheta_5 = sp.symbols(
        "dtheta_1 dtheta_2 dtheta_3 dtheta_4 dtheta_5"
    )

    # Define DH parameters
    d_1, d_5 = sp.symbols("d_1 d_5")
    a_2, a_3 = sp.symbols("a_2 a_3")
    alpha = [90, 0, 0, 90, 0]
    m1, m2, m3, m4, m5 = sp.symbols("m1 m2 m3 m4 m5")
    g = 9.81  # Gravity acceleration

    # Define inertia matrices (assuming simple diagonal form for simplicity)
    I1_xx, I1_yy, I1_zz = sp.symbols("I1_xx I1_yy I1_zz")
    I2_xx, I2_yy, I2_zz = sp.symbols("I2_xx I2_yy I2_zz")
    I3_xx, I3_yy, I3_zz = sp.symbols("I3_xx I3_yy I3_zz")
    I4_xx, I4_yy, I4_zz = sp.symbols("I4_xx I4_yy I4_zz")
    I5_xx, I5_yy, I5_zz = sp.symbols("I5_xx I5_yy I5_zz")

    I1 = sp.diag(I1_xx, I1_yy, I1_zz)
    I2 = sp.diag(I2_xx, I2_yy, I2_zz)
    I3 = sp.diag(I3_xx, I3_yy, I3_zz)
    I4 = sp.diag(I4_xx, I4_yy, I4_zz)
    I5 = sp.diag(I5_xx, I5_yy, I5_zz)

    # Helper function to create a transformation matrix from DH parameters
    def dh_matrix(theta, d, a, alpha):
        alpha_rad = sp.rad(alpha)
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

    # Create transformation matrices
    A1 = dh_matrix(theta_1, d_1, 0, alpha[0])
    A2 = dh_matrix(theta_2, 0, a_2, alpha[1])
    A3 = dh_matrix(theta_3, 0, a_3, alpha[2])
    A4 = dh_matrix(theta_4, 0, 0, alpha[3])
    A5 = dh_matrix(theta_5, d_5, 0, alpha[4])

    # Compute the individual transformation matrices
    T1 = A1
    T2 = T1 * A2
    T3 = T2 * A3
    T4 = T3 * A4
    T5 = T4 * A5

    # Extract positions and velocities of each link's center of mass
    # Assume center of mass at the middle of each link for simplicity
    p1 = T1[:3, 3] / 2
    p2 = T2[:3, 3] / 2
    p3 = T3[:3, 3] / 2
    p4 = T4[:3, 3] / 2
    p5 = T5[:3, 3] / 2

    dp1 = (
        p1.diff(theta_1) * dtheta_1
        + p1.diff(theta_2) * dtheta_2
        + p1.diff(theta_3) * dtheta_3
        + p1.diff(theta_4) * dtheta_4
        + p1.diff(theta_5) * dtheta_5
    )
    dp2 = (
        p2.diff(theta_1) * dtheta_1
        + p2.diff(theta_2) * dtheta_2
        + p2.diff(theta_3) * dtheta_3
        + p2.diff(theta_4) * dtheta_4
        + p2.diff(theta_5) * dtheta_5
    )
    dp3 = (
        p3.diff(theta_1) * dtheta_1
        + p3.diff(theta_2) * dtheta_2
        + p3.diff(theta_3) * dtheta_3
        + p3.diff(theta_4) * dtheta_4
        + p3.diff(theta_5) * dtheta_5
    )
    dp4 = (
        p4.diff(theta_1) * dtheta_1
        + p4.diff(theta_2) * dtheta_2
        + p4.diff(theta_3) * dtheta_3
        + p4.diff(theta_4) * dtheta_4
        + p4.diff(theta_5) * dtheta_5
    )
    dp5 = (
        p5.diff(theta_1) * dtheta_1
        + p5.diff(theta_2) * dtheta_2
        + p5.diff(theta_3) * dtheta_3
        + p5.diff(theta_4) * dtheta_4
        + p5.diff(theta_5) * dtheta_5
    )

    # Kinetic energy (translational and rotational) for each link
    K1 = (1 / 2) * m1 * dp1.dot(dp1) + (1 / 2) * sp.Matrix(
        [dtheta_1, dtheta_2, dtheta_3]
    ).T * I1 * sp.Matrix([dtheta_1, dtheta_2, dtheta_3])
    K2 = (1 / 2) * m2 * dp2.dot(dp2) + (1 / 2) * sp.Matrix(
        [dtheta_1, dtheta_2, dtheta_3]
    ).T * I2 * sp.Matrix([dtheta_1, dtheta_2, dtheta_3])
    K3 = (1 / 2) * m3 * dp3.dot(dp3) + (1 / 2) * sp.Matrix(
        [dtheta_1, dtheta_2, dtheta_3]
    ).T * I3 * sp.Matrix([dtheta_1, dtheta_2, dtheta_3])
    K4 = (1 / 2) * m4 * dp4.dot(dp4) + (1 / 2) * sp.Matrix(
        [dtheta_1, dtheta_2, dtheta_3]
    ).T * I4 * sp.Matrix([dtheta_1, dtheta_2, dtheta_3])
    K5 = (1 / 2) * (m5 + mass_camera + mass_lights) * dp5.dot(dp5) + (
        1 / 2
    ) * sp.Matrix([dtheta_1, dtheta_2, dtheta_3]).T * I5 * sp.Matrix(
        [dtheta_1, dtheta_2, dtheta_3]
    )

    # Total kinetic energy
    K = K1 + K2 + K3 + K4 + K5

    # Potential energy for each link
    P1 = m1 * g * p1[2]
    P2 = m2 * g * p2[2]
    P3 = m3 * g * p3[2]
    P4 = m4 * g * p4[2]
    P5 = (m5 + mass_camera + mass_lights) * g * p5[2]

    # Total potential energy
    P = P1 + P2 + P3 + P4 + P5

    # Lagrangian
    L = K - P

    # Derive the torques using the Euler-Lagrange equation
    torques = []
    for theta, dtheta in zip(
        [theta_1, theta_2, theta_3, theta_4, theta_5],
        [dtheta_1, dtheta_2, dtheta_3, dtheta_4, dtheta_5],
    ):
        dL_dtheta = L.diff(theta)
        dL_ddtheta = L.diff(dtheta)
        ddL_ddtheta_dt = (
            sp.diff(dL_ddtheta, theta_1) * dtheta_1
            + sp.diff(dL_ddtheta, theta_2) * dtheta_2
            + sp.diff(dL_ddtheta, theta_3) * dtheta_3
            + sp.diff(dL_ddtheta, theta_4) * dtheta_4
            + sp.diff(dL_ddtheta, theta_5) * dtheta_5
        )
        torque = ddL_ddtheta_dt - dL_dtheta
        torques.append(torque)

    # Simplify torques
    torques_simplified = [torque.simplify() for torque in torques]

    # Provide numerical values for testing
    values = {
        d_1: d_1_val,
        d_5: d_5_val,
        a_2: a_2_val,
        a_3: a_3_val,
        m1: masses[0],
        m2: masses[1],
        m3: masses[2],
        m4: masses[3],
        m5: masses[4],
        I1_xx: inertias[0][0],
        I1_yy: inertias[0][1],
        I1_zz: inertias[0][2],
        I2_xx: inertias[1][0],
        I2_yy: inertias[1][1],
        I2_zz: inertias[1][2],
        I3_xx: inertias[2][0],
        I3_yy: inertias[2][1],
        I3_zz: inertias[2][2],
        I4_xx: inertias[3][0],
        I4_yy: inertias[3][1],
        I4_zz: inertias[3][2],
        I5_xx: inertias[4][0],
        I5_yy: inertias[4][1],
        I5_zz: inertias[4][2],
        F_ext_x: external_forces[0],
        F_ext_y: external_forces[1],
        F_ext_z: external_forces[2],
        T_ext_1: external_torques[0],
        T_ext_2: external_torques[1],
        T_ext_3: external_torques[2],
        T_ext_4: external_torques[3],
        T_ext_5: external_torques[4],
    }

    # Initialize the maximum torque tracker
    max_torque_per_joint = [-float("inf")] * 5

    # Define the range for joint angles
    angle_range = np.linspace(-np.pi, np.pi, 10)  # 10 steps from -π to π

    for theta_1_val in angle_range:
        for theta_2_val in angle_range:
            for theta_3_val in angle_range:
                for theta_4_val in angle_range:
                    for theta_5_val in angle_range:
                        values.update(
                            {
                                theta_1: theta_1_val,
                                theta_2: theta_2_val,
                                theta_3: theta_3_val,
                                theta_4: theta_4_val,
                                theta_5: theta_5_val,
                                dtheta_1: 0,
                                dtheta_2: 0,
                                dtheta_3: 0,
                                dtheta_4: 0,
                                dtheta_5: 0,
                            }
                        )

                        # Compute numerical torques for the current configuration
                        numerical_torques = [
                            torque.subs(values) for torque in torques_simplified
                        ]

                        # Update the maximum torques observed
                        for i in range(5):
                            torque_val = float(numerical_torques[i])
                            if abs(torque_val) > abs(max_torque_per_joint[i]):
                                max_torque_per_joint[i] = torque_val

    # Plot the maximum torques
    joints = ["Joint 1", "Joint 2", "Joint 3", "Joint 4", "Joint 5"]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(joints, max_torque_per_joint, color="blue")
    plt.xlabel("Joints")
    plt.ylabel("Maximum Torque (Nm)")
    plt.title(
        "Maximum Static Torque on Each Joint Across All Configurations Using Lagrangian Formulation"
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
    print(f"Maximum Torques for given values: {max_torque_per_joint}")
    return max_torque_per_joint


# Experiment with different values
d_1_val = 0.1
d_5_val = 0.1
a_2_val = 0.5
a_3_val = 0.5
masses = [1.0, 1.0, 1.0, 1.0, 1.0]
inertias = [
    [0.1, 0.1, 0.1],
    [0.1, 0.1, 0.1],
    [0.1, 0.1, 0.1],
    [0.1, 0.1, 0.1],
    [0.1, 0.1, 0.1],
]
mass_camera = 0.5
mass_lights = 0.5
external_forces = [0, 0, 0]  # No external forces in this example
external_torques = [0, 0, 0, 0, 0]  # No external torques in this example

# Note: This is a static analysis
print(
    "Performing static torque analysis across all configurations using Lagrangian formulation..."
)
max_torque_per_joint = compute_torques_lagrangian(
    d_1_val,
    d_5_val,
    a_2_val,
    a_3_val,
    masses,
    inertias,
    [0, 0, 0, 0, 0],  # Placeholder, angles will be varied within the function
    mass_camera,
    mass_lights,
    external_forces,
    external_torques,
)

print(f"Maximum Torques for given values: {max_torque_per_joint}")

# Explanation for negative torques:
# The torques can be negative because they depend on the direction of the force/movement.
# Negative torque values indicate that the force/movement is in the opposite direction to the positive torque direction.
