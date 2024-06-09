import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from itertools import product


def compute_torques_lagrangian(
    d_1_val,
    d_5_val,
    a_2_val,
    a_3_val,
    masses,
    inertias,
    mass_camera,
    mass_lights,
):
    # Define symbolic variables for joint angles and velocities
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

    # Extract positions of each link's center of mass
    # Assume center of mass at the middle of each link for simplicity
    p1 = T1[:3, 3] / 2
    p2 = T2[:3, 3] / 2
    p3 = T3[:3, 3] / 2
    p4 = T4[:3, 3] / 2
    p5 = T5[:3, 3] / 2

    # Potential energy for each link
    P1 = m1 * g * p1[2]
    P2 = m2 * g * p2[2]
    P3 = m3 * g * p3[2]
    P4 = m4 * g * p4[2]
    P5 = (m5 + mass_camera + mass_lights) * g * p5[2]

    # Total potential energy
    P = P1 + P2 + P3 + P4 + P5

    # Kinetic energy for each link (translational + rotational)
    T1_trans = 0.5 * m1 * (sp.diff(p1, theta_1).T * sp.diff(p1, theta_1))[0]
    T1_rot = 0.5 * sp.Matrix([dtheta_1, 0, 0]).T * I1 * sp.Matrix([dtheta_1, 0, 0])
    T1 = T1_trans + T1_rot[0]

    T2_trans = 0.5 * m2 * (sp.diff(p2, theta_2).T * sp.diff(p2, theta_2))[0]
    T2_rot = 0.5 * sp.Matrix([0, dtheta_2, 0]).T * I2 * sp.Matrix([0, dtheta_2, 0])
    T2 = T2_trans + T2_rot[0]

    T3_trans = 0.5 * m3 * (sp.diff(p3, theta_3).T * sp.diff(p3, theta_3))[0]
    T3_rot = 0.5 * sp.Matrix([0, 0, dtheta_3]).T * I3 * sp.Matrix([0, 0, dtheta_3])
    T3 = T3_trans + T3_rot[0]

    T4_trans = 0.5 * m4 * (sp.diff(p4, theta_4).T * sp.diff(p4, theta_4))[0]
    T4_rot = 0.5 * sp.Matrix([dtheta_4, 0, 0]).T * I4 * sp.Matrix([dtheta_4, 0, 0])
    T4 = T4_trans + T4_rot[0]

    T5_trans = 0.5 * m5 * (sp.diff(p5, theta_5).T * sp.diff(p5, theta_5))[0]
    T5_rot = 0.5 * sp.Matrix([0, 0, dtheta_5]).T * I5 * sp.Matrix([0, 0, dtheta_5])
    T5 = T5_trans + T5_rot[0]

    # Total kinetic energy
    T = T1 + T2 + T3 + T4 + T5

    # Lagrangian
    L = T - P

    # Generalized coordinates and velocities
    thetas = [theta_1, theta_2, theta_3, theta_4, theta_5]
    dthetas = [dtheta_1, dtheta_2, dtheta_3, dtheta_4, dtheta_5]

    # Compute the equations of motion using Euler-Lagrange equation
    equations_of_motion = []
    for i in range(5):
        dL_dtheta = L.diff(thetas[i])
        dL_ddtheta = L.diff(dthetas[i])
        equation = sp.diff(dL_ddtheta, "t") - dL_dtheta
        equations_of_motion.append(equation)

    # Substitute numerical values into equations of motion
    substitutions = {
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
    }
    equations_of_motion_numeric = [eq.subs(substitutions) for eq in equations_of_motion]

    # Precompute the equations of motion function
    try:
        torques_func = sp.lambdify(
            (
                theta_1,
                theta_2,
                theta_3,
                theta_4,
                theta_5,
                dtheta_1,
                dtheta_2,
                dtheta_3,
                dtheta_4,
                dtheta_5,
            ),
            equations_of_motion_numeric,
            "numpy",
        )
    except Exception as e:
        print(f"Lambdify failed: {e}")
        return []

    # Initialize the maximum torque tracker
    max_torque_per_joint = np.zeros(5)

    # Define the range for joint angles and velocities
    angle_range = np.linspace(-np.pi, np.pi, 5)  # Reduced to 5 steps from -π to π
    velocity_range = np.linspace(-2, 2, 3)  # Reduced to 3 steps for angular velocities

    # Generate all combinations of joint angles and velocities
    angle_combinations = list(product(angle_range, repeat=5))
    velocity_combinations = list(product(velocity_range, repeat=5))

    # Convert combinations to numpy arrays for vectorized operations
    angle_combinations = np.array(angle_combinations)
    velocity_combinations = np.array(velocity_combinations)

    # Iterate over all angle combinations
    for angles in angle_combinations:
        # Compute torques for all velocity combinations at once using vectorized operations
        for velocities in velocity_combinations:
            numerical_torques = np.array(
                torques_func(*angles, *velocities), dtype=float
            ).flatten()
            max_torque_per_joint = np.maximum(
                max_torque_per_joint, np.abs(numerical_torques)
            )

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
    print(f"Maximum Torques for given values: {max_torque_per_joint.tolist()}")
    return max_torque_per_joint.tolist()


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

# Note: This is a dynamic analysis
print(
    "Performing dynamic torque analysis across all configurations using Lagrangian formulation..."
)
max_torque_per_joint = compute_torques_lagrangian(
    d_1_val, d_5_val, a_2_val, a_3_val, masses, inertias, mass_camera, mass_lights
)

print(f"Maximum Torques for given values: {max_torque_per_joint}")
