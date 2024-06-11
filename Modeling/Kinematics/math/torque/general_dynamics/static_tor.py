import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from itertools import product


def compute_torques(
    d_1_val,
    d_5_val,
    a_2_val,
    a_3_val,
    masses,
    mass_camera,
    mass_lights,
    external_forces,
    external_torques,
):
    """
    Compute the maximum static torques for a robotic arm across all configurations.

    Parameters:
    d_1_val (float): DH parameter d_1.
    d_5_val (float): DH parameter d_5.
    a_2_val (float): DH parameter a_2.
    a_3_val (float): DH parameter a_3.
    masses (list): List of masses for each link.
    mass_camera (float): Mass of the camera attached to the last link.
    mass_lights (float): Mass of the lights attached to the last link.
    external_forces (list): External forces acting on the end effector [F_ext_x, F_ext_y, F_ext_z].
    external_torques (list): External torques acting on each joint [T_ext_1, T_ext_2, T_ext_3, T_ext_4, T_ext_5].

    Returns:
    list: Maximum torques for each joint.
    """
    # Define symbolic variables for joint angles, DH parameters, and masses
    theta_1, theta_2, theta_3, theta_4, theta_5 = sp.symbols(
        "theta_1 theta_2 theta_3 theta_4 theta_5"
    )
    d_1, d_5 = sp.symbols("d_1 d_5")
    a_2, a_3 = sp.symbols("a_2 a_3")
    alpha = [90, 0, 0, 90, 0]
    m1, m2, m3, m4, m5 = sp.symbols("m1 m2 m3 m4 m5")
    g = sp.Matrix([0, 0, -9.81])

    def dh_matrix(theta, d, a, alpha):
        """Create a transformation matrix from DH parameters."""
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
    p1 = T1[:3, 3] / 2
    p2 = T2[:3, 3] / 2
    p3 = T3[:3, 3] / 2
    p4 = T4[:3, 3] / 2
    p5 = T5[:3, 3] / 2

    # Compute the Jacobians for each center of mass
    Jv1 = p1.jacobian([theta_1, theta_2, theta_3, theta_4, theta_5])
    Jv2 = p2.jacobian([theta_1, theta_2, theta_3, theta_4, theta_5])
    Jv3 = p3.jacobian([theta_1, theta_2, theta_3, theta_4, theta_5])
    Jv4 = p4.jacobian([theta_1, theta_2, theta_3, theta_4, theta_5])
    Jv5 = p5.jacobian([theta_1, theta_2, theta_3, theta_4, theta_5])

    # Compute the gravity vector for each link (assuming center of mass at the link origin)
    G1 = m1 * g
    G2 = m2 * g
    G3 = m3 * g
    G4 = m4 * g
    G5 = (
        m5 + mass_camera + mass_lights
    ) * g  # Adding camera and lights masses to the last link

    # Compute the torques due to gravity for each link
    tau_g1 = Jv1.T * G1
    tau_g2 = Jv2.T * G2
    tau_g3 = Jv3.T * G3
    tau_g4 = Jv4.T * G4
    tau_g5 = Jv5.T * G5

    # Define symbolic variables for external forces and torques
    F_ext_x, F_ext_y, F_ext_z = sp.symbols("F_ext_x F_ext_y F_ext_z")
    T_ext_1, T_ext_2, T_ext_3, T_ext_4, T_ext_5 = sp.symbols(
        "T_ext_1 T_ext_2 T_ext_3 T_ext_4 T_ext_5"
    )

    F_ext = sp.Matrix([F_ext_x, F_ext_y, F_ext_z])
    T_ext = sp.Matrix([T_ext_1, T_ext_2, T_ext_3, T_ext_4, T_ext_5])

    # Compute the Jacobian for the external force application point (assuming it is the end effector)
    Jv_ext = T5[:3, 3].jacobian([theta_1, theta_2, theta_3, theta_4, theta_5])

    # Compute the torques due to external forces and torques
    tau_ext_forces = Jv_ext.T * F_ext
    tau_ext = tau_ext_forces + T_ext

    # Sum the torques due to gravity and external forces/torques
    tau_total = tau_g1 + tau_g2 + tau_g3 + tau_g4 + tau_g5 + tau_ext

    # Simplify the total torques
    tau_total_simplified = sp.simplify(tau_total)

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
        F_ext_x: external_forces[0],
        F_ext_y: external_forces[1],
        F_ext_z: external_forces[2],
        T_ext_1: external_torques[0],
        T_ext_2: external_torques[1],
        T_ext_3: external_torques[2],
        T_ext_4: external_torques[3],
        T_ext_5: external_torques[4],
    }

    # Initialize the maximum torque tracker and store configurations
    max_torque_per_joint = np.zeros(5)
    top_configurations = []

    # Define the range for joint angles
    angle_range = np.linspace(-np.pi, np.pi, 10)  # 10 steps from -π to π

    # Generate all combinations of joint angles
    angle_combinations = list(product(angle_range, repeat=5))

    # Precompute torque function
    tau_total_func = sp.lambdify(
        (theta_1, theta_2, theta_3, theta_4, theta_5),
        tau_total_simplified.subs(values),
        "numpy",
    )

    # Iterate over all angle combinations and compute torques
    for angles in angle_combinations:
        numerical_torques = np.array(tau_total_func(*angles), dtype=float).flatten()
        if np.any(np.abs(numerical_torques) > max_torque_per_joint):
            max_torque_per_joint = np.maximum(
                max_torque_per_joint, np.abs(numerical_torques)
            )
            top_configurations.append((angles, numerical_torques))

    # Sort the configurations by the highest torque experienced
    top_configurations.sort(key=lambda x: np.max(np.abs(x[1])), reverse=True)
    top_configurations = top_configurations[:3]  # Get top three configurations

    # Plot the maximum torques
    joints = ["Joint 1", "Joint 2", "Joint 3", "Joint 4", "Joint 5"]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(joints, max_torque_per_joint.tolist(), color="blue")
    plt.xlabel("Joints")
    plt.ylabel("Maximum Torque (Nm)")
    plt.title("Maximum Static Torque on Each Joint Across All Configurations")
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

    # Plot the top three configurations
    fig = plt.figure(figsize=(15, 10))
    for i, (angles, torques) in enumerate(top_configurations):
        ax = fig.add_subplot(1, 3, i + 1, projection="3d")

        # Evaluate transformation matrices
        T1_eval = np.array(T1.subs(values).subs({theta_1: angles[0]})).astype(
            np.float64
        )
        T2_eval = np.array(
            T2.subs(values).subs({theta_1: angles[0], theta_2: angles[1]})
        ).astype(np.float64)
        T3_eval = np.array(
            T3.subs(values).subs(
                {theta_1: angles[0], theta_2: angles[1], theta_3: angles[2]}
            )
        ).astype(np.float64)
        T4_eval = np.array(
            T4.subs(values).subs(
                {
                    theta_1: angles[0],
                    theta_2: angles[1],
                    theta_3: angles[2],
                    theta_4: angles[3],
                }
            )
        ).astype(np.float64)
        T5_eval = np.array(
            T5.subs(values).subs(
                {
                    theta_1: angles[0],
                    theta_2: angles[1],
                    theta_3: angles[2],
                    theta_4: angles[3],
                    theta_5: angles[4],
                }
            )
        ).astype(np.float64)

        positions = np.vstack(
            [
                np.array([0, 0, 0]),
                T1_eval[:3, 3],
                T2_eval[:3, 3],
                T3_eval[:3, 3],
                T4_eval[:3, 3],
                T5_eval[:3, 3],
            ]
        )

        ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], "o-", markersize=10)
        ax.set_title(f"Configuration {i+1}\nTorques: {np.round(torques, 2)} Nm")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([0, 1])

    plt.show()

    return max_torque_per_joint.tolist()


# Experiment with different values
d_1_val = 0.1
d_5_val = 0.1
a_2_val = 0.5
a_3_val = 0.5
masses = [1.0, 1.0, 1.0, 1.0, 1.0]
mass_camera = 0.5
mass_lights = 0.5
external_forces = [0, 0, 0]  # No external forces in this example
external_torques = [0, 0, 0, 0, 0]  # No external torques in this example

# Note: This is a static analysis
print("Performing static torque analysis across all configurations...")
max_torque_per_joint = compute_torques(
    d_1_val,
    d_5_val,
    a_2_val,
    a_3_val,
    masses,
    mass_camera,
    mass_lights,
    external_forces,
    external_torques,
)

print(f"Maximum Torques for given values: {max_torque_per_joint}")
