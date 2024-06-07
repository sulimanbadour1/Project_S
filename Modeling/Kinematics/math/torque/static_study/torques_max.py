import sympy as sp
import numpy as np
import matplotlib.pyplot as plt


# Define the computation function
def compute_symbolic_dynamics():
    # Define symbolic variables for joint angles, DH parameters, masses, and inertia
    theta_1, theta_2, theta_3, theta_4, theta_5 = sp.symbols(
        "theta_1 theta_2 theta_3 theta_4 theta_5"
    )
    d_1, d_5 = sp.symbols("d_1 d_5")
    a_2, a_3 = sp.symbols("a_2 a_3")
    alpha = [90, 0, 0, 90, 0]
    m1, m2, m3, m4, m5 = sp.symbols("m1 m2 m3 m4 m5")
    g = sp.Matrix([0, 0, -9.81])
    Ixx1, Iyy1, Izz1 = sp.symbols("Ixx1 Iyy1 Izz1")
    Ixx2, Iyy2, Izz2 = sp.symbols("Ixx2 Iyy2 Izz2")
    Ixx3, Iyy3, Izz3 = sp.symbols("Ixx3 Iyy3 Izz3")
    Ixx4, Iyy4, Izz4 = sp.symbols("Ixx4 Iyy4 Izz4")
    Ixx5, Iyy5, Izz5 = sp.symbols("Ixx5 Iyy5 Izz5")

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
    G5 = m5 * g

    # Compute the torques due to gravity for each link
    tau_g1 = Jv1.T * G1
    tau_g2 = Jv2.T * G2
    tau_g3 = Jv3.T * G3
    tau_g4 = Jv4.T * G4
    tau_g5 = Jv5.T * G5

    # Sum the torques due to gravity
    tau_g = tau_g1 + tau_g2 + tau_g3 + tau_g4 + tau_g5

    # Define external forces and moments
    Fx, Fy, Fz, Mx, My, Mz = sp.symbols("Fx Fy Fz Mx My Mz")
    F_ext = sp.Matrix([Fx, Fy, Fz, Mx, My, Mz])

    # Jacobian for the end effector
    O5 = T5[:3, 3]
    Jv_ee = O5.jacobian([theta_1, theta_2, theta_3, theta_4, theta_5])

    # The angular part of the Jacobian is given by the z-axis of the previous frames
    z0 = sp.Matrix([0, 0, 1])  # z0 axis (base frame)
    z1 = A1[:3, :3] * z0
    z2 = (A1 * A2)[:3, :3] * z0
    z3 = (A1 * A2 * A3)[:3, :3] * z0
    z4 = (A1 * A2 * A3 * A4)[:3, :3] * z0

    # Assemble the angular velocity Jacobian
    Jw = sp.Matrix.hstack(z0, z1, z2, z3, z4)

    # Combine Jv and Jw to form the full Jacobian matrix for the end effector
    J_full = sp.Matrix.vstack(Jv_ee, Jw)

    # Compute the joint torques due to external forces/moments
    tau_ext = J_full.T * F_ext

    # Total torque in equilibrium (gravitational + external)
    tau_total = tau_g - tau_ext

    # Print symbolic torques
    sp.init_printing(use_unicode=True)
    print("Symbolic Total Joint Torques in Equilibrium:")
    sp.pprint(tau_total)

    return tau_total, [
        theta_1,
        theta_2,
        theta_3,
        theta_4,
        theta_5,
        d_1,
        d_5,
        a_2,
        a_3,
        m1,
        m2,
        m3,
        m4,
        m5,
        Fx,
        Fy,
        Fz,
        Mx,
        My,
        Mz,
        Ixx1,
        Iyy1,
        Izz1,
        Ixx2,
        Iyy2,
        Izz2,
        Ixx3,
        Iyy3,
        Izz3,
        Ixx4,
        Iyy4,
        Izz4,
        Ixx5,
        Iyy5,
        Izz5,
    ]


# Define parameters for the robot
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
# Add mass for camera and lights at the end effector
mass_camera = 0
mass_lights = 0
masses[-1] += mass_camera + mass_lights  # Add to the last mass (end effector)

external_forces = [0, 0, 0, 0, 0, 0]  # No external forces/moments

# Compute symbolic torques
tau_total, symbols = compute_symbolic_dynamics()

# Dictionary to store maximum and minimum torques for each joint
max_torques = {i: -np.inf for i in range(1, 6)}  # Initialize with negative infinity
min_torques = {i: np.inf for i in range(1, 6)}  # Initialize with positive infinity
max_torque_angles = {i: None for i in range(1, 6)}  # To store angles for max torques
min_torque_angles = {i: None for i in range(1, 6)}  # To store angles for min torques

# Compute torques for all combinations of joint angles
for theta_1_val, theta_2_val, theta_3_val, theta_4_val, theta_5_val in zip(
    theta_1_vals, theta_2_vals, theta_3_vals, theta_4_vals, theta_5_vals
):

    angles = [theta_1_val, theta_2_val, theta_3_val, theta_4_val, theta_5_val]
    values = {
        symbols[0]: angles[0],
        symbols[1]: angles[1],
        symbols[2]: angles[2],
        symbols[3]: angles[3],
        symbols[4]: angles[4],
        symbols[5]: d_1_val,
        symbols[6]: d_5_val,
        symbols[7]: a_2_val,
        symbols[8]: a_3_val,
        symbols[9]: masses[0],
        symbols[10]: masses[1],
        symbols[11]: masses[2],
        symbols[12]: masses[3],
        symbols[13]: masses[4],
        symbols[14]: external_forces[0],
        symbols[15]: external_forces[1],
        symbols[16]: external_forces[2],
        symbols[17]: external_forces[3],
        symbols[18]: external_forces[4],
        symbols[19]: external_forces[5],
    }
    numerical_torques = compute_numerical_torques(tau_total, symbols, values)
    numerical_torques = [float(torque) for torque in numerical_torques]

    for i in range(5):
        if numerical_torques[i] > max_torques[i + 1]:
            max_torques[i + 1] = numerical_torques[i]
            max_torque_angles[i + 1] = angles
        if numerical_torques[i] < min_torques[i + 1]:
            min_torques[i + 1] = numerical_torques[i]
            min_torque_angles[i + 1] = angles

# Compute the total required torque (max of absolute max and min torques)
total_torques = {i: max(abs(max_torques[i]), abs(min_torques[i])) for i in range(1, 6)}

# Plot the maximum torques for each joint
plt.figure(figsize=(12, 8))
joints = range(1, 6)
max_torque_values = [max_torques[joint] for joint in joints]
min_torque_values = [min_torques[joint] for joint in joints]
total_torque_values = [total_torques[joint] for joint in joints]

plt.plot(joints, max_torque_values, marker="o", label="Max Torque")
plt.plot(joints, min_torque_values, marker="o", label="Min Torque")
plt.plot(
    joints,
    total_torque_values,
    marker="o",
    label="Total Required Torque",
    linestyle="--",
)

plt.xlabel("Joint")
plt.ylabel("Torque (Nm)")
plt.title("Joint Torques in Equilibrium")
plt.xticks(joints)
plt.legend()
plt.grid(True)
plt.show()

# Print the maximum torques for each joint and the corresponding angles
print("Maximum and Minimum Joint Torques and Corresponding Angles:")
for joint in range(1, 6):
    print(
        f"Joint {joint}: Max {max_torques[joint]:.2f} Nm at angles {max_torque_angles[joint]}, "
        f"Min {min_torques[joint]:.2f} Nm at angles {min_torque_angles[joint]}"
    )
    print(f"Total required torque for Joint {joint}: {total_torques[joint]:.2f} Nm")
