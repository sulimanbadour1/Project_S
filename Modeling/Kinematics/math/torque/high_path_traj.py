import sympy as sp
import numpy as np
import matplotlib.pyplot as plt


# Define the computation function
def compute_symbolic_torques():
    # Define symbolic variables for joint angles, DH parameters, masses, and inertia
    theta_1, theta_2, theta_3, theta_4, theta_5 = sp.symbols(
        "theta_1 theta_2 theta_3 theta_4 theta_5"
    )
    d_1, d_5 = sp.symbols("d_1 d_5")
    a_2, a_3 = sp.symbols("a_2 a_3")
    alpha = [90, 0, 0, 90, 0]
    m1, m2, m3, m4, m5 = sp.symbols("m1 m2 m3 m4 m5")
    g = sp.Matrix([0, 0, -9.81])

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
    ]


def compute_numerical_torques(tau_total, symbols, values):
    # Compute numerical torques
    numerical_torques = tau_total.subs(values)
    return numerical_torques


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
mass_camera = 0.5
mass_lights = 0.5
masses[-1] += mass_camera + mass_lights  # Add to the last mass (end effector)

external_forces = [0, 0, 0, 0, 0, 0]  # No external forces/moments

# Compute symbolic torques
tau_total, symbols = compute_symbolic_torques()

# Define the start and end joint configurations (angles in degrees)
start_angles = [0, 0, 0, 0, 0]
end_angles = [350.0, -90.0, -90.0, -90.0, 0.0]


# Define the number of steps for interpolation
num_steps = 360

# Interpolate the joint angles
trajectory = np.linspace(start_angles, end_angles, num_steps)

# Compute torques for each step in the trajectory
torques_trajectory = []

for angles in trajectory:
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
    torques_trajectory.append([float(torque) for torque in numerical_torques])

# Convert torques_trajectory to a NumPy array for easier plotting
torques_trajectory = np.array(torques_trajectory)

# Plot the torques for each joint over the trajectory
plt.figure(figsize=(12, 8))
for joint in range(1, 6):
    plt.plot(torques_trajectory[:, joint - 1], label=f"Joint {joint}")

plt.xlabel("Trajectory Step")
plt.ylabel("Torque (Nm)")
plt.title("Joint Torques from Point 1 to Point 2")
plt.legend()
plt.grid(True)
plt.show()

# Print the torques for each joint at the start and end points
print("Torques at the start point (Point 1):")
print(torques_trajectory[0])

print("\nTorques at the end point (Point 2):")
print(torques_trajectory[-1])
