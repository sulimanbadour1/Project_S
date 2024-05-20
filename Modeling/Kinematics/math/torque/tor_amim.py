import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d import Axes3D
from sympy import symbols, cos, sin, pi, Matrix, N
import sympy as sp

# Define symbolic variables
theta1, theta2, theta3, theta4, theta5 = symbols("theta1 theta2 theta3 theta4 theta5")

# DH Parameters
d1, a1, alpha1 = 0.1, 0, pi / 2
d2, a2, alpha2 = 0, 0.5, 0
d3, a3, alpha3 = 0, 0.5, 0
d4, a4, alpha4 = 0, 0, pi / 2
d5, a5, alpha5 = 0.1, 0, 0


# Define the transformation matrix using DH parameters
def DH_matrix(theta, d, a, alpha):
    return Matrix(
        [
            [
                cos(theta),
                -sin(theta) * cos(alpha),
                sin(theta) * sin(alpha),
                a * cos(theta),
            ],
            [
                sin(theta),
                cos(theta) * cos(alpha),
                -cos(theta) * sin(alpha),
                a * sin(theta),
            ],
            [0, sin(alpha), cos(alpha), d],
            [0, 0, 0, 1],
        ]
    )


# Compute symbolic torques
def compute_symbolic_torques():
    theta_1, theta_2, theta_3, theta_4, theta_5 = symbols(
        "theta_1 theta_2 theta_3 theta_4 theta_5"
    )
    d_1, d_5 = symbols("d_1 d_5")
    a_2, a_3 = symbols("a_2 a_3")
    alpha = [90, 0, 0, 90, 0]
    m1, m2, m3, m4, m5 = symbols("m1 m2 m3 m4 m5")
    g = Matrix([0, 0, -9.81])

    # Define inertia matrices (assuming simple diagonal form for simplicity)
    I1_xx, I1_yy, I1_zz = symbols("I1_xx I1_yy I1_zz")
    I2_xx, I2_yy, I2_zz = symbols("I2_xx I2_yy I2_zz")
    I3_xx, I3_yy, I3_zz = symbols("I3_xx I3_yy I3_zz")
    I4_xx, I4_yy, I4_zz = symbols("I4_xx I4_yy I4_zz")
    I5_xx, I5_yy, I5_zz = symbols("I5_xx I5_yy I5_zz")

    I1 = sp.diag(I1_xx, I1_yy, I1_zz)
    I2 = sp.diag(I2_xx, I2_yy, I2_zz)
    I3 = sp.diag(I3_xx, I3_yy, I3_zz)
    I4 = sp.diag(I4_xx, I4_yy, I4_zz)
    I5 = sp.diag(I5_xx, I5_yy, I5_zz)

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

    A1 = dh_matrix(theta_1, d_1, 0, alpha[0])
    A2 = dh_matrix(theta_2, 0, a_2, alpha[1])
    A3 = dh_matrix(theta_3, 0, a_3, alpha[2])
    A4 = dh_matrix(theta_4, 0, 0, alpha[3])
    A5 = dh_matrix(theta_5, d_5, 0, alpha[4])

    T1 = A1
    T2 = T1 * A2
    T3 = T2 * A3
    T4 = T3 * A4
    T5 = T4 * A5

    p1 = T1[:3, 3] / 2
    p2 = T2[:3, 3] / 2
    p3 = T3[:3, 3] / 2
    p4 = T4[:3, 3] / 2
    p5 = T5[:3, 3] / 2

    Jv1 = p1.jacobian([theta_1, theta_2, theta_3, theta_4, theta_5])
    Jv2 = p2.jacobian([theta_1, theta_2, theta_3, theta_4, theta_5])
    Jv3 = p3.jacobian([theta_1, theta_2, theta_3, theta_4, theta_5])
    Jv4 = p4.jacobian([theta_1, theta_2, theta_3, theta_4, theta_5])
    Jv5 = p5.jacobian([theta_1, theta_2, theta_3, theta_4, theta_5])

    G1 = m1 * g
    G2 = m2 * g
    G3 = m3 * g
    G4 = m4 * g
    G5 = m5 * g

    tau_g1 = Jv1.T * G1
    tau_g2 = Jv2.T * G2
    tau_g3 = Jv3.T * G3
    tau_g4 = Jv4.T * G4
    tau_g5 = Jv5.T * G5

    tau_g = tau_g1 + tau_g2 + tau_g3 + tau_g4 + tau_g5

    Fx, Fy, Fz, Mx, My, Mz = sp.symbols("Fx Fy Fz Mx My Mz")
    F_ext = sp.Matrix([Fx, Fy, Fz, Mx, My, Mz])

    O5 = T5[:3, 3]
    Jv_ee = O5.jacobian([theta_1, theta_2, theta_3, theta_4, theta_5])

    z0 = sp.Matrix([0, 0, 1])
    z1 = A1[:3, :3] * z0
    z2 = (A1 * A2)[:3, :3] * z0
    z3 = (A1 * A2 * A3)[:3, :3] * z0
    z4 = (A1 * A2 * A3 * A4)[:3, :3] * z0

    Jw = sp.Matrix.hstack(z0, z1, z2, z3, z4)

    J_full = sp.Matrix.vstack(Jv_ee, Jw)

    tau_ext = J_full.T * F_ext

    tau_total = tau_g - tau_ext

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
    numerical_torques = tau_total.subs(values)
    return numerical_torques


# Define robot parameters
d_1_val = 0.1
d_5_val = 0.1
a_2_val = 0.5
a_3_val = 0.5
masses = [1.0, 1.0, 1.0, 1.0, 1.0]
mass_camera = 0.5
mass_lights = 0.5
masses[-1] += mass_camera + mass_lights

external_forces = [0, 0, 0, 0, 0, 0]

# Compute symbolic torques
tau_total, symbols = compute_symbolic_torques()

# Target angles for animation
target_angles = {
    theta1: -pi,
    theta2: -pi / 6,
    theta3: -pi / 2,
    theta4: -pi / 6,
    theta5: -pi / 6,
}

# Prepare the figure and axis
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])
ax.set_zlim([-0.5, 1])

# Store end effector positions and torques
ee_positions = []
torques_over_time = []


# Animation update function
def update(frame):
    ax.cla()
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-0.5, 1])

    if frame <= 100:
        interpolated_angles = {
            theta: float(frame * target / 100)
            for theta, target in target_angles.items()
        }
    else:
        interpolated_angles = {
            theta: float((200 - frame) * target / 100)
            for theta, target in target_angles.items()
        }

    T1 = DH_matrix(interpolated_angles[theta1], d1, a1, alpha1)
    T2 = DH_matrix(interpolated_angles[theta2], d2, a2, alpha2)
    T3 = DH_matrix(interpolated_angles[theta3], d3, a3, alpha3)
    T4 = DH_matrix(interpolated_angles[theta4], d4, a4, alpha4)
    T5 = DH_matrix(interpolated_angles[theta5], d5, a5, alpha5)

    T01 = T1
    T02 = T01 * T2
    T03 = T02 * T3
    T04 = T03 * T4
    T05 = T04 * T5

    positions = [
        Matrix([0, 0, 0, 1]),
        T01[:3, 3],
        T02[:3, 3],
        T03[:3, 3],
        T04[:3, 3],
        T05[:3, 3],
    ]
    positions = [N(p) for p in positions]

    x_vals = [p[0] for p in positions]
    y_vals = [p[1] for p in positions]
    z_vals = [p[2] for p in positions]

    ax.plot(x_vals, y_vals, z_vals, "ro-", label="Robot Arm")

    ee_positions.append(positions[-1])
    ee_trace_x = [p[0] for p in ee_positions]
    ee_trace_y = [p[1] for p in ee_positions]
    ee_trace_z = [p[2] for p in ee_positions]
    ax.plot(ee_trace_x, ee_trace_y, ee_trace_z, "b--", label="End Effector Trace")

    end_effector = positions[-1]
    ax.text(
        x_vals[-1],
        y_vals[-1],
        z_vals[-1],
        f"End Effector\nx: {end_effector[0]:.2f}, y: {end_effector[1]:.2f}, z: {end_effector[2]:.2f}",
        color="blue",
    )
    angle_texts = "\n".join(
        [
            f"{idx + 1}: {np.degrees(val):.2f}°"
            for idx, val in interpolated_angles.items()
        ]
    )
    ax.text2D(
        0.95,
        0.95,
        angle_texts,
        transform=ax.transAxes,
        color="black",
        ha="right",
        va="top",
    )
    ax.set_title("Robot Arm Configuration")
    ax.legend(loc="upper left")

    # Compute torques
    values = {
        symbols[0]: interpolated_angles[theta1],
        symbols[1]: interpolated_angles[theta2],
        symbols[2]: interpolated_angles[theta3],
        symbols[3]: interpolated_angles[theta4],
        symbols[4]: interpolated_angles[theta5],
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
    torques_over_time.append((numerical_torques, interpolated_angles, positions[-1]))


# Create animation
ani = FuncAnimation(fig, update, frames=200, repeat=False)

# Save the animation
ani.save("robot_arm_animation_cycle_with_torques.gif", writer=PillowWriter(fps=10))

plt.show()

# Extract torques, angles, and positions
torques = np.array([item[0] for item in torques_over_time])
angles = [item[1] for item in torques_over_time]
positions = [item[2] for item in torques_over_time]

# Find and print max and min torques
max_torques = np.max(torques, axis=0)
min_torques = np.min(torques, axis=0)

print("Maximum Torques for each joint:")
for i, max_torque in enumerate(max_torques):
    print(f"Joint {i + 1}: {max_torque:.2f} Nm")

print("\nMinimum Torques for each joint:")
for i, min_torque in enumerate(min_torques):
    print(f"Joint {i + 1}: {min_torque:.2f} Nm")

# Write the data to a text file
with open("torques_data.txt", "w") as file:
    file.write("Frame\tJoint Torques (Nm)\tAngles (rad)\tEnd Effector Position (m)\n")
    for frame, (torque, angle, position) in enumerate(torques_over_time):
        angle_values = [
            f"{angle[key]:.2f}" for key in [theta1, theta2, theta3, theta4, theta5]
        ]
        torque_values = [f"{t:.2f}" for t in torque]
        position_values = [f"{pos:.2f}" for pos in position]
        file.write(
            f"{frame}\t{', '.join(torque_values)}\t{', '.join(angle_values)}\t{', '.join(position_values)}\n"
        )

# Plot the torques over time for each joint
plt.figure(figsize=(10, 6))
for i in range(5):
    plt.plot(torques[:, i], label=f"Joint {i+1} Torque")
plt.xlabel("Frame")
plt.ylabel("Torque (Nm)")
plt.title("Joint Torques Over Time")
plt.legend()
plt.grid(True)
plt.show()
