import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d import Axes3D
from sympy import symbols, cos, sin, pi, Matrix, N

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

# Store end effector positions
ee_positions = []


# Animation update function
def update(frame):
    ax.cla()
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-0.5, 1])

    if frame <= 100:
        # Interpolate between 0 and the target angle based on the current frame
        interpolated_angles = {
            theta: float(frame * target / 100)
            for theta, target in target_angles.items()
        }
    else:
        # Interpolate between the target angle and 0 based on the current frame
        interpolated_angles = {
            theta: float((200 - frame) * target / 100)
            for theta, target in target_angles.items()
        }

    T1 = DH_matrix(interpolated_angles[theta1], d1, a1, alpha1)
    T2 = DH_matrix(interpolated_angles[theta2], d2, a2, alpha2)
    T3 = DH_matrix(interpolated_angles[theta3], d3, a3, alpha3)
    T4 = DH_matrix(interpolated_angles[theta4], d4, a4, alpha4)
    T5 = DH_matrix(interpolated_angles[theta5], d5, a5, alpha5)

    # Cumulative transformations
    T01 = T1
    T02 = T01 * T2
    T03 = T02 * T3
    T04 = T03 * T4
    T05 = T04 * T5

    # Extract positions
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

    # Plot the robot arm
    ax.plot(x_vals, y_vals, z_vals, "ro-", label="Robot Arm")

    # Store and plot the end effector trace
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


# Create animation
ani = FuncAnimation(fig, update, frames=200, repeat=False)

# Save the animation
ani.save("robot_arm_animation_cycle.gif", writer=PillowWriter(fps=10))

plt.show()
