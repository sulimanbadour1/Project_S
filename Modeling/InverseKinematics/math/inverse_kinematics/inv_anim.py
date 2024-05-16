import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib.animation import FuncAnimation

# Define DH parameters
d1, a2, a3, d5 = 0.1, 0.5, 0.5, 0.1
omega = -90


def inverse_kinematics(Px, Py, Pz, d1, a2, a3, d5, omega):
    R = d5 * math.cos(math.radians(omega))
    theta1 = math.degrees(math.atan2(Py, Px))
    theta1_rad = math.radians(theta1)

    Pxw = Px - R * math.cos(theta1_rad)
    Pyw = Py - R * math.sin(theta1_rad)
    Pzw = Pz - d5 * math.sin(math.radians(omega))

    Rw = math.sqrt(Pxw**2 + Pyw**2)
    S = math.sqrt((Pzw - d1) ** 2 + Rw**2)

    alpha = math.degrees(math.atan2(Pzw - d1, Rw))
    beta = math.degrees(math.acos((a2**2 + S**2 - a3**2) / (2 * a2 * S)))

    theta2 = alpha + beta
    theta2_alt = alpha - beta

    theta3 = math.degrees(math.acos((S**2 - a2**2 - a3**2) / (2 * a2 * a3)))
    theta3 = -theta3

    theta234 = 90 - omega
    theta4 = theta234 - theta2 - theta3

    return theta1, theta2, theta3, theta4


def forward_kinematics(d1, a2, a3, d5, theta1, theta2, theta3, theta4):
    theta1 = math.radians(theta1)
    theta2 = math.radians(theta2)
    theta3 = math.radians(theta3)
    theta4 = math.radians(theta4)

    x0, y0, z0 = 0, 0, 0
    x1, y1, z1 = 0, 0, d1
    x2 = a2 * math.cos(theta1) * math.cos(theta2)
    y2 = a2 * math.sin(theta1) * math.cos(theta2)
    z2 = d1 + a2 * math.sin(theta2)
    x3 = x2 + a3 * math.cos(theta1) * math.cos(theta2 + theta3)
    y3 = y2 + a3 * math.sin(theta1) * math.cos(theta2 + theta3)
    z3 = z2 + a3 * math.sin(theta2 + theta3)

    x4 = x3
    y4 = y3
    z4 = z3 - d5

    return [(x0, y0, z0), (x1, y1, z1), (x2, y2, z2), (x3, y3, z3), (x4, y4, z4)]


def animate_robot(Px, Py, Pz, d1, a2, a3, d5, omega, frames=100):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    def update(frame):
        ax.cla()
        t = frame / (frames - 1)
        Px_curr = t * Px
        Py_curr = t * Py
        Pz_curr = t * Pz
        theta1, theta2, theta3, theta4 = inverse_kinematics(
            Px_curr, Py_curr, Pz_curr, d1, a2, a3, d5, omega
        )
        joint_positions = forward_kinematics(
            d1, a2, a3, d5, theta1, theta2, theta3, theta4
        )
        x, y, z = zip(*joint_positions)
        ax.plot(x, y, z, "o-", markersize=10, label="Robot Arm")
        ax.scatter(x, y, z, c="k")
        for i in range(len(joint_positions) - 1):
            ax.text(
                (x[i] + x[i + 1]) / 2,
                (y[i] + y[i + 1]) / 2,
                (z[i] + z[i + 1]) / 2,
                f"Link {i + 1}",
                color="black",
            )
        ax.set_xlabel("X axis")
        ax.set_ylabel("Y axis")
        ax.set_zlabel("Z axis")
        ax.set_title("3D Robot Configuration")
        ax.set_box_aspect([1, 1, 1])

    ani = FuncAnimation(fig, update, frames=frames, repeat=False)
    plt.show()


# Example parameters for target position
Px, Py, Pz = 0.8, 0, -0.4
animate_robot(Px, Py, Pz, d1, a2, a3, d5, omega)
