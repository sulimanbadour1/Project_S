import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


class RobotArm:
    def __init__(self, d1, a2, a3, d5):
        self.d1 = d1
        self.a2 = a2
        self.a3 = a3
        self.d5 = d5

    def inverse_kinematics(self, Px, Py, Pz, omega):
        R = self.d5 * math.cos(math.radians(omega))
        theta1 = math.degrees(math.atan2(Py, Px))

        Pxw = Px - R * math.cos(math.radians(theta1))
        Pyw = Py - R * math.sin(math.radians(theta1))
        Pzw = Pz + self.d5 * math.sin(math.radians(omega))

        Rw = math.sqrt(Pxw**2 + Pyw**2)
        S = math.sqrt((Pzw - self.d1) ** 2 + Rw**2)

        alpha = math.degrees(math.atan2(Pzw - self.d1, Rw))
        beta = math.degrees(
            math.acos((self.a2**2 + S**2 - self.a3**2) / (2 * self.a2 * S))
        )

        theta2 = alpha + beta
        theta2_alt = alpha - beta

        theta3 = math.degrees(
            math.acos((S**2 - self.a2**2 - self.a3**2) / (2 * self.a2 * self.a3))
        )
        theta3 = -theta3

        theta234 = 90 - omega
        theta4 = theta234 - theta2 - theta3

        return theta1, theta2, theta3, theta4

    def forward_kinematics(self, theta1, theta2, theta3, theta4):
        theta1 = math.radians(theta1)
        theta2 = math.radians(theta2)
        theta3 = math.radians(theta3)
        theta4 = math.radians(theta4)

        x0, y0, z0 = 0, 0, 0
        x1, y1, z1 = 0, 0, self.d1
        x2 = self.a2 * math.cos(theta1) * math.cos(theta2)
        y2 = self.a2 * math.sin(theta1) * math.cos(theta2)
        z2 = self.d1 + self.a2 * math.sin(theta2)
        x3 = x2 + self.a3 * math.cos(theta1) * math.cos(theta2 + theta3)
        y3 = y2 + self.a3 * math.sin(theta1) * math.cos(theta2 + theta3)
        z3 = z2 + self.a3 * math.sin(theta2 + theta3)

        x4 = x3 + self.d5
        y4 = y3
        z4 = z3

        return [(x0, y0, z0), (x1, y1, z1), (x2, y2, z2), (x3, y3, z3), (x4, y4, z4)]

    def plot_robot(self, joint_positions):
        x, y, z = zip(*joint_positions)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
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
        ax.legend()
        ax.set_box_aspect([1, 1, 1])
        plt.show()


# Example parameters
Px, Py, Pz = 1, 0, 0
omega = 0

# Define DH parameters
d1, a1, alpha1 = 0.1, 0, math.pi / 2
d2, a2, alpha2 = 0, 0.5, 0
d3, a3, alpha3 = 0, 0.5, 0
d4, a4, alpha4 = 0, 0, math.pi / 2
d5, a5, alpha5 = 0.1, 0, 0

robot_arm = RobotArm(d1, a2, a3, d5)
theta1, theta2, theta3, theta4 = robot_arm.inverse_kinematics(Px, Py, Pz, omega)
joint_positions = robot_arm.forward_kinematics(theta1, theta2, theta3, theta4)
robot_arm.plot_robot(joint_positions)

print(f"Theta1: {theta1:.2f} degrees")
print(f"Theta2: {theta2:.2f} degrees")
print(f"Theta3: {theta3:.2f} degrees")
print(f"Theta4: {theta4:.2f} degrees")
print(f"Joint positions: {joint_positions}")
print(f"End effector position: {joint_positions[-1]}")
