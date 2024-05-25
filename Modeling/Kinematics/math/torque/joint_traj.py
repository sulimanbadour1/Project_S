import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import csv


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

        # Clamping the value for beta calculation
        cos_beta = (self.a2**2 + S**2 - self.a3**2) / (2 * self.a2 * S)
        cos_beta = min(1, max(-1, cos_beta))  # Clamp to the valid range
        beta = math.degrees(math.acos(cos_beta))

        theta2 = alpha + beta
        theta2_alt = alpha - beta

        # Clamping the value for theta3 calculation
        cos_theta3 = (S**2 - self.a2**2 - self.a3**2) / (2 * self.a2 * self.a3)
        cos_theta3 = min(1, max(-1, cos_theta3))  # Clamp to the valid range
        theta3 = math.degrees(math.acos(cos_theta3))
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

    def generate_trajectory(self, start_pos, end_pos, num_points):
        return np.linspace(start_pos, end_pos, num_points)

    def compute_trajectory_ik(self, trajectory):
        joint_angles = []
        for point in trajectory:
            angles = self.inverse_kinematics(point[0], point[1], point[2], point[3])
            joint_angles.append(angles)
        return np.array(joint_angles)

    def compute_velocities_and_accelerations(self, joint_angles, time_step):
        velocities = np.diff(joint_angles, axis=0) / time_step
        accelerations = np.diff(velocities, axis=0) / time_step
        # Padding to maintain array shapes
        velocities = np.vstack((velocities, velocities[-1]))
        accelerations = np.vstack((accelerations, accelerations[-1], accelerations[-1]))
        return velocities, accelerations


# Example parameters
Px_start, Py_start, Pz_start, omega_start = 0.5, 0, -0.5, 0
Px_end, Py_end, Pz_end, omega_end = 0.6, 0, -0.4, 0

# Define DH parameters
d1, a2, a3, d5 = 0.1, 0.5, 0.5, 0.1

robot_arm = RobotArm(d1, a2, a3, d5)

# Generate trajectory
num_points = 100
time_step = 0.1  # seconds
trajectory = robot_arm.generate_trajectory(
    [Px_start, Py_start, Pz_start, omega_start],
    [Px_end, Py_end, Pz_end, omega_end],
    num_points,
)

# Compute joint angles for the trajectory
joint_angles = robot_arm.compute_trajectory_ik(trajectory)

# Compute joint velocities and accelerations
velocities, accelerations = robot_arm.compute_velocities_and_accelerations(
    joint_angles, time_step
)

# Print joint angles, velocities, and accelerations
print("Joint Angles:\n", joint_angles)
print("Joint Velocities:\n", velocities)
print("Joint Accelerations:\n", accelerations)

# Write the values to a CSV file
with open("joint_trajectory.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(
        [
            "Time",
            "Theta1",
            "Theta2",
            "Theta3",
            "Theta4",
            "Theta1_dot",
            "Theta2_dot",
            "Theta3_dot",
            "Theta4_dot",
            "Theta1_ddot",
            "Theta2_ddot",
            "Theta3_ddot",
            "Theta4_ddot",
        ]
    )
    for i in range(num_points):
        time = i * time_step
        angles = joint_angles[i]
        vel = velocities[i]
        acc = accelerations[i]
        writer.writerow([time, *angles, *vel, *acc])

# Plot the robot configuration at the start and end positions
start_joint_positions = robot_arm.forward_kinematics(*joint_angles[0])
end_joint_positions = robot_arm.forward_kinematics(*joint_angles[-1])

robot_arm.plot_robot(start_joint_positions)
robot_arm.plot_robot(end_joint_positions)
