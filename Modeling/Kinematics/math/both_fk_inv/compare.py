import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sympy import symbols, cos, sin, pi, Matrix, N


# First set of code: RobotArm class for kinematics and plotting
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


# Second set of code: DH parameters and symbolic calculations
# Define symbolic variables
theta1, theta2, theta3, theta4, theta5 = symbols("theta1 theta2 theta3 theta4 theta5")

# DH Parameters
d1, a1, alpha1 = 0.1, 0, pi / 2
d2, a2, alpha2 = 0, 0.5, 0
d3, a3, alpha3 = 0, 0.5, 0
d4, a4, alpha4 = 0, 0, pi / 2
d5, a5, alpha5 = 0.1, 0, 0


# Define the transformation matrix function using DH parameters
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


# Joint angles in degrees
theta_in_degrees = {
    "theta1": 0,
    "theta2": 18.76,
    "theta3": -50.21,
    "theta4": 121.44,
    "theta5": 0,
}

print("Joint angles in degrees:\n")
print(theta_in_degrees)

# Convert angles to radians
angles = {key: math.radians(value) for key, value in theta_in_degrees.items()}

print("Joint angles in radians:\n")
print(angles)

# Compute transformation matrices
T1 = DH_matrix(theta1, d1, a1, alpha1).subs(angles)
T2 = DH_matrix(theta2, d2, a2, alpha2).subs(angles)
T3 = DH_matrix(theta3, d3, a3, alpha3).subs(angles)
T4 = DH_matrix(theta4, d4, a4, alpha4).subs(angles)
T5 = DH_matrix(theta5, d5, a5, alpha5).subs(angles)

# Calculate cumulative transformations
T01 = T1
T02 = T01 * T2
T03 = T02 * T3
T04 = T03 * T4
T05 = T04 * T5

# Extract positions
positions_inverse = [
    Matrix([0, 0, 0, 1]),  # Base
    T01[:3, 3],
    T02[:3, 3],
    T03[:3, 3],
    T04[:3, 3],
    T05[:3, 3],  # End effector
]
positions_inverse = [N(p) for p in positions_inverse]  # Evaluate numerically

# Extract positions for forward kinematics
robot_arm = RobotArm(d1, a2, a3, d5)
theta1, theta2, theta3, theta4 = robot_arm.inverse_kinematics(1, 0, 0, 0)
joint_positions_forward = robot_arm.forward_kinematics(theta1, theta2, theta3, theta4)

# Plotting
fig, axes = plt.subplots(1, 2, figsize=(12, 6), subplot_kw={"projection": "3d"})

# Plot end effector positions from inverse kinematics
x1_vals = [p[0] for p in positions_inverse]
y1_vals = [p[1] for p in positions_inverse]
z1_vals = [p[2] for p in positions_inverse]
axes[0].plot(x1_vals, y1_vals, z1_vals, "o-", markersize=10, label="Inverse Kinematics")
axes[0].set_title("Inverse Kinematics")
axes[0].set_xlabel("X axis")
axes[0].set_ylabel("Y axis")
axes[0].set_zlabel("Z axis")
axes[0].legend()

# Plot end effector positions from forward kinematics
x2, y2, z2 = zip(*joint_positions_forward)
axes[1].plot(x2, y2, z2, "o-", markersize=10, label="Forward Kinematics")
axes[1].set_title("Forward Kinematics")
axes[1].set_xlabel("X axis")
axes[1].set_ylabel("Y axis")
axes[1].set_zlabel("Z axis")
axes[1].legend()

plt.tight_layout()
plt.show()
