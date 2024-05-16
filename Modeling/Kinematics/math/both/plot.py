import math
import matplotlib.pyplot as plt
import numpy as np

# Define DH parameters
d1, a2, a3, d5 = 0.1, 0.5, 0.5, 0.1


# Inverse kinematics function
def inverse_kinematics(Px, Py, Pz, d1, a2, a3, d5, omega):
    R = d5 * math.cos(math.radians(omega))
    theta1 = math.degrees(math.atan2(Py, Px))
    theta1_rad = math.radians(theta1)

    Pxw = Px - R * math.cos(theta1_rad)
    Pyw = Py - R * math.sin(theta1_rad)
    Pzw = Pz + d5 * math.sin(math.radians(omega))

    Rw = math.sqrt(Pxw**2 + Pyw**2)
    S = math.sqrt((Pzw - d1) ** 2 + Rw**2)

    alpha = math.degrees(math.atan2(Pzw - d1, Rw))
    beta = math.degrees(math.acos((a2**2 + S**2 - a3**2) / (2 * a2 * S)))

    theta2 = alpha + beta
    theta3 = -math.degrees(math.acos((S**2 - a2**2 - a3**2) / (2 * a2 * a3)))

    theta234 = 90 - omega
    theta4 = theta234 - theta2 - theta3

    return theta1, theta2, theta3, theta4


# Forward kinematics function
def forward_kinematics(d1, a2, a3, d5, theta1, theta2, theta3, theta4):
    theta1, theta2, theta3, theta4 = map(math.radians, [theta1, theta2, theta3, theta4])

    x0, y0, z0 = 0, 0, 0
    x1, y1, z1 = 0, 0, d1
    x2 = a2 * math.cos(theta1) * math.cos(theta2)
    y2 = a2 * math.sin(theta1) * math.cos(theta2)
    z2 = d1 + a2 * math.sin(theta2)
    x3 = x2 + a3 * math.cos(theta1) * math.cos(theta2 + theta3)
    y3 = y2 + a3 * math.sin(theta1) * math.cos(theta2 + theta3)
    z3 = z2 + a3 * math.sin(theta2 + theta3)

    x4 = x3 + d5
    y4 = y3
    z4 = z3

    return [(x0, y0, z0), (x1, y1, z1), (x2, y2, z2), (x3, y3, z3), (x4, y4, z4)]


# DH matrix function
def DH_matrix(theta, d, a, alpha):
    theta_rad = np.radians(theta)
    alpha_rad = np.radians(alpha)
    return np.array(
        [
            [
                np.cos(theta_rad),
                -np.sin(theta_rad) * np.cos(alpha_rad),
                np.sin(theta_rad) * np.sin(alpha_rad),
                a * np.cos(theta_rad),
            ],
            [
                np.sin(theta_rad),
                np.cos(theta_rad) * np.cos(alpha_rad),
                -np.cos(theta_rad) * np.sin(alpha_rad),
                a * np.sin(theta_rad),
            ],
            [0, np.sin(alpha_rad), np.cos(alpha_rad), d],
            [0, 0, 0, 1],
        ]
    )


# Plotting function
def plot_robot(ax, joint_positions, method_name):
    x, y, z = zip(*joint_positions)
    ax.plot(x, y, z, "o-", markersize=10, label=method_name)
    ax.scatter(x, y, z, c="k")
    ax.set_xlabel("X axis")
    ax.set_ylabel("Y axis")
    ax.set_zlabel("Z axis")
    ax.set_title(method_name)
    ax.legend()
    ax.set_box_aspect([1, 1, 1])  # Aspect ratio is 1:1:1


# Compute joint positions using inverse kinematics and forward kinematics
Px, Py, Pz = 0.5, 0, -0.5
omega = 0
theta1, theta2, theta3, theta4 = inverse_kinematics(Px, Py, Pz, d1, a2, a3, d5, omega)
joint_positions_kinematics = forward_kinematics(
    d1, a2, a3, d5, theta1, theta2, theta3, theta4
)

# Compute joint positions using DH parameters
theta_degrees = [theta1, theta2, theta3, theta4, 0]
dh_parameters = [
    {"theta": theta_degrees[0], "d": 0.1, "a": 0, "alpha": 90},
    {"theta": theta_degrees[1], "d": 0, "a": 0.5, "alpha": 0},
    {"theta": theta_degrees[2], "d": 0, "a": 0.5, "alpha": 0},
    {"theta": theta_degrees[3], "d": 0, "a": 0, "alpha": 90},
    {"theta": theta_degrees[4], "d": 0.1, "a": 0, "alpha": 0},
]

T = np.eye(4)
joint_positions_dh = [(0, 0, 0)]
for params in dh_parameters:
    T = T @ DH_matrix(params["theta"], params["d"], params["a"], params["alpha"])
    joint_positions_dh.append((T[0, 3], T[1, 3], T[2, 3]))

# Print end effector positions
end_effector_kinematics = joint_positions_kinematics[-1]
end_effector_dh = joint_positions_dh[-1]
print(f"End Effector Position (Kinematics Method): {end_effector_kinematics}")
print(f"End Effector Position (DH Method): {end_effector_dh}")


# Calculate and print the difference
def calculate_difference(pos1, pos2):
    return math.sqrt(sum((p1 - p2) ** 2 for p1, p2 in zip(pos1, pos2)))


difference = calculate_difference(end_effector_kinematics, end_effector_dh)
print(f"Difference between end effector positions: {difference}")

# Ignore small differences
if difference < 0.1:
    print("Difference is less than 0.1, rounding the positions.")
    end_effector_kinematics = tuple(
        round(coord, 1) for coord in end_effector_kinematics
    )
    end_effector_dh = tuple(round(coord, 1) for coord in end_effector_dh)
    joint_positions_kinematics[-1] = end_effector_kinematics
    joint_positions_dh[-1] = end_effector_dh
else:
    print("Difference is greater than 0.1, requires correction.")

# Plot the robot arm using both methods
fig = plt.figure(figsize=(12, 6))

# Subplot 1: Kinematics Method
ax1 = fig.add_subplot(121, projection="3d")
plot_robot(ax1, joint_positions_kinematics, "Inverse Kinematics Method")

# Subplot 2: DH Method
ax2 = fig.add_subplot(122, projection="3d")
plot_robot(ax2, joint_positions_dh, "Forward Kinematics (DH) Method")

plt.show()
