import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Define DH parameters
d1, a1, alpha1 = 0.1, 0, math.pi / 2
d2, a2, alpha2 = 0, 0.5, 0
d3, a3, alpha3 = 0, 0.5, 0
d4, a4, alpha4 = 0, 0, math.pi / 2
d5, a5, alpha5 = 0.1, 0, 0


def inverse_kinematics(Px, Py, Pz, d1, a2, a3, d5, omega):
    # Calculate wrist position coordinates
    R = d5 * math.cos(math.radians(omega))
    theta1 = math.degrees(math.atan2(Py, Px))
    theta1_rad = math.radians(theta1)

    Pxw = Px - R * math.cos(theta1_rad)
    Pyw = Py - R * math.sin(theta1_rad)
    Pzw = Pz + d5 * math.sin(math.radians(omega))

    # Calculate Rw and S
    Rw = math.sqrt(Pxw**2 + Pyw**2)
    S = math.sqrt((Pzw - d1) ** 2 + Rw**2)

    # Calculate theta2 and theta3
    alpha = math.degrees(math.atan2(Pzw - d1, Rw))
    beta = math.degrees(math.acos((a2**2 + S**2 - a3**2) / (2 * a2 * S)))

    theta2 = alpha + beta
    # or
    theta2_alt = alpha - beta

    # Calculate theta3

    theta3 = math.degrees(math.acos((S**2 - a2**2 - a3**2) / (2 * a2 * a3)))

    # or
    theta3 = -theta3  # Adjust for proper direction

    # Calculate theta4
    theta234 = 90 - omega
    # theta234_alt = 90 + omega
    theta4 = theta234 - theta2 - theta3

    return theta1, theta2, theta3, theta4


def forward_kinematics(
    d1,
    a2,
    a3,
    d5,
    theta1,
    theta2,
    theta3,
    theta4,
):
    # Convert angles to radians
    theta1 = math.radians(theta1)
    theta2 = math.radians(theta2)
    theta3 = math.radians(theta3)
    theta4 = math.radians(theta4)

    omega = 90 - (theta2 + theta3 + theta4)
    print(f"Omega: {omega} degrees from the forward kinematics function\n")

    # Joint positions
    x0, y0, z0 = 0, 0, 0
    x1, y1, z1 = 0, 0, d1
    x2 = a2 * math.cos(theta1) * math.cos(theta2)
    y2 = a2 * math.sin(theta1) * math.cos(theta2)
    z2 = d1 + a2 * math.sin(theta2)
    x3 = x2 + a3 * math.cos(theta1) * math.cos(theta2 + theta3)
    y3 = y2 + a3 * math.sin(theta1) * math.cos(theta2 + theta3)
    z3 = z2 + a3 * math.sin(theta2 + theta3)
    x4 = x3 + d5 * math.cos(math.radians(omega)) * math.cos(theta1) * math.cos(
        theta2 + theta3
    )
    y4 = y3 + d5 * math.cos(math.radians(omega)) * math.sin(theta1) * math.cos(
        theta2 + theta3
    )
    z4 = z3 + d5

    print("Joint Positions:\n")
    print(f"x0: {x0}, y0: {y0}, z0: {z0}")
    print(f"x1: {x1}, y1: {y1}, z1: {z1}")
    print(f"x2: {x2}, y2: {y2}, z2: {z2}")
    print(f"x3: {x3}, y3: {y3}, z3: {z3}")
    print(f"x4: {x4}, y4: {y4}, z4: {z4}")
    print("\n")

    return [(x0, y0, z0), (x1, y1, z1), (x2, y2, z2), (x3, y3, z3), (x4, y4, z4)]


def plot_robot(joint_positions):
    # Unpack joint positions
    x, y, z = zip(*joint_positions)

    # Plot the robot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(x, y, z, "o-", markersize=10, label="Robot Arm")
    ax.scatter(x, y, z, c="k")

    # Add text labels for the links and lengths
    for i in range(len(joint_positions) - 1):
        ax.text(
            (x[i] + x[i + 1]) / 2,
            (y[i] + y[i + 1]) / 2,
            (z[i] + z[i + 1]) / 2,
            f"Link {i + 1}",
            color="black",
        )

    # Set labels and title
    ax.set_xlabel("X axis")
    ax.set_ylabel("Y axis")
    ax.set_zlabel("Z axis")
    ax.set_title("3D Robot Configuration")
    ax.legend()
    ax.set_box_aspect([1, 1, 1])
    plt.show()


# Example parameters
Px, Py, Pz = 1, 0, 0
omega = -90
theta1, theta2, theta3, theta4 = inverse_kinematics(Px, Py, Pz, d1, a2, a3, d5, omega)
print(f"Theta1: {theta1} degrees")
print(f"Theta2: {theta2} degrees")
print(f"Theta3: {theta3} degrees")
print(f"Theta4: {theta4} degrees")
print(f"omega: {omega} degrees")

# Compute forward kinematics for plotting
joint_positions = forward_kinematics(d1, a2, a3, d5, theta1, theta2, theta3, theta4)

# Print the end effector position
end_effector_position = joint_positions[-1]
print(f"End Effector Position: {end_effector_position}")

print("Plotting robot...")
plot_robot(joint_positions)
