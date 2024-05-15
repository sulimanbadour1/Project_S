import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def inverse_kinematics(Px, Py, Pz, d1, a2, a3, d5, omega):
    # Calculate R
    R = d5 * math.cos(math.radians(omega))

    # Calculate wrist position coordinates
    Pxw = Px - R * math.cos(math.atan2(Py, Px))
    Pyw = Py - R * math.sin(math.atan2(Py, Px))
    Pzw = Pz + d5 * math.sin(math.radians(omega))

    # Calculate Rw
    Rw = math.sqrt(Pxw**2 + Pyw**2)

    # Calculate S
    S = math.sqrt((Pzw - d1) ** 2 + Rw**2)

    # Calculate theta1
    theta1 = math.degrees(math.atan2(Py, Px)) % 360

    # Calculate alpha and beta
    beta = math.degrees(math.acos((S**2 + a2**2 - a3**2) / (2 * a2 * S)))
    alpha = math.degrees(math.atan2(Pzw - d1, Rw))

    # Calculate theta2
    theta2 = alpha + beta
    theta2_alt = alpha - beta
    theta2 = max(0, min(theta2, 60))
    theta2_alt = max(0, min(theta2_alt, 60))

    # Calculate theta3
    theta3 = math.degrees(math.acos((S**2 - a2**2 - a3**2) / (2 * a2 * a3)))
    theta3_alt = -math.degrees(math.acos((S**2 - a2**2 - a3**2) / (2 * a2 * a3)))
    theta3 = max(-45, min(theta3, 45))
    theta3_alt = max(-45, min(theta3_alt, 45))

    # Calculate theta234
    theta234 = 90 - omega

    # Calculate theta4
    theta4 = theta234 - theta2 - theta3
    theta4_alt = theta234 - theta2_alt - theta3_alt
    theta4 = max(-45, min(theta4, 45))
    theta4_alt = max(-45, min(theta4_alt, 45))

    return theta1, theta2, theta3, theta4


def plot_robot(Px, Py, Pz, d1, a2, a3, d5, theta1, theta2, theta3, theta4):
    # Base position
    x0, y0, z0 = 0, 0, 0

    # Joint 1 position
    x1 = 0
    y1 = 0
    z1 = d1

    # Joint 2 position
    x2 = a2 * math.cos(math.radians(theta1)) * math.cos(math.radians(theta2))
    y2 = a2 * math.sin(math.radians(theta1)) * math.cos(math.radians(theta2))
    z2 = d1 + a2 * math.sin(math.radians(theta2))

    # Joint 3 position
    x3 = x2 + a3 * math.cos(math.radians(theta1)) * math.cos(
        math.radians(theta2 + theta3)
    )
    y3 = y2 + a3 * math.sin(math.radians(theta1)) * math.cos(
        math.radians(theta2 + theta3)
    )
    z3 = z2 + a3 * math.sin(math.radians(theta2 + theta3))

    # End-effector position
    x4 = Px
    y4 = Py
    z4 = Pz

    # Plotting the robot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Plot the links
    ax.plot([x0, x1], [y0, y1], [z0, z1], "ro-", label="Joint 1 to Joint 2")
    ax.plot([x1, x2], [y1, y2], [z1, z2], "go-", label="Joint 2 to Joint 3")
    ax.plot([x2, x3], [y2, y3], [z2, z3], "bo-", label="Joint 3 to Joint 4")
    ax.plot([x3, x4], [y3, y4], [z3, z4], "mo-", label="Joint 4 to End-Effector")

    # Plot joints
    ax.scatter([x0, x1, x2, x3, x4], [y0, y1, y2, y3, y4], [z0, z1, z2, z3, z4], c="k")

    # Set labels and title
    ax.set_xlabel("X axis")
    ax.set_ylabel("Y axis")
    ax.set_zlabel("Z axis")
    ax.set_title("3D Robot Configuration")

    # Show legend
    ax.legend()

    # Set equal scaling
    ax.set_box_aspect([1, 1, 1])

    plt.show()


# Example parameters
Px = 0.4
Py = 0
Pz = -0.2
d1 = 0.1
a2 = 0.5
a3 = 0.5
d5 = 0.1
omega = 90

(
    theta1,
    theta2,
    theta3,
    theta4,
) = inverse_kinematics(Px, Py, Pz, d1, a2, a3, d5, omega)
print(f"Theta1: {theta1} degrees")
print(f"Theta2: {theta2} degrees")
print(f"Theta3: {theta3} degrees")
print(f"Theta4: {theta4} degrees")

# Print the end effector position
end_effector_position = [Px, Py, Pz]
print(f"End Effector Position: {end_effector_position}")

print("Plotting robot...")

plot_robot(Px, Py, Pz, d1, a2, a3, d5, theta1, theta2, theta3, theta4)
