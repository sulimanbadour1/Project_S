import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np

# Define DH parameters
d1, a2, a3, d5 = 0.1, 0.5, 0.5, 0.1


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
    theta3 = -theta3  # Adjust for proper direction

    # Calculate theta4
    theta234 = 90 - omega
    theta4 = theta234 - theta2 - theta3

    return theta1, theta2, theta3, theta4


def forward_kinematics(d1, a2, a3, d5, theta1, theta2, theta3, theta4):
    # Convert angles to radians
    theta1 = math.radians(theta1)
    theta2 = math.radians(theta2)
    theta3 = math.radians(theta3)
    theta4 = math.radians(theta4)

    omega = 90 - (theta2 + theta3 + theta4)

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

    return [(x0, y0, z0), (x1, y1, z1), (x2, y2, z2), (x3, y3, z3), (x4, y4, z4)]


def plot_robot(ax, joint_positions, theta1, theta2, theta3, theta4, end_effector_pos):
    # Unpack joint positions
    x, y, z = zip(*joint_positions)

    # Plot the robot
    ax.plot(x, y, z, "o-", markersize=10, label="Robot Arm", linewidth=2)
    ax.scatter(x, y, z, c="k")

    # Add text labels for the links and lengths
    for i in range(len(joint_positions) - 1):
        ax.text(
            (x[i] + x[i + 1]) / 2,
            (y[i] + y[i + 1]) / 2,
            (z[i] + z[i + 1]) / 2,
            f"Link {i + 1}",
            color="black",
            fontsize=10,
            fontweight="bold",
        )

    # Add text for joint angles
    ax.text2D(
        0.05,
        0.95,
        f"Theta1: {theta1:.2f}°",
        transform=ax.transAxes,
        color="red",
        fontsize=12,
        fontweight="bold",
    )
    ax.text2D(
        0.05,
        0.90,
        f"Theta2: {theta2:.2f}°",
        transform=ax.transAxes,
        color="red",
        fontsize=12,
        fontweight="bold",
    )
    ax.text2D(
        0.05,
        0.85,
        f"Theta3: {theta3:.2f}°",
        transform=ax.transAxes,
        color="red",
        fontsize=12,
        fontweight="bold",
    )
    ax.text2D(
        0.05,
        0.80,
        f"Theta4: {theta4:.2f}°",
        transform=ax.transAxes,
        color="red",
        fontsize=12,
        fontweight="bold",
    )

    # Add text for end effector position with units
    ax.text2D(
        0.05,
        0.75,
        f"End Effector Position: ({end_effector_pos[0]:.2f} m, "
        f"{end_effector_pos[1]:.2f} m, {end_effector_pos[2]:.2f} m)",
        transform=ax.transAxes,
        color="red",
        fontsize=12,
        fontweight="bold",
    )


# Define the sinusoidal path parameters
amplitude = 0.2
num_points = 100
x_range = np.linspace(-0.5, 0.5, num_points)
Py = 0.3
Px = x_range
Pz = amplitude * np.sin(2 * np.pi * x_range) + 0.5  # Sinusoidal path in X-Z plane

# Compute inverse kinematics for each point on the path
joint_angles = [
    inverse_kinematics(px, Py, pz, d1, a2, a3, d5, omega=-90) for px, pz in zip(Px, Pz)
]

# Create the static figure
fig_static = plt.figure()
ax_static = fig_static.add_subplot(111, projection="3d")
ax_static.set_xlabel("X axis (m)", fontsize=12, fontweight="bold")
ax_static.set_ylabel("Y axis (m)", fontsize=12, fontweight="bold")
ax_static.set_zlabel("Z axis (m)", fontsize=12, fontweight="bold")
ax_static.set_title("3D Robot Configuration", fontsize=14, fontweight="bold")
ax_static.set_xlim([-1, 1])
ax_static.set_ylim([-1, 1])
ax_static.set_zlim([0, 1])

# List to store end effector positions
path_x = []
path_y = []
path_z = []

# Plot the robot at 10 evenly spaced points along the trajectory
indices = np.linspace(0, num_points - 1, 10, dtype=int)
for i in indices:
    theta1, theta2, theta3, theta4 = joint_angles[i]
    joint_positions = forward_kinematics(d1, a2, a3, d5, theta1, theta2, theta3, theta4)
    end_effector_pos = joint_positions[-1]
    plot_robot(
        ax_static, joint_positions, theta1, theta2, theta3, theta4, end_effector_pos
    )

    # Store the end effector position
    path_x.append(end_effector_pos[0])
    path_y.append(end_effector_pos[1])
    path_z.append(end_effector_pos[2])

    # Print FK vs IK results for validation
    ik_result = (Px[i], Py, Pz[i])
    fk_result = end_effector_pos
    print(f"IK result: {ik_result}")
    print(f"FK result: {fk_result}")
    print(
        f"Theta1: {theta1:.2f}°, Theta2: {theta2:.2f}°, Theta3: {theta3:.2f}°, Theta4: {theta4:.2f}°"
    )
    print("-" * 30)

# Plot the path traced by the end effector
ax_static.plot(path_x, path_y, path_z, "r--", label="Path")

plt.legend()
plt.show()


# Create the animation
fig_anim = plt.figure()
ax_anim = fig_anim.add_subplot(111, projection="3d")
ax_anim.set_xlabel("X axis (m)", fontsize=12, fontweight="bold")
ax_anim.set_ylabel("Y axis (m)", fontsize=12, fontweight="bold")
ax_anim.set_zlabel("Z axis (m)", fontsize=12, fontweight="bold")
ax_anim.set_title("3D Robot Configuration", fontsize=14, fontweight="bold")
ax_anim.set_xlim([-1, 1])
ax_anim.set_ylim([-1, 1])
ax_anim.set_zlim([0, 1])


# Initialization function for the animation
def init():
    ax_anim.clear()
    ax_anim.set_xlabel("X axis (m)", fontsize=12, fontweight="bold")
    ax_anim.set_ylabel("Y axis (m)", fontsize=12, fontweight="bold")
    ax_anim.set_zlabel("Z axis (m)", fontsize=12, fontweight="bold")
    ax_anim.set_title("3D Robot Configuration", fontsize=14, fontweight="bold")
    ax_anim.set_xlim([-1, 1])
    ax_anim.set_ylim([-1, 1])
    ax_anim.set_zlim([0, 1])
    return []


# Animation function
def animate(i):
    ax_anim.clear()
    ax_anim.set_xlabel("X axis (m)", fontsize=12, fontweight="bold")
    ax_anim.set_ylabel("Y axis (m)", fontsize=12, fontweight="bold")
    ax_anim.set_zlabel("Z axis (m)", fontsize=12, fontweight="bold")
    ax_anim.set_title("3D Robot Configuration", fontsize=14, fontweight="bold")
    ax_anim.set_xlim([-1, 1])
    ax_anim.set_ylim([-1, 1])
    ax_anim.set_zlim([0, 1])
    theta1, theta2, theta3, theta4 = joint_angles[i]
    joint_positions = forward_kinematics(d1, a2, a3, d5, theta1, theta2, theta3, theta4)
    end_effector_pos = joint_positions[-1]
    plot_robot(
        ax_anim, joint_positions, theta1, theta2, theta3, theta4, end_effector_pos
    )

    # Store the end effector position
    path_x.append(end_effector_pos[0])
    path_y.append(end_effector_pos[1])
    path_z.append(end_effector_pos[2])

    # Plot the path traced by the end effector
    ax_anim.plot(path_x, path_y, path_z, "r--", label="Path")
    return []


# Create the animation
ani = FuncAnimation(fig_anim, animate, frames=num_points, init_func=init, blit=True)

# Save the animation as a GIF with the updated text properties
ani.save("robot_animation_sinusoidal_with_units.gif", writer=PillowWriter(fps=20))

# Show the animation
plt.show()
