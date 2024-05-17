import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
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
    # print(f"Omega: {omega} degrees from the forward kinematics function\n")

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


def plot_robot(ax, joint_positions):
    # Unpack joint positions
    x, y, z = zip(*joint_positions)

    # Plot the robot
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


# Define the circular path parameters
radius = 0.5
num_points = 100
z_level = 0.8

# Generate points on the circular path
angles = np.linspace(0, 2 * np.pi, num_points)
Px = radius * np.cos(angles)
Py = radius * np.sin(angles)
Pz = np.full_like(Px, z_level)

# Compute inverse kinematics for each point on the path
joint_angles = [
    inverse_kinematics(px, py, pz, d1, a2, a3, d5, omega=-90)
    for px, py, pz in zip(Px, Py, Pz)
]

# Create the animation
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.set_xlabel("X axis")
ax.set_ylabel("Y axis")
ax.set_zlabel("Z axis")
ax.set_title("3D Robot Configuration")
ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])
ax.set_zlim([0, 1])

# List to store end effector positions
path_x = []
path_y = []
path_z = []


# Initialization function for the animation
def init():
    ax.clear()
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([0, 1])
    return []


# Animation function
def animate(i):
    ax.clear()
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([0, 1])
    theta1, theta2, theta3, theta4 = joint_angles[i]
    joint_positions = forward_kinematics(d1, a2, a3, d5, theta1, theta2, theta3, theta4)
    plot_robot(ax, joint_positions)

    # Store the end effector position
    end_effector_pos = joint_positions[-1]
    path_x.append(end_effector_pos[0])
    path_y.append(end_effector_pos[1])
    path_z.append(end_effector_pos[2])

    # Plot the path traced by the end effector
    ax.plot(path_x, path_y, path_z, "r--", label="Path")
    return []


# Create the animation
ani = FuncAnimation(fig, animate, frames=num_points, init_func=init, blit=True)

# Show the animation
plt.show()
