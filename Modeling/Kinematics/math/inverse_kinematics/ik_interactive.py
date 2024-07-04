import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib.widgets import Slider

# Define DH parameters
d1, a1, alpha1 = 0.1, 0, math.pi / 2
d2, a2, alpha2 = 0, 0.5, 0
d3, a3, alpha3 = 0, 0.5, 0
d4, a4, alpha4 = 0, 0, math.pi / 2
d5, a5, alpha5 = 0.1, 0, 0


def inverse_kinematics(Px, Py, Pz, d1, a2, a3, d5, omega):
    try:
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
        theta2_alt = alpha - beta

        # Calculate theta3
        theta3 = math.degrees(math.acos((S**2 - a2**2 - a3**2) / (2 * a2 * a3)))
        theta3 = -theta3  # Adjust for proper direction

        # Calculate theta4
        theta234 = 90 - omega
        theta4 = theta234 - theta2 - theta3

        return theta1, theta2, theta3, theta4
    except ValueError as e:
        print(
            f"Warning: {e}. The position (Px={Px}, Py={Py}, Pz={Pz}) is out of reach."
        )
        return float("nan"), float("nan"), float("nan"), float("nan")


def forward_kinematics(d1, a2, a3, d5, theta1, theta2, theta3, theta4):
    if any(math.isnan(angle) for angle in [theta1, theta2, theta3, theta4]):
        return [(float("nan"), float("nan"), float("nan"))] * 5

    # Convert angles to radians
    theta1 = math.radians(theta1)
    theta2 = math.radians(theta2)
    theta3 = math.radians(theta3)
    theta4 = math.radians(theta4)

    omega = 90 - (theta2 + theta3 + theta4)
    print(f"Omega: {omega} degrees from the forward kinematics function\n")
    omega = -omega
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


def plot_robot(joint_positions, ax, Px, Py, Pz, theta1, theta2, theta3, theta4):
    ax.cla()
    x, y, z = zip(*joint_positions)

    if all(math.isnan(coord) for coord in x + y + z):
        ax.text2D(
            0.5,
            0.5,
            "Out of Reach",
            transform=ax.transAxes,
            fontsize=15,
            color="red",
            ha="center",
        )
    else:
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

        # Annotate end-effector position and joint angles
        ax.text2D(
            0.05,
            0.95,
            f"End Effector Position: ({Px:.2f}, {Py:.2f}, {Pz:.2f})",
            transform=ax.transAxes,
        )
        ax.text2D(0.05, 0.90, f"Theta1: {theta1:.2f}°", transform=ax.transAxes)
        ax.text2D(0.05, 0.85, f"Theta2: {theta2:.2f}°", transform=ax.transAxes)
        ax.text2D(0.05, 0.80, f"Theta3: {theta3:.2f}°", transform=ax.transAxes)
        ax.text2D(0.05, 0.75, f"Theta4: {theta4:.2f}°", transform=ax.transAxes)

    ax.set_xlabel("X axis")
    ax.set_ylabel("Y axis")
    ax.set_zlabel("Z axis")
    ax.set_title("3D Robot Configuration")
    ax.legend()
    ax.set_box_aspect([1, 1, 1])


# Determine workspace limits
max_reach = a2 + a3 + d5
min_reach = 0  # Considering the arm can fold onto itself

# Set realistic limits for Px, Py, Pz based on the arm's reach
Px_min, Px_max = -max_reach, max_reach
Py_min, Py_max = -max_reach, max_reach
Pz_min, Pz_max = -1, d1 + a2 + a3  # Assuming the arm cannot go below the base level

# Initial end-effector positions
Px_init, Py_init, Pz_init, omega_init = 0.5, 0, 0.5, -90

# Create a figure and axis for plotting
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

# Adjust the main plot to make space for sliders
plt.subplots_adjust(left=0.1, bottom=0.35)

# Create sliders for Px, Py, Pz, and omega
axcolor = "lightgoldenrodyellow"
ax_Px = plt.axes([0.1, 0.25, 0.65, 0.03], facecolor=axcolor)
ax_Py = plt.axes([0.1, 0.20, 0.65, 0.03], facecolor=axcolor)
ax_Pz = plt.axes([0.1, 0.15, 0.65, 0.03], facecolor=axcolor)
ax_omega = plt.axes([0.1, 0.10, 0.65, 0.03], facecolor=axcolor)

slider_Px = Slider(ax_Px, "Px", Px_min, Px_max, valinit=Px_init)
slider_Py = Slider(ax_Py, "Py", Py_min, Py_max, valinit=Py_init)
slider_Pz = Slider(ax_Pz, "Pz", Pz_min, Pz_max, valinit=Pz_init)
slider_omega = Slider(ax_omega, "Omega", -180.0, 180.0, valinit=omega_init)


def update(val):
    Px = slider_Px.val
    Py = slider_Py.val
    Pz = slider_Pz.val
    omega = slider_omega.val
    theta1, theta2, theta3, theta4 = inverse_kinematics(
        Px, Py, Pz, d1, a2, a3, d5, omega
    )
    joint_positions = forward_kinematics(d1, a2, a3, d5, theta1, theta2, theta3, theta4)
    plot_robot(joint_positions, ax, Px, Py, Pz, theta1, theta2, theta3, theta4)
    fig.canvas.draw_idle()


slider_Px.on_changed(update)
slider_Py.on_changed(update)
slider_Pz.on_changed(update)
slider_omega.on_changed(update)

# Perform an initial plot
update(None)

plt.show()
