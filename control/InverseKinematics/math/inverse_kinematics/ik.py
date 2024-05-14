import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# Helper function to clamp values within a specific range
def clamp(value, min_value, max_value):
    return max(min_value, min(value, max_value))


# Function to compute joint angles based on the desired position and constraints
def compute_angles(px, py, pz, a2, a3, d1, d5):
    # Step 1: Calculate theta1 within its range
    theta1 = math.degrees(math.atan2(py, px))
    theta1 = clamp(theta1, -180, 180)  # Joint 1 can rotate 360 degrees

    # Calculate the wrist center position
    wx = px - d5 * math.cos(math.radians(theta1))
    wy = py - d5 * math.sin(math.radians(theta1))
    wz = pz

    # Calculate distances
    r = math.sqrt(wx**2 + wy**2)
    s = wz - d1

    # Step 2: Calculate theta2 and theta3 within their ranges
    D = (r**2 + s**2 - a2**2 - a3**2) / (2 * a2 * a3)
    D = clamp(D, -1.0, 1.0)  # Ensure D is within valid range for acos
    theta3 = math.degrees(math.acos(D))
    theta3 = clamp(theta3, -90, 90)  # Joint 3 can rotate from -90 to 90

    theta2 = math.degrees(
        math.atan2(s, r)
        - math.atan2(
            a3 * math.sin(math.radians(theta3)),
            a2 + a3 * math.cos(math.radians(theta3)),
        )
    )
    theta2 = clamp(theta2, -10, 70)  # Joint 2 can rotate from -10 to 80

    # Step 3: Calculate theta4 within its range
    theta234 = math.degrees(math.atan2(s, r))
    theta4 = theta234 - theta2 - theta3
    theta4 = clamp(theta4, -90, 90)  # Joint 4 can rotate from -90 to 90

    return {
        "theta1": theta1,
        "theta2": theta2,
        "theta3": theta3,
        "theta4": theta4,
    }


# Function to plot the robot in 3D space
def plot_robot(joint_positions, ax, color="b", label=""):
    xs = [pos[0] for pos in joint_positions]
    ys = [pos[1] for pos in joint_positions]
    zs = [pos[2] for pos in joint_positions]

    ax.plot(xs, ys, zs, marker="o", color=color, label=label)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")


# Function to compare two positions with a tolerance
def compare_positions(pos1, pos2, tolerance=1e-3):
    diff = [abs(a - b) for a, b in zip(pos1, pos2)]
    return all(d < tolerance for d in diff)


# Example usage with constraints
px, py, pz = 0, 0, 1.2  # example coordinates
a2, a3 = 0.5, 0.5  # example link lengths
d1, d5 = 0.1, 0.1  # example offsets

angles = compute_angles(px, py, pz, a2, a3, d1, d5)
print("Computed Angles:", angles)

# The wrist center should consider d5
wrist_center = (
    px - d5 * math.cos(math.radians(angles["theta1"])),
    py - d5 * math.sin(math.radians(angles["theta1"])),
    pz - d5,
)

joint_positions_inverse = [
    (0, 0, 0),
    (0, 0, d1),
    (
        a2
        * math.cos(math.radians(angles["theta1"]))
        * math.cos(math.radians(angles["theta2"])),
        a2
        * math.sin(math.radians(angles["theta1"]))
        * math.cos(math.radians(angles["theta2"])),
        d1 + a2 * math.sin(math.radians(angles["theta2"])),
    ),
    wrist_center,
    (px, py, pz),
]

# Extract end effector positions
end_effector_inverse = joint_positions_inverse[-1]

# Compare positions
print("End Effector Position (Inverse):", end_effector_inverse)

fig = plt.figure()

ax2 = fig.add_subplot(111, projection="3d")
plot_robot(joint_positions_inverse, ax2, color="r", label="Inverse Kinematics")
ax2.legend()

plt.show()
