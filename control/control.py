import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from simple_pid import PID


def load_camera_points(filename):
    return np.loadtxt(filename, skiprows=1)


def simulate_rotating_base_with_arm(camera_points, printer_height, pid_base, pid_arm):
    # Assume starting with the base aligned with the first target point.
    target_angle_base = np.arctan2(camera_points[0, 1], camera_points[0, 0])
    current_angle_base = target_angle_base
    current_angle_arm = 0.0

    # Stores the positions of the camera (end of the arm)
    camera_positions = []

    # Go through each target point
    for point in camera_points:
        # Calculate the base rotation angle and arm angle required to point at the target
        target_angle_base = np.arctan2(point[1], point[0])
        target_angle_arm = np.arcsin(
            point[2] / np.sqrt(point[0] ** 2 + point[1] ** 2 + point[2] ** 2)
        )

        # Update the angles using the PID controllers
        current_angle_base += pid_base(target_angle_base - current_angle_base)
        current_angle_arm += pid_arm(target_angle_arm - current_angle_arm)

        # Calculate the position of the arm's end (camera position)
        distance_to_point = np.linalg.norm([point[0], point[1]])
        camera_x = distance_to_point * np.cos(current_angle_base)
        camera_y = distance_to_point * np.sin(current_angle_base)
        camera_z = printer_height + distance_to_point * np.sin(current_angle_arm)

        camera_positions.append((camera_x, camera_y, camera_z))

    return np.array(camera_positions)


# PID controllers for the rotating base and the arm
pid_base = PID(1.0, 0.1, 0.05, setpoint=0)
pid_arm = PID(1.0, 0.1, 0.05, setpoint=0)

# Load camera points
camera_points = load_camera_points("camera_points.txt")

# Printer height, adjust to the height of your 3D printer where the rotating base is located
printer_height = 650  # in mm

# Simulate the rotating base with the arm
camera_positions = simulate_rotating_base_with_arm(
    camera_points, printer_height, pid_base, pid_arm
)

# Create a 3D plot
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection="3d")

# Plot target points on the printer bed
ax.scatter(
    camera_points[:, 0],
    camera_points[:, 1],
    np.zeros_like(camera_points[:, 2]),
    color="green",
    label="Target Points",
)

# Plot the positions of the camera at the end of the arm
ax.scatter(
    camera_positions[:, 0],
    camera_positions[:, 1],
    camera_positions[:, 2],
    color="red",
    label="Camera Positions",
)

# Draw lines from the base to the camera position to represent the arm
for position in camera_positions:
    # Base to arm joint
    ax.plot(
        [0, position[0]],
        [0, position[1]],
        [printer_height, printer_height],
        "b-",
        label="Base to Arm Joint",
    )
    # Arm joint to camera
    ax.plot(
        [position[0], position[0]],
        [position[1], position[1]],
        [printer_height, position[2]],
        "r-",
        label="Arm Joint to Camera",
    )

# Setting the labels and titles for the axes
ax.set_xlabel("X Position (mm)")
ax.set_ylabel("Y Position (mm)")
ax.set_zlabel("Z Position (mm)")
ax.set_title("3D Simulation with Rotating Camera Base and Arm on 3D Printer")

# Limiting the axes ranges for better visibility
max_range = (
    np.array(
        [
            camera_points[:, 0].max() - camera_points[:, 0].min(),
            camera_points[:, 1].max() - camera_points[:, 1].min(),
            camera_points[:, 2].max() - camera_points[:, 2].min(),
        ]
    ).max()
    / 2.0
)
mid_x = (camera_points[:, 0].max() + camera_points[:, 0].min()) * 0.5
mid_y = (camera_points[:, 1].max() + camera_points[:, 1].min()) * 0.5
mid_z = (camera_points[:, 2].max() + camera_points[:, 2].min()) * 0.5
ax.set_xlim(mid_x - max_range, mid_x + max_range)
ax.set_ylim(mid_y - max_range, mid_y + max_range)
ax.set_zlim(0, printer_height + max_range)

# Show the plot
plt.show()
