import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Constants for the mechanical design
TILT_RANGE_DEGREES = 90.0  # Full range of tilt in degrees

# Constants for converting XYZ coordinates to movement degrees
MM_PER_STEP_LINEAR = 0.01  # Assumed value: 0.01mm per step for linear actuator
DEGREES_PER_STEP_ROTARY = 0.1  # Assumed value: 0.1 degrees per step for rotary base
DEGREES_PER_STEP_TILT = 0.1  # Assumed value: 0.1 degrees per step for tilt joint
# Roll will be omitted in movement commands due to its complexity and lack of details.
# Maximum X-coordinate representing the full rotation range
X_MAX = 550.0
# Maximum Y-coordinate representing the full tilt range
Y_MAX = 490.0
# Maximum Z-coordinate representing the maximum height
Z_MAX = 650.0


# Load camera points from the file
def load_camera_points(filename):
    return np.loadtxt(filename, skiprows=1)


# Convert XYZ coordinates to control commands
def xyz_to_control_commands(x, y, z):
    # Convert to polar coordinates for rotary base
    angle = (
        np.degrees(np.arctan2(y, x)) % 360.0
    )  # Convert to degrees and normalize to 0-360
    normalized_z = z / Z_MAX  # Normalize Z based on maximum printer height

    # Calculate the commands for each actuator
    azimuth = angle  # Azimuth angle for the turntable
    elevation = normalized_z * TILT_RANGE_DEGREES  # Elevation angle for the tilt
    linear_steps = z / MM_PER_STEP_LINEAR  # Linear steps for Z-axis height adjustment

    return azimuth, elevation, int(linear_steps)


# Write the control commands to a file
def write_control_commands(camera_points, output_file):
    with open(output_file, "w") as file:
        for x, y, z in camera_points:
            azimuth, elevation, linear_steps = xyz_to_control_commands(x, y, z)
            file.write(f"MOVE AZIMUTH TO {azimuth:.2f} DEGREES\n")
            file.write(f"MOVE ELEVATION TO {elevation:.2f} DEGREES\n")
            file.write(f"ADJUST HEIGHT TO {linear_steps} STEPS\n")
            file.write("WAIT FOR COMPLETION\n")


# Main function to generate the control system and plot the points
def generate_control_file_and_plot(input_filename, output_filename):
    camera_points = load_camera_points(input_filename)
    write_control_commands(camera_points, output_filename)

    # Plot the points in 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(camera_points[:, 0], camera_points[:, 1], camera_points[:, 2])
    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")
    ax.set_zlabel("Z Axis")
    plt.show()

    print(f"Control file '{output_filename}' generated successfully.")


# File paths
input_filename = "camera_points.txt"  # The text file with XYZ coordinates
output_filename = "control_commands.txt"  # The control file to be generated

# Generate the control file and plot
generate_control_file_and_plot(input_filename, output_filename)
