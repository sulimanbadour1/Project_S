import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation


# Load camera points from the file
def load_camera_points(filename):
    return np.loadtxt(filename, skiprows=1)


# Initialize the 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

# Load the camera points
camera_points = load_camera_points("camera_points.txt")

# Plot the path (points)
ax.scatter(
    camera_points[:, 0],
    camera_points[:, 1],
    camera_points[:, 2],
    c="r",
    label="Camera Path",
)

# Draw lines between the points to show the movement path
for i in range(len(camera_points) - 1):
    ax.plot(
        camera_points[i : i + 2, 0],
        camera_points[i : i + 2, 1],
        camera_points[i : i + 2, 2],
        "gray",
    )

# Set plot labels
ax.set_xlabel("X axis")
ax.set_ylabel("Y axis")
ax.set_zlabel("Z axis")
ax.set_title("3D Simulation of Camera Movement")

# Initialize the elements we want to animate
(camera,) = ax.plot([], [], [], "bo", label="Camera Position")


# Initialization function for the animation
def init():
    camera.set_data([], [])
    camera.set_3d_properties([])
    return (camera,)


# Animation function which updates figure data. This is called sequentially
def animate(i):
    camera.set_data(camera_points[i, 0], camera_points[i, 1])
    camera.set_3d_properties(camera_points[i, 2])
    return (camera,)


# Create the animation
anim = FuncAnimation(
    fig, animate, init_func=init, frames=len(camera_points), interval=100, blit=True
)

# Show the plot
plt.legend()
plt.show()
