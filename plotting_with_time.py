import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# Load the data
data = np.loadtxt("resTime.txt")

# Split the data into x, y, z, and time
x = data[:, 0]
y = data[:, 1]
z = data[:, 2]
time = data[:, 3]

# Create a figure and axis
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

# Initialize an empty line
(line,) = ax.plot([], [], [], color="gray")


# Function to update the animation
def update(frame):
    line.set_data(x[: frame + 1], y[: frame + 1])
    line.set_3d_properties(z[: frame + 1])
    return (line,)


# Animate
ani = FuncAnimation(fig, update, frames=len(x), blit=True)

# Set labels and title
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("Points in 3D with Time")

# Plot points
sc = ax.scatter(x, y, z, c=time, cmap="viridis")

# Add color bar
cbar = plt.colorbar(sc)
cbar.set_label("Time (seconds)")

plt.show()
