import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot_path_3d_from_file(file_path):
    # Read end-effector positions from the text file
    with open(file_path, "r") as file:
        lines = file.readlines()
        end_effector_positions = []
        for line in lines:
            values = line.strip().split(",")
            end_effector_positions.append([float(value) for value in values])

    # Convert end-effector positions to numpy array
    path = np.array(end_effector_positions)

    # Plot path in 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(path[:, 0], path[:, 1], path[:, 2], "-o")
    ax.set_title("Robot End-Effector Path (3D)")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.show()


# Path to the text file containing end-effector positions
file_path = "end_effector_positions.txt"

# Plot the path in 3D
plot_path_3d_from_file(file_path)
