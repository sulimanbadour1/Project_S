import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def read_points_from_file(filename):
    """
    Reads 3D points from a file. The file should have one point per line,
    with each coordinate separated by spaces.
    """
    points = []
    with open(filename, "r") as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) == 3:  # Ensure the line has exactly three elements
                points.append(tuple(float(p) for p in parts))
    return points


def plot_3d_points(points):
    """
    Plots the points in 3D space and connects them with lines.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Unpack points into x, y, and z lists
    x, y, z = zip(*points)

    # Plot points
    ax.scatter(x, y, z, color="b", label="Points")

    # Plot lines connecting points
    ax.plot(x, y, z, color="r", label="Path")

    # Setting labels and title
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    ax.set_zlabel("Z Coordinate")
    ax.set_title("3D Points Visualization with Path Connection")

    # Adding a legend
    ax.legend()

    # Show the plot
    plt.show()


# Read points from file
points = read_points_from_file(
    "Camera_Path_Mapping_Algorithm/plotting/camera_points.txt"
)

# Plot the points
plot_3d_points(points)
