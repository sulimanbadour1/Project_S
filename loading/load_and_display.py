import numpy as np
import matplotlib.pyplot as plt
import trimesh
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D


def project_model_to_2d(mesh):
    """
    Project the 3D model onto the 2D XY plane.
    """
    return mesh.vertices[:, :2]


def generate_circular_camera_points(mesh, radius=100, num_points=12, levels=3):
    """
    Generate camera points in multiple circular paths around the model based on its height.
    """
    min_z, max_z = mesh.bounds[:, 2]
    z_values = np.linspace(min_z, max_z, levels)
    points = []
    for z in z_values:
        center_x, center_y = mesh.centroid[0], mesh.centroid[1]
        angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
        points += [
            (center_x + radius * np.cos(angle), center_y + radius * np.sin(angle), z)
            for angle in angles
        ]
    return points


def map_camera_focus_points(mesh, radius=100, num_points=12, levels=3):
    """
    Create a mapping of camera focus points based on model geometry.
    """
    critical_points = mesh.vertices[mesh.faces].mean(
        axis=1
    )  # Example: using face centroids
    focus_points = []
    for cp in critical_points:
        theta = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
        focus_points.extend(
            [
                (cp[0] + radius * np.cos(t), cp[1] + radius * np.sin(t), cp[2])
                for t in theta
            ]
        )
    return focus_points


def visualize_camera_path_with_circle(ax, mesh, camera_points, radius):
    """
    Visualize the circular path and camera points in 2D.
    """
    projected_model = project_model_to_2d(mesh)
    ax.scatter(
        projected_model[:, 0],
        projected_model[:, 1],
        alpha=0.5,
        label="Model Projection",
    )

    # Draw the circle
    center_x, center_y = mesh.centroid[0], mesh.centroid[1]
    circle = plt.Circle(
        (center_x, center_y),
        radius,
        color="g",
        fill=False,
        linewidth=2,
        label="Camera Path Circle",
    )
    ax.add_artist(circle)

    # Draw the camera points
    camera_points = np.array(camera_points)
    ax.plot(camera_points[:, 0], camera_points[:, 1], "ro", label="Camera Points")
    ax.set_aspect("equal", "datalim")
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    ax.legend()


def animate_camera_movement_2d(ax, camera_points):
    """
    Create a 2D animation of the camera moving along the path defined by the selected points.
    """
    camera_points = np.array(camera_points)
    (line,) = ax.plot([], [], "r-", label="Camera Movement", linewidth=2)
    (point,) = ax.plot([], [], "ro")

    def init():
        line.set_data([], [])
        point.set_data([], [])
        return line, point

    def animate(i):
        x, y, z = camera_points[:, 0], camera_points[:, 1], camera_points[:, 2]
        line.set_data(x[: i % len(x) + 1], y[: i % len(y) + 1])
        point.set_data(x[i % len(x)], y[i % len(y)])
        return line, point

    anim = FuncAnimation(
        ax.figure,
        animate,
        init_func=init,
        frames=len(camera_points),
        interval=200,
        blit=False,
    )
    ax.set_title("Camera Movement in 2D")
    plt.legend()
    plt.show()


def animate_camera_movement_3d(mesh, camera_points):
    """
    Create a 3D animation of the camera moving around the model at multiple heights.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Plot the model
    vertices = mesh.vertices
    faces = mesh.faces
    ax.plot_trisurf(
        vertices[:, 0],
        vertices[:, 1],
        vertices[:, 2],
        triangles=faces,
        color="gray",
        alpha=0.5,
    )

    # Camera path setup
    camera_points = np.array(camera_points)
    (line,) = ax.plot([], [], [], "r-", label="Camera Movement", linewidth=2)
    (point,) = ax.plot([], [], [], "ro")

    def init():
        line.set_data([], [])
        line.set_3d_properties([])
        point.set_data([], [])
        point.set_3d_properties([])
        return line, point

    def animate(i):
        x, y, z = camera_points[:, 0], camera_points[:, 1], camera_points[:, 2]
        line.set_data(x[: i % len(x) + 1], y[: i % len(y) + 1])
        line.set_3d_properties(z[: i % len(z) + 1])
        point.set_data(x[i % len(x)], y[i % len(y)])
        point.set_3d_properties(z[i % len(z)])
        return line, point

    anim = FuncAnimation(
        fig,
        animate,
        init_func=init,
        frames=len(camera_points),
        interval=200,
        blit=False,
    )
    plt.legend()
    plt.show()


def main_analysis_and_path_generation(file_path):
    """
    Load the model, generate circular camera points for each height level, and visualize in 2D and 3D.
    """
    mesh = trimesh.load(file_path)
    print(f"Is the mesh watertight? {mesh.is_watertight}")

    # Generate camera points with multiple levels
    radius = 100  # radius in units
    num_points = 12  # points per circle
    levels = 5  # number of levels based on height
    camera_points = generate_circular_camera_points(mesh, radius, num_points, levels)

    # Set up for 2D visualization and animation
    fig2d, ax2d = plt.subplots()
    visualize_camera_path_with_circle(ax2d, mesh, camera_points, radius)
    animate_camera_movement_2d(ax2d, camera_points)

    # Animate the camera movement in 3D
    animate_camera_movement_3d(mesh, camera_points)


# Specify the model path and start the process
model_path = "loading/logo3d.stl"
main_analysis_and_path_generation(model_path)
