import numpy as np
import matplotlib.pyplot as plt
import trimesh
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.widgets as widgets
import tkinter as tk
from tkinter import simpledialog
import time
import matplotlib.animation as animation


### Get the user input
def get_user_input():
    """
    Opens a GUI to prompt the user for parameters.
    """
    root = tk.Tk()
    root.withdraw()  # Hide the main window

    radius = simpledialog.askinteger(
        "Input",
        "Enter the radius (mm):",
        parent=root,
        minvalue=10,
        maxvalue=200,
        initialvalue=100,
    )
    num_points = simpledialog.askinteger(
        "Input",
        "Enter the number of points per circle:",
        parent=root,
        minvalue=3,
        maxvalue=100,
        initialvalue=12,
    )
    levels = simpledialog.askinteger(
        "Input",
        "Enter the number of levels:",
        parent=root,
        minvalue=1,
        maxvalue=10,
        initialvalue=3,
    )
    z_distance = simpledialog.askfloat(
        "Input",
        "Enter the Z distance between levels (mm):",
        parent=root,
        minvalue=1.0,
        maxvalue=50.0,
        initialvalue=10.0,
    )
    time_per_cycle = simpledialog.askinteger(
        "Input",
        "Enter the time per cycle (seconds):",
        parent=root,
        minvalue=1,
        maxvalue=240,
        initialvalue=60,
    )
    initial_z = simpledialog.askinteger(
        "Input",
        "Enter the initial Z distance (mm):",
        parent=root,
        minvalue=0,
        maxvalue=200,
        initialvalue=10,
    )

    root.destroy()
    return radius, num_points, levels, z_distance, time_per_cycle, initial_z


def project_model_to_2d(mesh):
    """
    Project the 3D model onto the 2D XY plane.
    """
    return mesh.vertices[:, :2]


def generate_circular_camera_points(
    mesh, radius, num_points, levels, initial_z, z_distance
):
    """
    Generate camera points in multiple circular paths around the model based on its height, starting at initial_z and separating each level by z_distance.
    """
    z_values = [initial_z + i * z_distance for i in range(levels)]
    points = []
    for z in z_values:
        center_x, center_y = mesh.centroid[0], mesh.centroid[1]
        angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
        points.extend(
            [
                (
                    center_x + radius * np.cos(angle),
                    center_y + radius * np.sin(angle),
                    z,
                )
                for angle in angles
            ]
        )
    return np.array(points)


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
    projected_model = project_model_to_2d(mesh)
    ax.scatter(
        projected_model[:, 0],
        projected_model[:, 1],
        alpha=0.5,
        label="Model Projection",
    )
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


def setup_zoom_controls(ax, fig):
    ax_zoom = fig.add_axes([0.2, 0.05, 0.65, 0.03], facecolor="lightgoldenrodyellow")
    zoom_slider = widgets.Slider(ax_zoom, "Zoom", 0.5, 2.0, valinit=1.0)

    def update_zoom(val):
        ax.auto_scale_xyz([0, val * 100], [0, val * 100], [0, val * 100])

    zoom_slider.on_changed(update_zoom)


def animate_camera_movement_3d(mesh, camera_points, time_per_cycle, save_path):
    fig = plt.figure(figsize=(10, 8))  # Enlarge the figure
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title("3D Camera Movement Simulation")

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

    (line,) = ax.plot([], [], [], "r-", label="Camera Path", linewidth=2)
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
        point.set_data([x[i % len(x)]], [y[i % len(y)]])
        point.set_3d_properties([z[i % len(z)]])
        return line, point

    interval = (time_per_cycle * 1000) / len(camera_points)
    anim = FuncAnimation(
        fig,
        animate,
        init_func=init,
        frames=len(camera_points),
        interval=interval,
        blit=False,
    )
    axcolor = "lightgoldenrodyellow"
    ax_zoom = fig.add_axes([0.2, 0.01, 0.65, 0.03], facecolor=axcolor)
    zoom_slider = widgets.Slider(ax_zoom, "Zoom", 0.1, 10.0, valinit=1.0)

    def update_zoom(val):
        ax.set_xlim([mesh.centroid[0] - val * 50, mesh.centroid[0] + val * 50])
        ax.set_ylim([mesh.centroid[1] - val * 50, mesh.centroid[1] + val * 50])
        ax.set_zlim(
            [mesh.bounds[:, 2].min() - val * 50, mesh.bounds[:, 2].max() + val * 50]
        )

    zoom_slider.on_changed(update_zoom)

    plt.legend()
    plt.show()

    # Save the animation
    if save_path.endswith(".mp4"):
        Writer = animation.FFMpegWriter
        writer = Writer(fps=15, metadata=dict(artist="Me"), bitrate=1800)
    elif save_path.endswith(".gif"):
        Writer = animation.PillowWriter
        writer = Writer(fps=15)
    else:
        raise ValueError("Unsupported file format: Use .mp4 or .gif")

    anim.save(save_path, writer=writer)
    print(f"Saved animation to {save_path}")
    plt.close(fig)


def write_camera_points_to_file(camera_points, filename="camera_points.txt"):
    np.savetxt(filename, camera_points, fmt="%f", header="X Y Z")


def main_analysis_and_path_generation(file_path):
    """
    Load the model, prompt for parameters, generate camera points, and animate.
    """
    radius, num_points, levels, z_distance, time_per_cycle, initial_z = get_user_input()

    start_time = time.time()
    mesh = trimesh.load(file_path)
    print(f"Is the mesh watertight? {mesh.is_watertight}")

    camera_points = generate_circular_camera_points(
        mesh, radius, num_points, levels, initial_z, z_distance
    )

    # Set up for 2D visualization and animation
    fig2d, ax2d = plt.subplots()
    visualize_camera_path_with_circle(ax2d, mesh, camera_points, radius)
    animate_camera_movement_2d(ax2d, camera_points)

    # Specify the path for the saved animation file
    save_path = (
        "camera_movement_3d.gif"  # or "camera_movement_3d.gif/.mp4" for GIF format
    )
    animate_camera_movement_3d(mesh, camera_points, time_per_cycle, save_path)

    total_time = time.time() - start_time
    print(f"Total processing time: {total_time:.2f} seconds")
    print(f"Time per cycle: {time_per_cycle} seconds per cycle")
    print(f"Initial Z distance: {initial_z} mm")
    print(f"Z distance between levels: {z_distance} mm")

    # Write points to file
    write_camera_points_to_file(camera_points, "camera_points.txt")


# Specify the model path and start the process
model_path = "loading/logo3d.stl"
main_analysis_and_path_generation(model_path)
