import numpy as np
import trimesh
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import tkinter as tk
from tkinter import simpledialog


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
    initial_z = simpledialog.askinteger(
        "Input",
        "Enter the initial Z distance (mm):",
        parent=root,
        minvalue=0,
        maxvalue=200,
        initialvalue=10,
    )

    root.destroy()
    return radius, num_points, levels, z_distance, initial_z


def generate_circular_camera_points(
    mesh, radius, num_points, levels, initial_z, z_distance
):
    """
    Generate camera points in multiple circular paths around the model based on its height,
    starting at initial_z and separating each level by z_distance. Camera points generated
    between 10 degrees to 340 degrees, starting from the right to the left for the first level
    and then alternating directions by level.
    """
    start_angle = np.radians(10)  # Convert 10 degrees to radians
    end_angle = np.radians(340)  # Convert 340 degrees to radians
    z_values = [initial_z + i * z_distance for i in range(levels)]
    points = []
    for i, z in enumerate(z_values):
        center_x, center_y = mesh.centroid[0], mesh.centroid[1]
        if i % 2 == 0:
            # Even levels (starting with the first): rotate from right to left (340 to 10 degrees)
            angles = np.linspace(end_angle, start_angle, num_points, endpoint=True)
        else:
            # Odd levels: rotate from left to right (10 to 340 degrees)
            angles = np.linspace(start_angle, end_angle, num_points, endpoint=True)
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


def visualize_camera_path_with_circle(ax, mesh, camera_points, radius):
    projected_model = mesh.vertices[:, :2]
    ax.scatter(
        projected_model[:, 0],
        projected_model[:, 1],
        alpha=0.5,
        label="Model Projection",
    )
    center_x, center_y = mesh.centroid[0], mesh.centroid[1]
    angles = np.linspace(0, 2 * np.pi, 100, endpoint=True)
    circle_path = np.array(
        [
            (center_x + radius * np.cos(angle), center_y + radius * np.sin(angle))
            for angle in angles
        ]
    )
    ax.plot(
        circle_path[:, 0], circle_path[:, 1], "g--", label="Full Camera Path Circle"
    )
    camera_points = np.array(camera_points)
    ax.plot(camera_points[:, 0], camera_points[:, 1], "ro", label="Camera Points")
    ax.set_aspect("equal", adjustable="datalim")
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
        x, y = camera_points[:, 0], camera_points[:, 1]
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


def main():
    radius, num_points, levels, z_distance, initial_z = get_user_input()
    file_path = "Camera_Path_Mapping_Algorithm\logo3d.stl"  # Replace this with the path to your model file
    mesh = trimesh.load(file_path)

    camera_points = generate_circular_camera_points(
        mesh, radius, num_points, levels, initial_z, z_distance
    )

    fig, ax = plt.subplots()
    visualize_camera_path_with_circle(ax, mesh, camera_points, radius)
    animate_camera_movement_2d(ax, camera_points)


if __name__ == "__main__":
    main()
