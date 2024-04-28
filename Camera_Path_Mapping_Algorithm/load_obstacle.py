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


import numpy as np
import trimesh


def generate_circular_camera_points(
    mesh, radius, num_points, levels, initial_z, z_distance
):
    """
    Generate camera points in multiple circular paths around the model based on its height,
    starting at initial_z and separating each level by z_distance.
    Each level begins at 190 degrees and completes a 340-degree sweep. For the next Z-level,
    the direction is reversed until it returns to the original start point of the previous level.
    """
    start_angle_deg = 190  # Start at 190 degrees
    total_sweep_deg = 340  # Sweep of 340 degrees

    z_values = [initial_z + i * z_distance for i in range(levels)]
    points = []
    for i, z in enumerate(z_values):
        center_x, center_y = mesh.centroid[0], mesh.centroid[1]
        start_angle_rad = np.radians(start_angle_deg)
        total_sweep_rad = np.radians(total_sweep_deg)

        if i % 2 == 0:
            # Even levels: rotate clockwise from 190 degrees
            angles = start_angle_rad + np.linspace(
                0, total_sweep_rad, num_points, endpoint=True
            )
        else:
            # Odd levels: rotate anticlockwise, returning to the start point of the last level (190 degrees)
            end_angle_rad = (start_angle_rad + total_sweep_rad) % (2 * np.pi)
            angles = end_angle_rad - np.linspace(
                0, total_sweep_rad, num_points, endpoint=True
            )

        # Adjust angles to ensure they wrap correctly around the circle
        angles = np.mod(angles, 2 * np.pi)

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
