import sys
import numpy as np
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QSlider,
    QVBoxLayout,
    QWidget,
    QPushButton,
    QLabel,
    QHBoxLayout,
    QLineEdit,
    QFileDialog,
    QMessageBox,
    QInputDialog,
)
from PyQt5.QtCore import Qt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import trimesh


def load_stl_file(file_path, scale=1.0):
    """
    Load the STL file with optional scaling and center it at the origin with the bottom face aligned with the z=0 plane.
    """
    mesh = trimesh.load(file_path)

    # Scale the mesh
    mesh.apply_scale(scale)

    # Center the mesh at the origin
    mesh.vertices -= mesh.vertices.mean(axis=0)

    # Translate the mesh to align with the z=0 plane
    min_z = np.min(mesh.vertices[:, 2])
    mesh.vertices[:, 2] -= min_z

    return mesh


class Window(QMainWindow):
    def __init__(self):
        super().__init__()
        self.mesh = None
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Project_S - 3D Model Simulation")
        self.radius = 100
        self.num_points = 12
        self.levels = 3
        self.z_distance = 10
        self.initial_z = 10
        self.start_angle_deg = 210  # Default start angle in degrees
        self.total_sweep_deg = 300  # Default sweep in degrees

        widget = QWidget()
        self.setCentralWidget(widget)
        layout = QVBoxLayout()
        widget.setLayout(layout)

        self.info_label = QLabel("Loading... Please open an STL file to start.")
        layout.addWidget(self.info_label)

        self.canvas = FigureCanvas(plt.figure())
        layout.addWidget(self.canvas)

        self.setupSliders(layout)

        self.parameter_display = QLineEdit(self)
        self.parameter_display.setReadOnly(True)
        layout.addWidget(self.parameter_display)

        self.update_button = QPushButton("Update Simulation")
        self.update_button.clicked.connect(self.update_plot)
        layout.addWidget(self.update_button)

        self.file_button = QPushButton("Load STL File")
        self.file_button.clicked.connect(self.load_stl_file)
        layout.addWidget(self.file_button)

        self.export_button = QPushButton("Export Points to Text")
        self.export_button.clicked.connect(self.export_points_to_text)
        layout.addWidget(self.export_button)

        self.setGeometry(300, 300, 800, 600)
        self.show_startup_message()
        self.show()

    def show_startup_message(self):
        self.info_label.setText(
            "Welcome to Project_S by Suli1man - This program simulates camera paths around 3D models. Load an STL file to start."
        )

    def setupSliders(self, layout):
        sliders_layout = QHBoxLayout()
        layout.addLayout(sliders_layout)
        self.radius_slider = self.create_slider(
            10, 500, self.radius, "Radius (mm):", sliders_layout, self.update_radius
        )
        self.points_slider = self.create_slider(
            3,
            100,
            self.num_points,
            "Number of Points:",
            sliders_layout,
            self.update_points,
        )
        self.levels_slider = self.create_slider(
            1, 10, self.levels, "Levels:", sliders_layout, self.update_levels
        )
        self.z_distance_slider = self.create_slider(
            1,
            50,
            self.z_distance,
            "Z Distance (mm):",
            sliders_layout,
            self.update_z_distance,
        )
        self.initial_z_slider = self.create_slider(
            0,
            200,
            self.initial_z,
            "Initial Z (mm):",
            sliders_layout,
            self.update_initial_z,
        )
        self.start_angle_slider = self.create_slider(
            0,
            360,
            self.start_angle_deg,
            "Start Angle (deg):",
            sliders_layout,
            self.update_start_angle,
        )
        self.sweep_angle_slider = self.create_slider(
            0,
            360,
            self.total_sweep_deg,
            "Sweep Angle (deg):",
            sliders_layout,
            self.update_sweep_angle,
        )

    def create_slider(self, min_val, max_val, init_val, label, layout, callback):
        lbl = QLabel(label)
        slider = QSlider(Qt.Horizontal)
        slider.setRange(min_val, max_val)
        slider.setValue(init_val)
        slider.valueChanged.connect(callback)
        layout.addWidget(lbl)
        layout.addWidget(slider)
        return slider

    def update_plot(self):
        if not hasattr(self, "ax"):
            self.create_plot()
        else:
            self.scatter.remove()
            camera_points = self.generate_circular_camera_points(
                self.mesh,
                self.radius,
                self.num_points,
                self.levels,
                self.initial_z,
                self.z_distance,
            )
            self.scatter = self.ax.scatter(
                camera_points[:, 0],
                camera_points[:, 1],
                camera_points[:, 2],
                color="red",
            )
        self.canvas.draw()

    def create_plot(self):
        self.ax = self.canvas.figure.add_subplot(111, projection="3d")
        if self.mesh:
            self.ax.plot_trisurf(
                self.mesh.vertices[:, 0],
                self.mesh.vertices[:, 1],
                self.mesh.vertices[:, 2],
                triangles=self.mesh.faces,
                color="grey",
                alpha=0.5,
            )
        camera_points = self.generate_circular_camera_points(
            self.mesh,
            self.radius,
            self.num_points,
            self.levels,
            self.initial_z,
            self.z_distance,
        )
        self.scatter = self.ax.scatter(
            camera_points[:, 0], camera_points[:, 1], camera_points[:, 2], color="red"
        )
        self.canvas.draw()

    def load_stl_file(self):
        filename, _ = QFileDialog.getOpenFileName(
            self, "Open STL file", "", "STL Files (*.stl)"
        )
        if filename:
            scale = self.get_user_input()
            self.mesh = load_stl_file(filename, scale)
            self.update_plot()
            self.info_label.hide()

    def get_user_input(self):
        # Define a method to get user input for scaling factor
        scale, okPressed = QInputDialog.getDouble(
            self, "Get Scaling Factor", "Enter the scaling factor:", 1.0, 0.1, 100.0, 1
        )
        if okPressed:
            return scale
        else:
            return 1.0  # Default to no scaling if user cancels

    def export_points_to_text(self):
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save File", "", "Text Files (*.txt)"
        )
        if filename:
            try:
                camera_points = self.generate_circular_camera_points(
                    self.mesh,
                    self.radius,
                    self.num_points,
                    self.levels,
                    self.initial_z,
                    self.z_distance,
                )
                with open(filename, "w") as f:
                    for point in camera_points:
                        f.write(f"{point[0]}  {point[1]} {point[2]}\n")
                self.show_notification(f"Saved camera points to {filename}")
            except Exception as e:
                self.show_notification(f"Failed to save file: {str(e)}")

    def generate_circular_camera_points(
        self, mesh, radius, num_points, levels, initial_z, z_distance
    ):
        start_angle_rad = np.radians(self.start_angle_deg)
        total_sweep_rad = np.radians(self.total_sweep_deg)

        z_values = [initial_z + i * z_distance for i in range(levels)]
        points = []
        for i, z in enumerate(z_values):
            center_x, center_y = mesh.centroid[0], mesh.centroid[1]

            if i % 2 == 0:
                angles = start_angle_rad + np.linspace(
                    0, total_sweep_rad, num_points, endpoint=True
                )
            else:
                end_angle_rad = (start_angle_rad + total_sweep_rad) % (2 * np.pi)
                angles = end_angle_rad - np.linspace(
                    0, total_sweep_rad, num_points, endpoint=True
                )

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

    def update_radius(self, value):
        self.radius = value
        self.update_parameters_display()

    def update_points(self, value):
        self.num_points = value
        self.update_parameters_display()

    def update_levels(self, value):
        self.levels = value
        self.update_parameters_display()

    def update_z_distance(self, value):
        self.z_distance = value
        self.update_parameters_display()

    def update_initial_z(self, value):
        self.initial_z = value
        self.update_parameters_display()

    def update_start_angle(self, value):
        self.start_angle_deg = value
        self.update_parameters_display()

    def update_sweep_angle(self, value):
        self.total_sweep_deg = value
        self.update_parameters_display()

    def update_parameters_display(self):
        text = f"Radius: {self.radius} mm, Points: {self.num_points}, Levels: {self.levels}, Z Distance: {self.z_distance} mm, Initial Z: {self.initial_z} mm, Start Angle: {self.start_angle_deg} deg, Sweep: {self.total_sweep_deg} deg"
        self.parameter_display.setText(text)

    def show_notification(self, message):
        QMessageBox.information(self, "Notification", message)


def main():
    app = QApplication(sys.argv)
    ex = Window()
    ex.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
