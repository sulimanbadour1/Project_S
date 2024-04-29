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
from PyQt5.QtCore import Qt, QTimer
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import trimesh


def load_stl_file(file_path, scale=1.0):
    """
    Load the STL file with optional scaling and center it at the origin with the bottom face aligned with the z=0 plane.
    """
    mesh = trimesh.load(file_path)
    mesh.apply_scale(scale)
    mesh.vertices -= mesh.vertices.mean(axis=0)
    min_z = np.min(mesh.vertices[:, 2])
    mesh.vertices[:, 2] -= min_z
    return mesh


class Window(QMainWindow):
    def __init__(self):
        super().__init__()
        self.mesh = None
        self.timer = QTimer(self)
        self.path_duration = 60  # Default time for each cycle in seconds
        self.radius = 100
        self.num_points = 12
        self.levels = 3
        self.z_distance = 10
        self.initial_z = 10
        self.start_angle_deg = 210
        self.total_sweep_deg = 300
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Project_S - 3D Model Simulation")

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

        self.setupButtons(layout)

        self.setGeometry(300, 300, 800, 600)
        self.show_startup_message()
        self.show()

    def show_startup_message(self):
        self.info_label.setText(
            "Welcome to Project_S by Suli1man - This program simulates camera paths around 3D models. Load an STL file to start."
        )

    def setupSliders(self, layout):
        self.sliders_layout = QHBoxLayout()
        layout.addLayout(self.sliders_layout)

        self.radius_slider = self.create_slider(
            10,
            500,
            self.radius,
            "Radius (mm):",
            self.sliders_layout,
            self.update_radius,
        )
        self.points_slider = self.create_slider(
            3,
            100,
            self.num_points,
            "Number of Points:",
            self.sliders_layout,
            self.update_points,
        )
        self.levels_slider = self.create_slider(
            1, 10, self.levels, "Levels:", self.sliders_layout, self.update_levels
        )
        self.z_distance_slider = self.create_slider(
            1,
            50,
            self.z_distance,
            "Z Distance (mm):",
            self.sliders_layout,
            self.update_z_distance,
        )
        self.initial_z_slider = self.create_slider(
            0,
            200,
            self.initial_z,
            "Initial Z (mm):",
            self.sliders_layout,
            self.update_initial_z,
        )
        self.start_angle_slider = self.create_slider(
            0,
            360,
            self.start_angle_deg,
            "Start Angle (deg):",
            self.sliders_layout,
            self.update_start_angle,
        )
        self.sweep_angle_slider = self.create_slider(
            0,
            360,
            self.total_sweep_deg,
            "Sweep Angle (deg):",
            self.sliders_layout,
            self.update_sweep_angle,
        )
        self.time_slider = self.create_slider(
            1,
            1200,
            self.path_duration,
            "Time per Level (sec):",
            self.sliders_layout,
            self.update_time,
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

    def setupButtons(self, layout):  # Method to set up buttons for actions
        self.update_button = QPushButton(
            "Update Simulation"
        )  # Create a button for updating the simulation
        self.update_button.clicked.connect(
            self.update_plot
        )  # Connect the clicked signal to the update_plot method
        layout.addWidget(self.update_button)  # Add the button to the layout

        self.file_button = QPushButton(
            "Load STL File"
        )  # Create a button for loading an STL file
        self.file_button.clicked.connect(
            self.load_stl_file
        )  # Connect the clicked signal to the load_stl_file method
        layout.addWidget(self.file_button)  # Add the button to the layout

        self.export_button = QPushButton(
            "Export Points to Text with Timings"
        )  # Create a button for exporting points to text
        self.export_button.clicked.connect(
            self.export_points_to_text
        )  # Connect the clicked signal to the export_points_to_text method
        layout.addWidget(self.export_button)  # Add the button to the layout

    def update_radius(self, value):
        self.radius = value
        self.update_parameters_display()
        self.update_plot()

    def update_points(self, value):
        self.num_points = value
        self.update_parameters_display()
        self.update_plot()

    def update_levels(self, value):
        self.levels = value
        self.update_parameters_display()
        self.update_plot()

    def update_z_distance(self, value):
        self.z_distance = value
        self.update_parameters_display()
        self.update_plot()

    def update_initial_z(self, value):
        self.initial_z = value
        self.update_parameters_display()
        self.update_plot()

    def update_start_angle(self, value):
        self.start_angle_deg = value
        self.update_parameters_display()
        self.update_plot()

    def update_sweep_angle(self, value):
        self.total_sweep_deg = value
        self.update_parameters_display()
        self.update_plot()

    def update_time(self, value):
        self.path_duration = value
        self.update_parameters_display()
        self.timer.stop()  # Stop the timer to reset with new interval
        self.timer.start(self.path_duration * 1000)  # Timer interval in milliseconds

    def export_points_to_text(self):
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save File", "", "Text Files (*.txt)"
        )
        if filename:
            try:
                camera_points = self.generate_circular_camera_points()
                timings = self.calculate_timings()
                with open(filename, "w") as f:
                    for point, time in zip(camera_points, timings):
                        f.write(f"{point[0]} {point[1]} {point[2]} {time}\n")
                self.show_notification(f"Saved camera points to {filename}")
            except Exception as e:
                self.show_notification(f"Failed to save file: {str(e)}")

    def calculate_timings(self):
        """
        Calculate the timing for each point based on the path duration and number of levels.
        """
        total_points = self.levels * self.num_points
        total_time = self.path_duration * self.levels
        time_per_point = total_time / total_points
        timings = [time_per_point * i for i in range(total_points)]
        return timings

    def update_parameters_display(self):
        text = f"Radius: {self.radius} mm, Points: {self.num_points}, Levels: {self.levels}, Z Distance: {self.z_distance} mm, Initial Z: {self.initial_z} mm, Start Angle: {self.start_angle_deg} deg, Sweep: {self.total_sweep_deg} deg, Duration per Level: {self.path_duration} sec"
        self.parameter_display.setText(text)

    def update_plot(self):
        if not hasattr(self, "ax"):
            self.create_plot()
        else:
            self.scatter.remove()
            camera_points = self.generate_circular_camera_points()
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
        camera_points = self.generate_circular_camera_points()
        self.scatter = self.ax.scatter(
            camera_points[:, 0], camera_points[:, 1], camera_points[:, 2], color="red"
        )
        self.canvas.draw()

    def load_stl_file(self):
        filename, _ = QFileDialog.getOpenFileName(
            self, "Open STL file", "", "STL Files (*.stl)"
        )
        try:
            if filename:
                scale = self.get_user_input()
                self.mesh = load_stl_file(filename, scale)
                self.update_plot()
                self.info_label.hide()
        except Exception as e:
            self.show_notification(f"Failed to load file: {str(e)}")

    def get_user_input(self):
        scale, okPressed = QInputDialog.getDouble(
            self, "Get Scaling Factor", "Enter the scaling factor:", 1.0, 0.1, 100.0, 1
        )
        if okPressed:
            return scale
        else:
            return 1.0

    def generate_circular_camera_points(self):
        if self.mesh is None:
            self.show_notification("No mesh loaded. Please load an STL file first.")
            raise ValueError("No mesh loaded. Please load an STL file first.")

        start_angle_rad = np.radians(self.start_angle_deg)
        total_sweep_rad = np.radians(self.total_sweep_deg)

        z_values = [self.initial_z + i * self.z_distance for i in range(self.levels)]
        points = []
        for i, z in enumerate(z_values):
            center_x, center_y = self.mesh.centroid[0], self.mesh.centroid[1]

            if i % 2 == 0:
                angles = start_angle_rad + np.linspace(
                    0, total_sweep_rad, self.num_points, endpoint=True
                )
            else:
                end_angle_rad = (start_angle_rad + total_sweep_rad) % (2 * np.pi)
                angles = end_angle_rad - np.linspace(
                    0, total_sweep_rad, self.num_points, endpoint=True
                )

            angles = np.mod(angles, 2 * np.pi)

            points.extend(
                [
                    (
                        center_x + self.radius * np.cos(angle),
                        center_y + self.radius * np.sin(angle),
                        z,
                    )
                    for angle in angles
                ]
            )
        return np.array(points)

    def show_notification(self, message):
        QMessageBox.information(self, "Notification", message)


def main():
    app = QApplication(sys.argv)
    ex = Window()
    ex.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
