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
)
from PyQt5.QtCore import Qt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import trimesh
import pandas as pd


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

        # Initial placeholder content
        self.info_label = QLabel("Loading... Please open an STL file to start.")
        layout.addWidget(self.info_label)

        # Canvas Setup
        self.canvas = FigureCanvas(plt.figure())
        layout.addWidget(self.canvas)

        # Sliders and labels
        self.setupSliders(layout)

        # Parameter Display
        self.parameter_display = QLineEdit(self)
        self.parameter_display.setReadOnly(True)
        layout.addWidget(self.parameter_display)

        # Update and File Selection Buttons
        self.update_button = QPushButton("Update Simulation")
        self.update_button.clicked.connect(self.update_plot)
        layout.addWidget(self.update_button)

        self.file_button = QPushButton("Load STL File")
        self.file_button.clicked.connect(self.load_stl_file)
        layout.addWidget(self.file_button)

        self.export_button = QPushButton("Export Points to CSV")
        self.export_button.clicked.connect(self.export_points_to_csv)
        layout.addWidget(self.export_button)

        self.setGeometry(300, 300, 800, 600)
        self.show_startup_message()
        self.show()

    def show_startup_message(self):
        self.info_label.setText(
            f"Welcome to Project_S by \nThis program simulates camera paths around 3D models. Load an STL file to start."
        )

    def show_notification(self, message):
        QMessageBox.information(self, "Notification", message)

    def setupSliders(self, layout):
        sliders_layout = QHBoxLayout()
        layout.addLayout(sliders_layout)

        # Creating sliders
        self.radius_slider = self.create_slider(
            10, 200, self.radius, "Radius (mm):", sliders_layout, self.update_radius
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

    def update_plot(self):
        if not hasattr(self, "ax"):
            self.create_plot()
        else:
            # Only update the scatter plot
            self.scatter.remove()
            camera_points = self.generate_camera_points()
            self.scatter = self.ax.scatter(
                camera_points[:, 0],
                camera_points[:, 1],
                camera_points[:, 2],
                color="red",
            )
        self.canvas.draw()

    def create_plot(self):
        """
        Create the initial plot, which will only happen once to minimize overhead.
        """
        self.ax = self.canvas.figure.add_subplot(111, projection="3d")
        if self.mesh:  # Ensure mesh is loaded
            self.ax.plot_trisurf(
                self.mesh.vertices[:, 0],
                self.mesh.vertices[:, 1],
                self.mesh.vertices[:, 2],
                triangles=self.mesh.faces,
                color="grey",
                alpha=0.5,
            )
        camera_points = self.generate_camera_points()
        self.scatter = self.ax.scatter(
            camera_points[:, 0], camera_points[:, 1], camera_points[:, 2], color="red"
        )
        self.canvas.draw()

    def load_stl_file(self):
        filename, _ = QFileDialog.getOpenFileName(
            self, "Open STL file", "", "STL Files (*.stl)"
        )
        if filename:
            self.mesh = trimesh.load(filename)
            self.update_plot()
            self.info_label.hide()  # Hide the info label after loading a file

    def export_points_to_csv(self):
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save File", "", "CSV Files (*.csv)"
        )
        if filename:
            try:
                camera_points = self.generate_camera_points()
                pd.DataFrame(camera_points, columns=["X", "Y", "Z"]).to_csv(
                    filename, index=False
                )
                self.show_notification(f"Saved camera points to {filename}")
            except Exception as e:
                self.show_notification(f"Failed to save file: {str(e)}")

    def generate_camera_points(self):
        """
        Generate camera points in multiple circular paths around the model based on its height,
        starting at initial_z and separating each level by z_distance.
        Each level begins at start_angle_deg degrees and completes a total_sweep_deg sweep.
        For the next Z-level, the direction is reversed until it returns to the original start point of the previous level.
        """
        z_values = [self.initial_z + i * self.z_distance for i in range(self.levels)]
        points = []
        for i, z in enumerate(z_values):
            center_x, center_y = self.mesh.centroid[0], (
                self.mesh.centroid[1] if self.mesh else (0, 0)
            )
            start_angle_rad = np.radians(self.start_angle_deg)
            total_sweep_rad = np.radians(self.total_sweep_deg)

            if i % 2 == 0:
                angles = start_angle_rad + np.linspace(
                    0, total_sweep_rad, self.num_points, endpoint=True
                )
            else:
                end_angle_rad = (start_angle_rad + total_sweep_rad) % (2 * np.pi)
                angles = end_angle_rad - np.linspace(
                    0, total_sweep_rad, self.num_points, endpoint=True
                )

            angles = np.mod(angles, 2 * np.pi)  # Adjust angles to wrap around

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


def main():
    app = QApplication(sys.argv)
    ex = Window()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
