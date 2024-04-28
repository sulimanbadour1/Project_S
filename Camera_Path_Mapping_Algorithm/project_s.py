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
)
from PyQt5.QtCore import Qt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import trimesh


class Window(QMainWindow):
    def __init__(self, mesh):
        super().__init__()
        self.mesh = mesh
        self.initUI()

    def initUI(self):
        self.radius = 100
        self.num_points = 12
        self.levels = 3
        self.z_distance = 10
        self.initial_z = 10

        widget = QWidget()
        self.setCentralWidget(widget)
        layout = QVBoxLayout()
        widget.setLayout(layout)

        # Canvas Setup
        self.canvas = FigureCanvas(plt.figure())
        layout.addWidget(self.canvas)

        # Sliders and labels
        self.setupSliders(layout)

        # Parameter Display
        self.parameter_display = QLineEdit(self)
        self.parameter_display.setReadOnly(True)
        self.update_parameters_display()  # Initial display update
        layout.addWidget(self.parameter_display)

        # Update Button
        self.update_button = QPushButton("Update Simulation")
        self.update_button.clicked.connect(self.update_plot)
        layout.addWidget(self.update_button)

        self.setGeometry(300, 300, 800, 600)
        self.setWindowTitle("3D Model Simulation")
        self.show()

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

    def update_parameters_display(self):
        text = f"Radius: {self.radius} mm, Points: {self.num_points}, Levels: {self.levels}, Z Distance: {self.z_distance} mm, Initial Z: {self.initial_z} mm"
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

    def generate_camera_points(self):
        z_values = [self.initial_z + i * self.z_distance for i in range(self.levels)]
        points = []
        for z in z_values:
            center_x, center_y = self.mesh.centroid[0], self.mesh.centroid[1]
            angles = np.linspace(0, 2 * np.pi, self.num_points, endpoint=False)
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

    def log_points(self):
        points = self.generate_camera_points()
        np.savetxt("camera_points.txt", points, fmt="%f", header="X Y Z", comments="")


def main():
    app = QApplication(sys.argv)
    mesh = trimesh.load(
        "Camera_Path_Mapping_Algorithm\logo3d.stl"
    )  # Adjust the path to your STL file
    ex = Window(mesh)
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
