import ikpy.chain
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import math


def init_robot():
    """Initializes and returns the robot chain from a URDF file."""
    return ikpy.chain.Chain.from_urdf_file(
        "urdfs/s.urdf",
        base_elements=["base"],
        active_links_mask=[False, True, True, True, True, True],
    )


def create_plot():
    """Creates and returns a 3D plot and its axes for robot visualization."""
    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.1, bottom=0.35)
    ax3d = fig.add_subplot(111, projection="3d")
    return fig, ax3d


def setup_sliders(ax):
    """Sets up sliders for X, Y, Z axes manipulation and returns them."""
    axcolor = "lightgoldenrodyellow"
    slider_x = Slider(
        plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor), "X", -0.5, 0.5, valinit=0
    )
    slider_y = Slider(
        plt.axes([0.25, 0.15, 0.65, 0.03], facecolor=axcolor), "Y", -0.5, 0.5, valinit=0
    )
    slider_z = Slider(
        plt.axes([0.25, 0.2, 0.65, 0.03], facecolor=axcolor), "Z", -0.5, 0.5, valinit=0
    )  # Modified range
    return slider_x, slider_y, slider_z


def update_plot(chain, ax3d, target_position):
    """Updates the plot based on the current slider positions and calculates inverse kinematics."""
    target_orientation = [0, 0, 0]  # Assuming no rotation as a default
    ik = chain.inverse_kinematics(
        target_position, target_orientation, orientation_mode="Y"
    )
    ax3d.clear()
    chain.plot(ik, ax=ax3d, target=target_position)
    ax3d.set_xlim(-0.5, 0.5)
    ax3d.set_ylim(-0.5, 0.5)
    ax3d.set_zlim(-0.5, 0.5)  # Adjusted Z limits
    plt.title(f"Real-time Robot Arm Movement\nTarget: {target_position}")
    print(
        "The angles of each joint are : ",
        list(map(lambda r: math.degrees(r), ik.tolist())),
    )


def main():
    """Main function to initialize components and handle events."""
    robot_chain = init_robot()
    fig, ax3d = create_plot()
    slider_x, slider_y, slider_z = setup_sliders(ax3d)

    def on_change(val):
        target_position = [slider_x.val, slider_y.val, slider_z.val]
        update_plot(robot_chain, ax3d, target_position)

    slider_x.on_changed(on_change)
    slider_y.on_changed(on_change)
    slider_z.on_changed(on_change)
    plt.show()


if __name__ == "__main__":
    main()
