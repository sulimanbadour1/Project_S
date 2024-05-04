import ikpy.chain
import ikpy.utils.plot as plot_utils
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


def init_robot():
    return ikpy.chain.Chain.from_urdf_file(
        "urdfs/arm_urdf.urdf",
        active_links_mask=[False, True, True, True, True, True, True],
    )


def create_plot():
    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.1, bottom=0.35)
    ax3d = fig.add_subplot(111, projection="3d")
    return fig, ax3d


def setup_sliders(ax):
    axcolor = "lightgoldenrodyellow"
    ax_x = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
    ax_y = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor=axcolor)
    ax_z = plt.axes([0.25, 0.2, 0.65, 0.03], facecolor=axcolor)
    slider_x = Slider(ax_x, "X", -0.5, 0.5, valinit=0)
    slider_y = Slider(ax_y, "Y", -0.5, 0.5, valinit=0)
    slider_z = Slider(ax_z, "Z", 0, 0.6, valinit=0.58)
    return slider_x, slider_y, slider_z


def update_plot(chain, ax3d, target_position):
    ik = chain.inverse_kinematics(target_position)
    ax3d.clear()
    chain.plot(ik, ax3d, target=target_position)
    # Coloring the links
    colors = [
        "red",
        "blue",
        "green",
        "purple",
        "orange",
        "cyan",
    ]  # Colors for each link
    for idx, line in enumerate(ax3d.get_lines()):
        line.set_color(colors[idx % len(colors)])
    ax3d.set_xlim(-0.5, 0.5)
    ax3d.set_ylim(-0.5, 0.5)
    ax3d.set_zlim(0, 0.6)
    plt.title(f"Real-time Robot Arm Movement\nTarget: {target_position}")


def main():
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
