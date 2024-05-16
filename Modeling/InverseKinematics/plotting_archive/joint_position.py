import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot_end_effector_positions(filename):
    data = pd.read_csv(filename)
    data.columns = [
        col.strip() for col in data.columns  # Strip whitespace from headers
    ]
    x = data["X"]
    y = data["Y"]
    z = data["Z"]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(x, y, z, label="End-effector path")
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.set_zlabel("Z Position")
    ax.set_title("3D Path of Robot End-Effector")
    ax.legend()
    plt.show()


def plot_joint_angles(filename):
    data = pd.read_csv(filename)
    data.columns = [
        col.strip() for col in data.columns
    ]  # Strip whitespace from headers
    times = data["Time"]
    joints = data.filter(regex="Joint")

    plt.figure(figsize=(10, 6))
    colors = [
        "b",
        "g",
        "r",
        "c",
        "m",
        "y",
        "k",
    ]  # Define a list of colors for the plots

    for i, col in enumerate(joints.columns):
        plt.plot(
            times, joints[col], label=f"{col}", color=colors[i % len(colors)]
        )  # Cycle through colors

    plt.title("Joint Angles Over Time")
    plt.xlabel("Time (s)")
    plt.ylabel("Angle (rad)")
    plt.legend()  # Include a legend to identify each joint
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# Assuming the CSV files are available and correctly formatted, you would call the functions like this:
plot_end_effector_positions("plotting/robot_end_effector_positions.csv")
plot_joint_angles("plotting/robot_joint_angles.csv")
