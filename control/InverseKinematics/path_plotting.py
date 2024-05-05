import pandas as pd
import matplotlib.pyplot as plt


def plot_joint_data(filename):
    data = pd.read_csv(filename)
    # Strip whitespace from headers to ensure the column names are accessed correctly
    data.columns = data.columns.str.strip()
    time = data["Time"]

    # Plotting joint angles
    plt.figure(figsize=(12, 8))
    for i in range(1, 6):  # Assuming there are 5 joints
        plt.plot(time, data[f"Joint {i} Angle"], label=f"Joint {i} Angle")
    plt.title("Joint Angles over Time")
    plt.xlabel("Time (s)")
    plt.ylabel("Angle (rad)")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plotting joint velocities
    plt.figure(figsize=(12, 8))
    for i in range(1, 6):
        plt.plot(time, data[f"Joint {i} Vel"], label=f"Joint {i} Vel")
    plt.title("Joint Velocities over Time")
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (rad/s)")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plotting joint efforts (torques)
    plt.figure(figsize=(12, 8))
    for i in range(1, 6):
        plt.plot(time, data[f"Joint {i} Eff"], label=f"Joint {i} Eff")
    plt.title("Joint Efforts over Time")
    plt.xlabel("Time (s)")
    plt.ylabel("Effort (Nm)")
    plt.legend()
    plt.grid(True)
    plt.show()


#### Plot End-Effector Positions


def plot_end_effector_positions(filename):
    data = pd.read_csv(filename)

    data.columns = data.columns.str.strip()
    x = data["X"]
    y = data["Y"]
    z = data["Z"]
    time = data["Time"]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Plot end-effector path
    scatter = ax.scatter(x, y, z, c=time, cmap="viridis")
    cbar = fig.colorbar(scatter, ax=ax, orientation="vertical")
    cbar.set_label("Time (s)")

    # Mark start position
    ax.scatter(
        x.iloc[0],
        y.iloc[0],
        z.iloc[0],
        color="red",
        label="Start Position",
        s=100,
        marker="o",
    )
    # Mark end position
    ax.scatter(
        x.iloc[-1],
        y.iloc[-1],
        z.iloc[-1],
        color="green",
        label="End Position",
        s=100,
        marker="s",
    )

    ax.set_xlabel("X Position (m)")
    ax.set_ylabel("Y Position (m)")
    ax.set_zlabel("Z Position (m)")
    ax.set_title("3D Path of Robot End-Effector Over Time")
    ax.legend()
    plt.show()


plot_joint_data("robot_joint_data.csv")  # Replace with the filename of your joint data
plot_end_effector_positions(
    "robot_end_effector_positions.csv"
)  # Replace with the filename of your end-effector positions
