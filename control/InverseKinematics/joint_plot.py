import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# Load data from the file
data = np.loadtxt("robot_data.txt", delimiter=",")
time_steps = data[:, 0]  # Assuming the first column is time
end_effector_positions = data[:, 1:4]  # Adjusted for time column
joint_angles = data[:, 4:]  # Adjusted for time column

# Print some debug information
print("Time steps:")
print(time_steps)
print("End-effector positions:")
print(end_effector_positions)
print("Joint angles:")
print(joint_angles)

# Plot the end-effector positions
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

# Creating a scatter plot that will be updated
scat = ax.scatter(
    end_effector_positions[0, 0],
    end_effector_positions[0, 1],
    end_effector_positions[0, 2],
    color="blue",
    label="End-effector positions",
)


def update(frame):
    # Update the data of the scatter plot
    scat._offsets3d = (
        end_effector_positions[:frame, 0],
        end_effector_positions[:frame, 1],
        end_effector_positions[:frame, 2],
    )
    return (scat,)


# Animation
ani = FuncAnimation(fig, update, frames=len(time_steps), interval=50, blit=False)

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("End-effector Positions Over Time")
ax.legend()

plt.show()

# Static plot for the joint angles
fig, ax = plt.subplots()
for i in range(joint_angles.shape[1]):
    ax.plot(time_steps, joint_angles[:, i], label=f"Joint {i+1}")
ax.set_xlabel("Time step")
ax.set_ylabel("Joint Angle")
ax.set_title("Joint Angles")
ax.legend()
plt.show()
