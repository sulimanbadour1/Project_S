import ikpy.chain
import ikpy.utils.plot as plot_utils
import numpy as np
import matplotlib.pyplot as plt


"""refer to robot.txt for the correct URDF file"""


# Load the robot from the URDF file files #arm_urdf.urdf
robot_chain = ikpy.chain.Chain.from_urdf_file(
    "urdfs/s.urdf",
    base_elements=["base"],
    active_links_mask=[False, True, True, True, True, True],  # Adjusted to six elements
)


# Function to safely retrieve joint limits and handle possible None values or other anomalies
def get_joint_limits(chain):
    limits = []
    for link in chain.links:
        if link.bounds and all(type(b) is float for b in link.bounds):
            # Ensure bounds are valid and not excessively large
            if all(
                abs(b) <= 10 * np.pi for b in link.bounds
            ):  # Example threshold, adjust as needed
                limits.append(link.bounds)
            else:
                print(
                    f"Adjusted bounds for link {link.name} from {link.bounds} to (-pi, pi)"
                )
                limits.append((-np.pi, np.pi))
        else:
            print(f"No valid bounds for link {link.name}, using (-pi, pi)")
            limits.append((-np.pi, np.pi))
    return limits


limits = get_joint_limits(robot_chain)
print("Joint limits:", limits)


# Function to generate random joint configurations and compute the corresponding end effector positions
def generate_workspace(chain, n_samples, joint_limits):
    positions = []
    for _ in range(n_samples):
        random_angles = np.random.uniform(
            [limit[0] for limit in joint_limits], [limit[1] for limit in joint_limits]
        )
        fk = chain.forward_kinematics(random_angles)
        positions.append(fk[:3, 3])
    return np.array(positions)


# Generate workspace points
workspace = generate_workspace(robot_chain, 1000, limits)

# Plotting the workspace
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

# Scatter plot of the workspace
ax.scatter(workspace[:, 0], workspace[:, 1], workspace[:, 2], alpha=0.1, c="b")
ax.set_xlabel("X axis")
ax.set_ylabel("Y axis")
ax.set_zlabel("Z axis")
ax.set_title("Workspace of the Robot")
plt.show()
