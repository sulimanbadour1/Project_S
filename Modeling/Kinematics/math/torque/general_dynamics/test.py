import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from itertools import product
from joblib import Parallel, delayed

# Define DH parameters
d1 = 0.1
d5 = 0.1
a2 = 0.5
a3 = 0.5
alpha = np.deg2rad([90, 0, 0, 90, 0])


# Helper function to create a transformation matrix from DH parameters
def dh_matrix(theta, d, a, alpha):
    return np.array(
        [
            [
                np.cos(theta),
                -np.sin(theta) * np.cos(alpha),
                np.sin(theta) * np.sin(alpha),
                a * np.cos(theta),
            ],
            [
                np.sin(theta),
                np.cos(theta) * np.cos(alpha),
                -np.cos(theta) * np.sin(alpha),
                a * np.sin(theta),
            ],
            [0, np.sin(alpha), np.cos(alpha), d],
            [0, 0, 0, 1],
        ]
    )


# Function to compute the Jacobian matrix
def compute_jacobian(thetas):
    theta1, theta2, theta3, theta4, theta5 = thetas
    A1 = dh_matrix(theta1, d1, 0, alpha[0])
    A2 = dh_matrix(theta2, 0, a2, alpha[1])
    A3 = dh_matrix(theta3, 0, a3, alpha[2])
    A4 = dh_matrix(theta4, 0, 0, alpha[3])
    A5 = dh_matrix(theta5, d5, 0, alpha[4])

    T = A1 @ A2 @ A3 @ A4 @ A5

    position = T[:3, 3]
    z0 = np.array([0, 0, 1])
    z1 = A1[:3, :3] @ z0
    z2 = (A1 @ A2)[:3, :3] @ z0
    z3 = (A1 @ A2 @ A3)[:3, :3] @ z0
    z4 = (A1 @ A2 @ A3 @ A4)[:3, :3] @ z0

    Jv = np.column_stack(
        [
            np.cross(z0, position),
            np.cross(z1, position - A1[:3, 3]),
            np.cross(z2, position - (A1 @ A2)[:3, 3]),
            np.cross(z3, position - (A1 @ A2 @ A3)[:3, 3]),
            np.cross(z4, position - (A1 @ A2 @ A3 @ A4)[:3, 3]),
        ]
    )

    Jw = np.column_stack([z0, z1, z2, z3, z4])

    J = np.vstack([Jv, Jw])

    return J


# Function to compute the transformation matrices for visualization
def compute_transforms(thetas):
    theta1, theta2, theta3, theta4, theta5 = thetas
    A1 = dh_matrix(theta1, d1, 0, alpha[0])
    A2 = dh_matrix(theta2, 0, a2, alpha[1])
    A3 = dh_matrix(theta3, 0, a3, alpha[2])
    A4 = dh_matrix(theta4, 0, 0, alpha[3])
    A5 = dh_matrix(theta5, d5, 0, alpha[4])

    T1 = A1
    T2 = A1 @ A2
    T3 = A1 @ A2 @ A3
    T4 = A1 @ A2 @ A3 @ A4
    T5 = A1 @ A2 @ A3 @ A4 @ A5

    return [np.eye(4), T1, T2, T3, T4, T5]


# Function to compute the dynamic torques for a given set of angles, velocities, and accelerations
def compute_torques_for_combination(
    combination, masses, mass_camera, mass_lights, external_torques
):
    angles, velocities, accelerations = combination
    g = 9.81
    J = compute_jacobian(angles)
    Jv = J[:3, :]
    Jw = J[3:, :]

    m_total = np.array(masses + [mass_camera + mass_lights])
    V = np.array(velocities)
    A = np.array(accelerations)

    # Calculate kinetic energy related torques
    tau_kinetic = Jv.T @ (m_total[:3] * (Jv @ A)) + Jw.T @ (m_total[:3] * (Jw @ A))

    # Calculate potential energy related torques
    p = np.array([0, 0, d1]) + Jv @ angles
    tau_potential = Jv.T @ (m_total[:3] * g * p)

    # Total torques including external forces and torques
    tau_total = tau_kinetic + tau_potential + np.array(external_torques)

    return np.abs(tau_total), angles


# Function to compute the dynamic torques
def compute_dynamic_torques(
    masses,
    mass_camera,
    mass_lights,
    external_forces,
    external_torques,
    velocity_range,
    acceleration_range,
    num_samples=100,
    num_workers=4,
):
    max_torque_per_joint = np.zeros(5)
    top_configurations = []

    # Generate random samples of joint angles, velocities, and accelerations
    angle_combinations = np.random.uniform(-np.pi, np.pi, (num_samples, 5))
    velocity_combinations = np.random.uniform(
        min(velocity_range), max(velocity_range), (num_samples, 5)
    )
    acceleration_combinations = np.random.uniform(
        min(acceleration_range), max(acceleration_range), (num_samples, 5)
    )

    # Combine all samples into one array
    combinations = zip(
        angle_combinations, velocity_combinations, acceleration_combinations
    )

    # Use parallel processing to compute torques
    results = Parallel(n_jobs=num_workers)(
        delayed(compute_torques_for_combination)(
            comb, masses, mass_camera, mass_lights, external_torques
        )
        for comb in combinations
    )

    for result, angles in results:
        if np.any(result > max_torque_per_joint):
            max_torque_per_joint = np.maximum(max_torque_per_joint, result)
            top_configurations.append((angles, result))

    # Sort the configurations by the highest torque experienced
    top_configurations.sort(key=lambda x: np.max(x[1]), reverse=True)
    top_configurations = top_configurations[:3]  # Get top three configurations

    return max_torque_per_joint, top_configurations


# Function to plot the robot configuration in 3D
def plot_robot(thetas, title):
    transforms = compute_transforms(thetas)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    xs, ys, zs = [], [], []
    for T in transforms:
        xs.append(T[0, 3])
        ys.append(T[1, 3])
        zs.append(T[2, 3])

    ax.plot(xs, ys, zs, "o-", label="Robot Links")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title)
    ax.legend()
    plt.show()


# Parameters
masses = [1.0, 1.0, 1.0, 1.0, 1.0]
mass_camera = 0.5
mass_lights = 0.5
external_forces = [0, 0, 0]  # No external forces in this example
external_torques = [0, 0, 0, 0, 0]  # No external torques in this example
velocity_range = np.linspace(-1, 1, 3)  # Reduced steps for optimization
acceleration_range = np.linspace(-2, 2, 3)  # Reduced steps for optimization

# Compute dynamic torques
print("Performing dynamic torque analysis across all configurations...")
max_torque_per_joint, top_configurations = compute_dynamic_torques(
    masses,
    mass_camera,
    mass_lights,
    external_forces,
    external_torques,
    velocity_range,
    acceleration_range,
)

# Plot the maximum torques
joints = ["Joint 1", "Joint 2", "Joint 3", "Joint 4", "Joint 5"]

plt.figure(figsize=(10, 6))
bars = plt.bar(joints, max_torque_per_joint.tolist(), color="blue")
plt.xlabel("Joints")
plt.ylabel("Maximum Torque (Nm)")
plt.title("Maximum Dynamic Torque on Each Joint Across All Configurations")
plt.grid(True, linestyle="--", alpha=0.6)

# Annotate bars with their values
for bar in bars:
    yval = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        yval,
        f"{yval:.2f}",
        va="bottom" if yval < 0 else "top",
        ha="center",
        color="black",
    )

plt.show()

print(f"Maximum Dynamic Torques for given values: {max_torque_per_joint.tolist()}")

# Plot the top three configurations
for i, (angles, torques) in enumerate(top_configurations):
    plot_robot(angles, f"Configuration {i+1} with Torques: {np.round(torques, 2)} Nm")
