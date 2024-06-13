import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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

    p0 = np.array([0, 0, 0])
    p1 = A1[:3, 3]
    p2 = (A1 @ A2)[:3, 3]
    p3 = (A1 @ A2 @ A3)[:3, 3]
    p4 = (A1 @ A2 @ A3 @ A4)[:3, 3]

    Jv = np.column_stack(
        [
            np.cross(z0, position - p0),
            np.cross(z1, position - p1),
            np.cross(z2, position - p2),
            np.cross(z3, position - p3),
            np.cross(z4, position - p4),
        ]
    )

    Jw = np.column_stack([z0, z1, z2, z3, z4])

    J = np.vstack([Jv, Jw])

    return J


# Compute the transformation matrices for visualization
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


# Plot the robot configuration in 3D
def plot_robot(thetas):
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
    ax.set_title("Robot Configuration")
    ax.legend()
    plt.show()


# Numerical Jacobian using finite differences
def numerical_jacobian(thetas, delta=1e-8):
    num_jacobian = np.zeros((6, len(thetas)))
    T_original = compute_transforms(thetas)[-1]
    for i in range(len(thetas)):
        thetas_perturbed = np.copy(thetas)
        thetas_perturbed[i] += delta
        T_perturbed = compute_transforms(thetas_perturbed)[-1]

        # Compute position and orientation difference
        position_diff = (T_perturbed[:3, 3] - T_original[:3, 3]) / delta
        R_diff = (T_perturbed[:3, :3] - T_original[:3, :3]) / delta

        # Angular velocity vector from skew-symmetric matrix
        w_x = (R_diff[2, 1] - R_diff[1, 2]) / 2.0
        w_y = (R_diff[0, 2] - R_diff[2, 0]) / 2.0
        w_z = (R_diff[1, 0] - R_diff[0, 1]) / 2.0
        angular_velocity = np.array([w_x, w_y, w_z])

        num_jacobian[:, i] = np.hstack((position_diff, angular_velocity))

    return num_jacobian


# Verify the end-effector position and orientation
def forward_kinematics(thetas):
    A1 = dh_matrix(thetas[0], d1, 0, alpha[0])
    A2 = dh_matrix(thetas[1], 0, a2, alpha[1])
    A3 = dh_matrix(thetas[2], 0, a3, alpha[2])
    A4 = dh_matrix(thetas[3], 0, 0, alpha[3])
    A5 = dh_matrix(thetas[4], d5, 0, alpha[4])
    T = A1 @ A2 @ A3 @ A4 @ A5
    return T


# Specific configuration to check (angles in degrees)
theta_values_deg = [0, 0, 0, 0, 0]
theta_values_rad = np.deg2rad(theta_values_deg)

# Compute the Jacobian for the specific configuration
J = compute_jacobian(theta_values_rad)

# Determine the rank of the Jacobian
rank_J = np.linalg.matrix_rank(J)

# Print the Jacobian and its rank
print("\nJacobian matrix at specific configuration:")
print(J)
print("\nRank of the Jacobian matrix at specific configuration:")
print(rank_J)

# Check if the configuration is singular
is_singular = rank_J < 5
print(f"\nIs the configuration singular? {'Yes' if is_singular else 'No'}")

# Plot the robot configuration
plot_robot(theta_values_rad)

# Compute numerical Jacobian for the specific configuration
numerical_J = numerical_jacobian(theta_values_rad, delta=1e-8)  # Adjusted delta

# Compare numerical Jacobian with analytical Jacobian
print("\nNumerical Jacobian matrix:")
print(numerical_J)
print("\nAnalytical Jacobian matrix:")
print(J)
print("\nDifference between Numerical and Analytical Jacobian:")
difference = numerical_J - J
print(difference)

# Verify the end-effector position and orientation from forward kinematics
T = forward_kinematics(theta_values_rad)
print("\nEnd-effector position and orientation from forward kinematics:")
print(T)

# Check the maximum absolute difference
max_diff = np.max(np.abs(difference))
print(
    f"\nMaximum absolute difference between Numerical and Analytical Jacobians: {max_diff}"
)
