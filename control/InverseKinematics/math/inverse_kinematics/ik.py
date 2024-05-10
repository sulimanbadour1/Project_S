import numpy as np
from scipy.optimize import minimize
from numpy import cos, sin, pi

# Define the Denavit-Hartenberg parameters for each joint
# Each row: [theta, d, a, alpha]
DH_params = np.array(
    [
        [-pi, 0.05, 0, 0],  # Joint 1
        [-pi / 6, 0, 0.03, pi / 2],  # Joint 2
        [-pi / 6, 0, 0.25, 0],  # Joint 3
        [-pi / 6, 0, 0.28, 0],  # Joint 4
        [-pi / 6, 0, 0.28, 0],  # Joint 5
    ]
)


def transformation_matrix(theta, d, a, alpha):
    """Compute individual transformation matrix for given DH parameters"""
    return np.array(
        [
            [
                cos(theta),
                -sin(theta) * cos(alpha),
                sin(theta) * sin(alpha),
                a * cos(theta),
            ],
            [
                sin(theta),
                cos(theta) * cos(alpha),
                -cos(theta) * sin(alpha),
                a * sin(theta),
            ],
            [0, sin(alpha), cos(alpha), d],
            [0, 0, 0, 1],
        ]
    )


def forward_kinematics(thetas):
    """Compute the overall transformation matrix for given joint angles"""
    T = np.eye(4)
    for i, (theta, d, a, alpha) in enumerate(DH_params):
        T_i = transformation_matrix(theta + thetas[i], d, a, alpha)
        T = np.dot(T, T_i)
    return T


def objective_function(thetas, target_position):
    """Objective function to minimize: the Euclidean distance from the current end-effector position to the target"""
    T = forward_kinematics(thetas)
    end_effector_pos = T[
        :3, 3
    ]  # Extract position from the last column of the transformation matrix
    return np.linalg.norm(end_effector_pos - target_position)


# Target position for the end-effector
target_position = np.array([0.3, -0.2, 0.5])

# Initial guess for the joint angles (in radians)
initial_guess = np.array([-pi / 2, -pi / 6, -pi / 6, -pi / 6, -pi / 6])

# Minimize the objective function
result = minimize(
    objective_function, initial_guess, args=(target_position), method="BFGS"
)

# Validate the results with forward kinematics
joint_angles_optimized = result.x
T_final = forward_kinematics(joint_angles_optimized)
computed_position = T_final[:3, 3]

# Comparison and output
tolerance = 1e-3
print("Optimization Result:")
print("Success:", result.success)
print("Message:", result.message)
print("Joint Angles (Radians):", joint_angles_optimized)
print("Computed Position:", computed_position)
print("Target Position:", target_position)
print("Difference:", np.abs(computed_position - target_position))
print(
    "Is the computed position within tolerance?",
    np.all(np.abs(computed_position - target_position) < tolerance),
)
