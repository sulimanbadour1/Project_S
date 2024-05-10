import sympy as sp
import numpy as np


# Define symbolic variables for angles and dimensions
theta_1, theta_2, theta_3, theta_4, theta_5 = sp.symbols(
    "theta_1 theta_2 theta_3 theta_4 theta_5"
)
d_1 = sp.symbols("d_1")
a_2, a_3, a_4, a_5 = sp.symbols("a_2 a_3 a_4 a_5")
alpha = [
    90,
    0,
    0,
    0,
    0,
]  # alpha values in degrees converted to radians within the matrix


# Helper function to create a transformation matrix from DH parameters
def dh_matrix(theta, d, a, alpha):
    alpha_rad = sp.rad(alpha)  # Convert alpha to radians
    return sp.Matrix(
        [
            [
                sp.cos(theta),
                -sp.sin(theta) * sp.cos(alpha_rad),
                sp.sin(theta) * sp.sin(alpha_rad),
                a * sp.cos(theta),
            ],
            [
                sp.sin(theta),
                sp.cos(theta) * sp.cos(alpha_rad),
                -sp.cos(theta) * sp.sin(alpha_rad),
                a * sp.sin(theta),
            ],
            [0, sp.sin(alpha_rad), sp.cos(alpha_rad), d],
            [0, 0, 0, 1],
        ]
    )


# Define the transformation matrices for each joint
A1 = dh_matrix(theta_1, d_1, 0, 90)
A2 = dh_matrix(theta_2, 0, a_2, 0)
A3 = dh_matrix(theta_3, 0, a_3, 0)
A4 = dh_matrix(theta_4, 0, a_4, 0)
A5 = dh_matrix(theta_5, 0, a_5, 0)

# Compute the overall transformation matrix by multiplying individual matrices
T = A1 * A2 * A3 * A4 * A5

# Display the resulting transformation matrix
sp.init_printing(use_unicode=True)
# print("Transformation matrix:")
print(T)


############# Numerical values for the DH parameters #############


def DH_matrix(theta, d, a, alpha):
    """Creates the transformation matrix using Denavit-Hartenberg parameters."""
    # Convert angles from degrees to radians
    theta_rad = np.radians(theta)
    alpha_rad = np.radians(alpha)
    return np.array(
        [
            [
                np.cos(theta_rad),
                -np.sin(theta_rad) * np.cos(alpha_rad),
                np.sin(theta_rad) * np.sin(alpha_rad),
                a * np.cos(theta_rad),
            ],
            [
                np.sin(theta_rad),
                np.cos(theta_rad) * np.cos(alpha_rad),
                -np.cos(theta_rad) * np.sin(alpha_rad),
                a * np.sin(theta_rad),
            ],
            [0, np.sin(alpha_rad), np.cos(alpha_rad), d],
            [0, 0, 0, 1],
        ]
    )


# Joint angles in degrees
theta_degrees = [0, 0, 0, 0, 0]

# DH Parameters: theta (degrees), d, a, alpha (degrees)
dh_parameters = [
    {"theta": theta_degrees[0], "d": "d1", "a": 0, "alpha": 90},  # Joint 1
    {"theta": theta_degrees[1], "d": 0, "a": "a2", "alpha": 0},  # Joint 2
    {"theta": theta_degrees[2], "d": 0, "a": "a3", "alpha": 0},  # Joint 3
    {"theta": theta_degrees[3], "d": 0, "a": "a4", "alpha": 0},  # Joint 4
    {"theta": theta_degrees[4], "d": 0, "a": "a5", "alpha": 0},  # Joint 5
]

# Define actual values for d1, a2, a3, a4, a5 if known, or use placeholders
d1 = 0.1  # Example value for d1
a2 = a3 = a4 = 0.3  # Example values for ai
a5 = 0.1  # Example value for a5


# Compute individual transformation matrices with joint angles
T = np.eye(4)  # Initialize T as an identity matrix for the base
for params in dh_parameters:
    # Replace strings in parameters with actual values
    d = eval(params["d"]) if isinstance(params["d"], str) else params["d"]
    a = eval(params["a"]) if isinstance(params["a"], str) else params["a"]
    T = T @ DH_matrix(params["theta"], d, a, params["alpha"])

# Extracting the position and orientation from the transformation matrix
x, y, z = T[0, 3], T[1, 3], T[2, 3]
orientation = T[:3, :3]

# Display results
print("Position of the End Effector (x, y, z):")
print(f"x = {x}")
print(f"y = {y}")
print(f"z = {z}\n")

print("Orientation Matrix of the End Effector:")
print(orientation)
print("\n")

# Optional: Print the entire transformation matrix for clarity
print("Numerical Evaluation of the Transformation Matrix:")
print(T)
