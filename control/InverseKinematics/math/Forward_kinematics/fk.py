import sympy as sp

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
print("Transformation matrix:")
print(T)
