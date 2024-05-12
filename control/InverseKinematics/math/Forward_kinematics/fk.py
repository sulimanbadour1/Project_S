import sympy as sp

# Define symbolic variables for angles and dimensions
theta_1, theta_2, theta_3, theta_4, theta_5 = sp.symbols(
    "theta_1 theta_2 theta_3 theta_4 theta_5"
)
d_1, d_5 = sp.symbols("d_1 d_5")  # d_5 for the last joint
a_2, a_3 = sp.symbols(
    "a_2 a_3"
)  # a_2 and a_3 for the lengths of the second and third links

# Alpha values in degrees, with an updated value for alpha_4
alpha = [90, 0, 0, 90, 0]


# Helper function to create a transformation matrix from DH parameters
def dh_matrix(theta, d, a, alpha):
    alpha_rad = sp.rad(alpha)  # Convert alpha from degrees to radians
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


# Create transformation matrices for each joint using the updated parameters
A1 = dh_matrix(theta_1, d_1, 0, alpha[0])
A2 = dh_matrix(theta_2, 0, a_2, alpha[1])
A3 = dh_matrix(theta_3, 0, a_3, alpha[2])
A4 = dh_matrix(theta_4, 0, 0, alpha[3])  # a_4 is zero
A5 = dh_matrix(theta_5, d_5, 0, alpha[4])  # a_5 is zero, added d_5

# Compute the overall transformation matrix by multiplying individual matrices
T = A1 * A2 * A3 * A4 * A5

# Initialize pretty printing for better output readability
sp.init_printing(use_unicode=True)

# Print the resulting transformation matrix
print("Transformation matrix:")
# print(T)

print(T)
