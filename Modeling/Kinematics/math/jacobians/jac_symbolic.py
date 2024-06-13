import sympy as sp

# Define symbolic variables for joint angles and DH parameters
theta_1, theta_2, theta_3, theta_4, theta_5 = sp.symbols(
    "theta_1 theta_2 theta_3 theta_4 theta_5"
)
d_1, d_5 = sp.symbols("d_1 d_5")
a_2, a_3 = sp.symbols("a_2 a_3")
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
A4 = dh_matrix(theta_4, 0, 0, alpha[3])
A5 = dh_matrix(theta_5, d_5, 0, alpha[4])

# Compute the overall transformation matrix by multiplying individual matrices
T = A1 * A2 * A3 * A4 * A5

# Extract the position and orientation
position = T[:3, 3]
orientation = T[:3, :3]

# Compute the Jacobian for linear velocity
Jv = position.jacobian([theta_1, theta_2, theta_3, theta_4, theta_5])

# The angular part of the Jacobian is given by the z-axis of the previous frames
z0 = sp.Matrix([0, 0, 1])  # z0 axis (base frame)
z1 = A1[:3, :3] * z0
z2 = (A1 * A2)[:3, :3] * z0
z3 = (A1 * A2 * A3)[:3, :3] * z0
z4 = (A1 * A2 * A3 * A4)[:3, :3] * z0

# Assemble the angular velocity Jacobian
Jw = sp.Matrix.hstack(z0, z1, z2, z3, z4)

# Combine Jv and Jw to form the full Jacobian matrix
J_full = sp.Matrix.vstack(Jv, Jw)

# Initialize pretty printing for better output readability
sp.init_printing(use_unicode=True)

# Print the Jacobians
# print("Jacobian for linear velocity (Jv):")
# sp.pprint(Jv)

# print("\nJacobian for angular velocity (Jw):")
# sp.pprint(Jw)

print("\nFull Jacobian matrix (J):")
sp.pprint(J_full)
