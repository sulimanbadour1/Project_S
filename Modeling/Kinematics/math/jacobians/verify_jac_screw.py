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

# Extract the position vector from the transformation matrix
p = T[:3, 3]

# Define the joint variables
joint_vars = [theta_1, theta_2, theta_3, theta_4, theta_5]

# Compute the position Jacobian
J_v = sp.Matrix.hstack(*[sp.diff(p, var) for var in joint_vars])

# Compute the rotation matrices for each joint to find the z-axes in the base frame
R0 = sp.eye(3)
R1 = A1[:3, :3]
R2 = (A1 * A2)[:3, :3]
R3 = (A1 * A2 * A3)[:3, :3]
R4 = (A1 * A2 * A3 * A4)[:3, :3]

# Axes of rotation for each joint in the base frame
z0 = sp.Matrix([0, 0, 1])
z1 = R1[:, 2]
z2 = R2[:, 2]
z3 = R3[:, 2]
z4 = R4[:, 2]

# Compute the orientation Jacobian
J_w = sp.Matrix.hstack(z0, z1, z2, z3, z4)

# Combine J_v and J_w to form the full Jacobian
J = sp.Matrix.vstack(J_v, J_w)

# Initialize pretty printing for better output readability
sp.init_printing(use_unicode=True)

# Print the resulting transformation matrix
print("Transformation matrix:")
sp.pprint(T)

# Print the resulting Jacobian matrix
print("\nJacobian matrix:")
sp.pprint(J)

# Verification using screw axis representation
# Define position vectors to each joint
o0 = sp.Matrix([0, 0, 0])
o1 = A1[:3, 3]
o2 = (A1 * A2)[:3, 3]
o3 = (A1 * A2 * A3)[:3, 3]
o4 = (A1 * A2 * A3 * A4)[:3, 3]
o5 = (A1 * A2 * A3 * A4 * A5)[:3, 3]

# Position Jacobian using screw axis
J_v_screw = sp.Matrix.hstack(
    z0.cross(o5 - o0),
    z1.cross(o5 - o1),
    z2.cross(o5 - o2),
    z3.cross(o5 - o3),
    z4.cross(o5 - o4),
)

# Combine the position and orientation Jacobians
J_screw = sp.Matrix.vstack(J_v_screw, J_w)

# Print the screw axis Jacobian matrix
print("\nScrew Axis Jacobian matrix:")
sp.pprint(J_screw)

# check the size of the Jacobian matrices
print("\nSize of Jacobian matrix:")
print(J.shape)
print(J_screw.shape)


# Check if both Jacobians are equal
print("\nAre both Jacobians equal?")
print(J.equals(J_screw))
