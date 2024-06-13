import sympy as sp
import numpy as np

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

# Extract the position of the end effector
position = T[:3, 3]

# Define the joint variables in a vector
joint_vars = sp.Matrix([theta_1, theta_2, theta_3, theta_4, theta_5])

# Compute the Jacobian matrix for the position
J_v = position.jacobian(joint_vars)

# Extract the z-axis of each transformation matrix (which are the rotation axes)
z0 = sp.Matrix([0, 0, 1])
z1 = A1[:3, 2]
z2 = (A1 * A2)[:3, 2]
z3 = (A1 * A2 * A3)[:3, 2]
z4 = (A1 * A2 * A3 * A4)[:3, 2]

# Compute the angular part of the Jacobian
J_w = sp.Matrix.hstack(z0, z1, z2, z3, z4)

# Define twist axes for each joint (screw axes)
S1 = sp.Matrix([0, 0, 1, 0, 0, 0])
S2 = sp.Matrix([0, 1, 0, -d_1, 0, 0])
S3 = sp.Matrix([0, 1, 0, -d_1, 0, a_2])
S4 = sp.Matrix([0, 1, 0, -d_1, 0, a_2 + a_3])
S5 = sp.Matrix([0, 0, 1, 0, 0, 0])

# Compute the Jacobian matrix for the position using screw axes
J_v_exp = sp.Matrix.hstack(S1[:3], S2[:3], S3[:3], S4[:3], S5[:3])

# Compute the Jacobian matrix for the angular velocity using screw axes
J_w_exp = sp.Matrix.hstack(S1[3:], S2[3:], S3[3:], S4[3:], S5[3:])

# Initialize pretty printing for better output readability
sp.init_printing(use_unicode=True)

# Display the results
print("Transformation matrix using DH parameters:")
sp.pprint(T)

print("\nEnd Effector Position using DH parameters:")
sp.pprint(position)

print("\nJacobian (Linear Velocity) using DH parameters:")
sp.pprint(J_v)

print("\nJacobian (Angular Velocity) using DH parameters:")
sp.pprint(J_w)

print("\nJacobian (Linear Velocity) using screw axis method:")
sp.pprint(J_v_exp)

print("\nJacobian (Angular Velocity) using screw axis method:")
sp.pprint(J_w_exp)


# Numerical verification
def numerical_jacobian(theta_vals, a2_val, a3_val, d1_val, d5_val):
    subs = {
        theta_1: theta_vals[0],
        theta_2: theta_vals[1],
        theta_3: theta_vals[2],
        theta_4: theta_vals[3],
        theta_5: theta_vals[4],
        a_2: a2_val,
        a_3: a3_val,
        d_1: d1_val,
        d_5: d5_val,
    }
    J_v_num = J_v.evalf(subs=subs)
    J_w_num = J_w.evalf(subs=subs)
    J_v_exp_num = J_v_exp.evalf(subs=subs)
    J_w_exp_num = J_w_exp.evalf(subs=subs)
    return (
        np.array(J_v_num).astype(np.float64),
        np.array(J_w_num).astype(np.float64),
        np.array(J_v_exp_num).astype(np.float64),
        np.array(J_w_exp_num).astype(np.float64),
    )


# Example numerical values for joint angles and link dimensions
theta_vals = [np.pi / 4, np.pi / 6, np.pi / 3, np.pi / 4, np.pi / 6]
a2_val = 0.5
a3_val = 0.5
d1_val = 0.1
d5_val = 0.1

# Calculate numerical Jacobians
J_v_num, J_w_num, J_v_exp_num, J_w_exp_num = numerical_jacobian(
    theta_vals, a2_val, a3_val, d1_val, d5_val
)
print("\nNumerical Jacobian (Linear Velocity) using DH parameters:")
print(J_v_num)

print("\nNumerical Jacobian (Linear Velocity) using screw axis method:")
print(J_v_exp_num)

print("\nNumerical Jacobian (Angular Velocity) using DH parameters:")
print(J_w_num)

print("\nNumerical Jacobian (Angular Velocity) using screw axis method:")
print(J_w_exp_num)

# Compare the Jacobians
print("\nDifference in Linear Velocity Jacobians:")
print(J_v_num - J_v_exp_num)

print("\nDifference in Angular Velocity Jacobians:")
print(J_w_num - J_w_exp_num)
