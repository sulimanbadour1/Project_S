import numpy as np


def dh_matrix(theta, d, a, alpha):
    alpha_rad = np.deg2rad(alpha)  # Convert alpha from degrees to radians
    return np.array(
        [
            [
                np.cos(theta),
                -np.sin(theta) * np.cos(alpha_rad),
                np.sin(theta) * np.sin(alpha_rad),
                a * np.cos(theta),
            ],
            [
                np.sin(theta),
                np.cos(theta) * np.cos(alpha_rad),
                -np.cos(theta) * np.sin(alpha_rad),
                a * np.sin(theta),
            ],
            [0, np.sin(alpha_rad), np.cos(alpha_rad), d],
            [0, 0, 0, 1],
        ]
    )


def compute_jacobians(theta_vals, d1, a2, a3, d5):
    alpha = [90, 0, 0, 90, 0]

    # Create transformation matrices for each joint using the updated parameters
    A1 = dh_matrix(theta_vals[0], d1, 0, alpha[0])
    A2 = dh_matrix(theta_vals[1], 0, a2, alpha[1])
    A3 = dh_matrix(theta_vals[2], 0, a3, alpha[2])
    A4 = dh_matrix(theta_vals[3], 0, 0, alpha[3])
    A5 = dh_matrix(theta_vals[4], d5, 0, alpha[4])

    # Compute the overall transformation matrix by multiplying individual matrices
    T1 = A1
    T2 = A1 @ A2
    T3 = A1 @ A2 @ A3
    T4 = A1 @ A2 @ A3 @ A4
    T5 = A1 @ A2 @ A3 @ A4 @ A5

    # End-effector position
    position = T5[:3, 3]

    # Compute the Jacobian matrix for the position
    J_v = np.zeros((3, 5))

    z0 = np.array([0, 0, 1])
    z1 = T1[:3, 2]
    z2 = T2[:3, 2]
    z3 = T3[:3, 2]
    z4 = T4[:3, 2]

    p0 = np.array([0, 0, 0])
    p1 = T1[:3, 3]
    p2 = T2[:3, 3]
    p3 = T3[:3, 3]
    p4 = T4[:3, 3]
    pe = position

    J_v[:, 0] = np.cross(z0, (pe - p0))
    J_v[:, 1] = np.cross(z1, (pe - p1))
    J_v[:, 2] = np.cross(z2, (pe - p2))
    J_v[:, 3] = np.cross(z3, (pe - p3))
    J_v[:, 4] = np.cross(z4, (pe - p4))

    # Rotation axes for angular part of the Jacobian
    J_w = np.column_stack((z0, z1, z2, z3, z4))

    # Screw axis Jacobians
    S1 = np.array([0, 0, 1, 0, 0, 0])
    S2 = np.array([0, 1, 0, -d1, 0, 0])
    S3 = np.array([0, 1, 0, -d1, 0, a2])
    S4 = np.array([0, 1, 0, -d1, 0, a2 + a3])
    S5 = np.array([0, 0, 1, 0, 0, 0])

    J_v_exp = np.column_stack(
        (
            np.cross(S1[:3], pe) + S1[3:],
            np.cross(S2[:3], pe) + S2[3:],
            np.cross(S3[:3], pe) + S3[3:],
            np.cross(S4[:3], pe) + S4[3:],
            np.cross(S5[:3], pe) + S5[3:],
        )
    )
    J_w_exp = np.column_stack((S1[:3], S2[:3], S3[:3], S4[:3], S5[:3]))

    return J_v, J_w, J_v_exp, J_w_exp, position


# Example numerical values for joint angles and link dimensions
theta_vals = [np.pi / 4, np.pi / 6, np.pi / 3, np.pi / 4, np.pi / 6]
a2_val = 0.5
a3_val = 0.5
d1_val = 0.1
d5_val = 0.1

# Calculate numerical Jacobians
J_v_num, J_w_num, J_v_exp_num, J_w_exp_num, position = compute_jacobians(
    theta_vals, d1_val, a2_val, a3_val, d5_val
)

print("\nEnd Effector Position:")
print(position)

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
