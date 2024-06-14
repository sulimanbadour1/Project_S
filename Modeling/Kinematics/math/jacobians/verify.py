import numpy as np


def dh_matrix(theta, d, a, alpha):
    alpha_rad = np.deg2rad(alpha)
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


def compute_dh_jacobian(theta_vals, d1, a2, a3, d5):
    alpha = [90, 0, 0, 90, 0]

    A1 = dh_matrix(theta_vals[0], d1, 0, alpha[0])
    A2 = dh_matrix(theta_vals[1], 0, a2, alpha[1])
    A3 = dh_matrix(theta_vals[2], 0, a3, alpha[2])
    A4 = dh_matrix(theta_vals[3], 0, 0, alpha[3])
    A5 = dh_matrix(theta_vals[4], d5, 0, alpha[4])

    T1 = A1
    T2 = A1 @ A2
    T3 = A1 @ A2 @ A3
    T4 = A1 @ A2 @ A3 @ A4
    T5 = A1 @ A2 @ A3 @ A4 @ A5

    position = T5[:3, 3]

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

    J_v = np.column_stack(
        (
            np.cross(z0, pe - p0),
            np.cross(z1, pe - p1),
            np.cross(z2, pe - p2),
            np.cross(z3, pe - p3),
            np.cross(z4, pe - p4),
        )
    )

    J_w = np.column_stack((z0, z1, z2, z3, z4))

    return J_v, J_w, position


def compute_screw_jacobian(theta_vals, d1, a2, a3, d5):
    S1 = np.array([0, 0, 1, 0, 0, 0])
    S2 = np.array([0, 1, 0, -d1, 0, 0])
    S3 = np.array([0, 1, 0, -d1, 0, a2])
    S4 = np.array([0, 1, 0, -d1, 0, a2 + a3])
    S5 = np.array([0, 0, 1, 0, 0, 0])

    theta1, theta2, theta3, theta4, theta5 = theta_vals

    def skew(v):
        return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])

    def expm(s, theta):
        v = s[:3]
        w = s[3:]
        wx = skew(w)
        wx2 = wx @ wx
        I = np.eye(3)
        R = I + np.sin(theta) * wx + (1 - np.cos(theta)) * wx2
        V = I + (1 - np.cos(theta)) * wx + (theta - np.sin(theta)) * wx2
        p = V @ v
        return np.vstack((np.hstack((R, p[:, np.newaxis])), np.array([0, 0, 0, 1])))

    T = (
        expm(S1, theta1)
        @ expm(S2, theta2)
        @ expm(S3, theta3)
        @ expm(S4, theta4)
        @ expm(S5, theta5)
    )

    position = T[:3, 3]

    J_v = np.column_stack(
        (
            np.cross(S1[:3], position) + S1[3:],
            np.cross(S2[:3], position) + S2[3:],
            np.cross(S3[:3], position) + S3[3:],
            np.cross(S4[:3], position) + S4[3:],
            np.cross(S5[:3], position) + S5[3:],
        )
    )

    J_w = np.column_stack((S1[:3], S2[:3], S3[:3], S4[:3], S5[:3]))

    return J_v, J_w, position


def compute_geometric_jacobian(theta_vals, d1, a2, a3, d5):
    theta1, theta2, theta3, theta4, theta5 = theta_vals

    c1, s1 = np.cos(theta1), np.sin(theta1)
    c2, s2 = np.cos(theta2), np.sin(theta2)
    c3, s3 = np.cos(theta3), np.sin(theta3)
    c4, s4 = np.cos(theta4), np.sin(theta4)
    c5, s5 = np.cos(theta5), np.sin(theta5)

    p1 = np.array([0, 0, d1])
    p2 = p1 + np.array([a2 * c1 * c2, a2 * s1 * c2, a2 * s2])
    p3 = p2 + np.array(
        [
            a3 * c1 * (c2 * c3 - s2 * s3),
            a3 * s1 * (c2 * c3 - s2 * s3),
            a3 * (s2 * c3 + c2 * s3),
        ]
    )
    p4 = p3 + np.array([0, 0, 0])
    p5 = p4 + np.array([0, 0, d5])

    pe = p5

    z0 = np.array([0, 0, 1])
    z1 = np.array([c1, s1, 0])
    z2 = np.array([-s1, c1, 0])
    z3 = z2
    z4 = z0

    J_v = np.column_stack(
        (
            np.cross(z0, pe - np.array([0, 0, 0])),
            np.cross(z1, pe - p1),
            np.cross(z2, pe - p2),
            np.cross(z3, pe - p3),
            np.cross(z4, pe - p4),
        )
    )

    J_w = np.column_stack((z0, z1, z2, z3, z4))

    return J_v, J_w, pe


theta_vals = [np.pi / 4, np.pi / 6, np.pi / 3, np.pi / 4, np.pi / 6]
a2_val = 0.5
a3_val = 0.5
d1_val = 0.1
d5_val = 0.1

J_v_dh, J_w_dh, position_dh = compute_dh_jacobian(
    theta_vals, d1_val, a2_val, a3_val, d5_val
)
J_v_screw, J_w_screw, position_screw = compute_screw_jacobian(
    theta_vals, d1_val, a2_val, a3_val, d5_val
)
J_v_geo, J_w_geo, position_geo = compute_geometric_jacobian(
    theta_vals, d1_val, a2_val, a3_val, d5_val
)

print("\nEnd Effector Position (DH Method):")
print(position_dh)

print("\nEnd Effector Position (Screw Axis Method):")
print(position_screw)

print("\nEnd Effector Position (Geometric Method):")
print(position_geo)

print("\nNumerical Jacobian (Linear Velocity) using DH parameters:")
print(J_v_dh)

print("\nNumerical Jacobian (Linear Velocity) using screw axis method:")
print(J_v_screw)

print("\nGeometric Jacobian (Linear Velocity):")
print(J_v_geo)

print("\nNumerical Jacobian (Angular Velocity) using DH parameters:")
print(J_w_dh)

print("\nNumerical Jacobian (Angular Velocity) using screw axis method:")
print(J_w_screw)

print("\nGeometric Jacobian (Angular Velocity):")
print(J_w_geo)

print("\nDifference in Linear Velocity Jacobians (DH vs Screw):")
print(J_v_dh - J_v_screw)

print("\nDifference in Linear Velocity Jacobians (DH vs Geometric):")
print(J_v_dh - J_v_geo)

print("\nDifference in Angular Velocity Jacobians (DH vs Screw):")
print(J_w_dh - J_w_screw)

print("\nDifference in Angular Velocity Jacobians (DH vs Geometric):")
print(J_w_dh - J_w_geo)
