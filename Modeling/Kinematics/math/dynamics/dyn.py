import numpy as np

# DH Parameters
d1, a1, alpha1 = 0.1, 0, np.pi / 2
d2, a2, alpha2 = 0, 0.5, 0
d3, a3, alpha3 = 0, 0.5, 0
d4, a4, alpha4 = 0, 0, np.pi / 2
d5, a5, alpha5 = 0.1, 0, 0

# Robot parameters
d_1_val = 0.1
d_5_val = 0.1
a_2_val = 0.5
a_3_val = 0.5
masses = [1.0, 1.0, 1.0, 1.0, 2.0]  # Includes camera and lights mass
g = np.array([0, 0, -9.81])

# External forces
external_forces = np.array([0, 0, 0, 0, 0, 0])


# Define the transformation matrix using DH parameters
def DH_matrix(theta, d, a, alpha):
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


# Compute the Jacobian
def jacobian(theta1, theta2, theta3, theta4, theta5):
    T1 = DH_matrix(theta1, d1, a1, alpha1)
    T2 = DH_matrix(theta2, d2, a2, alpha2)
    T3 = DH_matrix(theta3, d3, a3, alpha3)
    T4 = DH_matrix(theta4, d4, a4, alpha4)
    T5 = DH_matrix(theta5, d5, a5, alpha5)

    T01 = T1
    T02 = np.dot(T01, T2)
    T03 = np.dot(T02, T3)
    T04 = np.dot(T03, T4)
    T05 = np.dot(T04, T5)

    p1 = T01[:3, 3] / 2
    p2 = T02[:3, 3] / 2
    p3 = T03[:3, 3] / 2
    p4 = T04[:3, 3] / 2
    p5 = T05[:3, 3] / 2

    p = [p1, p2, p3, p4, p5]
    Jv = []

    for i in range(5):
        Jvi = np.zeros((3, 5))
        for j in range(i + 1):
            T = [T01, T02, T03, T04, T05][j]
            z = T[:3, 2]
            Jvi[:, j] = np.cross(z, (p[i] - T[:3, 3]))
        Jv.append(Jvi)

    return np.vstack(Jv)


# Define range of angles for sampling
theta1_range = np.linspace(-np.pi, np.pi, 40)
theta2_range = np.linspace(-np.pi / 2, np.pi / 2, 40)
theta3_range = np.linspace(-np.pi, np.pi, 40)
theta4_range = np.linspace(-np.pi / 2, np.pi / 2, 40)
theta5_range = np.linspace(-np.pi, np.pi, 40)

# Initialize variable to hold maximum torque
max_torque = 0

# Sample all combinations of joint angles
for t1 in theta1_range:
    for t2 in theta2_range:
        for t3 in theta3_range:
            for t4 in theta4_range:
                for t5 in theta5_range:
                    J = jacobian(t1, t2, t3, t4, t5)
                    torques = []
                    for i in range(5):
                        torque_g = masses[i] * np.dot(J[i * 3 : (i + 1) * 3, :].T, g)
                        torques.append(np.linalg.norm(torque_g))
                    max_torque = max(max_torque, max(torques))

print(
    f"The maximum torque the robot has to move in any configuration is: {max_torque:.2f} Nm"
)
