from sympy import symbols, cos, sin, pi, Matrix, simplify

# Define symbolic variables for joint angles
theta1, theta2, theta3, theta4, theta5 = symbols("theta1 theta2 theta3 theta4 theta5")

# DH Parameters
d1, a1, alpha1 = 0.05, 0.03, 0
d2, a2, alpha2 = 0, 0.25, 0
d3, a3, alpha3 = 0, 0.28, 0
d4, a4, alpha4 = 0, 0.28, 0
d5, a5, alpha5 = 0, 0.15, 0


# Define the transformation matrix using DH parameters
def DH_matrix(theta, d, a, alpha):
    return Matrix(
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


# Compute individual transformation matrices
T1 = DH_matrix(theta1, d1, a1, alpha1)
T2 = DH_matrix(theta2, d2, a2, alpha2)
T3 = DH_matrix(theta3, d3, a3, alpha3)
T4 = DH_matrix(theta4, d4, a4, alpha4)
T5 = DH_matrix(theta5, d5, a5, alpha5)

# Overall transformation matrix from the base to the end-effector
T = T1 * T2 * T3 * T4 * T5

# Test at zero angles
zero_angles = {theta1: 0, theta2: 0, theta3: 0, theta4: 0, theta5: 0}
expected_translations = [
    Matrix([[1, 0, 0, a1], [0, 1, 0, 0], [0, 0, 1, d1], [0, 0, 0, 1]]),
    Matrix([[1, 0, 0, a2], [0, 1, 0, 0], [0, 0, 1, d2], [0, 0, 0, 1]]),
    Matrix([[1, 0, 0, a3], [0, 1, 0, 0], [0, 0, 1, d3], [0, 0, 0, 1]]),
    Matrix([[1, 0, 0, a4], [0, 1, 0, 0], [0, 0, 1, d4], [0, 0, 0, 1]]),
    Matrix([[1, 0, 0, a5], [0, 1, 0, 0], [0, 0, 1, d5], [0, 0, 0, 1]]),
]

# Compute transformation matrices at zero angles
T1_zero = DH_matrix(0, d1, a1, alpha1).subs(zero_angles)
T2_zero = DH_matrix(0, d2, a2, alpha2).subs(zero_angles)
T3_zero = DH_matrix(0, d3, a3, alpha3).subs(zero_angles)
T4_zero = DH_matrix(0, d4, a4, alpha4).subs(zero_angles)
T5_zero = DH_matrix(0, d5, a5, alpha5).subs(zero_angles)

# List of computed matrices
computed_matrices = [T1_zero, T2_zero, T3_zero, T4_zero, T5_zero]

# Verification of transformation matrices
for i, (computed, expected) in enumerate(zip(computed_matrices, expected_translations)):
    if simplify(computed - expected) == Matrix.zeros(4, 4):
        print(f"Matrix T{i+1} at zero angles is correct.")
    else:
        print(f"Matrix T{i+1} at zero angles is incorrect.")
