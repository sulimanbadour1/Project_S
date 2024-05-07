from sympy import symbols, cos, sin, pi, Matrix

# Define the symbolic variables for joint angles
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
T
# print(T)


# Define some example joint angles in radians
theta1_val = pi / 4  # 45 degrees
theta2_val = pi / 6  # 30 degrees
theta3_val = -pi / 6  # -30 degrees
theta4_val = pi / 3  # 60 degrees
theta5_val = -pi / 4  # -45 degrees

# Substitute the angles into the transformation matrix
T_numerical = T.subs(
    {
        theta1: theta1_val,
        theta2: theta2_val,
        theta3: theta3_val,
        theta4: theta4_val,
        theta5: theta5_val,
    }
)

T_numerical.evalf()  # Numerically evaluate the matrix
print(T_numerical.evalf())
