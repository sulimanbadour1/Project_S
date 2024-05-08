from sympy import symbols, cos, sin, pi, Matrix, rad

# Define the symbolic variables for joint angles
theta1, theta2, theta3, theta4, theta5 = symbols("theta1 theta2 theta3 theta4 theta5")

# DH Parameters
d1, a1, alpha1 = 0.05, 0, 0  # Joint 1 rotates about z-axis
d2, a2, alpha2 = 0, 0.03, pi / 2  # Joint 2 shifts axis to y, hence 90-degree shift
d3, a3, alpha3 = 0, 0.25, 0  # Following joints rotate around y-axis, no alpha shift
d4, a4, alpha4 = 0, 0.28, 0
d5, a5, alpha5 = (
    0,
    0.28,
    0,
)  # Note: This should be the link before the end, if any confusion clarify


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
# Uncomment to display the matrix
print(T)


################  Numerical Evaluation  ################


# Compute individual transformation matrices with joint angles
T1 = DH_matrix(-pi, d1, a1, alpha1)
T2 = DH_matrix(-pi / 6, d2, a2, alpha2)
T3 = DH_matrix(-pi / 6, d3, a3, alpha3)
T4 = DH_matrix(-pi / 6, d4, a4, alpha4)
T5 = DH_matrix(-pi / 6, d5, a5, alpha5)

# Overall transformation matrix from the base to the end-effector
T = T1 * T2 * T3 * T4 * T5

# Extracting the position and orientation from the transformation matrix
x, y, z = T[0, 3], T[1, 3], T[2, 3]
orientation = T[:3, :3]

# Display results
print("Position of the End Effector (x, y, z):")
print(f"x = {x.evalf()}")
print(f"y = {y.evalf()}")
print(f"z = {z.evalf()}\n")

print("Orientation Matrix of the End Effector:")
print(orientation.evalf())
print("\n")
# Overall transformation matrix from the base to the end-effector
T = T1 * T2 * T3 * T4 * T5
T.evalf()  # Evaluate the matrix numerically for clarity
print("Numerical Evaluation of the Transformation Matrix:")
print(T.evalf())
