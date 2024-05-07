import sympy as sp

# Define symbols for joint angles
theta1, theta2, theta3, theta4, theta5 = sp.symbols(
    "theta1 theta2 theta3 theta4 theta5"
)


# Define transformation matrix using DH parameters
def DH_transform(a, alpha, d, theta):
    return sp.Matrix(
        [
            [
                sp.cos(theta),
                -sp.sin(theta) * sp.cos(alpha),
                sp.sin(theta) * sp.sin(alpha),
                a * sp.cos(theta),
            ],
            [
                sp.sin(theta),
                sp.cos(theta) * sp.cos(alpha),
                -sp.cos(theta) * sp.sin(alpha),
                a * sp.sin(theta),
            ],
            [0, sp.sin(alpha), sp.cos(alpha), d],
            [0, 0, 0, 1],
        ]
    )


# DH Parameters for each joint
params = [
    (0, 0, 0.05, theta1),  # Joint 1
    (0.03, sp.pi / 2, 0, theta2),  # Joint 2
    (0.25, 0, 0, theta3),  # Joint 3
    (0.28, 0, 0, theta4),  # Joint 4
    (0.28, 0, 0, theta5),  # Joint 5
]

# Calculate the total transformation matrix
T = sp.eye(4)
for p in params:
    T *= DH_transform(*p)

T  # Display the resulting transformation matrix from base to end-effector
print(T)


### verify the result
import sympy as sp

# Define symbols for joint angles
theta1, theta2, theta3, theta4, theta5 = sp.symbols(
    "theta1 theta2 theta3 theta4 theta5"
)


# Define transformation matrix using DH parameters
def DH_transform(a, alpha, d, theta):
    return sp.Matrix(
        [
            [
                sp.cos(theta),
                -sp.sin(theta) * sp.cos(alpha),
                sp.sin(theta) * sp.sin(alpha),
                a * sp.cos(theta),
            ],
            [
                sp.sin(theta),
                sp.cos(theta) * sp.cos(alpha),
                -sp.cos(theta) * sp.sin(alpha),
                a * sp.sin(theta),
            ],
            [0, sp.sin(alpha), sp.cos(alpha), d],
            [0, 0, 0, 1],
        ]
    )


# DH Parameters for each joint
params = [
    (0, 0, 0.05, theta1),  # Joint 1
    (0.03, sp.pi / 2, 0, theta2),  # Joint 2
    (0.25, 0, 0, theta3),  # Joint 3
    (0.28, 0, 0, theta4),  # Joint 4
    (0.28, 0, 0, theta5),  # Joint 5
]

# Calculate the total transformation matrix
T = sp.eye(4)
for a, alpha, d, theta in params:
    T *= DH_transform(a, alpha, d, theta)

# Substitute specific joint angles to evaluate
joint_values = {
    theta1: sp.pi / 6,
    theta2: -sp.pi / 4,
    theta3: sp.pi / 4,
    theta4: -sp.pi / 6,
    theta5: sp.pi / 12,
}
T_evaluated = T.subs(joint_values)

# Display the evaluated transformation matrix
print("Evaluated Transformation Matrix:")
print(T_evaluated)

# Calculate specific position for verification
# Extract the position part of the transformation matrix
position = T_evaluated[:3, 3]
print("Position of the end-effector:")
print(position)
