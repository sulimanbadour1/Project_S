import numpy as np
import sympy as sp

# Define DH parameters
d1 = 0.1  # Link offset
d5 = 0.1  # Link offset
a2 = 0.5  # Link length
a3 = 0.5  # Link length
alpha = [np.pi / 2, 0, 0, np.pi / 2, 0]  # Twist angles in radians

# Define symbolic variables for joint angles and velocities
theta1, theta2, theta3, theta4, theta5 = sp.symbols(
    "theta1 theta2 theta3 theta4 theta5", real=True
)
dtheta1, dtheta2, dtheta3, dtheta4, dtheta5 = sp.symbols(
    "dtheta1 dtheta2 dtheta3 dtheta4 dtheta5", real=True
)
ddtheta1, ddtheta2, ddtheta3, ddtheta4, ddtheta5 = sp.symbols(
    "ddtheta1 ddtheta2 ddtheta3 ddtheta4 ddtheta5", real=True
)

# Masses of the links and additional components
masses = [1.0, 1.0, 1.0, 1.0, 1.0]
mass_camera = 0.5
mass_lights = 0.5
g = 9.81  # Gravitational acceleration

# Define lengths of the links
L1 = d1
L2 = a2
L3 = a3
L4 = d5
L5 = d5  # Assuming the last link has length d5


# Define DH transformation matrix function
def dh(theta, d, a, alpha):
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


# Compute transformation matrices
A1 = dh(theta1, d1, 0, alpha[0])
A2 = dh(theta2, 0, a2, alpha[1])
A3 = dh(theta3, 0, a3, alpha[2])
A4 = dh(theta4, 0, 0, alpha[3])
A5 = dh(theta5, d5, 0, alpha[4])

# Forward kinematics
T1 = A1
T2 = T1 * A2
T3 = T2 * A3
T4 = T3 * A4
T5 = T4 * A5

# Positions of each link's center of mass
p1 = T1[:3, 3] / 2
p2 = T2[:3, 3] / 2
p3 = T3[:3, 3] / 2
p4 = T4[:3, 3] / 2
p5 = T5[:3, 3] / 2

# Compute Jacobians for each link
Jv1 = p1.jacobian([theta1, theta2, theta3, theta4, theta5])
Jv2 = p2.jacobian([theta1, theta2, theta3, theta4, theta5])
Jv3 = p3.jacobian([theta1, theta2, theta3, theta4, theta5])
Jv4 = p4.jacobian([theta1, theta2, theta3, theta4, theta5])
Jv5 = p5.jacobian([theta1, theta2, theta3, theta4, theta5])

# Inertia matrices for each link (simplified as diagonal matrices for demonstration)
I1 = sp.eye(3) * (1 / 12) * masses[0] * (d1**2)
I2 = sp.eye(3) * (1 / 12) * masses[1] * (a2**2)
I3 = sp.eye(3) * (1 / 12) * masses[2] * (a3**2)
I4 = sp.eye(3) * (1 / 12) * masses[3] * (d1**2)
I5 = sp.eye(3) * (1 / 12) * (masses[4] + mass_camera + mass_lights) * (d5**2)

# Compute the inertia matrix
M = (
    Jv1.T * masses[0] * Jv1
    + Jv2.T * masses[1] * Jv2
    + Jv3.T * masses[2] * Jv3
    + Jv4.T * masses[3] * Jv4
    + Jv5.T * (masses[4] + mass_camera + mass_lights) * Jv5
)

# Compute the gravity vector
G = (
    Jv1.T * masses[0] * sp.Matrix([0, 0, -g])
    + Jv2.T * masses[1] * sp.Matrix([0, 0, -g])
    + Jv3.T * masses[2] * sp.Matrix([0, 0, -g])
    + Jv4.T * masses[3] * sp.Matrix([0, 0, -g])
    + Jv5.T * (masses[4] + mass_camera + mass_lights) * sp.Matrix([0, 0, -g])
)

# Define q, dq, and ddq
q = sp.Matrix([theta1, theta2, theta3, theta4, theta5])
dq = sp.Matrix([dtheta1, dtheta2, dtheta3, dtheta4, dtheta5])
ddq = sp.Matrix([ddtheta1, ddtheta2, ddtheta3, ddtheta4, ddtheta5])

# Kinetic energy
T = 0.5 * dq.T * M * dq

# Potential energy
U_g1 = -masses[0] * g * L1
U_g2 = -masses[1] * g * (L1 + 1 / 2 * L2 * sp.sin(theta2))
U_g3 = (
    -masses[2] * g * (L1 + L2 * sp.sin(theta2) + 1 / 2 * L3 * sp.sin(theta2 + theta3))
)
U_g4 = (
    -masses[3]
    * g
    * (
        L1
        + L2 * sp.sin(theta2)
        + L3 * sp.sin(theta2 + theta3)
        + 1 / 2 * L4 * sp.sin(theta2 + theta3 + theta4)
    )
)
U_g5 = (
    -(masses[4] + mass_camera + mass_lights)
    * g
    * (
        L1
        + L2 * sp.sin(theta2)
        + L3 * sp.sin(theta2 + theta3)
        + L4 * sp.sin(theta2 + theta3 + theta4)
        + 1 / 2 * L5 * sp.sin(theta2 + theta3 + theta4)
    )
)

# Total potential energy
V = U_g1 + U_g2 + U_g3 + U_g4 + U_g5

# Total energy
E = T + V

# Define target end-effector position
target_position = sp.Matrix([0.5, 0.5, 0.2])  # Column vector for target position

# Solve for two sets of joint angles that reach the target position with the constraint
# theta2 + theta3 + theta4 = 0

# Constraint equation
constraint = theta2 + theta3 + theta4

# Define two sets of initial guesses for the joint angles
initial_guess1 = [0, np.pi / 6, -np.pi / 6, -np.pi / 6, 0]
initial_guess2 = [0, -np.pi / 6, np.pi / 6, -np.pi / 6, 0]

# Define quintic polynomial coefficients and time
t, tf = sp.symbols("t tf", real=True)
tf_val = 10  # Final time (you can set it to any desired duration)


def calculate_quintic_coeffs(theta_f, tf):
    a0, a1, a2, a3, a4, a5 = sp.symbols("a0 a1 a2 a3 a4 a5", real=True)
    theta_t = a0 + a1 * t + a2 * t**2 + a3 * t**3 + a4 * t**4 + a5 * t**5
    dtheta_t = sp.diff(theta_t, t)
    ddtheta_t = sp.diff(dtheta_t, t)

    theta_0 = 0
    dtheta_0 = 0
    ddtheta_0 = 0
    dtheta_f = 0
    ddtheta_f = 0

    eqns = [
        theta_t.subs(t, 0) - theta_0,
        dtheta_t.subs(t, 0) - dtheta_0,
        ddtheta_t.subs(t, 0) - ddtheta_0,
        theta_t.subs(t, tf) - theta_f,
        dtheta_t.subs(t, tf) - dtheta_f,
        ddtheta_t.subs(t, tf) - ddtheta_f,
    ]

    coeffs = sp.solve(eqns, (a0, a1, a2, a3, a4, a5))
    return coeffs


# Calculate quintic polynomial coefficients for each joint for both configurations
coeffs1 = [calculate_quintic_coeffs(theta_f, tf_val) for theta_f in final_config1]
coeffs2 = [calculate_quintic_coeffs(theta_f, tf_val) for theta_f in final_config2]

# Define symbolic expressions for joint angles, velocities, and accelerations for each configuration
theta_t1_sym = [
    coeffs[a0]
    + coeffs[a1] * t
    + coeffs[a2] * t**2
    + coeffs[a3] * t**3
    + coeffs[a4] * t**4
    + coeffs[a5] * t**5
    for coeffs in coeffs1
]
theta_t2_sym = [
    coeffs[a0]
    + coeffs[a1] * t
    + coeffs[a2] * t**2
    + coeffs[a3] * t**3
    + coeffs[a4] * t**4
    + coeffs[a5] * t**5
    for coeffs in coeffs2
]

dtheta_t1_sym = [sp.diff(theta_t, t) for theta_t in theta_t1_sym]
dtheta_t2_sym = [sp.diff(theta_t, t) for theta_t in theta_t2_sym]

ddtheta_t1_sym = [sp.diff(dtheta_t, t) for dtheta_t in dtheta_t1_sym]
ddtheta_t2_sym = [sp.diff(dtheta_t, t) for dtheta_t in dtheta_t2_sym]

# Display the symbolic expressions for both configurations
print("Configuration 1 Joint Angle Polynomials:")
for theta_t in theta_t1_sym:
    sp.pprint(theta_t)

print("Configuration 2 Joint Angle Polynomials:")
for theta_t in theta_t2_sym:
    sp.pprint(theta_t)

# Generate LaTeX code for the symbolic expressions for inclusion in the paper
latex_code1 = [sp.latex(theta_t) for theta_t in theta_t1_sym]
latex_code2 = [sp.latex(theta_t) for theta_t in theta_t2_sym]

print("LaTeX Code for Configuration 1:")
for code in latex_code1:
    print(code)

print("LaTeX Code for Configuration 2:")
for code in latex_code2:
    print(code)
