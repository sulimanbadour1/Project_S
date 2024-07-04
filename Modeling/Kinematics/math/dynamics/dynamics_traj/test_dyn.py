import numpy as np
import sympy as sp

# Define symbolic variables for joints
q1, q2, q3, q4, q5 = sp.symbols("q1 q2 q3 q4 q5")
q1_dot, q2_dot, q3_dot, q4_dot, q5_dot = sp.symbols(
    "q1_dot q2_dot q3_dot q4_dot q5_dot"
)
q1_ddot, q2_ddot, q3_ddot, q4_ddot, q5_ddot = sp.symbols(
    "q1_ddot q2_ddot q3_ddot q4_ddot q5_ddot"
)

# Define DH parameters
d1, a1, alpha1 = 0.1, 0, np.pi / 2
d2, a2, alpha2 = 0, 0.5, 0
d3, a3, alpha3 = 0, 0.5, 0
d4, a4, alpha4 = 0, 0, np.pi / 2
d5, a5, alpha5 = 0.1, 0, 0

# Physical properties
masses = [1, 1, 1, 1, 1]  # Mass of each link in kg
lengths = [d1, a2, a3, d4, d5]  # Lengths corresponding to DH parameters
inertia_tensors = [
    sp.Matrix([[0.00083, 0, 0], [0, 0.00083, 0], [0, 0, 0]]),
    sp.Matrix([[0, 0, 0], [0, 0, 0], [0, 0, 0.02083]]),
    sp.Matrix([[0, 0, 0], [0, 0, 0], [0, 0, 0.02083]]),
    sp.Matrix([[0.00083, 0, 0], [0, 0.00083, 0], [0, 0, 0]]),
    sp.Matrix([[0.00083, 0, 0], [0, 0.00083, 0], [0, 0, 0]]),
]

# Gravity vector
g = sp.Matrix([0, 0, -9.81])

# Initial and final conditions for trajectory
q_initial = [0, 0, 0, 0, 0]
q_dot_initial = [0, 0, 0, 0, 0]
q_ddot_initial = [0, 0, 0, 0, 0]

q_final = [np.pi / 4, np.pi / 2, np.pi / 2, np.pi / 4, np.pi / 4]
q_dot_final = [0, 0, 0, 0, 0]
q_ddot_final = [0, 0, 0, 0, 0]

# Time duration
T = 2  # seconds


# Function to compute DH transformation matrix
def dh_transformation(a, alpha, d, theta):
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


# Transformation matrices for each joint
T1 = dh_transformation(a1, alpha1, d1, q1)
T2 = dh_transformation(a2, alpha2, d2, q2)
T3 = dh_transformation(a3, alpha3, d3, q3)
T4 = dh_transformation(a4, alpha4, d4, q4)
T5 = dh_transformation(a5, alpha5, d5, q5)

# Full transformation from base to end-effector
T_matrices = [T1, T2, T3, T4, T5]


# Function to compute the Jacobian
def compute_jacobian(T_matrices, joints):
    J = sp.zeros(6, len(joints))
    T_0_i = sp.eye(4)
    z = sp.Matrix([0, 0, 1])
    p_0_n = T_matrices[-1][:3, 3]

    for i in range(len(joints)):
        T_0_i = T_0_i * T_matrices[i]
        z_0_i = T_0_i[:3, 2]
        p_0_i = T_0_i[:3, 3]

        J[:3, i] = z_0_i.cross(p_0_n - p_0_i)
        J[3:, i] = z_0_i

    return J


# List of joint variables
joints = sp.Matrix([q1, q2, q3, q4, q5])

# Compute Jacobian
J = compute_jacobian(T_matrices, joints)


# Function to compute the inertia matrix
def compute_inertia_matrix(inertia_tensors, T_matrices, masses, J):
    M = sp.zeros(len(joints), len(joints))

    for i in range(len(joints)):
        R_i = T_matrices[i][:3, :3]
        I_i = R_i * inertia_tensors[i] * R_i.T
        Jvi = J[:3, i]
        Jwi = J[3:, i]
        M += masses[i] * (Jvi * Jvi.T) + (Jwi.T * I_i * Jwi)

    return M


# Compute inertia matrix
M = compute_inertia_matrix(inertia_tensors, T_matrices, masses, J)


# Function to compute the Coriolis matrix
def compute_coriolis_matrix(M, joints, q_dot):
    C = sp.zeros(len(joints), len(joints))

    for i in range(len(joints)):
        for j in range(len(joints)):
            for k in range(len(joints)):
                C[i, j] += (
                    0.5
                    * (
                        sp.diff(M[i, j], joints[k])
                        + sp.diff(M[i, k], joints[j])
                        - sp.diff(M[j, k], joints[i])
                    )
                    * q_dot[k]
                )

    return C


# Compute Coriolis matrix
q_dot = sp.Matrix([q1_dot, q2_dot, q3_dot, q4_dot, q5_dot])
C = compute_coriolis_matrix(M, joints, q_dot)


# Function to compute the gravity vector
def compute_gravity_vector(masses, g, T_matrices):
    G = sp.zeros(len(joints), 1)

    for i in range(len(masses)):
        R_i = T_matrices[i][:3, :3]
        p_i = T_matrices[i][:3, 3]
        G += masses[i] * (R_i.T * g)

    return G


# Compute gravity vector
G = compute_gravity_vector(masses, g, T_matrices)


# Function to generate cubic trajectory
def cubic_trajectory(q0, qT, T):
    t = sp.symbols("t")
    a0 = q0
    a1 = 0
    a2 = 3 * (qT - q0) / T**2
    a3 = -2 * (qT - q0) / T**3
    q_t = a0 + a1 * t + a2 * t**2 + a3 * t**3
    q_dot_t = sp.diff(q_t, t)
    q_ddot_t = sp.diff(q_dot_t, t)
    return q_t, q_dot_t, q_ddot_t


# Generate trajectories for each joint
trajectories = [cubic_trajectory(q_initial[i], q_final[i], T) for i in range(5)]

# Define symbolic time variable
t = sp.symbols("t")

# Substitute trajectory into dynamic equations
q_ddot_sym = sp.Matrix([q1_ddot, q2_ddot, q3_ddot, q4_ddot, q5_ddot])
tau = M * q_ddot_sym + C * q_dot + G


# Function to evaluate torque at a specific time
def evaluate_torque(t_val):
    subs = {
        q1: trajectories[0][0].subs(t, t_val),
        q2: trajectories[1][0].subs(t, t_val),
        q3: trajectories[2][0].subs(t, t_val),
        q4: trajectories[3][0].subs(t, t_val),
        q5: trajectories[4][0].subs(t, t_val),
        q1_dot: trajectories[0][1].subs(t, t_val),
        q2_dot: trajectories[1][1].subs(t, t_val),
        q3_dot: trajectories[2][1].subs(t, t_val),
        q4_dot: trajectories[3][1].subs(t, t_val),
        q5_dot: trajectories[4][1].subs(t, t_val),
        q1_ddot: trajectories[0][2].subs(t, t_val),
        q2_ddot: trajectories[1][2].subs(t, t_val),
        q3_ddot: trajectories[2][2].subs(t, t_val),
        q4_ddot: trajectories[3][2].subs(t, t_val),
        q5_ddot: trajectories[4][2].subs(t, t_val),
    }
    return tau.subs(subs)


# Example evaluation at t = 1 second
tau_at_t1 = evaluate_torque(1)
print(tau_at_t1)
