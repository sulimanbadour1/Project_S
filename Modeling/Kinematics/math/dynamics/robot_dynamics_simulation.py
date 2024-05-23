"""
robot_dynamics_simulation.py

This script calculates and simulates the dynamics of a 5-DOF robotic manipulator using the Euler-Lagrange formulation.
The simulation provides the joint angles of the robot over time based on given initial conditions and physical parameters.

### Overview

The script performs the following tasks:
1. Defines symbolic variables for the joint angles, velocities, and accelerations.
2. Specifies the Denavit-Hartenberg (DH) parameters for the robot's links.
3. Computes the transformation matrices for each link using the DH parameters.
4. Calculates the positions and velocities of the centers of mass for each link.
5. Derives the kinetic and potential energies of the robot.
6. Formulates the Euler-Lagrange equations of motion.
7. Substitutes given physical parameters (masses, lengths, etc.) into the equations.
8. Converts the equations of motion into a system of first-order differential equations.
9. Solves the differential equations numerically using `scipy.integrate.solve_ivp`.
10. Plots the joint angles over time.
"""

import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Define symbolic variables for joint angles, velocities, accelerations
theta_1, theta_2, theta_3, theta_4, theta_5 = sp.symbols(
    "theta_1 theta_2 theta_3 theta_4 theta_5"
)
dtheta_1, dtheta_2, dtheta_3, dtheta_4, dtheta_5 = sp.symbols(
    "dtheta_1 dtheta_2 dtheta_3 dtheta_4 dtheta_5"
)
ddtheta_1, ddtheta_2, ddtheta_3, ddtheta_4, ddtheta_5 = sp.symbols(
    "ddtheta_1 ddtheta_2 ddtheta_3 ddtheta_4 ddtheta_5"
)

# DH parameters
d_1, d_5 = 0.1, 0.1
a_2, a_3 = 0.5, 0.5
alpha = [90, 0, 0, 90, 0]

# Masses and inertia tensors of the links
m1, m2, m3, m4, m5 = 1, 1, 1, 1, 1
I1, I2, I3, I4, I5 = 0.1, 0.1, 0.1, 0.1, 0.1
# Gravity
g = 9.81


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


# Transformation matrices for each joint
A1 = dh_matrix(theta_1, d_1, 0, alpha[0])
A2 = dh_matrix(theta_2, 0, a_2, alpha[1])
A3 = dh_matrix(theta_3, 0, a_3, alpha[2])
A4 = dh_matrix(theta_4, 0, 0, alpha[3])
A5 = dh_matrix(theta_5, d_5, 0, alpha[4])

# Compute the overall transformation matrix by multiplying individual matrices
T1 = A1
T2 = T1 * A2
T3 = T2 * A3
T4 = T3 * A4
T5 = T4 * A5

# Position vectors of the center of mass of each link (assuming center of mass is at the middle of each link)
P1 = T1[:3, 3] / 2
P2 = T2[:3, 3] / 2
P3 = T3[:3, 3] / 2
P4 = T4[:3, 3] / 2
P5 = T5[:3, 3] / 2

# Velocity vectors of the center of mass of each link
V1 = (
    P1.diff(theta_1) * dtheta_1
    + P1.diff(theta_2) * dtheta_2
    + P1.diff(theta_3) * dtheta_3
    + P1.diff(theta_4) * dtheta_4
    + P1.diff(theta_5) * dtheta_5
)
V2 = (
    P2.diff(theta_1) * dtheta_1
    + P2.diff(theta_2) * dtheta_2
    + P2.diff(theta_3) * dtheta_3
    + P2.diff(theta_4) * dtheta_4
    + P2.diff(theta_5) * dtheta_5
)
V3 = (
    P3.diff(theta_1) * dtheta_1
    + P3.diff(theta_2) * dtheta_2
    + P3.diff(theta_3) * dtheta_3
    + P3.diff(theta_4) * dtheta_4
    + P3.diff(theta_5) * dtheta_5
)
V4 = (
    P4.diff(theta_1) * dtheta_1
    + P4.diff(theta_2) * dtheta_2
    + P4.diff(theta_3) * dtheta_3
    + P4.diff(theta_4) * dtheta_4
    + P4.diff(theta_5) * dtheta_5
)
V5 = (
    P5.diff(theta_1) * dtheta_1
    + P5.diff(theta_2) * dtheta_2
    + P5.diff(theta_3) * dtheta_3
    + P5.diff(theta_4) * dtheta_4
    + P5.diff(theta_5) * dtheta_5
)

# Kinetic energy of each link
T1 = 0.5 * m1 * (V1.dot(V1)) + 0.5 * I1 * dtheta_1**2
T2 = 0.5 * m2 * (V2.dot(V2)) + 0.5 * I2 * dtheta_2**2
T3 = 0.5 * m3 * (V3.dot(V3)) + 0.5 * I3 * dtheta_3**2
T4 = 0.5 * m4 * (V4.dot(V4)) + 0.5 * I4 * dtheta_4**2
T5 = 0.5 * m5 * (V5.dot(V5)) + 0.5 * I5 * dtheta_5**2

# Total kinetic energy
T = T1 + T2 + T3 + T4 + T5

# Potential energy of each link
V1 = m1 * g * P1[2]
V2 = m2 * g * P2[2]
V3 = m3 * g * P3[2]
V4 = m4 * g * P4[2]
V5 = m5 * g * P5[2]

# Total potential energy
V = V1 + V2 + V3 + V4 + V5
# Lagrangian
L = T - V

# Generalized coordinates and their derivatives
q = sp.Matrix([theta_1, theta_2, theta_3, theta_4, theta_5])
dq = sp.Matrix([dtheta_1, dtheta_2, dtheta_3, dtheta_4, dtheta_5])
ddq = sp.Matrix([ddtheta_1, ddtheta_2, ddtheta_3, ddtheta_4, ddtheta_5])

# Euler-Lagrange equations
EOM = sp.Matrix(
    [sp.diff(sp.diff(L, dq[i]), "t") - sp.diff(L, q[i]) for i in range(len(q))]
)

# Substitute time derivatives of generalized coordinates
EOM = EOM.subs({q[i].diff("t"): dq[i] for i in range(len(q))}).subs(
    {dq[i].diff("t"): ddq[i] for i in range(len(dq))}
)

# Substitute given values
params = {
    d_1: 0.1,
    d_5: 0.1,
    a_2: 0.5,
    a_3: 0.5,
    m1: 1,
    m2: 1,
    m3: 1,
    m4: 1,
    m5: 1,
    I1: 0.1,
    I2: 0.1,
    I3: 0.1,
    I4: 0.1,
    I5: 0.1,
    g: 9.81,
}
EOM_numeric = EOM.subs(params)

# Convert EOM to a system of first-order ODEs
state_vars = q.tolist() + dq.tolist()
state_funcs = sp.lambdify(state_vars, EOM_numeric, "numpy")


def dynamics(t, state):
    q_vals = state[:5]
    dq_vals = state[5:]
    ddq_vals = np.array(state_funcs(*q_vals, *dq_vals)).flatten()
    return np.concatenate([dq_vals, ddq_vals])


# Initial conditions (angles and angular velocities)
initial_conditions = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

# Time span for the simulation
t_span = (0, 10)
t_eval = np.linspace(t_span[0], t_span[1], 100)

# Solve the system of ODEs
sol = solve_ivp(dynamics, t_span, initial_conditions, t_eval=t_eval)

# Plot the results
plt.figure(figsize=(12, 8))
for i in range(5):
    plt.plot(sol.t, sol.y[i], label=f"Theta_{i+1}")
plt.xlabel("Time (s)")
plt.ylabel("Joint Angles (rad)")
plt.title("Joint Angles vs. Time")
plt.legend()
plt.grid()
plt.show()
