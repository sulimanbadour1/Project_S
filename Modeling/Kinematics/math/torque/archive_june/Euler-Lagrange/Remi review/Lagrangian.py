import numpy as np

# Parameters
m_values = [1.0, 2.0, 1.5, 1.2, 1.8]  # Masses of the segments (kg)
v_ci_values = [0.5, 0.6, 0.55, 0.52, 0.58]  # Linear velocities of the centers of mass (m/s)
I_ci_values = [0.02, 0.03, 0.025, 0.022, 0.028]  # Moments of inertia about the centers of mass (kg*m^2)
omega_values = [1.0, 1.2, 1.1, 1.05, 1.15]  # Angular velocities of the segments (rad/s)
g = 9.81  # Acceleration due to gravity (m/s^2)
h_values = [0.2, 0.25, 0.23, 0.22, 0.24]  # Heights of the centers of mass (m)
q_ddot_values = [0.8, 0.9, 0.85, 0.82, 0.88]  # Accelerations (rad/s^2)
q_dot_values = [0.5, 0.6, 0.55, 0.52, 0.58]  # Angular velocities (rad/s)
tau_values = [10, 12, 11, 10.5, 11.5]  # Joint torques (Nm)

# Kinetic Energy (T)
T = 0
for i in range(5):
    T += 0.5 * m_values[i] * v_ci_values[i]**2 + 0.5 * I_ci_values[i] * omega_values[i]**2

# Potential Energy (V)
V = 0
for i in range(5):
    V += m_values[i] * g * h_values[i]

# Lagrangian (L)
L = T - V

# Dynamic Equations
M = np.zeros((5, 5))  # Mass matrix
C = np.zeros((5, 5))  # Coriolis and centrifugal matrix
G = np.zeros(5)       # Gravitational forces vector

# Filling the matrices and vectors with example values
for i in range(5):
    for j in range(5):
        M[i, j] = np.random.random()  # Replace with actual computation
        C[i, j] = np.random.random()  # Replace with actual computation
    G[i] = m_values[i] * g * h_values[i]

# Generalized accelerations
q_ddot = np.array(q_ddot_values)

# Generalized velocities
q_dot = np.array(q_dot_values)

# Joint torques
tau = np.array(tau_values)

# Compute the left-hand side of the equations
lhs = M @ q_ddot + C @ q_dot + G


# Print the results
for i in range(5):
    M_terms = " + ".join([f"{M[i, j]:.2f}*q{j+1}_ddot" for j in range(5)])
    C_terms = " + ".join([f"{C[i, j]:.2f}*q{j+1}_dot" for j in range(5)])
    equation = f"{M_terms} + {C_terms} + {G[i]:.2f} = {tau[i]}"
    print(f"Equation {i+1}:")
    print(equation)