import sympy as sp
from sympy import pprint

# Definition of theoretical parameters
m_values = [1.0, 2.0, 1.5, 1.2, 1.8]  # Masses of the segments (in kg)
a_values = [0.1, 0.2, 0.15, 0.12, 0.18]  # Linear accelerations of the center of mass of each segment (in m/s^2)
R_values = [1.0, 1.0, 1.0, 1.0, 1.0]  # Rotation coefficients (unitless)
d_values = [0.15, 0.2, 0.175, 0.125, 0.15]  # Positions of the center of mass of each segment relative to the rotation axes (in meters)

# Axes of rotation of the joints (hypothetical)
z_values = [sp.Matrix([0, 1, 0])] * 5

# Initialization of variables for the Newton-Euler equations
forces = [0.0] * 5  # Forces on each segment
moments = [0.0] * 5  # Moments on each segment
torques = [0.0] * 5  # Joint torques

forces[4] = m_values[4] * a_values[4]
moments[4] = 0.25  # Placeholder value for moments on the fifth segment
torques[4] = sp.transpose(z_values[4]) * moments[4]

# Calculation of forces and moments for each segment
for i in range(3, -1, -1):
    forces[i] = forces[i+1] + m_values[i] * a_values[i]
    moments[i] = moments[i+1] + R_values[i+1] * (moments[i+1] + d_values[i+1] * forces[i+1]) + d_values[i] * forces[i]
    # Calculation of joint torques (projection of moments onto the rotation axes of the joints)
    torques[i] = sp.transpose(z_values[i]) * moments[i]

# Displaying the results
print("Forces on each segment:", forces)
print("Moments on each segment:", moments)
print("Joint torques on each joint:")
for i, torque in enumerate(torques, start=1):
    print(f"T{i}:", end=" ")
    pprint(torque)
