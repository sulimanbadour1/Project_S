import numpy as np
from scipy.optimize import fsolve

# Constants (these need to be defined or estimated)
a2 = 0.3  # Link length
a3 = 0.3  # Link length
a4 = 0.3  # Link length
a5 = 0.1  # Link length
d1 = 1.1  # Base height


# Define the system of equations
def equations(variables):
    theta1, theta2, theta3 = variables
    c1, c2, c3 = np.cos([theta1, theta2, theta3])
    s1, s2, s3 = np.sin([theta1, theta2, theta3])
    s23 = np.sin(theta2 + theta3)
    c23 = np.cos(theta2 + theta3)

    # Assuming theta4 and theta5 are zero for simplicity in this model
    theta4, theta5 = 0, 0
    c4, c5 = np.cos([theta4, theta5])
    s4, s5 = np.sin([theta4, theta5])

    # Simplified forward kinematics equations for position
    T14 = 0  # Assuming simplification or symmetry
    T24 = 0  # Assuming simplification or symmetry
    T34 = (
        a2 * s2 + a3 * s2 * c3 + a3 * s3 * c2 + d1
    )  # Simplified without s23, c23 dependencies

    # Equations representing the desired end-effector position
    return [
        T14,  # Simplified to be always zero, ideally should be expanded with actual kinematic relations
        T24,  # Simplified to be always zero, ideally should be expanded with actual kinematic relations
        T34 - 1.1,
    ]


# Initial guesses for theta values
initial_guesses = [0, 0, 0]

# Solve the system of equations
solution = fsolve(equations, initial_guesses)
print("Solution for the joint angles in radians:")
print("Theta1: {:.4f}, Theta2: {:.4f}, Theta3: {:.4f}".format(*solution))
