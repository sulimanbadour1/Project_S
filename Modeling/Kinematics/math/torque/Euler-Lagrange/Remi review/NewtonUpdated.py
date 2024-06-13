import sympy as sp
from sympy import pprint
import numpy as np
import matplotlib.pyplot as plt
from itertools import product

def dynamic_analysis_newton_euler(
    m_values, a_values, R_values, d_values
):
    # Define symbolic variables for angular accelerations and velocities
    theta_1, theta_2, theta_3, theta_4, theta_5 = sp.symbols(
        "theta_1 theta_2 theta_3 theta_4 theta_5"
    )
    dtheta_1, dtheta_2, dtheta_3, dtheta_4, dtheta_5 = sp.symbols(
        "dtheta_1 dtheta_2 dtheta_3 dtheta_4 dtheta_5"
    )
    
    # Axes of rotation of the joints (hypothetical)
    z_values = [sp.Matrix([0, 1, 0])] * 5

    # Initialization of variables for the Newton-Euler equations
    forces = [0.0] * 5  # Forces on each segment
    moments = [0.0] * 5  # Moments on each segment
    torques = [0.0] * 5  # Joint torques

    # Calculate forces and moments for each segment
    forces[4] = m_values[4] * a_values[4]
    moments[4] = 0.0  # Placeholder value for moments on the fifth segment
    torques[4] = sp.transpose(z_values[4]) * moments[4]

    # Calculation of forces and moments for each segment
    for i in range(3, -1, -1):
        forces[i] = forces[i+1] + m_values[i] * a_values[i]
        moments[i] = moments[i+1] + R_values[i+1] * (moments[i+1] + d_values[i+1] * forces[i+1]) + d_values[i] * forces[i]
        # Calculation of joint torques (projection of moments onto the rotation axes of the joints)
        torques[i] = sp.transpose(z_values[i]) * moments[i]

    # Substitute numerical values into the torques
    substitutions = {
        theta_1: 0, theta_2: 0, theta_3: 0, theta_4: 0, theta_5: 0,
        dtheta_1: 0, dtheta_2: 0, dtheta_3: 0, dtheta_4: 0, dtheta_5: 0
    }
    torques_numeric = [torque.subs(substitutions) for torque in torques]

    # Precompute the torques function
    try:
        torques_func = sp.lambdify(
            (
                theta_1,
                theta_2,
                theta_3,
                theta_4,
                theta_5,
                dtheta_1,
                dtheta_2,
                dtheta_3,
                dtheta_4,
                dtheta_5,
            ),
            torques_numeric,
            "numpy",
        )
    except Exception as e:
        print(f"Lambdify failed: {e}")
        return []

    # Initialize the maximum torque tracker
    max_torque_per_joint = np.zeros(5)

    # Define the range for joint angles and velocities
    angle_range = np.linspace(-np.pi, np.pi, 5)  # Reduced to 5 steps from -π to π
    velocity_range = np.linspace(-2, 2, 3)  # Reduced to 3 steps for angular velocities

    # Generate all combinations of joint angles and velocities
    angle_combinations = list(product(angle_range, repeat=5))
    velocity_combinations = list(product(velocity_range, repeat=5))

    # Convert combinations to numpy arrays for vectorized operations
    angle_combinations = np.array(angle_combinations)
    velocity_combinations = np.array(velocity_combinations)

    # Iterate over all angle combinations
    for angles in angle_combinations:
        # Compute torques for all velocity combinations at once using vectorized operations
        for velocities in velocity_combinations:
            numerical_torques = np.array(
                torques_func(*angles, *velocities), dtype=float
            ).flatten()
            max_torque_per_joint = np.maximum(
                max_torque_per_joint, np.abs(numerical_torques[:5])
            )

    # Plot the maximum torques
    joints = ["Joint 1", "Joint 2", "Joint 3", "Joint 4", "Joint 5"]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(joints, max_torque_per_joint.tolist(), color="blue")
    plt.xlabel("Joints")
    plt.ylabel("Maximum Torque (Nm)")
    plt.title(
        "Maximum Dynamic Torque on Each Joint Across All Configurations Using Newton-Euler Formulation"
    )
    plt.grid(True, linestyle="--", alpha=0.6)

    # Annotate bars with their values
    for bar in bars:
        yval = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            yval,
            f"{yval:.2f}",
            va="bottom" if yval < 0 else "top",
            ha="center",
            color="black",
        )

    plt.show()

    # Print the maximum torques
    print(f"Maximum Torques for given values: {max_torque_per_joint.tolist()}")
    return max_torque_per_joint.tolist()


# Parameters
m_values = [1.0, 1.0, 1.0, 1.0, 2.0]  # Masses of the segments (kg)
a_values = [0.5, 0.5, 0.5, 0.5, 0.5]  # Linear accelerations of the center of mass (m/s^2)
R_values = [1.0, 1.0, 1.0, 1.0, 1.0]  # Rotation coefficients (unitless)
d_values = [0.1, 0.5, 0.5, 0.1, 0.1]  # Positions of the center of mass (m)

print("Performing dynamic torque analysis across all configurations using Newton-Euler formulation...")
max_torque_per_joint = dynamic_analysis_newton_euler(
    m_values, a_values, R_values, d_values
)

print(f"Maximum Torques for given values: {max_torque_per_joint}")
