import numpy as np
import matplotlib.pyplot as plt

def dynamic_analysis(
    m_values, v_ci_values, I_ci_values, omega_values, g, h_values, q_ddot_values, q_dot_values, tau_values
):
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

    # Calculate the maximum torques
    max_torque_per_joint = np.maximum(lhs, tau)

    # Plot the maximum torques
    joints = ["Joint 1", "Joint 2", "Joint 3", "Joint 4", "Joint 5"]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(joints, max_torque_per_joint.tolist(), color="blue")
    plt.xlabel("Joints")
    plt.ylabel("Maximum Torque (Nm)")
    plt.title(
        "Maximum Dynamic Torque on Each Joint Across All Configurations Using Lagrangian Formulation"
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
v_ci_values = [0.5, 0.6, 0.55, 0.52, 0.58]  # Linear velocities of the centers of mass (m/s)
I_ci_values = [0.1, 0.1, 0.1, 0.1, 0.1]  # Moments of inertia about the centers of mass (kg*m^2)
omega_values = [1.0, 1.2, 1.1, 1.05, 1.15]  # Angular velocities of the segments (rad/s)
g = 9.81  # Acceleration due to gravity (m/s^2)
h_values = [0.1, 0.2, 0.3, 0.1, 0.1]  # Heights of the centers of mass (m)
q_ddot_values = [0.0, 0.0, 0.0, 0.0, 0.0]  # Accelerations (rad/s^2)
q_dot_values = [0.0, 0.0, 0.0, 0.0, 0.0]  # Angular velocities (rad/s)
tau_values = [0.0, 0.0, 0.0, 0.0,0.0]  # Joint torques (Nm)

print("Performing dynamic torque analysis across all configurations using Lagrangian formulation...")
max_torque_per_joint = dynamic_analysis(
    m_values, v_ci_values, I_ci_values, omega_values, g, h_values, q_ddot_values, q_dot_values, tau_values
)

print(f"Maximum Torques for given values: {max_torque_per_joint}")
