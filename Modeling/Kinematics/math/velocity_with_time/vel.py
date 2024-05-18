import numpy as np
import matplotlib.pyplot as plt

# Define DH parameters
d1 = 0.1
d5 = 0.1
a2 = 0.5
a3 = 0.5
alpha = np.deg2rad([90, 0, 0, 90, 0])


# Helper function to create a transformation matrix from DH parameters
def dh_matrix(theta, d, a, alpha):
    return np.array(
        [
            [
                np.cos(theta),
                -np.sin(theta) * np.cos(alpha),
                np.sin(theta) * np.sin(alpha),
                a * np.cos(theta),
            ],
            [
                np.sin(theta),
                np.cos(theta) * np.cos(alpha),
                -np.cos(theta) * np.sin(alpha),
                a * np.sin(theta),
            ],
            [0, np.sin(alpha), np.cos(alpha), d],
            [0, 0, 0, 1],
        ]
    )


# Function to compute the Jacobian matrix
def compute_jacobian(thetas):
    theta1, theta2, theta3, theta4, theta5 = thetas
    A1 = dh_matrix(theta1, d1, 0, alpha[0])
    A2 = dh_matrix(theta2, 0, a2, alpha[1])
    A3 = dh_matrix(theta3, 0, a3, alpha[2])
    A4 = dh_matrix(theta4, 0, 0, alpha[3])
    A5 = dh_matrix(theta5, d5, 0, alpha[4])

    T = A1 @ A2 @ A3 @ A4 @ A5

    position = T[:3, 3]
    z0 = np.array([0, 0, 1])
    z1 = A1[:3, :3] @ z0
    z2 = (A1 @ A2)[:3, :3] @ z0
    z3 = (A1 @ A2 @ A3)[:3, :3] @ z0
    z4 = (A1 @ A2 @ A3 @ A4)[:3, :3] @ z0

    Jv = np.column_stack(
        [
            np.cross(z0, position),
            np.cross(z1, position - A1[:3, 3]),
            np.cross(z2, position - (A1 @ A2)[:3, 3]),
            np.cross(z3, position - (A1 @ A2 @ A3)[:3, 3]),
            np.cross(z4, position - (A1 @ A2 @ A3 @ A4)[:3, 3]),
        ]
    )

    Jw = np.column_stack([z0, z1, z2, z3, z4])

    J = np.vstack([Jv, Jw])

    return J


# Time range for the simulation
time = np.linspace(0, 10, 100)

# Define time-dependent joint angles (e.g., sinusoidal functions)
theta1 = np.deg2rad(45 * np.sin(0.5 * time))
theta2 = np.deg2rad(30 * np.sin(0.5 * time + np.pi / 4))
theta3 = np.deg2rad(15 * np.sin(0.5 * time + np.pi / 2))
theta4 = np.deg2rad(10 * np.sin(0.5 * time + 3 * np.pi / 4))
theta5 = np.deg2rad(5 * np.sin(0.5 * time + np.pi))

# Compute joint velocities (time derivatives)
theta1_dot = np.gradient(theta1, time)
theta2_dot = np.gradient(theta2, time)
theta3_dot = np.gradient(theta3, time)
theta4_dot = np.gradient(theta4, time)
theta5_dot = np.gradient(theta5, time)

# Compute end effector velocities
v = np.zeros((len(time), 3))
omega = np.zeros((len(time), 3))

for i in range(len(time)):
    thetas = [theta1[i], theta2[i], theta3[i], theta4[i], theta5[i]]
    thetas_dot = [
        theta1_dot[i],
        theta2_dot[i],
        theta3_dot[i],
        theta4_dot[i],
        theta5_dot[i],
    ]
    J = compute_jacobian(thetas)
    end_effector_velocity = J @ thetas_dot
    v[i, :] = end_effector_velocity[:3]
    omega[i, :] = end_effector_velocity[3:]

# Plot linear velocities
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(time, v[:, 0], label="v_x")
plt.plot(time, v[:, 1], label="v_y")
plt.plot(time, v[:, 2], label="v_z")
plt.xlabel("Time [s]")
plt.ylabel("Linear Velocity [m/s]")
plt.title("End Effector Linear Velocities")
plt.legend()
plt.grid(True)

# Plot angular velocities
plt.subplot(2, 1, 2)
plt.plot(time, omega[:, 0], label="ω_x")
plt.plot(time, omega[:, 1], label="ω_y")
plt.plot(time, omega[:, 2], label="ω_z")
plt.xlabel("Time [s]")
plt.ylabel("Angular Velocity [rad/s]")
plt.title("End Effector Angular Velocities")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
