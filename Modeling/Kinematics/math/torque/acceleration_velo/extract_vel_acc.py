import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# Define DH parameters and lengths
d_1 = 0.1
d_5 = 0.1
a_2 = 0.5
a_3 = 0.5


# Helper function to create a transformation matrix from DH parameters
def dh_matrix(theta, d, a, alpha):
    alpha_rad = np.radians(alpha)
    return np.array(
        [
            [
                np.cos(theta),
                -np.sin(theta) * np.cos(alpha_rad),
                np.sin(theta) * np.sin(alpha_rad),
                a * np.cos(theta),
            ],
            [
                np.sin(theta),
                np.cos(theta) * np.cos(alpha_rad),
                -np.cos(theta) * np.sin(alpha_rad),
                a * np.sin(theta),
            ],
            [0, np.sin(alpha_rad), np.cos(alpha_rad), d],
            [0, 0, 0, 1],
        ]
    )


# Calculate transformation matrices for each joint
def forward_kinematics(theta_1, theta_2, theta_3, theta_4, theta_5):
    A1 = dh_matrix(theta_1, d_1, 0, 90)
    A2 = dh_matrix(theta_2, 0, a_2, 0)
    A3 = dh_matrix(theta_3, 0, a_3, 0)
    A4 = dh_matrix(theta_4, 0, 0, 90)
    A5 = dh_matrix(theta_5, d_5, 0, 0)
    T = A1 @ A2 @ A3 @ A4 @ A5
    return T


# Calculate Jacobian matrix for the current joint angles
def jacobian(theta_1, theta_2, theta_3, theta_4, theta_5):
    A1 = dh_matrix(theta_1, d_1, 0, 90)
    A2 = dh_matrix(theta_2, 0, a_2, 0)
    A3 = dh_matrix(theta_3, 0, a_3, 0)
    A4 = dh_matrix(theta_4, 0, 0, 90)
    A5 = dh_matrix(theta_5, d_5, 0, 0)

    T1 = A1
    T2 = T1 @ A2
    T3 = T2 @ A3
    T4 = T3 @ A4
    T5 = T4 @ A5

    O0 = np.array([0, 0, 0])
    O1 = T1[:3, 3]
    O2 = T2[:3, 3]
    O3 = T3[:3, 3]
    O4 = T4[:3, 3]
    O5 = T5[:3, 3]

    Z0 = np.array([0, 0, 1])
    Z1 = T1[:3, 2]
    Z2 = T2[:3, 2]
    Z3 = T3[:3, 2]
    Z4 = T4[:3, 2]

    Jv = np.hstack(
        [
            np.cross(Z0, O5 - O0).reshape(-1, 1),
            np.cross(Z1, O5 - O1).reshape(-1, 1),
            np.cross(Z2, O5 - O2).reshape(-1, 1),
            np.cross(Z3, O5 - O3).reshape(-1, 1),
            np.cross(Z4, O5 - O4).reshape(-1, 1),
        ]
    )

    Jw = np.hstack(
        [
            Z0.reshape(-1, 1),
            Z1.reshape(-1, 1),
            Z2.reshape(-1, 1),
            Z3.reshape(-1, 1),
            Z4.reshape(-1, 1),
        ]
    )

    J = np.vstack([Jv, Jw])
    return J


# Generate a trajectory (example)
num_points = 100
time = np.linspace(0, 10, num_points)
theta_1_traj = np.deg2rad(90 * np.sin(time))
theta_2_traj = np.deg2rad(45 * np.cos(time))
theta_3_traj = np.deg2rad(45 * np.sin(time / 2))
theta_4_traj = np.deg2rad(0 * np.cos(time / 2))
theta_5_traj = np.deg2rad(0 * np.sin(time / 3))

theta_dot_1_traj = np.gradient(theta_1_traj, time)
theta_dot_2_traj = np.gradient(theta_2_traj, time)
theta_dot_3_traj = np.gradient(theta_3_traj, time)
theta_dot_4_traj = np.gradient(theta_4_traj, time)
theta_dot_5_traj = np.gradient(theta_5_traj, time)

theta_ddot_1_traj = np.gradient(theta_dot_1_traj, time)
theta_ddot_2_traj = np.gradient(theta_dot_2_traj, time)
theta_ddot_3_traj = np.gradient(theta_dot_3_traj, time)
theta_ddot_4_traj = np.gradient(theta_dot_4_traj, time)
theta_ddot_5_traj = np.gradient(theta_dot_5_traj, time)

max_linear_velocity = np.zeros(3)
max_angular_velocity = np.zeros(3)
max_linear_acceleration = np.zeros(3)
max_angular_acceleration = np.zeros(3)

traj_data = []

for i in range(num_points):
    joint_vals = [
        theta_1_traj[i],
        theta_2_traj[i],
        theta_3_traj[i],
        theta_4_traj[i],
        theta_5_traj[i],
    ]
    velocity_vals = [
        theta_dot_1_traj[i],
        theta_dot_2_traj[i],
        theta_dot_3_traj[i],
        theta_dot_4_traj[i],
        theta_dot_5_traj[i],
    ]
    acceleration_vals = [
        theta_ddot_1_traj[i],
        theta_ddot_2_traj[i],
        theta_ddot_3_traj[i],
        theta_ddot_4_traj[i],
        theta_ddot_5_traj[i],
    ]

    J = jacobian(*joint_vals)
    if i > 0:
        J_prev = jacobian(
            theta_1_traj[i - 1],
            theta_2_traj[i - 1],
            theta_3_traj[i - 1],
            theta_4_traj[i - 1],
            theta_5_traj[i - 1],
        )
        J_dot = (J - J_prev) / (time[i] - time[i - 1])
    else:
        J_dot = np.zeros_like(J)

    v_numeric = np.dot(J[:3, :], velocity_vals)
    w_numeric = np.dot(J[3:, :], velocity_vals)
    a_numeric = np.dot(J[:3, :], acceleration_vals) + np.dot(
        J_dot[:3, :], velocity_vals
    )
    alpha_numeric = np.dot(J[3:, :], acceleration_vals) + np.dot(
        J_dot[3:, :], velocity_vals
    )

    max_linear_velocity = np.maximum(max_linear_velocity, np.abs(v_numeric))
    max_angular_velocity = np.maximum(max_angular_velocity, np.abs(w_numeric))
    max_linear_acceleration = np.maximum(max_linear_acceleration, np.abs(a_numeric))
    max_angular_acceleration = np.maximum(
        max_angular_acceleration, np.abs(alpha_numeric)
    )

    traj_data.append(np.hstack([v_numeric, w_numeric, a_numeric, alpha_numeric]))

traj_data = np.array(traj_data)

# Save trajectory data to CSV
df_traj = pd.DataFrame(
    traj_data,
    columns=[
        "v_x",
        "v_y",
        "v_z",
        "w_x",
        "w_y",
        "w_z",
        "a_x",
        "a_y",
        "a_z",
        "alpha_x",
        "alpha_y",
        "alpha_z",
    ],
)
df_traj.to_csv("robot_trajectory_data.csv", index=False)

# Save max values to CSV
max_data = {
    "Max Linear Velocity X": [max_linear_velocity[0]],
    "Max Linear Velocity Y": [max_linear_velocity[1]],
    "Max Linear Velocity Z": [max_linear_velocity[2]],
    "Max Angular Velocity X": [max_angular_velocity[0]],
    "Max Angular Velocity Y": [max_angular_velocity[1]],
    "Max Angular Velocity Z": [max_angular_velocity[2]],
    "Max Linear Acceleration X": [max_linear_acceleration[0]],
    "Max Linear Acceleration Y": [max_linear_acceleration[1]],
    "Max Linear Acceleration Z": [max_linear_acceleration[2]],
    "Max Angular Acceleration X": [max_angular_acceleration[0]],
    "Max Angular Acceleration Y": [max_angular_acceleration[1]],
    "Max Angular Acceleration Z": [max_angular_acceleration[2]],
}

df_max = pd.DataFrame(max_data)
df_max.to_csv("robot_max_values.csv", index=False)

print("Max values saved to 'robot_max_values.csv'")
print("Trajectory data saved to 'robot_trajectory_data.csv'")

# Animation
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])
ax.set_zlim([0, 1.5])
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

lines = [ax.plot([], [], [], "o-", markersize=8)[0] for _ in range(num_points)]


def update(num, theta_1_traj, theta_2_traj, theta_3_traj, theta_4_traj, theta_5_traj):
    theta_1 = theta_1_traj[num]
    theta_2 = theta_2_traj[num]
    theta_3 = theta_3_traj[num]
    theta_4 = theta_4_traj[num]
    theta_5 = theta_5_traj[num]

    joint_positions = [
        [0, 0, 0],
        [0, 0, d_1],
        [
            a_2 * np.cos(theta_1) * np.cos(theta_2),
            a_2 * np.sin(theta_1) * np.cos(theta_2),
            d_1 + a_2 * np.sin(theta_2),
        ],
        [
            a_2 * np.cos(theta_1) * np.cos(theta_2)
            + a_3 * np.cos(theta_1) * np.cos(theta_2 + theta_3),
            a_2 * np.sin(theta_1) * np.cos(theta_2)
            + a_3 * np.sin(theta_1) * np.cos(theta_2 + theta_3),
            d_1 + a_2 * np.sin(theta_2) + a_3 * np.sin(theta_2 + theta_3),
        ],
        [
            a_2 * np.cos(theta_1) * np.cos(theta_2)
            + a_3 * np.cos(theta_1) * np.cos(theta_2 + theta_3)
            + d_5 * np.cos(theta_1) * np.cos(theta_2 + theta_3 + theta_4),
            a_2 * np.sin(theta_1) * np.cos(theta_2)
            + a_3 * np.sin(theta_1) * np.cos(theta_2 + theta_3)
            + d_5 * np.sin(theta_1) * np.cos(theta_2 + theta_3 + theta_4),
            d_1
            + a_2 * np.sin(theta_2)
            + a_3 * np.sin(theta_2 + theta_3)
            + d_5 * np.sin(theta_2 + theta_3 + theta_4),
        ],
    ]

    joint_positions = np.array(joint_positions)

    lines[num].set_data(joint_positions[:, 0], joint_positions[:, 1])
    lines[num].set_3d_properties(joint_positions[:, 2])
    return lines


ani = FuncAnimation(
    fig,
    update,
    frames=num_points,
    fargs=(theta_1_traj, theta_2_traj, theta_3_traj, theta_4_traj, theta_5_traj),
    interval=100,
)

plt.show()
