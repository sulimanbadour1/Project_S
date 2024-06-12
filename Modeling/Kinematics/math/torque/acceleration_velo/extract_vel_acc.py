import sympy as sp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Forward Kinematics
# Define symbolic variables for angles and dimensions
theta_1, theta_2, theta_3, theta_4, theta_5 = sp.symbols(
    "theta_1 theta_2 theta_3 theta_4 theta_5"
)
d_1, d_5 = sp.symbols("d_1 d_5")
a_2, a_3 = sp.symbols("a_2 a_3")

alpha = [90, 0, 0, 90, 0]


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


# Create transformation matrices for each joint using the updated parameters
A1 = dh_matrix(theta_1, d_1, 0, alpha[0])
A2 = dh_matrix(theta_2, 0, a_2, alpha[1])
A3 = dh_matrix(theta_3, 0, a_3, alpha[2])
A4 = dh_matrix(theta_4, 0, 0, alpha[3])
A5 = dh_matrix(theta_5, d_5, 0, alpha[4])

# Compute the overall transformation matrix by multiplying individual matrices
T = A1 * A2 * A3 * A4 * A5

# Extract the position and orientation
position = T[:3, 3]
orientation = T[:3, :3]

# Compute the Jacobian for linear velocity
Jv = position.jacobian([theta_1, theta_2, theta_3, theta_4, theta_5])

# The angular part of the Jacobian is given by the z-axis of the previous frames
z0 = sp.Matrix([0, 0, 1])  # z0 axis (base frame)
z1 = A1[:3, :3] * z0
z2 = (A1 * A2)[:3, :3] * z0
z3 = (A1 * A2 * A3)[:3, :3] * z0
z4 = (A1 * A2 * A3 * A4)[:3, :3] * z0

# Assemble the angular velocity Jacobian
Jw = sp.Matrix.hstack(z0, z1, z2, z3, z4)

# Combine Jv and Jw to form the full Jacobian matrix
J_full = sp.Matrix.vstack(Jv, Jw)

# Velocities and Accelerations
# Define symbolic variables for joint velocities and accelerations
theta_dot = sp.symbols("theta_dot_1 theta_dot_2 theta_dot_3 theta_dot_4 theta_dot_5")
theta_ddot = sp.symbols(
    "theta_ddot_1 theta_ddot_2 theta_ddot_3 theta_ddot_4 theta_ddot_5"
)

# Calculate linear and angular velocities
v = Jv * sp.Matrix(theta_dot)
w = Jw * sp.Matrix(theta_dot)

# Calculate linear and angular accelerations
a = Jv * sp.Matrix(theta_ddot)
alpha = Jw * sp.Matrix(theta_ddot)

# Zero matrices for summation
zero_matrix_v = sp.Matrix([0, 0, 0])
zero_matrix_w = sp.Matrix([0, 0, 0, 0, 0])

for theta in [theta_1, theta_2, theta_3, theta_4, theta_5]:
    a += Jv.diff(theta).subs(
        zip([theta_1, theta_2, theta_3, theta_4, theta_5], theta_dot)
    ) * sp.Matrix(theta_dot)
    alpha += Jw.diff(theta).subs(
        zip([theta_1, theta_2, theta_3, theta_4, theta_5], theta_dot)
    ) * sp.Matrix(theta_dot)

# Create numerical functions using lambdify
joint_symbols = [theta_1, theta_2, theta_3, theta_4, theta_5]
velocity_symbols = [
    theta_dot[0],
    theta_dot[1],
    theta_dot[2],
    theta_dot[3],
    theta_dot[4],
]
acceleration_symbols = [
    theta_ddot[0],
    theta_ddot[1],
    theta_ddot[2],
    theta_ddot[3],
    theta_ddot[4],
]

v_func = sp.lambdify(joint_symbols + velocity_symbols, v, "numpy")
w_func = sp.lambdify(joint_symbols + velocity_symbols, w, "numpy")
a_func = sp.lambdify(
    joint_symbols + velocity_symbols + acceleration_symbols, a, "numpy"
)
alpha_func = sp.lambdify(
    joint_symbols + velocity_symbols + acceleration_symbols, alpha, "numpy"
)

# Generate a trajectory (example)
num_points = 100
time = np.linspace(0, 10, num_points)
theta_1_traj = np.deg2rad(30 * np.sin(time))
theta_2_traj = np.deg2rad(45 * np.cos(time))
theta_3_traj = np.deg2rad(60 * np.sin(time / 2))
theta_4_traj = np.deg2rad(75 * np.cos(time / 2))
theta_5_traj = np.deg2rad(90 * np.sin(time / 3))

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

max_total_acceleration = 0
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

    v_numeric = np.array(v_func(*(joint_vals + velocity_vals)))
    w_numeric = np.array(w_func(*(joint_vals + velocity_vals)))
    a_numeric = np.array(a_func(*(joint_vals + velocity_vals + acceleration_vals)))
    alpha_numeric = np.array(
        alpha_func(*(joint_vals + velocity_vals + acceleration_vals))
    )

    total_acceleration = np.linalg.norm(a_numeric) + np.linalg.norm(alpha_numeric)
    if total_acceleration > max_total_acceleration:
        max_total_acceleration = total_acceleration

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

# Maximum values for velocities and accelerations
max_linear_velocity = np.max(np.abs(traj_data[:, :3]))
max_angular_velocity = np.max(np.abs(traj_data[:, 3:6]))
max_linear_acceleration = np.max(np.abs(traj_data[:, 6:9]))
max_angular_acceleration = np.max(np.abs(traj_data[:, 9:]))

# Save max values to CSV
max_data = {
    "Max Linear Velocity": [max_linear_velocity],
    "Max Angular Velocity": [max_angular_velocity],
    "Max Linear Acceleration": [max_linear_acceleration],
    "Max Angular Acceleration": [max_angular_acceleration],
    "Max Total Acceleration": [max_total_acceleration],
}

df_max = pd.DataFrame(max_data)
df_max.to_csv("robot_max_values.csv", index=False)

print("Max values saved to 'robot_max_values.csv'")
print("Trajectory data saved to 'robot_trajectory_data.csv'")

# Plotting
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

# Plot trajectory
ax.plot3D(traj_data[:, 0], traj_data[:, 1], traj_data[:, 2], "gray")

# Plot robot configurations at different time points
time_points = [
    0,
    int(num_points / 4),
    int(num_points / 2),
    int(3 * num_points / 4),
    num_points - 1,
]
for i in time_points:
    joint_positions = np.array(
        [
            [0, 0, 0],
            [0, 0, float(d_1.evalf())],
            [
                float(a_2.evalf() * np.cos(theta_1_traj[i]) * np.cos(theta_2_traj[i])),
                float(a_2.evalf() * np.sin(theta_1_traj[i]) * np.cos(theta_2_traj[i])),
                float(d_1.evalf() + a_2.evalf() * np.sin(theta_2_traj[i])),
            ],
            [
                float(
                    a_2.evalf() * np.cos(theta_1_traj[i]) * np.cos(theta_2_traj[i])
                    + a_3.evalf()
                    * np.cos(theta_1_traj[i])
                    * np.cos(theta_2_traj[i] + theta_3_traj[i])
                ),
                float(
                    a_2.evalf() * np.sin(theta_1_traj[i]) * np.cos(theta_2_traj[i])
                    + a_3.evalf()
                    * np.sin(theta_1_traj[i])
                    * np.cos(theta_2_traj[i] + theta_3_traj[i])
                ),
                float(
                    d_1.evalf()
                    + a_2.evalf() * np.sin(theta_2_traj[i])
                    + a_3.evalf() * np.sin(theta_2_traj[i] + theta_3_traj[i])
                ),
            ],
            [
                float(
                    a_2.evalf() * np.cos(theta_1_traj[i]) * np.cos(theta_2_traj[i])
                    + a_3.evalf()
                    * np.cos(theta_1_traj[i])
                    * np.cos(theta_2_traj[i] + theta_3_traj[i])
                    + d_5.evalf()
                    * np.cos(theta_1_traj[i])
                    * np.cos(theta_2_traj[i] + theta_3_traj[i] + theta_4_traj[i])
                ),
                float(
                    a_2.evalf() * np.sin(theta_1_traj[i]) * np.cos(theta_2_traj[i])
                    + a_3.evalf()
                    * np.sin(theta_1_traj[i])
                    * np.cos(theta_2_traj[i] + theta_3_traj[i])
                    + d_5.evalf()
                    * np.sin(theta_1_traj[i])
                    * np.cos(theta_2_traj[i] + theta_3_traj[i] + theta_4_traj[i])
                ),
                float(
                    d_1.evalf()
                    + a_2.evalf() * np.sin(theta_2_traj[i])
                    + a_3.evalf() * np.sin(theta_2_traj[i] + theta_3_traj[i])
                    + d_5.evalf()
                    * np.sin(theta_2_traj[i] + theta_3_traj[i] + theta_4_traj[i])
                ),
            ],
        ]
    )

    ax.plot3D(
        joint_positions[:, 0],
        joint_positions[:, 1],
        joint_positions[:, 2],
        "o-",
        markersize=8,
        label=f"Time {time[i]:.2f}s",
    )

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("Robot Trajectory and Configurations")
ax.legend()

plt.show()
