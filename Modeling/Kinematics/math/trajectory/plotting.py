import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

# Define symbolic variables for time and joint angles as functions of time
t = sp.symbols("t")
theta_1 = sp.Function("theta_1")(t)
theta_2 = sp.Function("theta_2")(t)
theta_3 = sp.Function("theta_3")(t)
theta_4 = sp.Function("theta_4")(t)
theta_5 = sp.Function("theta_5")(t)

# Define other DH parameters
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
A4 = dh_matrix(theta_4, 0, 0, alpha[3])  # a_4 is zero
A5 = dh_matrix(theta_5, d_5, 0, alpha[4])  # a_5 is zero, added d_5

# Compute the overall transformation matrix by multiplying individual matrices
T = A1 * A2 * A3 * A4 * A5

# Extract the position of the end effector
position = T[:3, 3]

# Compute the time derivatives of the position to get the velocity
velocity = sp.diff(position, t)

# Compute the second derivatives of the position to get the acceleration
acceleration = sp.diff(velocity, t)

# Initialize pretty printing for better output readability
sp.init_printing(use_unicode=True)

# Print the position, velocity, and acceleration equations
print("Position of the end effector as a function of time:")
sp.pprint(position)

print("\nVelocity of the end effector as a function of time:")
sp.pprint(velocity)

print("\nAcceleration of the end effector as a function of time:")
sp.pprint(acceleration)

# Define numerical values for DH parameters
numerical_values = {
    d_1: 0.1,
    d_5: 0.1,
    a_2: 0.5,
    a_3: 0.5,
}

# Define joint angles as functions of time (example: simple linear functions)
theta_1_func = sp.lambdify(
    t, 30 * t * sp.pi / 180
)  # theta_1(t) = 30t degrees to radians
theta_2_func = sp.lambdify(
    t, 45 * t * sp.pi / 180
)  # theta_2(t) = 45t degrees to radians
theta_3_func = sp.lambdify(
    t, 60 * t * sp.pi / 180
)  # theta_3(t) = 60t degrees to radians
theta_4_func = sp.lambdify(
    t, 90 * t * sp.pi / 180
)  # theta_4(t) = 90t degrees to radians
theta_5_func = sp.lambdify(t, 0 * t * sp.pi / 180)  # theta_5(t) = 0t degrees to radians

# Select a specific time to evaluate
time_value = 1  # Example: t = 1 second

# Compute numerical values for the joint angles at the selected time
numerical_theta_values = {
    theta_1: theta_1_func(time_value),
    theta_2: theta_2_func(time_value),
    theta_3: theta_3_func(time_value),
    theta_4: theta_4_func(time_value),
    theta_5: theta_5_func(time_value),
}

# Substitute numerical values into the position, velocity, and acceleration equations
numerical_position = position.subs(numerical_values).subs(numerical_theta_values)
numerical_velocity = velocity.subs(numerical_values).subs(numerical_theta_values)
numerical_acceleration = acceleration.subs(numerical_values).subs(
    numerical_theta_values
)

# Evaluate the numerical position, velocity, and acceleration
numerical_position = numerical_position.evalf()
numerical_velocity = numerical_velocity.evalf()
numerical_acceleration = numerical_acceleration.evalf()

# Print numerical results
print("\nNumerical position of the end effector at t = 1 second:")
sp.pprint(numerical_position)

print("\nNumerical velocity of the end effector at t = 1 second:")
sp.pprint(numerical_velocity)

print("\nNumerical acceleration of the end effector at t = 1 second:")
sp.pprint(numerical_acceleration)

# Define the range of time values for plotting
time_values = np.linspace(0, 2, 100)  # from 0 to 2 seconds

# Compute the numerical values for joint angles over the time range
theta_1_values = np.array([theta_1_func(t) for t in time_values])
theta_2_values = np.array([theta_2_func(t) for t in time_values])
theta_3_values = np.array([theta_3_func(t) for t in time_values])
theta_4_values = np.array([theta_4_func(t) for t in time_values])
theta_5_values = np.array([theta_5_func(t) for t in time_values])

# Compute numerical position over the time range
position_values = np.array(
    [
        position.subs(numerical_values)
        .subs(
            {
                theta_1: theta_1_func(t),
                theta_2: theta_2_func(t),
                theta_3: theta_3_func(t),
                theta_4: theta_4_func(t),
                theta_5: theta_5_func(t),
            }
        )
        .evalf()
        for t in time_values
    ],
    dtype=float,
)

# Compute numerical velocity over the time range using numerical differentiation
dt = time_values[1] - time_values[0]
velocity_values = np.gradient(position_values, dt, axis=0)

# Compute numerical acceleration over the time range using numerical differentiation
acceleration_values = np.gradient(velocity_values, dt, axis=0)

# Extract x, y, z components for position, velocity, and acceleration
position_x = position_values[:, 0]
position_y = position_values[:, 1]
position_z = position_values[:, 2]

velocity_x = velocity_values[:, 0]
velocity_y = velocity_values[:, 1]
velocity_z = velocity_values[:, 2]

acceleration_x = acceleration_values[:, 0]
acceleration_y = acceleration_values[:, 1]
acceleration_z = acceleration_values[:, 2]

# Plot the position of the end effector
plt.figure()
plt.plot(time_values, position_x, label="x")
plt.plot(time_values, position_y, label="y")
plt.plot(time_values, position_z, label="z")
plt.title("Position of the End Effector Over Time")
plt.xlabel("Time (s)")
plt.ylabel("Position (m)")
plt.legend()
plt.grid(True)
plt.show()

# Plot the velocity of the end effector
plt.figure()
plt.plot(time_values, velocity_x, label="x")
plt.plot(time_values, velocity_y, label="y")
plt.plot(time_values, velocity_z, label="z")
plt.title("Velocity of the End Effector Over Time")
plt.xlabel("Time (s)")
plt.ylabel("Velocity (m/s)")
plt.legend()
plt.grid(True)
plt.show()

# Plot the acceleration of the end effector
plt.figure()
plt.plot(time_values, acceleration_x, label="x")
plt.plot(time_values, acceleration_y, label="y")
plt.plot(time_values, acceleration_z, label="z")
plt.title("Acceleration of the End Effector Over Time")
plt.xlabel("Time (s)")
plt.ylabel("Acceleration (m/s^2)")
plt.legend()
plt.grid(True)
plt.show()
