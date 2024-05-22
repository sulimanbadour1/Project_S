import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

# Define DH parameters and masses
d_1_val = 0.1
d_5_val = 0.1
a_2_val = 0.5
a_3_val = 0.5
masses = [
    1.0,
    1.0,
    1.0,
    1.0,
    2.0,
]  # Including the additional mass for the camera and lights


# Helper function to create a transformation matrix from DH parameters
def dh_matrix(theta, d, a, alpha):
    alpha_rad = np.deg2rad(alpha)
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


# Function to compute the transformation matrices for given joint angles
def compute_transformation_matrices(theta_1, theta_2, theta_3, theta_4, theta_5):
    alpha = [90, 0, 0, 90, 0]
    A1 = dh_matrix(theta_1, d_1_val, 0, alpha[0])
    A2 = dh_matrix(theta_2, 0, a_2_val, alpha[1])
    A3 = dh_matrix(theta_3, 0, a_3_val, alpha[2])
    A4 = dh_matrix(theta_4, 0, 0, alpha[3])
    A5 = dh_matrix(theta_5, d_5_val, 0, alpha[4])

    T1 = A1
    T2 = T1 @ A2
    T3 = T2 @ A3
    T4 = T3 @ A4
    T5 = T4 @ A5

    return [T1, T2, T3, T4, T5]


# Placeholder function for computing torques (replace this with your actual torque computation logic)
def compute_torque(theta_1, theta_2, theta_3, theta_4, theta_5):
    # Dummy computation for torque (replace this with actual torque calculation logic)
    return [theta_1, theta_2, theta_3, theta_4, theta_5]


# Define the ranges for each joint
theta_1_range = np.deg2rad(np.linspace(0, 360, 7))
theta_2_range = np.deg2rad(np.linspace(-90, 90, 7))
theta_3_range = np.deg2rad(np.linspace(-90, 90, 7))
theta_4_range = np.deg2rad(np.linspace(-90, 90, 7))
theta_5_range = np.deg2rad(np.linspace(0, 360, 7))

# Create meshgrid for joint angles
theta_1_vals, theta_2_vals, theta_3_vals, theta_4_vals, theta_5_vals = np.meshgrid(
    theta_1_range,
    theta_2_range,
    theta_3_range,
    theta_4_range,
    theta_5_range,
    indexing="ij",
)

# Flatten the meshgrid arrays for easier iteration
theta_1_vals = theta_1_vals.flatten()
theta_2_vals = theta_2_vals.flatten()
theta_3_vals = theta_3_vals.flatten()
theta_4_vals = theta_4_vals.flatten()
theta_5_vals = theta_5_vals.flatten()

# Dictionary to store maximum and minimum torques for each joint
max_torques = {i: -np.inf for i in range(1, 6)}  # Initialize with negative infinity
min_torques = {i: np.inf for i in range(1, 6)}  # Initialize with positive infinity
max_torque_angles = {i: None for i in range(1, 6)}  # To store angles for max torques
min_torque_angles = {i: None for i in range(1, 6)}  # To store angles for min torques

# Compute torques for all combinations of joint angles
for theta_1_val, theta_2_val, theta_3_val, theta_4_val, theta_5_val in zip(
    theta_1_vals, theta_2_vals, theta_3_vals, theta_4_vals, theta_5_vals
):
    angles = [theta_1_val, theta_2_val, theta_3_val, theta_4_val, theta_5_val]
    numerical_torques = compute_torque(*angles)

    for i in range(5):
        if numerical_torques[i] > max_torques[i + 1]:
            max_torques[i + 1] = numerical_torques[i]
            max_torque_angles[i + 1] = angles
        if numerical_torques[i] < min_torques[i + 1]:
            min_torques[i + 1] = numerical_torques[i]
            min_torque_angles[i + 1] = angles


# Convert angles from radians to degrees for animation
def rad2deg(angle_list):
    return [np.rad2deg(angle) for angle in angle_list]


max_torque_angles_deg = rad2deg(max_torque_angles[1])
min_torque_angles_deg = rad2deg(min_torque_angles[1])

# Print maximum and minimum configurations
print("Maximum Torque Configuration (degrees):", max_torque_angles_deg)
print("Minimum Torque Configuration (degrees):", min_torque_angles_deg)


# Function to animate the robot movement for maximum and minimum torque configurations
def animate_robot_movement(start_angles, max_angles, min_angles):
    fig, axs = plt.subplots(1, 2, figsize=(15, 8), subplot_kw={"projection": "3d"})
    titles = ["Maximum Torque Configuration", "Minimum Torque Configuration"]

    # Create a list to store text annotations
    texts = [[], []]

    for ax, end_angles, title, text_list in zip(
        axs, [max_angles, min_angles], titles, texts
    ):
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([0, 1.5])
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title(title)
        text_list.append(ax.text2D(0.05, 0.95, "", transform=ax.transAxes))

    # Define the number of frames for the animation
    frames = 100

    # Create a line object for each link in both subplots
    lines = [[ax.plot([], [], [], marker="o")[0] for _ in range(5)] for ax in axs]

    # Initialize function for the animation
    def init():
        for line_set in lines:
            for line in line_set:
                line.set_data([], [])
                line.set_3d_properties([])
        for text_list in texts:
            for text in text_list:
                text.set_text("")
        return [line for line_set in lines for line in line_set] + [
            text for text_list in texts for text in text_list
        ]

    # Update function for the animation
    def update(frame):
        t = frame / (frames - 1)
        for ax, end_angles, line_set, text_list in zip(
            axs, [max_angles, min_angles], lines, texts
        ):
            current_angles = [
                (1 - t) * start_angle + t * end_angle
                for start_angle, end_angle in zip(start_angles, end_angles)
            ]
            Ts = compute_transformation_matrices(*np.deg2rad(current_angles))

            points = [[0, 0, 0]] + [T[:3, 3] for T in Ts]
            xs, ys, zs = zip(*points)

            for i, line in enumerate(line_set):
                line.set_data(xs[i : i + 2], ys[i : i + 2])
                line.set_3d_properties(zs[i : i + 2])

            # Update the text annotation with current torque values
            text_list[0].set_text(f"Torques: {np.round(current_angles, 2)}")

        return [line for line_set in lines for line in line_set] + [
            text for text_list in texts for text in text_list
        ]

    ani = animation.FuncAnimation(fig, update, frames=frames, init_func=init, blit=True)

    plt.show()


# Maximum and minimum torque angles
start_angles_deg = [0, 0, 0, 0, 0]

# Animate robot movement for both maximum and minimum torque configurations
animate_robot_movement(start_angles_deg, max_torque_angles_deg, min_torque_angles_deg)
