import numpy as np
import matplotlib.pyplot as plt
import random
import sympy as sp

# Define DH parameters
d1 = 0.1
d5 = 0.1
a2 = 0.5
a3 = 0.5
alpha = np.deg2rad([90, 0, 0, 90, 0])

# Define the range of joint angles (in radians)
theta_range = np.deg2rad(np.arange(-180, 181, 5))

# Number of random samples to generate
num_samples = 1000


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


# Analyze singularities and workspace using random sampling
def analyze_singularities_workspace(theta_range, num_samples, compute_jacobian_func):
    singularities = []
    workspace_points = []
    are_jacobians_equal = True

    for _ in range(num_samples):
        thetas = [random.choice(theta_range) for _ in range(5)]
        J = compute_jacobian_func(thetas)
        J_screw = compute_jacobian_screw(thetas)

        if not np.allclose(J, J_screw):
            are_jacobians_equal = False

        rank = np.linalg.matrix_rank(J)
        transforms = compute_transforms(thetas)
        end_effector_pos = transforms[-1][:3, 3]
        if rank < 5:
            singularities.append(end_effector_pos)
        else:
            workspace_points.append(end_effector_pos)

    return singularities, workspace_points, are_jacobians_equal


# Compute the transformation matrices for visualization
def compute_transforms(thetas):
    theta1, theta2, theta3, theta4, theta5 = thetas
    A1 = dh_matrix(theta1, d1, 0, alpha[0])
    A2 = dh_matrix(theta2, 0, a2, alpha[1])
    A3 = dh_matrix(theta3, 0, a3, alpha[2])
    A4 = dh_matrix(theta4, 0, 0, alpha[3])
    A5 = dh_matrix(theta5, d5, 0, alpha[4])

    T1 = A1
    T2 = A1 @ A2
    T3 = A1 @ A2 @ A3
    T4 = A1 @ A2 @ A3 @ A4
    T5 = A1 @ A2 @ A3 @ A4 @ A5

    return [np.eye(4), T1, T2, T3, T4, T5]


# Plot the singularity map and workspace
def plot_singularity_workspace_map(singularities, workspace_points, method_name):
    fig = plt.figure(figsize=(18, 6))

    # Define the views
    views = [
        ("Top View", 0, 90),
        ("Front View", 0, 0),
        ("Side View", 90, 0),
    ]

    # Plot each view
    for i, (view_name, elev, azim) in enumerate(views, 1):
        ax = fig.add_subplot(1, 3, i, projection="3d")

        # Plot workspace points
        workspace_points_np = np.array(workspace_points)
        ax.scatter(
            workspace_points_np[:, 0],
            workspace_points_np[:, 1],
            workspace_points_np[:, 2],
            c="b",
            marker="^",
            label="Workspace",
            alpha=0.3,
        )

        # Plot singularities in x, y, z workspace
        singularities_np = np.array(singularities)
        ax.scatter(
            singularities_np[:, 0],
            singularities_np[:, 1],
            singularities_np[:, 2],
            c="r",
            marker="o",
            label="Singularity Point",
        )

        # Set view
        ax.view_init(elev=elev, azim=azim)

        # Add labels and title
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.set_title(f"{view_name} - {method_name}")

        # Add legend
        handles, labels = ax.get_legend_handles_labels()
        unique_labels = dict(zip(labels, handles))
        ax.legend(unique_labels.values(), unique_labels.keys())

    plt.tight_layout()
    plt.show()


# Symbolic Jacobian verification using sympy
# Define symbolic variables for angles and dimensions
theta_1, theta_2, theta_3, theta_4, theta_5 = sp.symbols(
    "theta_1 theta_2 theta_3 theta_4 theta_5"
)
d_1, d_5 = sp.symbols("d_1 d_5")  # d_5 for the last joint
a_2, a_3 = sp.symbols(
    "a_2 a_3"
)  # a_2 and a_3 for the lengths of the second and third links

# Alpha values in degrees, with an updated value for alpha_4
alpha_sym = [90, 0, 0, 90, 0]


# Helper function to create a transformation matrix from DH parameters
def dh_matrix_sym(theta, d, a, alpha):
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
A1_sym = dh_matrix_sym(theta_1, d_1, 0, alpha_sym[0])
A2_sym = dh_matrix_sym(theta_2, 0, a_2, alpha_sym[1])
A3_sym = dh_matrix_sym(theta_3, 0, a_3, alpha_sym[2])
A4_sym = dh_matrix_sym(theta_4, 0, 0, alpha_sym[3])  # a_4 is zero
A5_sym = dh_matrix_sym(theta_5, d_5, 0, alpha_sym[4])  # a_5 is zero, added d_5

# Compute the overall transformation matrix by multiplying individual matrices
T_sym = A1_sym * A2_sym * A3_sym * A4_sym * A5_sym

# Extract the position vector from the transformation matrix
p_sym = T_sym[:3, 3]

# Define the joint variables
joint_vars_sym = [theta_1, theta_2, theta_3, theta_4, theta_5]

# Compute the position Jacobian
J_v_sym = sp.Matrix.hstack(*[sp.diff(p_sym, var) for var in joint_vars_sym])

# Compute the rotation matrices for each joint to find the z-axes in the base frame
R0_sym = sp.eye(3)
R1_sym = A1_sym[:3, :3]
R2_sym = (A1_sym * A2_sym)[:3, :3]
R3_sym = (A1_sym * A2_sym * A3_sym)[:3, :3]
R4_sym = (A1_sym * A2_sym * A3_sym * A4_sym)[:3, :3]

# Axes of rotation for each joint in the base frame
z0_sym = sp.Matrix([0, 0, 1])
z1_sym = R1_sym[:, 2]
z2_sym = R2_sym[:, 2]
z3_sym = R3_sym[:, 2]
z4_sym = R4_sym[:, 2]

# Compute the orientation Jacobian
J_w_sym = sp.Matrix.hstack(z0_sym, z1_sym, z2_sym, z3_sym, z4_sym)

# Combine J_v_sym and J_w_sym to form the full Jacobian
J_sym = sp.Matrix.vstack(J_v_sym, J_w_sym)

# Convert symbolic Jacobian to numerical function
J_screw_func = sp.lambdify(
    (theta_1, theta_2, theta_3, theta_4, theta_5, d_1, d_5, a_2, a_3), J_sym
)


# Function to compute the Jacobian using screw axis representation
def compute_jacobian_screw(thetas):
    theta1, theta2, theta3, theta4, theta5 = thetas
    J_screw = J_screw_func(theta1, theta2, theta3, theta4, theta5, d1, d5, a2, a3)
    return np.array(J_screw).astype(np.float64)


# Analyze the singularities and workspace using traditional Jacobian
(
    singularities_traditional,
    workspace_points_traditional,
    are_jacobians_equal_traditional,
) = analyze_singularities_workspace(theta_range, num_samples, compute_jacobian)

# Analyze the singularities and workspace using screw axis Jacobian
singularities_screw, workspace_points_screw, are_jacobians_equal_screw = (
    analyze_singularities_workspace(theta_range, num_samples, compute_jacobian_screw)
)

# Verify if Jacobians are equal for all samples
are_jacobians_equal = are_jacobians_equal_traditional and are_jacobians_equal_screw

print("\nAre both Jacobians equal for all samples?")
print(are_jacobians_equal)

# Plot the singularity map and workspace for traditional method
plot_singularity_workspace_map(
    singularities_traditional, workspace_points_traditional, "Traditional Method"
)

# Plot the singularity map and workspace for screw axis method
plot_singularity_workspace_map(
    singularities_screw, workspace_points_screw, "Screw Axis Method"
)
