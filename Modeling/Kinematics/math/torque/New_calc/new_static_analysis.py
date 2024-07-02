# In[1]: Define DH Parameters and symbolic variables
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define symbolic variables for angles and dimensions
theta_1, theta_2, theta_3, theta_4, theta_5 = sp.symbols(
    "theta_1 theta_2 theta_3 theta_4 theta_5"
)
d_1, d_5 = sp.symbols("d_1 d_5")
a_2, a_3 = sp.symbols("a_2 a_3")

# Define DH parameters
theta = [theta_1, theta_2, theta_3, theta_4, theta_5]
d = [d_1, 0, 0, 0, d_5]
a = [0, a_2, a_3, 0, 0]
alpha = [sp.pi / 2, 0, 0, sp.pi / 2, 0]


# Helper function to create a transformation matrix from DH parameters
def dh_matrix(theta, d, a, alpha):
    return sp.Matrix(
        [
            [
                sp.cos(theta),
                -sp.sin(theta) * sp.cos(alpha),
                sp.sin(theta) * sp.sin(alpha),
                a * sp.cos(theta),
            ],
            [
                sp.sin(theta),
                sp.cos(theta) * sp.cos(alpha),
                -sp.cos(theta) * sp.sin(alpha),
                a * sp.sin(theta),
            ],
            [0, sp.sin(alpha), sp.cos(alpha), d],
            [0, 0, 0, 1],
        ]
    )


# Create transformation matrices for each joint
A = [dh_matrix(theta[i], d[i], a[i], alpha[i]) for i in range(5)]

# Compute the overall transformation matrix
T = [A[0]]
for i in range(1, 5):
    T.append(T[i - 1] * A[i])

# In[2]: Define Link Masses and Center of Masses
masses = [1.0, 1.0, 1.0, 1.0, 1.0]  # Masses of the links
com_positions = [
    sp.Matrix([0, 0, d_1 / 2]),  # Center of mass positions in link frames
    sp.Matrix([a_2 / 2, 0, 0]),
    sp.Matrix([a_3 / 2, 0, 0]),
    sp.Matrix([0, 0, 0]),
    sp.Matrix([0, 0, d_5 / 2]),
]

g = sp.Matrix([0, 0, -9.81])  # Gravitational acceleration

# In[3]: Compute Center of Mass Positions in the Base Frame
com_base_positions = []
for i in range(5):
    com_base_positions.append(T[i][:3, :3] * com_positions[i] + T[i][:3, 3])

# In[4]: Compute the Gravitational Forces on Each Link
gravitational_forces = []
for i in range(5):
    gravitational_forces.append(masses[i] * g)

# In[5]: Calculate Torques at Each Joint
torques = sp.Matrix([0, 0, 0, 0, 0])
for i in range(5):
    r = com_base_positions[i]
    F = gravitational_forces[i]
    for j in range(i + 1):
        torque = r.cross(F)
        torques[j] += torque.dot(T[j][:3, 2])  # Project torque onto joint axis

# Numerical values for the symbolic variables
numerical_values = {
    theta_1: np.deg2rad(30),
    theta_2: np.deg2rad(45),
    theta_3: np.deg2rad(60),
    theta_4: np.deg2rad(90),
    theta_5: np.deg2rad(0),
    d_1: 0.1,
    d_5: 0.1,
    a_2: 0.5,
    a_3: 0.5,
}

# Substitute numerical values into the torque equations
torques_num = torques.subs(numerical_values).evalf()

# Print the torques
print("Torques required at each joint to maintain equilibrium under gravity (in Nm):")
sp.pprint(torques_num)

# In[6]: Plot the torques
torques_np = np.array(torques_num).astype(np.float64).flatten()
joint_indices = np.arange(1, 6)

plt.figure(figsize=(10, 6))
plt.bar(joint_indices, torques_np, color="blue")
plt.xlabel("Joint Index")
plt.ylabel("Torque (Nm)")
plt.title("Torques Required at Each Joint to Maintain Equilibrium Under Gravity")
plt.xticks(joint_indices, [f"Joint {i}" for i in joint_indices])
plt.grid(True)
plt.show()


# In[7]: Plot the Robot Configuration
def plot_robot(joint_positions):
    # Unpack joint positions
    x, y, z = zip(*joint_positions)

    # Plot the robot
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(x, y, z, "o-", markersize=10, label="Robot Arm")
    ax.scatter(x, y, z, c="k")

    # Add text labels for the links and lengths
    for i in range(len(joint_positions) - 1):
        ax.text(
            (x[i] + x[i + 1]) / 2,
            (y[i] + y[i + 1]) / 2,
            (z[i] + z[i + 1]) / 2,
            f"Link {i + 1}",
            color="black",
        )

    # Set labels and title
    ax.set_xlabel("X axis")
    ax.set_ylabel("Y axis")
    ax.set_zlabel("Z axis")
    ax.set_title("3D Robot Configuration")
    ax.legend()
    ax.set_box_aspect([1, 1, 1])
    plt.show()


# Calculate joint positions for plotting
def calculate_joint_positions():
    joint_positions = []
    current_position = sp.Matrix([0, 0, 0, 1])
    joint_positions.append(current_position[:3])

    for i in range(5):
        current_position = T[i] * sp.Matrix([0, 0, 0, 1])
        joint_positions.append(current_position[:3])

    return joint_positions


joint_positions_sym = calculate_joint_positions()
joint_positions_num = [
    np.array(jp.subs(numerical_values)).astype(np.float64).flatten()
    for jp in joint_positions_sym
]

plot_robot(joint_positions_num)
