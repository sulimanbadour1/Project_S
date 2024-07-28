import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sympy import symbols, cos, sin, pi, Matrix, N

# Define symbolic variables
theta1, theta2, theta3, theta4, theta5 = symbols("theta1 theta2 theta3 theta4 theta5")

# DH Parameters
d1, a1, alpha1 = 0.1, 0, pi / 2
d2, a2, alpha2 = 0, 0.5, 0
d3, a3, alpha3 = 0, 0.5, 0
d4, a4, alpha4 = 0, 0, pi / 2
d5, a5, alpha5 = 0.1, 0, 0


# Define the transformation matrix function using DH parameters
def DH_matrix(theta, d, a, alpha):
    return Matrix(
        [
            [
                cos(theta),
                -sin(theta) * cos(alpha),
                sin(theta) * sin(alpha),
                a * cos(theta),
            ],
            [
                sin(theta),
                cos(theta) * cos(alpha),
                -cos(theta) * sin(alpha),
                a * sin(theta),
            ],
            [0, sin(alpha), cos(alpha), d],
            [0, 0, 0, 1],
        ]
    )


# Define two sets of joint angles in radians
angles_max_static_torque = {
    theta1: -3.14,
    theta2: -3.14,
    theta3: -0.35,
    theta4: 1.75,
    theta5: -3.14,
}
angles_max_dynamic_torque = {
    theta1: 0.3491,
    theta2: 3.1416,
    theta3: 0.3491,
    theta4: 1.0472,
    theta5: -1.7453,
}


# Function to compute positions for a given set of joint angles
def compute_positions(angles):
    T1 = DH_matrix(theta1, d1, a1, alpha1).subs(angles)
    T2 = DH_matrix(theta2, d2, a2, alpha2).subs(angles)
    T3 = DH_matrix(theta3, d3, a3, alpha3).subs(angles)
    T4 = DH_matrix(theta4, d4, a4, alpha4).subs(angles)
    T5 = DH_matrix(theta5, d5, a5, alpha5).subs(angles)

    # Calculate cumulative transformations
    T01 = T1
    T02 = T01 * T2
    T03 = T02 * T3
    T04 = T03 * T4
    T05 = T04 * T5

    # Extract positions
    positions = [
        Matrix([0, 0, 0, 1]),  # Base
        T01[:3, 3],
        T02[:3, 3],
        T03[:3, 3],
        T04[:3, 3],
        T05[:3, 3],  # End effector
    ]
    return [N(p) for p in positions]  # Evaluate numerically


# Compute positions for both configurations
positions_static = compute_positions(angles_max_static_torque)
positions_dynamic = compute_positions(angles_max_dynamic_torque)

# Print the end effector positions
end_effector_position_static = positions_static[-1]
end_effector_position_dynamic = positions_dynamic[-1]
print(f"End Effector Position (Static): {end_effector_position_static}")
print(f"End Effector Position (Dynamic): {end_effector_position_dynamic}")

# Plotting
fig = plt.figure(figsize=(20, 10))

# Subplot for Max Static Torque Config
ax_static = fig.add_subplot(121, projection="3d")
x_static = [p[0] for p in positions_static]
y_static = [p[1] for p in positions_static]
z_static = [p[2] for p in positions_static]
ax_static.plot(
    x_static, y_static, z_static, "bo-", label="Max Static Torque Config", linewidth=2
)
ax_static.plot(x_static[0], y_static[0], z_static[0], "go")  # Base in green
ax_static.plot(
    x_static[-1], y_static[-1], z_static[-1], "mo"
)  # End effector in magenta

# Labeling the base and end effector
ax_static.text(
    x_static[0],
    y_static[0],
    z_static[0],
    "Base",
    color="green",
    fontsize=10,
    fontweight="bold",
)
ax_static.text(
    x_static[-1],
    y_static[-1],
    z_static[-1],
    "End Effector",
    color="magenta",
    fontsize=10,
    fontweight="bold",
)

# Enhancing the plot
ax_static.set_title(
    "3D View of Max Static Torque Config", fontsize=16, fontweight="bold"
)
ax_static.set_xlabel("X (meters)", fontsize=14, fontweight="bold")
ax_static.set_ylabel("Y (meters)", fontsize=14, fontweight="bold")
ax_static.set_zlabel("Z (meters)", fontsize=14, fontweight="bold")
ax_static.legend(fontsize=12)
ax_static.grid(True)

# Adjusting the viewing angle for better visualization
ax_static.view_init(elev=20, azim=-45)

# Subplot for Max Dynamic Torque Config
ax_dynamic = fig.add_subplot(122, projection="3d")
x_dynamic = [p[0] for p in positions_dynamic]
y_dynamic = [p[1] for p in positions_dynamic]
z_dynamic = [p[2] for p in positions_dynamic]
ax_dynamic.plot(
    x_dynamic,
    y_dynamic,
    z_dynamic,
    "ro-",
    label="Max Dynamic Torque Config",
    linewidth=2,
)
ax_dynamic.plot(x_dynamic[0], y_dynamic[0], z_dynamic[0], "go")  # Base in green
ax_dynamic.plot(
    x_dynamic[-1], y_dynamic[-1], z_dynamic[-1], "mo"
)  # End effector in magenta

# Labeling the base and end effector
ax_dynamic.text(
    x_dynamic[0],
    y_dynamic[0],
    z_dynamic[0],
    "Base",
    color="green",
    fontsize=10,
    fontweight="bold",
)
ax_dynamic.text(
    x_dynamic[-1],
    y_dynamic[-1],
    z_dynamic[-1],
    "End Effector",
    color="magenta",
    fontsize=10,
    fontweight="bold",
)

# Enhancing the plot
ax_dynamic.set_title(
    "3D View of Max Dynamic Torque Config", fontsize=16, fontweight="bold"
)
ax_dynamic.set_xlabel("X (meters)", fontsize=14, fontweight="bold")
ax_dynamic.set_ylabel("Y (meters)", fontsize=14, fontweight="bold")
ax_dynamic.set_zlabel("Z (meters)", fontsize=14, fontweight="bold")
ax_dynamic.legend(fontsize=12)
ax_dynamic.grid(True)

# Adjusting the viewing angle for better visualization
ax_dynamic.view_init(elev=20, azim=-45)

# Combined plot
fig_combined = plt.figure(figsize=(10, 8))
ax_combined = fig_combined.add_subplot(111, projection="3d")

# Max Static Torque Config
ax_combined.plot(
    x_static, y_static, z_static, "bo-", label="Max Static Torque Config", linewidth=2
)
ax_combined.plot(x_static[0], y_static[0], z_static[0], "go")  # Base in green
ax_combined.plot(
    x_static[-1], y_static[-1], z_static[-1], "mo"
)  # End effector in magenta

# Max Dynamic Torque Config
ax_combined.plot(
    x_dynamic,
    y_dynamic,
    z_dynamic,
    "ro-",
    label="Max Dynamic Torque Config",
    linewidth=2,
)
ax_combined.plot(x_dynamic[0], y_dynamic[0], z_dynamic[0], "go")  # Base in green
ax_combined.plot(
    x_dynamic[-1], y_dynamic[-1], z_dynamic[-1], "mo"
)  # End effector in magenta

# Labeling the base and end effector
ax_combined.text(
    x_static[0],
    y_static[0],
    z_static[0],
    "Base (Static)",
    color="green",
    fontsize=10,
    fontweight="bold",
)
ax_combined.text(
    x_static[-1],
    y_static[-1],
    z_static[-1],
    "End Effector (Static)",
    color="magenta",
    fontsize=10,
    fontweight="bold",
)
ax_combined.text(
    x_dynamic[0],
    y_dynamic[0],
    z_dynamic[0],
    "Base (Dynamic)",
    color="green",
    fontsize=10,
    fontweight="bold",
)
ax_combined.text(
    x_dynamic[-1],
    y_dynamic[-1],
    z_dynamic[-1],
    "End Effector (Dynamic)",
    color="magenta",
    fontsize=10,
    fontweight="bold",
)

# Enhancing the plot
ax_combined.set_title("3D View of Robot Configurations", fontsize=16, fontweight="bold")
ax_combined.set_xlabel("X (meters)", fontsize=14, fontweight="bold")
ax_combined.set_ylabel("Y (meters)", fontsize=14, fontweight="bold")
ax_combined.set_zlabel("Z (meters)", fontsize=14, fontweight="bold")
ax_combined.legend(fontsize=12)
ax_combined.grid(True)

# Adjusting the viewing angle for better visualization
ax_combined.view_init(elev=20, azim=-45)

# Show the plots
plt.show()
