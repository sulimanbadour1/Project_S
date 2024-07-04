import numpy as np
from sympy import symbols, cos, sin, pi, Matrix, N
import matplotlib.pyplot as plt
import math

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


# Joint angles in degrees
theta_in_degrees = {
    "theta1": 90,
    "theta2": 10,
    "theta3": 20,
    "theta4": 0,
    "theta5": 0,
}

# Convert angles to radians
angles = {key: math.radians(value) for key, value in theta_in_degrees.items()}

# Compute transformation matrices
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
positions = [N(p) for p in positions]  # Evaluate numerically

# Calculate omega (pitch angle) from the final orientation matrix
R05 = T05[:3, :3]
omega = math.degrees(math.atan2(R05[2, 0], R05[2, 2]))

# Plotting
fig = plt.figure()
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122, projection="3d")

# 2D plot
x_vals = [p[0] for p in positions]
y_vals = [p[1] for p in positions]
ax1.plot(x_vals, y_vals, "bo-")
ax1.plot(x_vals[0], y_vals[0], "go")  # Base in green
ax1.plot(x_vals[-1], y_vals[-1], "mo")  # End effector in magenta
ax1.set_title("2D View")
ax1.set_xlabel("X")
ax1.set_ylabel("Y")
ax1.legend(["Path", "Base", "End Effector"])

# 3D plot
x_vals = [p[0] for p in positions]
y_vals = [p[1] for p in positions]
z_vals = [p[2] for p in positions]
ax2.plot(x_vals, y_vals, z_vals, "ro-")
ax2.plot(x_vals[0], y_vals[0], z_vals[0], "go")  # Base in green
ax2.plot(x_vals[-1], y_vals[-1], z_vals[-1], "mo")  # End effector in magenta

# Annotate omega
ax2.text(x_vals[-1], y_vals[-1], z_vals[-1], f"ω = {omega:.2f}°", color="blue")

ax2.set_title("3D View")
ax2.set_xlabel("X")
ax2.set_ylabel("Y")
ax2.set_zlabel("Z")
ax2.legend(["Path", "Base", "End Effector"])

print(f"End effector position: {positions[-1]}")
print("\n")
# print(f"End effector orientation: {T05[:3, :3]}")
print(f"Pitch angle omega: {omega} degrees")

plt.show()
