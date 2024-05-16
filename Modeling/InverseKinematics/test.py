import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def dh_transform(theta, d, a, alpha):
    """
    Returns the transformation matrix for the given DH parameters.
    """
    theta = np.radians(theta)
    alpha = np.radians(alpha)
    T = np.array(
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
    return T


# Define DH parameters
params = [
    (0, "l0", 0, 0),
    (0, "l1", 0, 90),
    (0, "l2", 0, 0),
    (0, "l3", 0, 0),
    (0, 0, 0, 90),
    (0, "l4", 0, 0),
]

# Initialize transformation from base to each joint
base_to_joint = np.eye(4)
points = [base_to_joint[:3, 3]]

# Substitute real values for l0 to l5 if known
l0 = 1
l1 = 1
l2 = 1
l3 = 1
l4 = 1


for theta, d, a, alpha in params:
    d = eval(d) if isinstance(d, str) else d
    T = dh_transform(theta, d, a, alpha)
    base_to_joint = np.dot(base_to_joint, T)
    points.append(base_to_joint[:3, 3])

points = np.array(points).T

# Plotting
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.plot(points[0], points[1], points[2], marker="o")

for i, point in enumerate(points.T):
    ax.text(point[0], point[1], point[2], f"J{i}", color="red")

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
plt.title("3D Plot of Robot Configuration")
plt.show()
