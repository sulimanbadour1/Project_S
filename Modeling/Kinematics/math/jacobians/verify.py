import sympy as sp

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
T = sp.eye(4)
for i in range(5):
    T = T * A[i]

# Position of the end-effector
p = T[:3, 3]

# Compute the linear velocity Jacobian Jv
Jv = p.jacobian(theta)

# Compute the angular velocity Jacobian Jw
R = [A[i][:3, :3] for i in range(5)]
z = [sp.Matrix([0, 0, 1])]  # Initial z vector (z0)
o = [sp.Matrix([0, 0, 0])]  # Initial origin vector (o0)

for i in range(1, 5):
    z.append(R[i - 1] * z[i - 1])
    o.append(T[:3, 3])

# Angular velocity Jacobian Jw
Jw = sp.Matrix.hstack(z[0], z[1], z[2], z[3], z[4])

# Initialize pretty printing for better output readability
sp.init_printing(use_unicode=True)

# Print the Jacobians
print("Linear velocity Jacobian (Jv):")
sp.pprint(Jv)

print("Angular velocity Jacobian (Jw):")
sp.pprint(Jw)
