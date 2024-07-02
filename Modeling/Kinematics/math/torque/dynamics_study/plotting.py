import sympy as sp
import numpy as np
import matplotlib.pyplot as plt


# Function to create a transformation matrix from DH parameters
def dh_matrix(theta, d, a, alpha):
    alpha_rad = sp.rad(alpha)
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


# Function to compute the dynamic analysis
def compute_dynamic_analysis(
    d_1_val,
    d_5_val,
    a_2_val,
    a_3_val,
    masses,
    inertias,
    angles,
    velocities,
    accelerations,
):
    # Symbolic variables
    theta = sp.symbols("theta_1:6")
    d_1, d_5 = sp.symbols("d_1 d_5")
    a_2, a_3 = sp.symbols("a_2 a_3")
    m = sp.symbols("m1:6")
    I_xx, I_yy, I_zz = sp.symbols("I_xx I_yy I_zz")
    g = sp.Matrix([0, 0, -9.81])

    # DH Parameters
    alpha = [90, 0, 0, 90, 0]

    # Inertia matrices (diagonal for simplicity)
    I = [sp.diag(I_xx, I_yy, I_zz) for _ in range(5)]

    # Transformation matrices
    A = [
        dh_matrix(theta[0], d_1, 0, alpha[0]),
        dh_matrix(theta[1], 0, a_2, alpha[1]),
        dh_matrix(theta[2], 0, a_3, alpha[2]),
        dh_matrix(theta[3], 0, 0, alpha[3]),
        dh_matrix(theta[4], d_5, 0, alpha[4]),
    ]

    # Compute transformation matrices
    T = [A[0]]
    for i in range(1, 5):
        T.append(T[i - 1] * A[i])

    # Positions of centers of mass (assume center for simplicity)
    p = [T[i][:3, 3] / 2 for i in range(5)]

    # Jacobians
    Jv = [p[i].jacobian(theta) for i in range(5)]

    # Angular velocities
    R = [T[i][:3, :3] for i in range(5)]
    omega = [R[i] * sp.Matrix([0, 0, sp.symbols(f"theta_dot_{i+1}")]) for i in range(5)]

    # Kinetic and Potential energies
    K = sum(
        0.5
        * m[i]
        * (Jv[i] * sp.Matrix([sp.symbols(f"theta_dot_{j+1}") for j in range(5)])).dot(
            Jv[i] * sp.Matrix([sp.symbols(f"theta_dot_{j+1}") for j in range(5)])
        )
        + 0.5 * omega[i].dot(I[i] * omega[i])
        for i in range(5)
    )
    P = sum(m[i] * g.dot(p[i]) for i in range(5))

    # Lagrangian
    L = K - P

    # Equations of motion
    tau = sp.Matrix(
        [
            sp.diff(sp.diff(L, sp.symbols(f"theta_dot_{i+1}")), "t")
            - sp.diff(L, theta[i])
            for i in range(5)
        ]
    )

    # Numerical values for testing
    values = {
        d_1: d_1_val,
        d_5: d_5_val,
        a_2: a_2_val,
        a_3: a_3_val,
        I_xx: inertias[0][0],
        I_yy: inertias[0][1],
        I_zz: inertias[0][2],
        m[0]: masses[0],
        m[1]: masses[1],
        m[2]: masses[2],
        m[3]: masses[3],
        m[4]: masses[4],
        theta[0]: angles[0],
        theta[1]: angles[1],
        theta[2]: angles[2],
        theta[3]: angles[3],
        theta[4]: angles[4],
        sp.symbols("theta_dot_1"): velocities[0],
        sp.symbols("theta_dot_2"): velocities[1],
        sp.symbols("theta_dot_3"): velocities[2],
        sp.symbols("theta_dot_4"): velocities[3],
        sp.symbols("theta_dot_5"): velocities[4],
        sp.symbols("theta_ddot_1"): accelerations[0],
        sp.symbols("theta_ddot_2"): accelerations[1],
        sp.symbols("theta_ddot_3"): accelerations[2],
        sp.symbols("theta_ddot_4"): accelerations[3],
        sp.symbols("theta_ddot_5"): accelerations[4],
    }

    # Compute numerical torques
    numerical_torques = tau.subs(values)
    numerical_torques = np.array(numerical_torques).astype(np.float64).flatten()
    print("Numerical Torques: ", numerical_torques)
    return numerical_torques


# Experiment with different values
d_1_val = 0.1
d_5_val = 0.1
a_2_val = 0.5
a_3_val = 0.5
masses = [1.0, 1.0, 1.0, 1.0, 1.0]
inertias = [
    [0.1, 0.1, 0.1],
    [0.1, 0.1, 0.1],
    [0.1, 0.1, 0.1],
    [0.1, 0.1, 0.1],
    [0.1, 0.1, 0.1],
]
angles_set = [
    [0, 0, 0, 0, 0],
    [np.radians(30), np.radians(45), np.radians(60), np.radians(0), np.radians(0)],
]
velocities_set = [[0, 0, 0, 0, 0], [1, 1, 1, 1, 1]]
accelerations_set = [[0, 0, 0, 0, 0], [0.1, 0.1, 0.1, 0.1, 0.1]]

results = []

for angles, velocities, accelerations in zip(
    angles_set, velocities_set, accelerations_set
):
    torques = compute_dynamic_analysis(
        d_1_val,
        d_5_val,
        a_2_val,
        a_3_val,
        masses,
        inertias,
        angles,
        velocities,
        accelerations,
    )
    results.append(torques)

# Plot the results
fig, ax = plt.subplots()
labels = [f"Joint {i+1}" for i in range(5)]
x = np.arange(5)

for i, result in enumerate(results):
    ax.plot(x, result, marker="o", label=f"Set {i+1}")

ax.set_xlabel("Joints")
ax.set_ylabel("Torque")
ax.set_title("Torque due to Dynamics for Different Angle Sets")
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

plt.show()
