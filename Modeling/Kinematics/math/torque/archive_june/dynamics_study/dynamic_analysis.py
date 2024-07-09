import sympy as sp


# Define the computation function for dynamic analysis
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
    # Define symbolic variables for joint angles, velocities, accelerations, DH parameters, masses, and inertia
    theta_1, theta_2, theta_3, theta_4, theta_5 = sp.symbols(
        "theta_1 theta_2 theta_3 theta_4 theta_5"
    )
    d_1, d_5 = sp.symbols("d_1 d_5")
    a_2, a_3 = sp.symbols("a_2 a_3")
    alpha = [90, 0, 0, 90, 0]
    m1, m2, m3, m4, m5 = sp.symbols("m1 m2 m3 m4 m5")
    g = sp.Matrix([0, 0, -9.81])

    theta = sp.Matrix([theta_1, theta_2, theta_3, theta_4, theta_5])
    theta_dot = sp.Matrix([sp.symbols(f"theta_dot_{i+1}") for i in range(5)])
    theta_ddot = sp.Matrix([sp.symbols(f"theta_ddot_{i+1}") for i in range(5)])

    # Define inertia matrices (assuming simple diagonal form for simplicity)
    I1_xx, I1_yy, I1_zz = sp.symbols("I1_xx I1_yy I1_zz")
    I2_xx, I2_yy, I2_zz = sp.symbols("I2_xx I2_yy I2_zz")
    I3_xx, I3_yy, I3_zz = sp.symbols("I3_xx I3_yy I3_zz")
    I4_xx, I4_yy, I4_zz = sp.symbols("I4_xx I4_yy I4_zz")
    I5_xx, I5_yy, I5_zz = sp.symbols("I5_xx I5_yy I5_zz")

    I1 = sp.diag(I1_xx, I1_yy, I1_zz)
    I2 = sp.diag(I2_xx, I2_yy, I2_zz)
    I3 = sp.diag(I3_xx, I3_yy, I3_zz)
    I4 = sp.diag(I4_xx, I4_yy, I4_zz)
    I5 = sp.diag(I5_xx, I5_yy, I5_zz)

    # Helper function to create a transformation matrix from DH parameters
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

    # Create transformation matrices
    A1 = dh_matrix(theta_1, d_1, 0, alpha[0])
    A2 = dh_matrix(theta_2, 0, a_2, alpha[1])
    A3 = dh_matrix(theta_3, 0, a_3, alpha[2])
    A4 = dh_matrix(theta_4, 0, 0, alpha[3])
    A5 = dh_matrix(theta_5, d_5, 0, alpha[4])

    # Compute the individual transformation matrices
    T1 = A1
    T2 = T1 * A2
    T3 = T2 * A3
    T4 = T3 * A4
    T5 = T4 * A5

    # Extract positions of each link's center of mass
    # Assume center of mass at the middle of each link for simplicity
    p1 = T1[:3, 3] / 2
    p2 = T2[:3, 3] / 2
    p3 = T3[:3, 3] / 2
    p4 = T4[:3, 3] / 2
    p5 = T5[:3, 3] / 2

    # Compute the Jacobians for each center of mass
    Jv1 = p1.jacobian(theta)
    Jv2 = p2.jacobian(theta)
    Jv3 = p3.jacobian(theta)
    Jv4 = p4.jacobian(theta)
    Jv5 = p5.jacobian(theta)

    # Compute angular velocities
    R1 = T1[:3, :3]
    R2 = T2[:3, :3]
    R3 = T3[:3, :3]
    R4 = T4[:3, :3]
    R5 = T5[:3, :3]

    omega1 = R1 * sp.Matrix([0, 0, theta_dot[0]])
    omega2 = R2 * sp.Matrix([0, 0, theta_dot[1]])
    omega3 = R3 * sp.Matrix([0, 0, theta_dot[2]])
    omega4 = R4 * sp.Matrix([0, 0, theta_dot[3]])
    omega5 = R5 * sp.Matrix([0, 0, theta_dot[4]])

    # Compute the kinetic energy of each link
    K1 = 0.5 * m1 * (Jv1 * theta_dot).dot(Jv1 * theta_dot) + 0.5 * omega1.dot(
        I1 * omega1
    )
    K2 = 0.5 * m2 * (Jv2 * theta_dot).dot(Jv2 * theta_dot) + 0.5 * omega2.dot(
        I2 * omega2
    )
    K3 = 0.5 * m3 * (Jv3 * theta_dot).dot(Jv3 * theta_dot) + 0.5 * omega3.dot(
        I3 * omega3
    )
    K4 = 0.5 * m4 * (Jv4 * theta_dot).dot(Jv4 * theta_dot) + 0.5 * omega4.dot(
        I4 * omega4
    )
    K5 = 0.5 * m5 * (Jv5 * theta_dot).dot(Jv5 * theta_dot) + 0.5 * omega5.dot(
        I5 * omega5
    )

    K = K1 + K2 + K3 + K4 + K5

    # Compute the potential energy of each link
    P1 = m1 * g.dot(p1)
    P2 = m2 * g.dot(p2)
    P3 = m3 * g.dot(p3)
    P4 = m4 * g.dot(p4)
    P5 = m5 * g.dot(p5)

    P = P1 + P2 + P3 + P4 + P5

    # Compute the Lagrangian
    L = K - P

    # Compute the equations of motion
    tau = sp.Matrix(
        [
            sp.diff(sp.diff(L, theta_dot[i]), "t") - sp.diff(L, theta[i])
            for i in range(5)
        ]
    )

    # Provide numerical values for testing
    values = {
        d_1: d_1_val,
        d_5: d_5_val,
        a_2: a_2_val,
        a_3: a_3_val,
        m1: masses[0],
        m2: masses[1],
        m3: masses[2],
        m4: masses[3],
        m5: masses[4],
        I1_xx: inertias[0][0],
        I1_yy: inertias[0][1],
        I1_zz: inertias[0][2],
        I2_xx: inertias[1][0],
        I2_yy: inertias[1][1],
        I2_zz: inertias[1][2],
        I3_xx: inertias[2][0],
        I3_yy: inertias[2][1],
        I3_zz: inertias[2][2],
        I4_xx: inertias[3][0],
        I4_yy: inertias[3][1],
        I4_zz: inertias[3][2],
        I5_xx: inertias[4][0],
        I5_yy: inertias[4][1],
        I5_zz: inertias[4][2],
        theta_1: angles[0],
        theta_2: angles[1],
        theta_3: angles[2],
        theta_4: angles[3],
        theta_5: angles[4],
        theta_dot[0]: velocities[0],
        theta_dot[1]: velocities[1],
        theta_dot[2]: velocities[2],
        theta_dot[3]: velocities[3],
        theta_dot[4]: velocities[4],
        theta_ddot[0]: accelerations[0],
        theta_ddot[1]: accelerations[1],
        theta_ddot[2]: accelerations[2],
        theta_ddot[3]: accelerations[3],
        theta_ddot[4]: accelerations[4],
    }

    # Compute numerical torques due to dynamics
    numerical_torques = tau.subs(values)
    print("\nNumerical Torques due to Dynamics:")
    sp.pprint(numerical_torques)


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
angles = [0, 0, 0, 0, 0]
velocities = [0, 0, 0, 0, 0]
accelerations = [0, 0, 0, 0, 0]

compute_dynamic_analysis(
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

# Experiment with another set of values
angles = [sp.rad(30), sp.rad(45), sp.rad(60), sp.rad(90), sp.rad(120)]
velocities = [1, 1, 1, 1, 1]
accelerations = [0.1, 0.1, 0.1, 0.1, 0.1]

compute_dynamic_analysis(
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
