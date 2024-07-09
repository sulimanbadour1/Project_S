% Define symbolic variables for masses and inertias
syms m1 m2 m3 m4 m5 m_camera m_lights real
syms I1 I2 I3 I4 I5 real

% Define DH parameters
d1 = 0.1;
d5 = 0.1;
a2 = 0.5;
a3 = 0.5;
alpha = [pi/2, 0, 0, pi/2, 0];  % Convert to radians

% Define symbolic variables
syms theta1 theta2 theta3 theta4 theta5 real
syms dtheta1 dtheta2 dtheta3 dtheta4 dtheta5 real
syms ddtheta1 ddtheta2 ddtheta3 ddtheta4 ddtheta5 real

% External parameters
g = 9.81;

% Define DH transformation matrix function
dh = @(theta, d, a, alpha) [
    cos(theta) -sin(theta)*cos(alpha)  sin(theta)*sin(alpha) a*cos(theta);
    sin(theta)  cos(theta)*cos(alpha) -cos(theta)*sin(alpha) a*sin(theta);
    0           sin(alpha)             cos(alpha)             d;
    0           0                       0                       1
];

% Compute transformation matrices
A1 = dh(theta1, d1, 0, alpha(1));
A2 = dh(theta2, 0, a2, alpha(2));
A3 = dh(theta3, 0, a3, alpha(3));
A4 = dh(theta4, 0, 0, alpha(4));
A5 = dh(theta5, d5, 0, alpha(5));

% Forward kinematics
T1 = A1;
T2 = T1 * A2;
T3 = T2 * A3;
T4 = T3 * A4;
T5 = T4 * A5;

% Positions of each link's center of mass
p1 = T1(1:3, 4) / 2;
p2 = T2(1:3, 4) / 2;
p3 = T3(1:3, 4) / 2;
p4 = T4(1:3, 4) / 2;
p5 = T5(1:3, 4) / 2;

% Compute Jacobians for each link
Jv1 = jacobian(p1, [theta1, theta2, theta3, theta4, theta5]);
Jv2 = jacobian(p2, [theta1, theta2, theta3, theta4, theta5]);
Jv3 = jacobian(p3, [theta1, theta2, theta3, theta4, theta5]);
Jv4 = jacobian(p4, [theta1, theta2, theta3, theta4, theta5]);
Jv5 = jacobian(p5, [theta1, theta2, theta3, theta4, theta5]);

% Inertia matrices for each link (simplified as diagonal matrices for demonstration)
I1_matrix = eye(3) * I1;
I2_matrix = eye(3) * I2;
I3_matrix = eye(3) * I3;
I4_matrix = eye(3) * I4;
I5_matrix = eye(3) * I5;

% Compute the inertia matrix
M = Jv1' * m1 * Jv1 + Jv2' * m2 * Jv2 + Jv3' * m3 * Jv3 + ...
    Jv4' * m4 * Jv4 + Jv5' * (m5 + m_camera + m_lights) * Jv5;

% Compute the gravity vector
G = Jv1' * m1 * [0; 0; -g] + ...
    Jv2' * m2 * [0; 0; -g] + ...
    Jv3' * m3 * [0; 0; -g] + ...
    Jv4' * m4 * [0; 0; -g] + ...
    Jv5' * (m5 + m_camera + m_lights) * [0; 0; -g];

% Define q, dq, and ddq
q = [theta1; theta2; theta3; theta4; theta5];
dq = [dtheta1; dtheta2; dtheta3; dtheta4; dtheta5];
ddq = [ddtheta1; ddtheta2; ddtheta3; ddtheta4; ddtheta5];

% Compute Coriolis and centrifugal matrix
C = sym(zeros(5));
for k = 1:5
    for j = 1:5
        for i = 1:5
            C(k, j) = C(k, j) + 0.5 * (diff(M(k, j), q(i)) + diff(M(k, i), q(j)) - diff(M(i, j), q(k))) * dq(i);
        end
    end
end

% Total torque calculation
tau = M * ddq + C * dq + G;

% Display symbolic results
disp('Inertia Matrix M:');
disp(M);

disp('Coriolis and Centrifugal Matrix C:');
disp(C);

disp('Gravity Vector G:');
disp(G);

disp('Equations of Motion tau:');
disp(tau);
