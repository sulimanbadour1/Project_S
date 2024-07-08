% Define DH parameters
d1 = 0.1;
d5 = 0.1;
a2 = 0.5;
a3 = 0.5;
alpha = [pi/2, 0, 0, pi/2, 0];  % Convert to radians

% Define symbolic variables
syms theta1 theta2 theta3 theta4 theta5 real
syms dtheta1 dtheta2 dtheta3 dtheta4 dtheta5 real

% Masses of the links and external parameters
masses = [1.0, 1.0, 1.0, 1.0, 1.0];
mass_camera = 0.5;
mass_lights = 0.5;
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
I1 = eye(3) * (1/12) * masses(1) * (d1^2);
I2 = eye(3) * (1/12) * masses(2) * (a2^2);
I3 = eye(3) * (1/12) * masses(3) * (a3^2);
I4 = eye(3) * (1/12) * masses(4) * (d1^2);
I5 = eye(3) * (1/12) * (masses(5) + mass_camera + mass_lights) * (d5^2);

% Compute the inertia matrix
M = Jv1' * masses(1) * Jv1 + Jv2' * masses(2) * Jv2 + Jv3' * masses(3) * Jv3 + ...
    Jv4' * masses(4) * Jv4 + Jv5' * (masses(5) + mass_camera + mass_lights) * Jv5;

% Compute the gravity vector
G = Jv1' * masses(1) * [0; 0; -g] + ...
    Jv2' * masses(2) * [0; 0; -g] + ...
    Jv3' * masses(3) * [0; 0; -g] + ...
    Jv4' * masses(4) * [0; 0; -g] + ...
    Jv5' * (masses(5) + mass_camera + mass_lights) * [0; 0; -g];

% Define q and dq
q = [theta1; theta2; theta3; theta4; theta5];
dq = [dtheta1; dtheta2; dtheta3; dtheta4; dtheta5];

% Kinetic energy
T = 0.5 * dq.' * M * dq;

% Potential energy
V = masses(1) * g * p1(3) + masses(2) * g * p2(3) + masses(3) * g * p3(3) + ...
    masses(4) * g * p4(3) + (masses(5) + mass_camera + mass_lights) * g * p5(3);

% Total energy
E = T + V;

% Convert symbolic expressions to numeric functions
T_func = matlabFunction(T, 'Vars', {theta1, theta2, theta3, theta4, theta5, ...
                                    dtheta1, dtheta2, dtheta3, dtheta4, dtheta5});
V_func = matlabFunction(V, 'Vars', {theta1, theta2, theta3, theta4, theta5});
E_func = matlabFunction(E, 'Vars', {theta1, theta2, theta3, theta4, theta5, ...
                                    dtheta1, dtheta2, dtheta3, dtheta4, dtheta5});

% Define initial and final configurations
initial_config = [0, 0, 0, 0, 0];  % Initial configuration (all joint angles at 0)
final_config = [-1.0472, 3.1416, 0.3491, 1.0472, 2.4435];  % Configuration for max torques

% Simulation time
t_final = 10;  % Final time in seconds
num_steps = 100;  % Number of time steps
time = linspace(0, t_final, num_steps);

% Quintic polynomial trajectory coefficients for each joint
coeffs = zeros(6, 5);
for i = 1:5
    coeffs(:, i) = polyfit([0, t_final], [initial_config(i), final_config(i)], 5);
end

% Compute joint angles and velocities using polynomial derivatives
theta_traj = zeros(num_steps, 5);
dtheta_traj = zeros(num_steps, 5);
for i = 1:num_steps
    t = time(i);
    for j = 1:5
        theta_traj(i, j) = polyval(coeffs(:, j), t);
        dtheta_traj(i, j) = polyval(polyder(coeffs(:, j)), t);
    end
end

% Initialize energy arrays
kinetic_energy = zeros(num_steps, 1);
potential_energy = zeros(num_steps, 1);
total_energy = zeros(num_steps, 1);

for i = 1:num_steps
    t1 = theta_traj(i, 1);
    t2 = theta_traj(i, 2);
    t3 = theta_traj(i, 3);
    t4 = theta_traj(i, 4);
    t5 = theta_traj(i, 5);
    
    dt1 = dtheta_traj(i, 1);
    dt2 = dtheta_traj(i, 2);
    dt3 = dtheta_traj(i, 3);
    dt4 = dtheta_traj(i, 4);
    dt5 = dtheta_traj(i, 5);
    
    kinetic_energy(i) = T_func(t1, t2, t3, t4, t5, dt1, dt2, dt3, dt4, dt5);
    potential_energy(i) = V_func(t1, t2, t3, t4, t5);
    total_energy(i) = E_func(t1, t2, t3, t4, t5, dt1, dt2, dt3, dt4, dt5);
end

% Plot energies
figure;
subplot(3, 1, 1);
plot(time, kinetic_energy, 'r', 'LineWidth', 2);
xlabel('Time (s)');
ylabel('Kinetic Energy (J)');
title('Kinetic Energy over Time');
grid on;

subplot(3, 1, 2);
plot(time, potential_energy, 'b', 'LineWidth', 2);
xlabel('Time (s)');
ylabel('Potential Energy (J)');
title('Potential Energy over Time');
grid on;

subplot(3, 1, 3);
plot(time, total_energy, 'k', 'LineWidth', 2);
xlabel('Time (s)');
ylabel('Total Energy (J)');
title('Total Energy over Time');
grid on;

% Display energies
disp('Kinetic Energy (J):');
disp(kinetic_energy);

disp('Potential Energy (J):');
disp(potential_energy);

disp('Total Energy (J):');
disp(total_energy);
