% Define DH parameters
d1 = 0.1;  % Link offset
d5 = 0.1;  % Link offset
a2 = 0.5;  % Link length
a3 = 0.5;  % Link length
alpha = [pi/2, 0, 0, pi/2, 0];  % Twist angles in radians

% Define symbolic variables for joint angles and velocities
syms theta1 theta2 theta3 theta4 theta5 real
syms dtheta1 dtheta2 dtheta3 dtheta4 dtheta5 real

% Masses of the links and additional components
masses = [1.0, 1.0, 1.0, 1.0, 1.0];
mass_camera = 0.5;
mass_lights = 0.5;
g = 9.81;  % Gravitational acceleration

% Define lengths of the links (assuming these are given)
L1 = d1;
L2 = a2;
L3 = a3;
L4 = d5;
L5 = d5;  % Assuming the last link has length d5


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
% Potential Energy (gravitational)
% Using the provided equations:
U_g1 = -masses(1) * g * L1;
U_g2 = -masses(2) * g * (L1 + 1/2 * L2 * sin(theta2));
U_g3 = -masses(3) * g * (L1 + L2 * sin(theta2) + 1/2 * L3 * sin(theta2 + theta3));
U_g4 = -masses(4) * g * (L1 + L2 * sin(theta2) + L3 * sin(theta2 + theta3) + 1/2 * L4 * sin(theta2 + theta3 + theta4));
U_g5 = -(masses(5) + mass_camera + mass_lights) * g * (L1 + L2 * sin(theta2) + L3 * sin(theta2 + theta3) + L4 * sin(theta2 + theta3 + theta4) + 1/2 * L5 * sin(theta2 + theta3 + theta4));

% Total potential energy
V = U_g1 + U_g2 + U_g3 + U_g4 + U_g5;

% Total energy
E = T + V;

% Convert symbolic expressions to numeric functions
T_func = matlabFunction(T, 'Vars', {q, dq});
V_func = matlabFunction(V, 'Vars', {q});
E_func = matlabFunction(E, 'Vars', {q, dq});
tau_func = matlabFunction(G + M * dq, 'Vars', {q, dq});

% Define initial and final configurations
initial_config = [0, 0, 0, 0, 0];  % Initial configuration (all joint angles at 0)
final_config = [-0.3491, 3.1416, -0.3491, 1.0472, -2.4435];  % Configuration for max torques

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

% Initialize energy and torque arrays
kinetic_energy = zeros(num_steps, 1);
potential_energy = zeros(num_steps, 1);
total_energy = zeros(num_steps, 1);
torques = zeros(num_steps, 5);

% Calculate energies and torques
for i = 1:num_steps
    q_vals = theta_traj(i, :)';
    dq_vals = dtheta_traj(i, :)';
    
    kinetic_energy(i) = T_func(q_vals, dq_vals);
    potential_energy(i) = V_func(q_vals);
    total_energy(i) = E_func(q_vals, dq_vals);
    torques(i, :) = tau_func(q_vals, dq_vals);
end

% Plot joint angles
figure;
for j = 1:5
    subplot(5, 1, j);
    plot(time, theta_traj(:, j), 'LineWidth', 2);
    xlabel('Time (s)');
    ylabel(['Theta ' num2str(j) ' (rad)']);
    title(['Joint Angle ' num2str(j) ' over Time']);
    grid on;
end

% Plot joint velocities
figure;
for j = 1:5
    subplot(5, 1, j);
    plot(time, dtheta_traj(:, j), 'LineWidth', 2);
    xlabel('Time (s)');
    ylabel(['dTheta ' num2str(j) ' (rad/s)']);
    title(['Joint Velocity ' num2str(j) ' over Time']);
    grid on;
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

% Plot torques
figure;
for j = 1:5
    subplot(5, 1, j);
    plot(time, torques(:, j), 'LineWidth', 2);
    xlabel('Time (s)');
    ylabel(['Torque ' num2str(j) ' (Nm)']);
    title(['Torque at Joint ' num2str(j) ' over Time']);
    grid on;
end

% Energy conservation validation
energy_difference = abs(total_energy - total_energy(1));
figure;
plot(time, energy_difference, 'g', 'LineWidth', 2);
xlabel('Time (s)');
ylabel('Energy Difference (J)');
title('Energy Conservation Validation');
grid on;

% Summarize energy values
disp('Maximum Kinetic Energy (J):');
disp(max(kinetic_energy));

disp('Minimum Kinetic Energy (J):');
disp(min(kinetic_energy));

disp('Maximum Potential Energy (J):');
disp(max(potential_energy));

disp('Minimum Potential Energy (J):');
disp(min(potential_energy));

disp('Maximum Total Energy (J):');
disp(max(total_energy));

disp('Minimum Total Energy (J):');
disp(min(total_energy));

% Summarize torques
disp('Maximum Torques (Nm):');
disp(max(torques));

disp('Minimum Torques (Nm):');
disp(min(torques));
