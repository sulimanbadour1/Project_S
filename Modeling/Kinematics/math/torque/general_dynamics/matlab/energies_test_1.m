% Define DH parameters
d1 = 0.1;  % Link offset
d5 = 0.1;  % Link offset
a2 = 0.5;  % Link length
a3 = 0.5;  % Link length
alpha = [pi/2, 0, 0, pi/2, 0];  % Twist angles in radians

% Define symbolic variables for joint angles and velocities
syms theta1 theta2 theta3 theta4 theta5 real
syms dtheta1 dtheta2 dtheta3 dtheta4 dtheta5 real
syms ddtheta1 ddtheta2 ddtheta3 ddtheta4 ddtheta5 real

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

% Define q, dq, and ddq
q = [theta1; theta2; theta3; theta4; theta5];
dq = [dtheta1; dtheta2; dtheta3; dtheta4; dtheta5];
ddq = [ddtheta1; ddtheta2; ddtheta3; ddtheta4; ddtheta5];

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
tau_func = matlabFunction(G + M * ddq, 'Vars', {q, dq, ddq});

% Define initial and final configurations
initial_config = [0, 0, 0, 0, 0];  % Initial configuration (all joint angles at 0)
final_config = [-3.1416, 2.4435, 1.0472, 1.7453, 0.3491];  % Configuration for max torques

% Simulation time
t_final = 10;  % Final time in seconds
num_steps = 100;  % Number of time steps
time = linspace(0, t_final, num_steps);

% Quintic polynomial trajectory coefficients for each joint
coeffs = zeros(6, 5);
for i = 1:5
    coeffs(:, i) = polyfit([0, t_final], [initial_config(i), final_config(i)], 5);
end

% Compute joint angles, velocities, and accelerations using polynomial derivatives
theta_traj = zeros(num_steps, 5);
dtheta_traj = zeros(num_steps, 5);
ddtheta_traj = zeros(num_steps, 5);
for i = 1:num_steps
    t = time(i);
    for j = 1:5
        theta_traj(i, j) = polyval(coeffs(:, j), t);
        dtheta_traj(i, j) = polyval(polyder(coeffs(:, j)), t);
        ddtheta_traj(i, j) = polyval(polyder(polyder(coeffs(:, j))), t);
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
    ddq_vals = ddtheta_traj(i, :)';
    
    kinetic_energy(i) = T_func(q_vals, dq_vals);
    potential_energy(i) = V_func(q_vals);
    total_energy(i) = E_func(q_vals, dq_vals);
    torques(i, :) = tau_func(q_vals, dq_vals, ddq_vals);
end

% Create figure for animation
figure;
hold on;
grid on;
axis equal;
xlabel('X');
ylabel('Y');
zlabel('Z');
title('End Effector Trajectory and Robot Animation');
view(3);

% Plot the end effector trajectory
plot3(end_effector_positions(:, 1), end_effector_positions(:, 2), end_effector_positions(:, 3), 'k--', 'LineWidth', 2);

% Initialize plot for robot
robot_plot = plot3(0, 0, 0, '-o', 'LineWidth', 2, 'MarkerSize', 10);
end_effector_plot = plot3(0, 0, 0, 'r*', 'MarkerSize', 10);

% Add labels for the base and end effector
base_label = text(0, 0, 0, 'Base', 'FontSize', 12, 'FontWeight', 'bold', 'Color', 'blue');
end_effector_label = text(0, 0, 0, 'End Effector', 'FontSize', 12, 'FontWeight', 'bold', 'Color', 'red');

% Animate the robot
for i = 1:num_steps
    % Extract joint positions
    T_matrices = robot_configurations(i, :);
    points = [0, 0, 0; T_matrices{1}(1:3, 4)'; T_matrices{2}(1:3, 4)'; T_matrices{3}(1:3, 4)'; T_matrices{4}(1:3, 4)'; T_matrices{5}(1:3, 4)'];
    % Update robot plot
    set(robot_plot, 'XData', points(:, 1), 'YData', points(:, 2), 'ZData', points(:, 3));
    % Update end effector plot
    set(end_effector_plot, 'XData', points(end, 1), 'YData', points(end, 2), 'ZData', points(end, 3));
    % Update end effector label
    set(end_effector_label, 'Position', points(end, :));
    drawnow;
end

% Plot joint angles in a new figure
figure;
for j = 1:5
    subplot(5, 1, j);
    plot(time, theta_traj(:, j), 'LineWidth', 2);
    xlabel('Time (s)');
    ylabel(['Theta ' num2str(j) ' (rad)']);
    title(['Joint Angle ' num2str(j) ' over Time']);
    grid on;
end

% Plot joint velocities in a new figure
figure;
for j = 1:5
    subplot(5, 1, j);
    plot(time, dtheta_traj(:, j), 'LineWidth', 2);
    xlabel('Time (s)');
    ylabel(['dTheta ' num2str(j) ' (rad/s)']);
    title(['Joint Velocity ' num2str(j) ' over Time']);
    grid on;
end

% Plot joint accelerations in a new figure
figure;
for j = 1:5
    subplot(5, 1, j);
    plot(time, ddtheta_traj(:, j), 'LineWidth', 2);
    xlabel('Time (s)');
    ylabel(['ddTheta ' num2str(j) ' (rad/s^2)']);
    title(['Joint Acceleration ' num2str(j) ' over Time']);
    grid on;
end

% Plot torques in a new figure
figure;
hold on;
for j = 1:5
    plot(time, torques(:, j), 'LineWidth', 2, 'DisplayName', ['Torque ' num2str(j)]);
end
xlabel('Time (s)');
ylabel('Torque (Nm)');
title('Torques at Joints over Time');
legend('show');
grid on;

% Plot energies in a new figure
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

% Energy conservation validation in a new figure
figure;
plot(time, energy_difference, 'g', 'LineWidth', 2);
xlabel('Time (s)');
ylabel('Energy Difference (J)');
title('Energy Conservation Validation');
grid on;

% Function to plot the robot's configuration
function plot_robot(T_matrices, ax)
    hold(ax, 'on');
    % Base point
    origin = [0; 0; 0];
    % Extract positions of the joints
    points = [origin, T_matrices{1}(1:3, 4), T_matrices{2}(1:3, 4), T_matrices{3}(1:3, 4), T_matrices{4}(1:3, 4), T_matrices{5}(1:3, 4)];
    % Plot links
    plot3(ax, points(1, :), points(2, :), points(3, :), '-o', 'LineWidth', 2, 'MarkerSize', 10);
    % Plot end effector
    plot3(ax, T_matrices{5}(1, 4), T_matrices{5}(2, 4), T_matrices{5}(3, 4), 'r*', 'MarkerSize', 10);
    hold(ax, 'off');
    xlabel(ax, 'X');
    ylabel(ax, 'Y');
    zlabel(ax, 'Z');
    grid(ax, 'on');
    axis(ax, 'equal');
    view(ax, 3);
end
