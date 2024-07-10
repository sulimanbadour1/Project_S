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
final_config1 = [0, -1.1, 0, 0, 0];  % Configuration for max torques
final_config2 = [0, 1.1, 0, 0, 0];  % Another configuration for max torques

% Simulation time
t_final = 10;  % Final time in seconds
num_steps = 100;  % Number of time steps
time = linspace(0, t_final, num_steps);

% Compute trajectories, energies, and torques for both configurations
[theta_traj1, dtheta_traj1, ddtheta_traj1, kinetic_energy1, potential_energy1, total_energy1, torques1, end_effector_positions1, robot_configurations1] = compute_trajectory(final_config1, t_final, num_steps, time, q, T5, T_func, V_func, E_func, tau_func, A1, A2, A3, A4, A5);
[theta_traj2, dtheta_traj2, ddtheta_traj2, kinetic_energy2, potential_energy2, total_energy2, torques2, end_effector_positions2, robot_configurations2] = compute_trajectory(final_config2, t_final, num_steps, time, q, T5, T_func, V_func, E_func, tau_func, A1, A2, A3, A4, A5);

% Plot joint angles for both configurations in a new figure
figure;
subplot(2, 1, 1);
hold on;
for j = 1:5
    plot(time, theta_traj1(:, j), 'LineWidth', 2, 'DisplayName', ['Theta ' num2str(j)]);
end
xlabel('Time (s)');
ylabel('Joint Angles (rad)');
title('Joint Angles over Time for Config 1');
legend('show');
grid on;

subplot(2, 1, 2);
hold on;
for j = 1:5
    plot(time, theta_traj2(:, j), 'LineWidth', 2, 'DisplayName', ['Theta ' num2str(j)]);
end
xlabel('Time (s)');
ylabel('Joint Angles (rad)');
title('Joint Angles over Time for Config 2');
legend('show');
grid on;

% Plot joint velocities for both configurations in a new figure
figure;
subplot(2, 1, 1);
hold on;
for j = 1:5
    plot(time, dtheta_traj1(:, j), 'LineWidth', 2, 'DisplayName', ['dTheta ' num2str(j)]);
end
xlabel('Time (s)');
ylabel('Joint Velocities (rad/s)');
title('Joint Velocities over Time for Config 1');
legend('show');
grid on;

subplot(2, 1, 2);
hold on;
for j = 1:5
    plot(time, dtheta_traj2(:, j), 'LineWidth', 2, 'DisplayName', ['dTheta ' num2str(j)]);
end
xlabel('Time (s)');
ylabel('Joint Velocities (rad/s)');
title('Joint Velocities over Time for Config 2');
legend('show');
grid on;

% Plot joint accelerations for both configurations in a new figure
figure;
subplot(2, 1, 1);
hold on;
for j = 1:5
    plot(time, ddtheta_traj1(:, j), 'LineWidth', 2, 'DisplayName', ['ddTheta ' num2str(j)]);
end
xlabel('Time (s)');
ylabel('Joint Accelerations (rad/s^2)');
title('Joint Accelerations over Time for Config 1');
legend('show');
grid on;

subplot(2, 1, 2);
hold on;
for j = 1:5
    plot(time, ddtheta_traj2(:, j), 'LineWidth', 2, 'DisplayName', ['ddTheta ' num2str(j)]);
end
xlabel('Time (s)');
ylabel('Joint Accelerations (rad/s^2)');
title('Joint Accelerations over Time for Config 2');
legend('show');
grid on;

% Plot torques for both configurations in a new figure
figure;
subplot(2, 1, 1);
hold on;
for j = 1:5
    plot(time, torques1(:, j), 'LineWidth', 2, 'DisplayName', ['Torque ' num2str(j)]);
end
xlabel('Time (s)');
ylabel('Torque (Nm)');
title('Torques at Joints over Time for Config 1');
legend('show');
grid on;

subplot(2, 1, 2);
hold on;
for j = 1:5
    plot(time, torques2(:, j), 'LineWidth', 2, 'DisplayName', ['Torque ' num2str(j)]);
end
xlabel('Time (s)');
ylabel('Torque (Nm)');
title('Torques at Joints over Time for Config 2');
legend('show');
grid on;

% Plot energies for both configurations in a new figure
figure;
subplot(3, 2, 1);
plot(time, kinetic_energy1, 'r', 'LineWidth', 2);
xlabel('Time (s)');
ylabel('Kinetic Energy (J)');
title('Kinetic Energy over Time for Config 1');
grid on;

subplot(3, 2, 3);
plot(time, potential_energy1, 'b', 'LineWidth', 2);
xlabel('Time (s)');
ylabel('Potential Energy (J)');
title('Potential Energy over Time for Config 1');
grid on;

subplot(3, 2, 5);
plot(time, total_energy1, 'k', 'LineWidth', 2);
xlabel('Time (s)');
ylabel('Total Energy (J)');
title('Total Energy over Time for Config 1');
grid on;

subplot(3, 2, 2);
plot(time, kinetic_energy2, 'r', 'LineWidth', 2);
xlabel('Time (s)');
ylabel('Kinetic Energy (J)');
title('Kinetic Energy over Time for Config 2');
grid on;

subplot(3, 2, 4);
plot(time, potential_energy2, 'b', 'LineWidth', 2);
xlabel('Time (s)');
ylabel('Potential Energy (J)');
title('Potential Energy over Time for Config 2');
grid on;

subplot(3, 2, 6);
plot(time, total_energy2, 'k', 'LineWidth', 2);
xlabel('Time (s)');
ylabel('Total Energy (J)');
title('Total Energy over Time for Config 2');
grid on;

% Energy conservation validation for both configurations in a new figure
figure;
subplot(2, 1, 1);
energy_difference1 = abs(total_energy1 - total_energy1(1));
plot(time, energy_difference1, 'g', 'LineWidth', 2);
xlabel('Time (s)');
ylabel('Energy Difference (J)');
title('Energy Conservation Validation for Config 1');
grid on;

subplot(2, 1, 2);
energy_difference2 = abs(total_energy2 - total_energy2(1));
plot(time, energy_difference2, 'g', 'LineWidth', 2);
xlabel('Time (s)');
ylabel('Energy Difference (J)');
title('Energy Conservation Validation for Config 2');
grid on;

% Animate the robot for both configurations
figure;
subplot(1, 2, 1);
hold on;
grid on;
axis equal;
xlabel('X');
ylabel('Y');
zlabel('Z');
title('End Effector Trajectory and Robot Animation for Config 1');
view(3);
robot_plot1 = plot3(0, 0, 0, '-o', 'LineWidth', 2, 'MarkerSize', 10);
end_effector_plot1 = plot3(0, 0, 0, 'r*', 'MarkerSize', 10);
base_label1 = text(0, 0, 0, 'Base', 'FontSize', 12, 'FontWeight', 'bold', 'Color', 'blue');
end_effector_label1 = text(0, 0, 0, 'End Effector', 'FontSize', 12, 'FontWeight', 'bold', 'Color', 'red');

% Plot the end effector trajectory for config 1
plot3(end_effector_positions1(:, 1), end_effector_positions1(:, 2), end_effector_positions1(:, 3), 'k--', 'LineWidth', 2);

subplot(1, 2, 2);
hold on;
grid on;
axis equal;
xlabel('X');
ylabel('Y');
zlabel('Z');
title('End Effector Trajectory and Robot Animation for Config 2');
view(3);
robot_plot2 = plot3(0, 0, 0, '-o', 'LineWidth', 2, 'MarkerSize', 10);
end_effector_plot2 = plot3(0, 0, 0, 'r*', 'MarkerSize', 10);
base_label2 = text(0, 0, 0, 'Base', 'FontSize', 12, 'FontWeight', 'bold', 'Color', 'blue');
end_effector_label2 = text(0, 0, 0, 'End Effector', 'FontSize', 12, 'FontWeight', 'bold', 'Color', 'red');

% Plot the end effector trajectory for config 2
plot3(end_effector_positions2(:, 1), end_effector_positions2(:, 2), end_effector_positions2(:, 3), 'k--', 'LineWidth', 2);

% Animate both configurations
for i = 1:num_steps
    % Config 1
    T_matrices1 = robot_configurations1(i, :);
    points1 = [0, 0, 0; T_matrices1{1}(1:3, 4)'; T_matrices1{2}(1:3, 4)'; T_matrices1{3}(1:3, 4)'; T_matrices1{4}(1:3, 4)'; T_matrices1{5}(1:3, 4)'];
    set(robot_plot1, 'XData', points1(:, 1), 'YData', points1(:, 2), 'ZData', points1(:, 3));
    set(end_effector_plot1, 'XData', points1(end, 1), 'YData', points1(end, 2), 'ZData', points1(end, 3));
    set(end_effector_label1, 'Position', points1(end, :));

    % Config 2
    T_matrices2 = robot_configurations2(i, :);
    points2 = [0, 0, 0; T_matrices2{1}(1:3, 4)'; T_matrices2{2}(1:3, 4)'; T_matrices2{3}(1:3, 4)'; T_matrices2{4}(1:3, 4)'; T_matrices2{5}(1:3, 4)'];
    set(robot_plot2, 'XData', points2(:, 1), 'YData', points2(:, 2), 'ZData', points2(:, 3));
    set(end_effector_plot2, 'XData', points2(end, 1), 'YData', points2(end, 2), 'ZData', points2(end, 3));
    set(end_effector_label2, 'Position', points2(end, :));

    drawnow;
end

% Function to compute trajectory, energies, and torques for a given final configuration
function [theta_traj, dtheta_traj, ddtheta_traj, kinetic_energy, potential_energy, total_energy, torques, end_effector_positions, robot_configurations] = compute_trajectory(final_config, t_final, num_steps, time, q, T5, T_func, V_func, E_func, tau_func, A1, A2, A3, A4, A5)
    % Quintic polynomial trajectory coefficients for each joint
    coeffs = zeros(6, 5);
    for i = 1:5
        coeffs(:, i) = polyfit([0, t_final], [0, final_config(i)], 5);
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

    % Compute end effector positions for the trajectory
    end_effector_positions = zeros(num_steps, 3);
    robot_configurations = cell(num_steps, 5);  % Store robot configurations for plotting
    for i = 1:num_steps
        q_vals = theta_traj(i, :)';
        dq_vals = dtheta_traj(i, :)';
        ddq_vals = ddtheta_traj(i, :)';

        kinetic_energy(i) = T_func(q_vals, dq_vals);
        potential_energy(i) = V_func(q_vals);
        total_energy(i) = E_func(q_vals, dq_vals);
        torques(i, :) = tau_func(q_vals, dq_vals, ddq_vals);

        T_current = double(subs(T5, {q(1), q(2), q(3), q(4), q(5)}, q_vals.'));
        end_effector_positions(i, :) = T_current(1:3, 4);
        % Store current configuration
        robot_configurations{i, 1} = double(subs(A1, q(1), q_vals(1)));
        robot_configurations{i, 2} = double(robot_configurations{i, 1} * subs(A2, q(2), q_vals(2)));
        robot_configurations{i, 3} = double(robot_configurations{i, 2} * subs(A3, q(3), q_vals(3)));
        robot_configurations{i, 4} = double(robot_configurations{i, 3} * subs(A4, q(4), q_vals(4)));
        robot_configurations{i, 5} = double(robot_configurations{i, 4} * subs(A5, q(5), q_vals(5)));
    end
end
