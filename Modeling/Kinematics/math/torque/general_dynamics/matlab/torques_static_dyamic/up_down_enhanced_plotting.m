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
g = -9.81;  % Gravitational acceleration

% Define lengths of the links
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
    0           0                      0                      1
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

% Define target end-effector position
target_position = [0.5; 0.5; 0.2];  % Column vector for target position

% Solve for two sets of joint angles that reach the target position with the constraint
% \theta_2 + \theta_3 + \theta_4 = 0

% Constraint equation
constraint = theta2 + theta3 + theta4 == 0;

% Define two sets of initial guesses for the joint angles
initial_guess1 = [0, pi/6, -pi/6, -pi/6, 0];
initial_guess2 = [0, -pi/6, pi/6, -pi/6, 0];

% Solve for the joint angles using the initial guesses
eqns1 = [T5(1:3, 4) == target_position; constraint];
eqns2 = [T5(1:3, 4) == target_position; constraint];

vars = [theta1, theta2, theta3, theta4, theta5];

sol1 = vpasolve(eqns1, vars, initial_guess1);
sol2 = vpasolve(eqns2, vars, initial_guess2);

% Convert solutions to numeric values
final_config1 = double([sol1.theta1, sol1.theta2, sol1.theta3, sol1.theta4, sol1.theta5]);
final_config2 = double([sol2.theta1, sol2.theta2, sol2.theta3, sol2.theta4, sol2.theta5]);

% Simulation parameters
t_final = 10;  % Final time in seconds
num_steps = 100;  % Number of time steps
time = linspace(0, t_final, num_steps);

% Compute trajectories, energies, and torques for both configurations
[theta_traj1, dtheta_traj1, ddtheta_traj1, kinetic_energy1, potential_energy1, total_energy1, torques1, end_effector_positions1, robot_configurations1, coeffs1] = compute_trajectory(final_config1, t_final, num_steps, time, q, T5, T_func, V_func, E_func, tau_func, A1, A2, A3, A4, A5);
[theta_traj2, dtheta_traj2, ddtheta_traj2, kinetic_energy2, potential_energy2, total_energy2, torques2, end_effector_positions2, robot_configurations2, coeffs2] = compute_trajectory(final_config2, t_final, num_steps, time, q, T5, T_func, V_func, E_func, tau_func, A1, A2, A3, A4, A5);

% Print trajectory equations
disp('Configuration 1 - Trajectory Equations:');
for j = 1:5
    fprintf('Theta %d(t) = %f*t^5 + %f*t^4 + %f*t^3 + %f*t^2 + %f*t + %f\n', j, coeffs1(:, j));
end

disp('Configuration 2 - Trajectory Equations:');
for j = 1:5
    fprintf('Theta %d(t) = %f*t^5 + %f*t^4 + %f*t^3 + %f*t^2 + %f*t + %f\n', j, coeffs2(:, j));
end

% Convert radians to degrees
theta_traj1_deg = rad2deg(theta_traj1);
theta_traj2_deg = rad2deg(theta_traj2);

% Print out values at 10 points along the trajectory for position and angles
disp('Configuration 1 - Position and Angles at 10 points:');
indices = round(linspace(1, num_steps, 10));
for i = 1:10
    idx = indices(i);
    fprintf('Point %d:\n', i);
    fprintf('End Effector Position: [%f, %f, %f]\n', end_effector_positions1(idx, :));
    fprintf('Joint Angles (deg): [%f, %f, %f, %f, %f]\n', theta_traj1_deg(idx, :));
end

disp('Configuration 2 - Position and Angles at 10 points:');
for i = 1:10
    idx = indices(i);
    fprintf('Point %d:\n', i);
    fprintf('End Effector Position: [%f, %f, %f]\n', end_effector_positions2(idx, :));
    fprintf('Joint Angles (deg): [%f, %f, %f, %f, %f]\n', theta_traj2_deg(idx, :));
end

% Print out maximum and minimum torque values for all joints
for j = 1:5
    [max_torque1, max_torque_idx1] = max(torques1(:, j));
    [min_torque1, min_torque_idx1] = min(torques1(:, j));
    fprintf('Configuration 1 - Joint %d Maximum Torque: %f Nm at time %f s\n', j, max_torque1, time(max_torque_idx1));
    fprintf('Configuration 1 - Joint %d Minimum Torque: %f Nm at time %f s\n', j, min_torque1, time(min_torque_idx1));
    
    [max_torque2, max_torque_idx2] = max(torques2(:, j));
    [min_torque2, min_torque_idx2] = min(torques2(:, j));
    fprintf('Configuration 2 - Joint %d Maximum Torque: %f Nm at time %f s\n', j, max_torque2, time(max_torque_idx2));
    fprintf('Configuration 2 - Joint %d Minimum Torque: %f Nm at time %f s\n', j, min_torque2, time(min_torque_idx2));
end

% Print out initial and final torques and remaining parameters
fprintf('Configuration 1 - Initial Torque: [%f, %f, %f, %f, %f] Nm\n', torques1(1, :));
fprintf('Configuration 1 - Final Torque: [%f, %f, %f, %f, %f] Nm\n', torques1(end, :));
fprintf('Configuration 1 - Initial Kinetic Energy: %f J\n', kinetic_energy1(1));
fprintf('Configuration 1 - Final Kinetic Energy: %f J\n', kinetic_energy1(end));
fprintf('Configuration 1 - Initial Potential Energy: %f J\n', potential_energy1(1));
fprintf('Configuration 1 - Final Potential Energy: %f J\n', potential_energy1(end));

fprintf('Configuration 2 - Initial Torque: [%f, %f, %f, %f, %f] Nm\n', torques2(1, :));
fprintf('Configuration 2 - Final Torque: [%f, %f, %f, %f, %f] Nm\n', torques2(end, :));
fprintf('Configuration 2 - Initial Kinetic Energy: %f J\n', kinetic_energy2(1));
fprintf('Configuration 2 - Final Kinetic Energy: %f J\n', kinetic_energy2(end));
fprintf('Configuration 2 - Initial Potential Energy: %f J\n', potential_energy2(1));
fprintf('Configuration 2 - Final Potential Energy: %f J\n', potential_energy2(end));

% Extract peak values
fprintf('Configuration 1 - Peak Kinetic Energy: %f J\n', max(kinetic_energy1));
fprintf('Configuration 1 - Peak Potential Energy: %f J\n', max(potential_energy1));
fprintf('Configuration 1 - Peak Total Energy: %f J\n', max(total_energy1));

fprintf('Configuration 2 - Peak Kinetic Energy: %f J\n', max(kinetic_energy2));
fprintf('Configuration 2 - Peak Potential Energy: %f J\n', max(potential_energy2));
fprintf('Configuration 2 - Peak Total Energy: %f J\n', max(total_energy2));

% Subplot for joint angles, velocities, and accelerations
figure;
subplot(3,1,1);
hold on;
for j = 1:5
    plot(time, theta_traj1(:, j), 'LineWidth', 2, 'DisplayName', ['Elbow Up - Theta ' num2str(j)]);
    plot(time, theta_traj2(:, j), '--', 'LineWidth', 2, 'DisplayName', ['Elbow down - Theta ' num2str(j)]);
end
xlabel('Time (s)', 'FontSize', 14, 'FontWeight', 'bold');
ylabel('Joint Angles (rad)', 'FontSize', 14, 'FontWeight', 'bold');
title('Joint Angles over Time', 'FontSize', 16, 'FontWeight', 'bold');
legend('show', 'FontSize', 12, 'FontWeight', 'bold');
grid on;
set(gca, 'FontSize', 12, 'FontWeight', 'bold');
hold off;

subplot(3,1,2);
hold on;
for j = 1:5
    plot(time, dtheta_traj1(:, j), 'LineWidth', 2, 'DisplayName', ['Elbow up - dTheta ' num2str(j)]);
    plot(time, dtheta_traj2(:, j), '--', 'LineWidth', 2, 'DisplayName', ['Elbow down - dTheta ' num2str(j)]);
end
xlabel('Time (s)', 'FontSize', 14, 'FontWeight', 'bold');
ylabel('Joint Velocities (rad/s)', 'FontSize', 14, 'FontWeight', 'bold');
title('Joint Velocities over Time', 'FontSize', 16, 'FontWeight', 'bold');
legend('show', 'FontSize', 12, 'FontWeight', 'bold');
grid on;
set(gca, 'FontSize', 12, 'FontWeight', 'bold');
hold off;

subplot(3,1,3);
hold on;
for j = 1:5
    plot(time, ddtheta_traj1(:, j), 'LineWidth', 2, 'DisplayName', ['Elbow up - ddTheta ' num2str(j)]);
    plot(time, ddtheta_traj2(:, j), '--', 'LineWidth', 2, 'DisplayName', ['Elbow down - ddTheta ' num2str(j)]);
end
xlabel('Time (s)', 'FontSize', 14, 'FontWeight', 'bold');
ylabel('Joint Accelerations (rad/s^2)', 'FontSize', 14, 'FontWeight', 'bold');
title('Joint Accelerations over Time', 'FontSize', 16, 'FontWeight', 'bold');
legend('show', 'FontSize', 12, 'FontWeight', 'bold');
grid on;
set(gca, 'FontSize', 12, 'FontWeight', 'bold');
hold off;

% Subplot for torques
figure;
hold on;
for j = 1:5
    plot(time, torques1(:, j), 'LineWidth', 2, 'DisplayName', ['Elbow up - Torque ' num2str(j)]);
    plot(time, torques2(:, j), '--', 'LineWidth', 2, 'DisplayName', ['Elbow down - Torque ' num2str(j)]);
end
xlabel('Time (s)', 'FontSize', 14, 'FontWeight', 'bold');
ylabel('Torque (Nm)', 'FontSize', 14, 'FontWeight', 'bold');
title('Torques at Joints over Time', 'FontSize', 16, 'FontWeight', 'bold');
legend('show', 'FontSize', 12, 'FontWeight', 'bold');
grid on;
set(gca, 'FontSize', 12, 'FontWeight', 'bold');
hold off;

% Subplot for energies
figure;
subplot(3,1,1);
hold on;
plot(time, kinetic_energy1, 'r', 'LineWidth', 2, 'DisplayName', 'Elbow up - Kinetic Energy');
plot(time, kinetic_energy2, '--r', 'LineWidth', 2, 'DisplayName', 'Elbow down - Kinetic Energy');
xlabel('Time (s)', 'FontSize', 14, 'FontWeight', 'bold');
ylabel('Kinetic Energy (J)', 'FontSize', 14, 'FontWeight', 'bold');
title('Kinetic Energy over Time', 'FontSize', 16, 'FontWeight', 'bold');
legend('show', 'FontSize', 12, 'FontWeight', 'bold');
grid on;
set(gca, 'FontSize', 12, 'FontWeight', 'bold');
hold off;

subplot(3,1,2);
hold on;
plot(time, potential_energy1, 'b', 'LineWidth', 2, 'DisplayName', 'Elbow up - Potential Energy');
plot(time, potential_energy2, '--b', 'LineWidth', 2, 'DisplayName', 'Elbow down - Potential Energy');
xlabel('Time (s)', 'FontSize', 14, 'FontWeight', 'bold');
ylabel('Potential Energy (J)', 'FontSize', 14, 'FontWeight', 'bold');
title('Potential Energy over Time', 'FontSize', 16, 'FontWeight', 'bold');
legend('show', 'FontSize', 12, 'FontWeight', 'bold');
grid on;
set(gca, 'FontSize', 12, 'FontWeight', 'bold');
hold off;

subplot(3,1,3);
hold on;
plot(time, total_energy1, 'k', 'LineWidth', 2, 'DisplayName', 'Elbow up - Total Energy');
plot(time, total_energy2, '--k', 'LineWidth', 2, 'DisplayName', 'Elbow down - Total Energy');
xlabel('Time (s)', 'FontSize', 14, 'FontWeight', 'bold');
ylabel('Total Energy (J)', 'FontSize', 14, 'FontWeight', 'bold');
title('Total Energy over Time', 'FontSize', 16, 'FontWeight', 'bold');
legend('show', 'FontSize', 12, 'FontWeight', 'bold');
grid on;
set(gca, 'FontSize', 12, 'FontWeight', 'bold');
hold off;

% Energy conservation validation for both configurations in a new figure
figure;
hold on;
energy_difference1 = abs(total_energy1 - total_energy1(1));
plot(time, energy_difference1, 'g', 'LineWidth', 2, 'DisplayName', 'Elbow up - Energy Difference');
energy_difference2 = abs(total_energy2 - total_energy2(1));
plot(time, energy_difference2, '--g', 'LineWidth', 2, 'DisplayName', 'Elbow down - Energy Difference');
xlabel('Time (s)', 'FontSize', 14, 'FontWeight', 'bold');
ylabel('Energy Difference (J)', 'FontSize', 14, 'FontWeight', 'bold');
title('Energy Conservation Validation', 'FontSize', 16, 'FontWeight', 'bold');
legend('show', 'FontSize', 12, 'FontWeight', 'bold');
grid on;
set(gca, 'FontSize', 12, 'FontWeight', 'bold');
hold off;

% Animate the robot for both configurations
figure;
subplot(1, 2, 1);
hold on;
grid on;
axis equal;
xlabel('X', 'FontSize', 14, 'FontWeight', 'bold');
ylabel('Y', 'FontSize', 14, 'FontWeight', 'bold');
zlabel('Z', 'FontSize', 14, 'FontWeight', 'bold');
title('End Effector Trajectory and Robot Animation for Elbow up', 'FontSize', 16, 'FontWeight', 'bold');
view(3);
robot_plot1 = plot3(0, 0, 0, '-o', 'LineWidth', 2, 'MarkerSize', 10);
end_effector_plot1 = plot3(0, 0, 0, 'r*', 'MarkerSize', 10);
base_label1 = text(-0.1, 0, 0, 'Base', 'FontSize', 14, 'FontWeight', 'bold', 'Color', 'blue', 'HorizontalAlignment', 'right');
end_effector_label1 = text(-0.1, 0, 0, 'End Effector', 'FontSize', 14, 'FontWeight', 'bold', 'Color', 'red', 'HorizontalAlignment', 'right');

% Plot the end effector trajectory for Elbow up
plot3(end_effector_positions1(:, 1), end_effector_positions1(:, 2), end_effector_positions1(:, 3), 'k--', 'LineWidth', 2);

subplot(1, 2, 2);
hold on;
grid on;
axis equal;
xlabel('X', 'FontSize', 14, 'FontWeight', 'bold');
ylabel('Y', 'FontSize', 14, 'FontWeight', 'bold');
zlabel('Z', 'FontSize', 14, 'FontWeight', 'bold');
title('End Effector Trajectory and Robot Animation for Elbow down', 'FontSize', 16, 'FontWeight', 'bold');
view(3);
robot_plot2 = plot3(0, 0, 0, '-o', 'LineWidth', 2, 'MarkerSize', 10);
end_effector_plot2 = plot3(0, 0, 0, 'r*', 'MarkerSize', 10);
base_label2 = text(-0.1, 0, 0, 'Base', 'FontSize', 14, 'FontWeight', 'bold', 'Color', 'blue', 'HorizontalAlignment', 'right');
end_effector_label2 = text(-0.1, 0, 0, 'End Effector', 'FontSize', 14, 'FontWeight', 'bold', 'Color', 'red', 'HorizontalAlignment', 'right');

% Plot the end effector trajectory for Elbow down
plot3(end_effector_positions2(:, 1), end_effector_positions2(:, 2), end_effector_positions2(:, 3), 'k--', 'LineWidth', 2);

% Animate both configurations
for i = 1:num_steps
    % Elbow up
    T_matrices1 = robot_configurations1(i, :);
    points1 = [0, 0, 0; T_matrices1{1}(1:3, 4)'; T_matrices1{2}(1:3, 4)'; T_matrices1{3}(1:3, 4)'; T_matrices1{4}(1:3, 4)'; T_matrices1{5}(1:3, 4)'];
    set(robot_plot1, 'XData', points1(:, 1), 'YData', points1(:, 2), 'ZData', points1(:, 3));
    set(end_effector_plot1, 'XData', points1(end, 1), 'YData', points1(end, 2), 'ZData', points1(end, 3));
    set(end_effector_label1, 'Position', [points1(end, 1) - 0.1, points1(end, 2), points1(end, 3)]);

    % Elbow down
    T_matrices2 = robot_configurations2(i, :);
    points2 = [0, 0, 0; T_matrices2{1}(1:3, 4)'; T_matrices2{2}(1:3, 4)'; T_matrices2{3}(1:3, 4)'; T_matrices2{4}(1:3, 4)'; T_matrices2{5}(1:3, 4)'];
    set(robot_plot2, 'XData', points2(:, 1), 'YData', points2(:, 2), 'ZData', points2(:, 3));
    set(end_effector_plot2, 'XData', points2(end, 1), 'YData', points2(end, 2), 'ZData', points2(end, 3));
    set(end_effector_label2, 'Position', [points2(end, 1) - 0.1, points2(end, 2), points2(end, 3)]);

    drawnow;
end

% Print end effector positions
disp('End Effector Positions for Elbow up:');
disp(end_effector_positions1);

disp('End Effector Positions for Elbow down:');
disp(end_effector_positions2);

fprintf(" Writing new ")

% Print initial and final torques and remaining parameters for Configuration 1
fprintf('Configuration 1 - Initial Torque: [%f, %f, %f, %f, %f] Nm\n', torques1(1, :));
fprintf('Configuration 1 - Final Torque: [%f, %f, %f, %f, %f] Nm\n', torques1(end, :));

% Print initial and final torques and remaining parameters for Configuration 2
fprintf('Configuration 2 - Initial Torque: [%f, %f, %f, %f, %f] Nm\n', torques2(1, :));
fprintf('Configuration 2 - Final Torque: [%f, %f, %f, %f, %f] Nm\n', torques2(end, :));


% Print peak torque values for Configuration 1 with original signs
for j = 1:5
    [peak_torque1_value, idx] = max(abs(torques1(:, j)));
    peak_torque1 = torques1(idx, j);
    fprintf('Configuration 1 - Joint %d Peak Torque: %f Nm\n', j, peak_torque1);
end

% Print peak torque values for Configuration 2 with original signs
for j = 1:5
    [peak_torque2_value, idx] = max(abs(torques2(:, j)));
    peak_torque2 = torques2(idx, j);
    fprintf('Configuration 2 - Joint %d Peak Torque: %f Nm\n', j, peak_torque2);
end

function [theta_traj, dtheta_traj, ddtheta_traj, kinetic_energy, potential_energy, total_energy, torques, end_effector_positions, robot_configurations, coeffs] = compute_trajectory(final_config, t_final, num_steps, time, q, T5, T_func, V_func, E_func, tau_func, A1, A2, A3, A4, A5)
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
