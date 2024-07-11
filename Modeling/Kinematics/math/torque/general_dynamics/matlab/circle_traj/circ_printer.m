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

% Define circular path parameters
radius = 0.2;  % Radius of the circular path around the printer bed in meters
height = 0.2;  % Height of the circular path from the base of the printer bed in meters
center = [0.3, 0.3, height];  % Center of the printer bed in meters
num_points = 100;  % Number of waypoints in the circular path
angle_step = 2 * pi / num_points;  % Angle step for each waypoint

% Initialize arrays for storing waypoints
waypoints = zeros(num_points, 3);

% Generate waypoints for the circular path
for i = 1:num_points
    angle = (i - 1) * angle_step;
    waypoints(i, 1) = center(1) + radius * cos(angle);
    waypoints(i, 2) = center(2) + radius * sin(angle);
    waypoints(i, 3) = center(3);
end

% Define initial guess for joint angles
initial_guess = [0, pi/6, -pi/6, -pi/6, 0];

% Initialize arrays to store joint angles for the trajectory
theta_traj = zeros(num_points, 5);

% Solve inverse kinematics for each waypoint
for i = 1:num_points
    target_position = waypoints(i, :).';
    eqns = [T5(1:3, 4) == target_position; constraint];
    sol = vpasolve(eqns, vars, initial_guess);
    theta_traj(i, :) = double([sol.theta1, sol.theta2, sol.theta3, sol.theta4, sol.theta5]);
    initial_guess = theta_traj(i, :);  % Update initial guess for next iteration
end

% Compute joint velocities and accelerations
time = linspace(0, t_final, num_points);
dtheta_traj = zeros(size(theta_traj));
ddtheta_traj = zeros(size(theta_traj));
for j = 1:5
    dtheta_traj(:, j) = gradient(theta_traj(:, j), time);
    ddtheta_traj(:, j) = gradient(dtheta_traj(:, j), time);
end

% Compute torques, energies, and end effector positions
[kinetic_energy, potential_energy, total_energy, torques, end_effector_positions, robot_configurations] = ...
    compute_trajectory(theta_traj, dtheta_traj, ddtheta_traj, time, q, T5, T_func, V_func, E_func, tau_func, A1, A2, A3, A4, A5);

% Extract values at 10 specific points
indices = round(linspace(1, num_points, 10));

fprintf('Extracted Values at 10 Points:\n');
for i = 1:10
    idx = indices(i);
    fprintf('Point %d (time = %.2f s):\n', i, time(idx));
    fprintf('  End Effector Position: [%f, %f, %f]\n', end_effector_positions(idx, :));
    fprintf('  Joint Angles (rad): [%f, %f, %f, %f, %f]\n', theta_traj(idx, :));
    fprintf('  Joint Velocities (rad/s): [%f, %f, %f, %f, %f]\n', dtheta_traj(idx, :));
    fprintf('  Joint Accelerations (rad/s^2): [%f, %f, %f, %f, %f]\n', ddtheta_traj(idx, :));
    fprintf('  Torques (Nm): [%f, %f, %f, %f, %f]\n', torques(idx, :));
end

% Plot joint angles over time
figure;
hold on;
for j = 1:5
    plot(time, rad2deg(theta_traj(:, j)), 'LineWidth', 2, 'DisplayName', ['Theta ' num2str(j)]);
end
xlabel('Time (s)');
ylabel('Joint Angles (deg)');
title('Joint Angles over Time');
legend('show');
grid on;

% Plot joint velocities over time
figure;
hold on;
for j = 1:5
    plot(time, dtheta_traj(:, j), 'LineWidth', 2, 'DisplayName', ['dTheta ' num2str(j)]);
end
xlabel('Time (s)');
ylabel('Joint Velocities (rad/s)');
title('Joint Velocities over Time');
legend('show');
grid on;

% Plot joint accelerations over time
figure;
hold on;
for j = 1:5
    plot(time, ddtheta_traj(:, j), 'LineWidth', 2, 'DisplayName', ['ddTheta ' num2str(j)]);
end
xlabel('Time (s)');
ylabel('Joint Accelerations (rad/s^2)');
title('Joint Accelerations over Time');
legend('show');
grid on;

% Plot torques over time
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

% Plot energies over time
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

% Plot the 3D trajectory of the end effector
figure;
plot3(waypoints(:, 1), waypoints(:, 2), waypoints(:, 3), 'k--', 'LineWidth', 2);
hold on;
plot3(end_effector_positions(:, 1), end_effector_positions(:, 2), end_effector_positions(:, 3), 'r', 'LineWidth', 2);
xlabel('X (m)');
ylabel('Y (m)');
zlabel('Z (m)');
title('3D Trajectory of the End Effector');
legend('Desired Path', 'Actual Path');
grid on;
axis equal;

% Save animation as a video
video_filename = 'robot_animation.avi';
video_writer = VideoWriter(video_filename);
open(video_writer);

% Animate the robot along the trajectory and write frames to video
figure;
hold on;
grid on;
axis equal;
xlabel('X');
ylabel('Y');
zlabel('Z');
title('Robot Animation');
view(3);
robot_plot = plot3(0, 0, 0, '-o', 'LineWidth', 2, 'MarkerSize', 10);
end_effector_plot = plot3(0, 0, 0, 'r*', 'MarkerSize', 10);
base_label = text(-0.1, 0, 0, 'Base', 'FontSize', 12, 'FontWeight', 'bold', 'Color', 'blue', 'HorizontalAlignment', 'right');
end_effector_label = text(-0.1, 0, 0, 'End Effector', 'FontSize', 12, 'FontWeight', 'bold', 'Color', 'red', 'HorizontalAlignment', 'right');

% Plot the end effector trajectory
plot3(end_effector_positions(:, 1), end_effector_positions(:, 2), end_effector_positions(:, 3), 'k--', 'LineWidth', 2);

for i = 1:num_points
    T_matrices = robot_configurations(i, :);
    points = [0, 0, 0; T_matrices{1}(1:3, 4)'; T_matrices{2}(1:3, 4)'; T_matrices{3}(1:3, 4)'; T_matrices{4}(1:3, 4)'; T_matrices{5}(1:3, 4)'];
    set(robot_plot, 'XData', points(:, 1), 'YData', points(:, 2), 'ZData', points(:, 3));
    set(end_effector_plot, 'XData', points(end, 1), 'YData', points(end, 2), 'ZData', points(end, 3));
    set(end_effector_label, 'Position', [points(end, 1) - 0.1, points(end, 2), points(end, 3)]);
    drawnow;

    % Capture frame and write to video
    frame = getframe(gcf);
    writeVideo(video_writer, frame);
end

close(video_writer);

fprintf('Animation saved as %s\n', video_filename);

% Function to compute trajectory, energies, and torques for a given configuration
function [kinetic_energy, potential_energy, total_energy, torques, end_effector_positions, robot_configurations] = compute_trajectory(theta_traj, dtheta_traj, ddtheta_traj, time, q, T5, T_func, V_func, E_func, tau_func, A1, A2, A3, A4, A5)
    num_steps = length(time);
    kinetic_energy = zeros(num_steps, 1);
    potential_energy = zeros(num_steps, 1);
    total_energy = zeros(num_steps, 1);
    torques = zeros(num_steps, 5);
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

        robot_configurations{i, 1} = double(subs(A1, q(1), q_vals(1)));
        robot_configurations{i, 2} = double(robot_configurations{i, 1} * subs(A2, q(2), q_vals(2)));
        robot_configurations{i, 3} = double(robot_configurations{i, 2} * subs(A3, q(3), q_vals(3)));
        robot_configurations{i, 4} = double(robot_configurations{i, 3} * subs(A4, q(4), q_vals(4)));
        robot_configurations{i, 5} = double(robot_configurations{i, 4} * subs(A5, q(5), q_vals(5)));
    end
end
