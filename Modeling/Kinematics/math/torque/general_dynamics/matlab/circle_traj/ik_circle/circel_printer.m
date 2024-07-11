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

% Define the circular path around the 3D printer bed
bed_center = [150, 150, 200];  % Center of the bed (300x300x400 bed dimensions)
radius = 200;  % Radius of the circle around the bed
height = 200;  % Height at which the camera will move

num_points = 100;  % Number of points in the circular path
theta_circle = linspace(0, 2*pi, num_points);
circle_path = zeros(num_points, 3);
for i = 1:num_points
    circle_path(i, 1) = bed_center(1) + radius * cos(theta_circle(i));
    circle_path(i, 2) = bed_center(2) + radius * sin(theta_circle(i));
    circle_path(i, 3) = height;
end

% Compute inverse kinematics for each point on the circular path
q_traj = zeros(num_points, 5);
for i = 1:num_points
    Px = circle_path(i, 1);
    Py = circle_path(i, 2);
    Pz = circle_path(i, 3);
    
    % Solve inverse kinematics
    theta1 = atan2(Py, Px);
    r = sqrt(Px^2 + Py^2);
    D = (r^2 + (Pz - d1)^2 - a2^2 - a3^2) / (2 * a2 * a3);
    theta3 = atan2(sqrt(1 - D^2), D);
    theta2 = atan2(Pz - d1, r) - atan2(a3 * sin(theta3), a2 + a3 * cos(theta3));
    theta4 = -theta2 - theta3;
    theta5 = 0;  % Assume no rotation around the end-effector's axis
    
    q_traj(i, :) = [theta1, theta2, theta3, theta4, theta5];
end

% Simulate the trajectory
t_final = 10;  % Duration of the simulation
time = linspace(0, t_final, num_points);

% Function to compute trajectory, energies, and torques
[theta_traj, dtheta_traj, ddtheta_traj, kinetic_energy, potential_energy, total_energy, torques, end_effector_positions, robot_configurations] = compute_trajectory(q_traj, t_final, num_points, time, q, T5, T_func, V_func, E_func, tau_func, A1, A2, A3, A4, A5);

% Plot the circular path and animate the robot
figure;
hold on;
grid on;
axis equal;
xlabel('X');
ylabel('Y');
zlabel('Z');
title('End Effector Trajectory and Robot Animation');
view(3);

% Plot the 3D printer bed
plot3([0, 300, 300, 0, 0], [0, 0, 300, 300, 0], [0, 0, 0, 0, 0], 'k-', 'LineWidth', 2);  % Bed outline
plot3([0, 300, 300, 0, 0], [0, 0, 300, 300, 0], [400, 400, 400, 400, 400], 'k-', 'LineWidth', 2);  % Top of the bed
for i = 1:4
    plot3([0, 0], [0, 0], [0, 400], 'k-', 'LineWidth', 2);  % Vertical lines at the corners
end

% Plot the circular path
plot3(circle_path(:, 1), circle_path(:, 2), circle_path(:, 3), 'r--', 'LineWidth', 2);

% Initialize plot for the robot
robot_plot = plot3(0, 0, 0, '-o', 'LineWidth', 2, 'MarkerSize', 10);
end_effector_plot = plot3(0, 0, 0, 'r*', 'MarkerSize', 10);

% Animate the robot along the circular path
for i = 1:num_points
    T_matrices = robot_configurations(i, :);
    points = [0, 0, 0; T_matrices{1}(1:3, 4)'; T_matrices{2}(1:3, 4)'; T_matrices{3}(1:3, 4)'; T_matrices{4}(1:3, 4)'; T_matrices{5}(1:3, 4)'];
    set(robot_plot, 'XData', points(:, 1), 'YData', points(:, 2), 'ZData', points(:, 3));
    set(end_effector_plot, 'XData', points(end, 1), 'YData', points(end, 2), 'ZData', points(end, 3));
    drawnow;
end

% Print out values at 10 points along the trajectory for position and angles
disp('Position and Angles at 10 points:');
indices = round(linspace(1, num_points, 10));
for i = 1:10
    idx = indices(i);
    fprintf('Point %d:\n', i);
    fprintf('End Effector Position: [%f, %f, %f]\n', end_effector_positions(idx, :));
    fprintf('Joint Angles (deg): [%f, %f, %f, %f, %f]\n', rad2deg(theta_traj(idx, :)));
end

% Print out maximum and minimum torque values for all joints
for j = 1:5
    [max_torque, max_torque_idx] = max(torques(:, j));
    [min_torque, min_torque_idx] = min(torques(:, j));
    fprintf('Joint %d Maximum Torque: %f Nm at time %f s\n', j, max_torque, time(max_torque_idx));
    fprintf('Joint %d Minimum Torque: %f Nm at time %f s\n', j, min_torque, time(min_torque_idx));
end

% Print out initial and final torques and remaining parameters
fprintf('Initial Torque: [%f, %f, %f, %f, %f] Nm\n', torques(1, :));
fprintf('Final Torque: [%f, %f, %f, %f, %f] Nm\n', torques(end, :));
fprintf('Initial Kinetic Energy: %f J\n', kinetic_energy(1));
fprintf('Final Kinetic Energy: %f J\n', kinetic_energy(end));
fprintf('Initial Potential Energy: %f J\n', potential_energy(1));
fprintf('Final Potential Energy: %f J\n', potential_energy(end));

% Function to compute trajectory, energies, and torques for a given final configuration
function [theta_traj, dtheta_traj, ddtheta_traj, kinetic_energy, potential_energy, total_energy, torques, end_effector_positions, robot_configurations] = compute_trajectory(q_traj, t_final, num_steps, time, q, T5, T_func, V_func, E_func, tau_func, A1, A2, A3, A4, A5)
    theta_traj = q_traj;
    dtheta_traj = gradient(theta_traj, time);
    ddtheta_traj = gradient(dtheta_traj, time);

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
