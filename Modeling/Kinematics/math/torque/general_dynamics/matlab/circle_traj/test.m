% Define DH parameters
d1 = 0.1;  % Link offset
d5 = 0.1;  % Link offset
a2 = 0.6;  % Link length
a3 = 0.6;  % Link length
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

% Define the simulation parameters
t_final = 60;  % Total simulation time in seconds
num_points = 50;  % Number of points along the trajectory
time = linspace(0, t_final, num_points);  % Time vector

% Define 3D Printer Dimensions and Position
printer_width = 0.3;  % 300 mm
printer_depth = 0.3;  % 300 mm
printer_height = 0.4; % 400 mm
printer_offset = 0.1; % 100 mm from the robot

% Define the new center of the circle path, avoiding the 3D printer
circle_radius = 0.32;  % Radius of the circle
circle_center = [0.5 + printer_offset, 0.5 + printer_depth/2, 0.2];  % Center of the circle

% Adjusting the theta_circle for a circular trajectory
theta_circle = linspace(0, 2*pi, num_points);

% New circle positions avoiding the 3D printer
circle_positions = circle_center + [circle_radius * cos(theta_circle)', circle_radius * sin(theta_circle)', zeros(num_points, 1)];

% Initialize arrays for storing joint angles
theta_traj = zeros(num_points, 5);
dtheta_traj = zeros(num_points, 5);
ddtheta_traj = zeros(num_points, 5);

% Solve for joint angles to follow the circle trajectory
initial_guess = [0, pi/6, -pi/6, -pi/6, 0];  % Initial guess for the solver
for i = 1:num_points
    target_position = circle_positions(i, :)';
    constraint = theta2 + theta3 + theta4 == 0;
    
    eqns = [T5(1:3, 4) == target_position; constraint];
    
    sol = vpasolve(eqns, [theta1, theta2, theta3, theta4, theta5], initial_guess);
    
    if isempty(sol.theta1)
        fprintf('No solution found for point %d. Using initial guess.\n', i);
        theta_traj(i, :) = initial_guess;  % Use initial guess
    else
        theta_traj(i, :) = double([sol.theta1, sol.theta2, sol.theta3, sol.theta4, sol.theta5]);
        initial_guess = theta_traj(i, :);  % Update initial guess for next iteration
    end
end

% Compute joint velocities and accelerations using finite differences
for i = 2:num_points-1
    dtheta_traj(i, :) = (theta_traj(i+1, :) - theta_traj(i-1, :)) / (2 * (t_final / num_points));
    ddtheta_traj(i, :) = (theta_traj(i+1, :) - 2 * theta_traj(i, :) + theta_traj(i-1, :)) / ((t_final / num_points)^2);
end

% Initialize energy and torque arrays
kinetic_energy = zeros(num_points, 1);
potential_energy = zeros(num_points, 1);
total_energy = zeros(num_points, 1);
torques = zeros(num_points, 5);

% Compute end effector positions for the trajectory
end_effector_positions = zeros(num_points, 3);
robot_configurations = cell(num_points, 5);  % Store robot configurations for plotting
for i = 1:num_points
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

% Animate the robot for the circle trajectory
figure;
hold on;
grid on;
axis equal;
xlabel('X');
ylabel('Y');
zlabel('Z');
title('End Effector Trajectory and Robot Animation for Circular Motion');
view(3);
robot_plot = plot3(0, 0, 0, '-o', 'LineWidth', 2, 'MarkerSize', 10);
end_effector_plot = plot3(0, 0, 0, 'r*', 'MarkerSize', 10);
base_label = text(-0.1, 0, 0, 'Base', 'FontSize', 12, 'FontWeight', 'bold', 'Color', 'blue', 'HorizontalAlignment', 'right');
end_effector_label = text(-0.1, 0, 0, 'End Effector', 'FontSize', 12, 'FontWeight', 'bold', 'Color', 'red', 'HorizontalAlignment', 'right');

% Plot the end effector trajectory
plot3(end_effector_positions(:, 1), end_effector_positions(:, 2), end_effector_positions(:, 3), 'k--', 'LineWidth', 2);

% Animate the trajectory
for i = 1:num_points
    T_matrices = robot_configurations(i, :);
    points = [0, 0, 0; T_matrices{1}(1:3, 4)'; T_matrices{2}(1:3, 4)'; T_matrices{3}(1:3, 4)'; T_matrices{4}(1:3, 4)'; T_matrices{5}(1:3, 4)'];
    set(robot_plot, 'XData', points(:, 1), 'YData', points(:, 2), 'ZData', points(:, 3));
    set(end_effector_plot, 'XData', points(end, 1), 'YData', points(end, 2), 'ZData', points(end, 3));
    set(end_effector_label, 'Position', [points(end, 1) - 0.1, points(end, 2), points(end, 3)]);

    drawnow;
end

% Print end effector positions
disp('End Effector Positions:');
disp(end_effector_positions);

% Print initial and final torques and remaining parameters
fprintf('Initial Torque: [%f, %f, %f, %f, %f] Nm\n', torques(1, :));
fprintf('Final Torque: [%f, %f, %f, %f, %f] Nm\n', torques(end, :));
fprintf('Initial Kinetic Energy: %f J\n', kinetic_energy(1));
fprintf('Final Kinetic Energy: %f J\n', kinetic_energy(end));
fprintf('Initial Potential Energy: %f J\n', potential_energy(1));
fprintf('Final Potential Energy: %f J\n', potential_energy(end));

% Extract peak values
fprintf('Peak Kinetic Energy: %f J\n', max(kinetic_energy));
fprintf('Peak Potential Energy: %f J\n', max(potential_energy));
fprintf('Peak Total Energy: %f J\n', max(total_energy));

% Print peak torque values with original signs
for j = 1:5
    [peak_torque_value, idx] = max(abs(torques(:, j)));
    peak_torque = torques(idx, j);
    fprintf('Joint %d Peak Torque: %f Nm\n', j, peak_torque);
end

% Plot joint angles, velocities, accelerations, torques, and energies
figure;
subplot(3, 1, 1);
hold on;
for j = 1:5
    plot(time, theta_traj(:, j), 'LineWidth', 2, 'DisplayName', ['Theta ' num2str(j)]);
end
xlabel('Time (s)');
ylabel('Joint Angles (rad)');
title('Joint Angles over Time');
legend('show');
grid on;

subplot(3, 1, 2);
hold on;
for j = 1:5
    plot(time, dtheta_traj(:, j), 'LineWidth', 2, 'DisplayName', ['dTheta ' num2str(j)]);
end
xlabel('Time (s)');
ylabel('Joint Velocities (rad/s)');
title('Joint Velocities over Time');
legend('show');
grid on;

subplot(3, 1, 3);
hold on;
for j = 1:5
    plot(time, ddtheta_traj(:, j), 'LineWidth', 2, 'DisplayName', ['ddTheta ' num2str(j)]);
end
xlabel('Time (s)');
ylabel('Joint Accelerations (rad/s^2)');
title('Joint Accelerations over Time');
legend('show');
grid on;

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

% Energy conservation validation
figure;
energy_difference = abs(total_energy - total_energy(1));
plot(time, energy_difference, 'g', 'LineWidth', 2);
xlabel('Time (s)');
ylabel('Energy Difference (J)');
title('Energy Conservation Validation');
grid on;
