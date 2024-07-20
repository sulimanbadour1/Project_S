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

% Define sampling ranges for joint positions, velocities, and accelerations
num_samples = 10; % Reduced number of samples for simplicity
theta_samples = linspace(-pi, pi, num_samples);
dtheta_samples = linspace(-2, 2, num_samples);
ddtheta_samples = linspace(-2, 2, num_samples);

% Initialize maximum torque array and store configuration
max_torques = zeros(5, 1);
max_config = [];

% Iterate over samples to find maximum torques
for i = 1:num_samples
    for j = 1:num_samples
        for k = 1:num_samples
            q_vals = [theta_samples(i); theta_samples(i); theta_samples(i); theta_samples(i); theta_samples(i)];
            dq_vals = [dtheta_samples(j); dtheta_samples(j); dtheta_samples(j); dtheta_samples(j); dtheta_samples(j)];
            ddq_vals = [ddtheta_samples(k); ddtheta_samples(k); ddtheta_samples(k); ddtheta_samples(k); ddtheta_samples(k)];
            
            tau_vals = tau_func(q_vals, dq_vals, ddq_vals);
            if any(abs(tau_vals) > max_torques)
                max_torques = max(max_torques, abs(tau_vals));
                max_config = [q_vals; dq_vals; ddq_vals];
            end
        end
    end
end

% Display maximum torques
disp('Maximum Dynamic Torques (Nm):');
disp(max_torques);

% Extract the configuration for the maximum torque
max_q = max_config(1:5);
max_dq = max_config(6:10);
max_ddq = max_config(11:15);

% Simulate the robot movement for the configuration with maximum torque
num_steps = 100;
time = linspace(0, 10, num_steps); % Simulation time 10 seconds
theta_traj = repmat(max_q', num_steps, 1);
dtheta_traj = repmat(max_dq', num_steps, 1);
ddtheta_traj = repmat(max_ddq', num_steps, 1);

% Compute energies and torques
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
    
    T_current = double(subs(T5, {theta1, theta2, theta3, theta4, theta5}, q_vals.'));
    end_effector_positions(i, :) = T_current(1:3, 4);
    % Store current configuration
    robot_configurations{i, 1} = double(subs(A1, theta1, q_vals(1)));
    robot_configurations{i, 2} = double(robot_configurations{i, 1} * subs(A2, theta2, q_vals(2)));
    robot_configurations{i, 3} = double(robot_configurations{i, 2} * subs(A3, theta3, q_vals(3)));
    robot_configurations{i, 4} = double(robot_configurations{i, 3} * subs(A4, theta4, q_vals(4)));
    robot_configurations{i, 5} = double(robot_configurations{i, 4} * subs(A5, theta5, q_vals(5)));
end

% Create figure for animation
figure;

% Subplot for robot animation
subplot(3, 2, 1);
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
base_label = text(0, 0, 0, 'Base', 'FontSize', 12, 'FontWeight', 'bold', 'Color', 'blue');
end_effector_label = text(0, 0, 0, 'End Effector', 'FontSize', 12, 'FontWeight', 'bold', 'Color', 'red');

% Plot the end effector trajectory
plot3(end_effector_positions(:, 1), end_effector_positions(:, 2), end_effector_positions(:, 3), 'k--', 'LineWidth', 2);

% Initialize plots for joint angles, velocities, energies, and torques
subplot(3, 2, 2);
angle_plots = plot(nan(num_steps, 5), 'LineWidth', 2);
xlabel('Time (s)');
ylabel('Angle (rad)');
title('Joint Angles');
grid on;

subplot(3, 2, 3);
velocity_plots = plot(nan(num_steps, 5), 'LineWidth', 2);
xlabel('Time (s)');
ylabel('Velocity (rad/s)');
title('Joint Velocities');
grid on;

subplot(3, 2, 4);
energy_plots = plot(nan(num_steps, 3), 'LineWidth', 2);
legend('Kinetic Energy', 'Potential Energy', 'Total Energy');
xlabel('Time (s)');
ylabel('Energy (J)');
title('Energies');
grid on;

subplot(3, 2, 5);
torque_plots = plot(nan(num_steps, 5), 'LineWidth', 2);
xlabel('Time (s)');
ylabel('Torque (Nm)');
title('Torques');
grid on;

% Animate the robot and update plots
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
    
    % Update joint angles plot
    for j = 1:5
        set(angle_plots(j), 'XData', time(1:i), 'YData', theta_traj(1:i, j));
    end
    
    % Update joint velocities plot
    for j = 1:5
        set(velocity_plots(j), 'XData', time(1:i), 'YData', dtheta_traj(1:i, j));
    end
    
    % Update energies plot
    set(energy_plots(1), 'XData', time(1:i), 'YData', kinetic_energy(1:i));
    set(energy_plots(2), 'XData', time(1:i), 'YData', potential_energy(1:i));
    set(energy_plots(3), 'XData', time(1:i), 'YData', total_energy(1:i));
    
    % Update torques plot
    for j = 1:5
        set(torque_plots(j), 'XData', time(1:i), 'YData', torques(1:i, j));
    end
    
    drawnow;
end

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
