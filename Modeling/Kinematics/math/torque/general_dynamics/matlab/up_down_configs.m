
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

% Positions of the end effector
end_effector_1 = zeros(3, num_steps);
end_effector_2 = zeros(3, num_steps);

% Simulation time
t_final = 10;  % Final time in seconds
num_steps = 100;  % Number of time steps
time = linspace(0, t_final, num_steps);

% Quintic polynomial trajectory coefficients for each joint
coeffs_1 = zeros(6, 5);
coeffs_2 = zeros(6, 5);
for i = 1:5
    coeffs_1(:, i) = polyfit([0, t_final], [initial_config(i), final_config_1(i)], 5);
    coeffs_2(:, i) = polyfit([0, t_final], [initial_config(i), final_config_2(i)], 5);
end

% Compute joint angles and velocities using polynomial derivatives
theta_traj_1 = zeros(num_steps, 5);
dtheta_traj_1 = zeros(num_steps, 5);
theta_traj_2 = zeros(num_steps, 5);
dtheta_traj_2 = zeros(num_steps, 5);
for i = 1:num_steps
    t = time(i);
    for j = 1:5
        theta_traj_1(i, j) = polyval(coeffs_1(:, j), t);
        dtheta_traj_1(i, j) = polyval(polyder(coeffs_1(:, j)), t);
        theta_traj_2(i, j) = polyval(coeffs_2(:, j), t);
        dtheta_traj_2(i, j) = polyval(polyder(coeffs_2(:, j)), t);
    end
end

% Initialize energy and torque arrays
kinetic_energy_1 = zeros(num_steps, 1);
potential_energy_1 = zeros(num_steps, 1);
total_energy_1 = zeros(num_steps, 1);
torques_1 = zeros(num_steps, 5);

kinetic_energy_2 = zeros(num_steps, 1);
potential_energy_2 = zeros(num_steps, 1);
total_energy_2 = zeros(num_steps, 1);
torques_2 = zeros(num_steps, 5);

% Calculate energies and torques for both configurations
for i = 1:num_steps
    q_vals_1 = theta_traj_1(i, :)';
    dq_vals_1 = dtheta_traj_1(i, :)';
    q_vals_2 = theta_traj_2(i, :)';
    dq_vals_2 = dtheta_traj_2(i, :)';
    
    kinetic_energy_1(i) = T_func(q_vals_1, dq_vals_1);
    potential_energy_1(i) = V_func(q_vals_1);
    total_energy_1(i) = E_func(q_vals_1, dq_vals_1);
    torques_1(i, :) = tau_func(q_vals_1, dq_vals_1);
    
    kinetic_energy_2(i) = T_func(q_vals_2, dq_vals_2);
    potential_energy_2(i) = V_func(q_vals_2);
    total_energy_2(i) = E_func(q_vals_2, dq_vals_2);
    torques_2(i, :) = tau_func(q_vals_2, dq_vals_2);
    
    % End effector positions
    end_effector_1(:, i) = double(subs(T5(1:3, 4), q, q_vals_1));
    end_effector_2(:, i) = double(subs(T5(1:3, 4), q, q_vals_2));
end

% Plot end effector trajectories
figure;
plot3(end_effector_1(1, :), end_effector_1(2, :), end_effector_1(3, :), 'r--', 'LineWidth', 2);
hold on;
plot3(end_effector_2(1, :), end_effector_2(2, :), end_effector_2(3, :), 'b--', 'LineWidth', 2);
xlabel('X');
ylabel('Y');
zlabel('Z');
title('End Effector Trajectories');
legend('Trajectory 1', 'Trajectory 2');
grid on;

% Plot energies
figure;
subplot(3, 1, 1);
plot(time, kinetic_energy_1, 'r', 'LineWidth', 2);
hold on;
plot(time, kinetic_energy_2, 'r--', 'LineWidth', 2);
xlabel('Time (s)');
ylabel('Kinetic Energy (J)');
title('Kinetic Energy over Time');
legend('Trajectory 1', 'Trajectory 2');
grid on;

subplot(3, 1, 2);
plot(time, potential_energy_1, 'b', 'LineWidth', 2);
hold on;
plot(time, potential_energy_2, 'b--', 'LineWidth', 2);
xlabel('Time (s)');
ylabel('Potential Energy (J)');
title('Potential Energy over Time');
legend('Trajectory 1', 'Trajectory 2');
grid on;

subplot(3, 1, 3);
plot(time, total_energy_1, 'k', 'LineWidth', 2);
hold on;
plot(time, total_energy_2, 'k--', 'LineWidth', 2);
xlabel('Time (s)');
ylabel('Total Energy (J)');
title('Total Energy over Time');
legend('Trajectory 1', 'Trajectory 2');
grid on;

% Plot torques
figure;
for j = 1:5
    subplot(5, 1, j);
    plot(time, torques_1(:, j), 'LineWidth', 2);
    hold on;
    plot(time, torques_2(:, j), 'LineWidth', 2);
    xlabel('Time (s)');
    ylabel(['Torque ' num2str(j) ' (Nm)']);
    title(['Torque at Joint ' num2str(j) ' over Time']);
    legend('Trajectory 1', 'Trajectory 2');
    grid on;
end