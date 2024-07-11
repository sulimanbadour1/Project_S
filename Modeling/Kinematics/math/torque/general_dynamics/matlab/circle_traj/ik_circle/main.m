
% Main script to simulate the trajectory
% Define the circular trajectory parameters
radius = 0.15;  % Radius of the circle around the model
height = 0.2;   % Height at which the camera circles around the model
center = [0.15, 0.15, height];  % Center of the circular path
num_points = 100;  % Number of points to define the circle
omega = 90;  % Camera orientation angle

% Generate points on the circular path
theta_circle = linspace(0, 2*pi, num_points);
x = center(1) + radius * cos(theta_circle);
y = center(2) + radius * sin(theta_circle);
z = center(3) * ones(1, num_points);

% Initialize the robot arm
robot = RobotArm(0.1, 0.5, 0.5, 0.1);

% Calculate joint angles for each target position
joint_angles = zeros(4, num_points);
for i = 1:num_points
    [theta1, theta2, theta3, theta4] = robot.inverse_kinematics(x(i), y(i), z(i), omega);
    joint_angles(:, i) = [theta1; theta2; theta3; theta4];
end

% Compute joint positions for each set of joint angles
joint_positions_all = zeros(num_points, 5, 3);
for i = 1:num_points
    joint_positions_all(i, :, :) = robot.forward_kinematics(joint_angles(1, i), joint_angles(2, i), joint_angles(3, i), joint_angles(4, i));
end

% Animate the robot for the circular trajectory
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
plot3(x, y, z, 'k--', 'LineWidth', 2);

for i = 1:num_points
    joint_positions = squeeze(joint_positions_all(i, :, :));
    plot3(joint_positions(:, 1), joint_positions(:, 2), joint_positions(:, 3), 'o-', 'MarkerSize', 10, 'DisplayName', 'Robot Arm');
    scatter3(joint_positions(:, 1), joint_positions(:, 2), joint_positions(:, 3), 'k');
    drawnow;
    pause(0.1);
end

disp('Simulation complete.');

% Calculate torques, energies, and other parameters
masses = [1.0, 1.0, 1.0, 1.0, 1.0];
mass_camera = 0.5;
mass_lights = 0.5;
g = 9.81;  % Gravitational acceleration

% Inertia matrices for each link (simplified as diagonal matrices for demonstration)
I1 = eye(3) * (1/12) * masses(1) * (robot.d1^2);
I2 = eye(3) * (1/12) * masses(2) * (robot.a2^2);
I3 = eye(3) * (1/12) * masses(3) * (robot.a3^2);
I4 = eye(3) * (1/12) * masses(4) * (robot.d1^2);
I5 = eye(3) * (1/12) * (masses(5) + mass_camera + mass_lights) * (robot.d5^2);

% Placeholder for Jacobian and mass matrix computations
Jv1 = eye(3, 4);
Jv2 = eye(3, 4);
Jv3 = eye(3, 4);
Jv4 = eye(3, 4);
Jv5 = eye(3, 4);

M = Jv1' * masses(1) * Jv1 + Jv2' * masses(2) * Jv2 + Jv3' * masses(3) * Jv3 + ...
    Jv4' * masses(4) * Jv4 + Jv5' * (masses(5) + mass_camera + mass_lights) * Jv5;

% Placeholder for energy calculations
kinetic_energy = zeros(num_points, 1);
potential_energy = zeros(num_points, 1);
total_energy = zeros(num_points, 1);
torques = zeros(num_points, 4);

for i = 1:num_points
    q_vals = joint_angles(:, i);
    dq_vals = zeros(4, 1);  % Placeholder for joint velocities
    ddq_vals = zeros(4, 1); % Placeholder for joint accelerations

    % Ensure M has the correct dimensions
    M = eye(4); % Update this with the actual mass matrix for your system

    % Compute energies and torques
    kinetic_energy(i) = 0.5 * dq_vals' * M * dq_vals;
    potential_energy(i) = -masses(1) * g * robot.d1;  % Simplified example
    total_energy(i) = kinetic_energy(i) + potential_energy(i);
    torques(i, :) = (M * ddq_vals + Jv1' * masses(1) * [0; 0; -g; 0])';
end

% Plot energies and torques
figure;
subplot(3, 1, 1);
plot(1:num_points, kinetic_energy, 'r', 'LineWidth', 2);
xlabel('Time Step');
ylabel('Kinetic Energy (J)');
title('Kinetic Energy');

subplot(3, 1, 2);
plot(1:num_points, potential_energy, 'b', 'LineWidth', 2);
xlabel('Time Step');
ylabel('Potential Energy (J)');
title('Potential Energy');

subplot(3, 1, 3);
plot(1:num_points, total_energy, 'k', 'LineWidth', 2);
xlabel('Time Step');
ylabel('Total Energy (J)');
title('Total Energy');

figure;
hold on;
for j = 1:4
    plot(1:num_points, torques(:, j), 'LineWidth', 2, 'DisplayName', ['Torque ' num2str(j)]);
end
xlabel('Time Step');
ylabel('Torque (Nm)');
title('Torques at Joints');
legend;
grid on;

disp('Energy and torque calculations complete.');