% Example parameters
Px = 0.5; Py = 0; Pz = 0.5; omega = 0;

% Define DH parameters
d1 = 0.1;
a2 = 0.5;
a3 = 0.5;
d5 = 0.1;

robot_arm = RobotArm(d1, a2, a3, d5);
[theta1, theta2, theta3, theta4] = robot_arm.inverse_kinematics(Px, Py, Pz, omega);
joint_positions = robot_arm.forward_kinematics(theta1, theta2, theta3, theta4);
robot_arm.plot_robot(joint_positions);

fprintf('Theta1: %.2f degrees\n', theta1);
fprintf('Theta2: %.2f degrees\n', theta2);
fprintf('Theta3: %.2f degrees\n', theta3);
fprintf('Theta4: %.2f degrees\n', theta4);
fprintf('Joint positions: \n');
disp(joint_positions);
fprintf('End effector position: [%.2f, %.2f, %.2f]\n', joint_positions(end, 1), joint_positions(end, 2), joint_positions(end, 3));

% Convert to radians
theta1 = deg2rad(theta1);
theta2 = deg2rad(theta2);
theta3 = deg2rad(theta3);
theta4 = deg2rad(theta4);

fprintf('Theta1: %.2f radians, Theta2: %.2f radians, Theta3: %.2f radians, Theta4: %.2f radians\n', theta1, theta2, theta3, theta4);
