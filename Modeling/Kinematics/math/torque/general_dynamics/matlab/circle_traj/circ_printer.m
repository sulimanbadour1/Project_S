classdef RobotArm
    properties
        d1
        a2
        a3
        d5
    end

    methods
        function obj = RobotArm(d1, a2, a3, d5)
            obj.d1 = d1;
            obj.a2 = a2;
            obj.a3 = a3;
            obj.d5 = d5;
        end

        function [theta1, theta2, theta3, theta4] = inverse_kinematics(obj, Px, Py, Pz, omega)
            R = obj.d5 * cosd(omega);
            theta1 = atan2d(Py, Px);

            Pxw = Px - R * cosd(theta1);
            Pyw = Py - R * sind(theta1);
            Pzw = Pz + obj.d5 * sind(omega);

            Rw = sqrt(Pxw^2 + Pyw^2);
            S = sqrt((Pzw - obj.d1)^2 + Rw^2);

            alpha = atan2d(Pzw - obj.d1, Rw);
            beta = acosd((obj.a2^2 + S^2 - obj.a3^2) / (2 * obj.a2 * S));

            theta2 = alpha + beta;
            theta3 = -acosd((S^2 - obj.a2^2 - obj.a3^2) / (2 * obj.a2 * obj.a3));

            theta234 = 90 - omega;
            theta4 = theta234 - theta2 - theta3;
        end

        function joint_positions = forward_kinematics(obj, theta1, theta2, theta3, theta4)
            theta1 = deg2rad(theta1);
            theta2 = deg2rad(theta2);
            theta3 = deg2rad(theta3);
            theta4 = deg2rad(theta4);

            x0 = 0; y0 = 0; z0 = 0;
            x1 = 0; y1 = 0; z1 = obj.d1;
            x2 = obj.a2 * cos(theta1) * cos(theta2);
            y2 = obj.a2 * sin(theta1) * cos(theta2);
            z2 = obj.d1 + obj.a2 * sin(theta2);
            x3 = x2 + obj.a3 * cos(theta1) * cos(theta2 + theta3);
            y3 = y2 + obj.a3 * sin(theta1) * cos(theta2 + theta3);
            z3 = z2 + obj.a3 * sin(theta2 + theta3);

            x4 = x3 + obj.d5 * cos(theta1) * cos(theta2 + theta3 + theta4);
            y4 = y3 + obj.d5 * sin(theta1) * cos(theta2 + theta3 + theta4);
            z4 = z3 + obj.d5 * sin(theta2 + theta3 + theta4);

            joint_positions = [x0, y0, z0; x1, y1, z1; x2, y2, z2; x3, y3, z3; x4, y4, z4];
        end

        function plot_robot(~, joint_positions)
            x = joint_positions(:, 1);
            y = joint_positions(:, 2);
            z = joint_positions(:, 3);

            figure;
            plot3(x, y, z, 'o-', 'MarkerSize', 10, 'DisplayName', 'Robot Arm');
            hold on;
            scatter3(x, y, z, 'k');

            for i = 1:length(joint_positions) - 1
                text((x(i) + x(i+1)) / 2, (y(i) + y(i+1)) / 2, (z(i) + z(i+1)) / 2, ...
                    ['Link ', num2str(i)], 'Color', 'black');
            end

            xlabel('X axis');
            ylabel('Y axis');
            zlabel('Z axis');
            title('3D Robot Configuration');
            legend;
            axis equal;
            grid on;
            hold off;
        end
    end
end

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
