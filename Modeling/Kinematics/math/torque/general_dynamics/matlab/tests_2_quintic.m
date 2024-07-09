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
            theta2_alt = alpha - beta;
            
            theta3 = acosd((S^2 - obj.a2^2 - obj.a3^2) / (2 * obj.a2 * obj.a3));
            theta3 = -theta3;
            
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
            
            x4 = x3 + obj.d5;
            y4 = y3;
            z4 = z3;
            
            joint_positions = [x0, y0, z0; x1, y1, z1; x2, y2, z2; x3, y3, z3; x4, y4, z4];
        end
        
        function plot_robot(obj, joint_positions)
            x = joint_positions(:, 1);
            y = joint_positions(:, 2);
            z = joint_positions(:, 3);
            
            figure;
            plot3(x, y, z, 'o-', 'MarkerSize', 10, 'DisplayName', 'Robot Arm');
            hold on;
            scatter3(x, y, z, 'k');
            
            for i = 1:length(joint_positions) - 1
                midx = (x(i) + x(i + 1)) / 2;
                midy = (y(i) + y(i + 1)) / 2;
                midz = (z(i) + z(i + 1)) / 2;
                text(midx, midy, midz, sprintf('Link %d', i), 'Color', 'black');
            end
            
            xlabel('X axis');
            ylabel('Y axis');
            zlabel('Z axis');
            title('3D Robot Configuration');
            legend;
            axis equal;
            grid on;
        end
    end
end

% Example parameters
Px = 0.5;
Py = 0;
Pz = 0.5;
omega = 0;

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
fprintf('Joint positions:\n');
disp(joint_positions);
fprintf('End effector position: %.2f, %.2f, %.2f\n', joint_positions(end, :));

% Convert to radians
theta1 = deg2rad(theta1);
theta2 = deg2rad(theta2);
theta3 = deg2rad(theta3);
theta4 = deg2rad(theta4);

fprintf('Theta1: %.2f radians, Theta2: %.2f radians, Theta3: %.2f radians, Theta4: %.2f radians\n', theta1, theta2, theta3, theta4);
