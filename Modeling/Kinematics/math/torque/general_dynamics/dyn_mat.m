% Define DH parameters
d1 = 0.1;
d5 = 0.1;
a2 = 0.5;
a3 = 0.5;
alpha = [90, 0, 0, 90, 0];

% Define symbolic variables
syms theta1 theta2 theta3 theta4 theta5 real
syms dtheta1 dtheta2 dtheta3 dtheta4 dtheta5 real
syms ddtheta1 ddtheta2 ddtheta3 ddtheta4 ddtheta5 real

% Masses of the links and external parameters
masses = [1.0, 1.0, 1.0, 1.0, 1.0];
mass_camera = 0.5;
mass_lights = 0.5;
external_forces = [0, 0, 0];  % No external forces in this example
external_torques = [0, 0, 0, 0, 0];  % No external torques in this example
g = 9.81;

% Define DH transformation matrix function
dh = @(theta, d, a, alpha) [
    cos(theta) -sin(theta)*cosd(alpha)  sin(theta)*sind(alpha) a*cos(theta);
    sin(theta)  cos(theta)*cosd(alpha) -cos(theta)*sind(alpha) a*sin(theta);
    0           sind(alpha)             cosd(alpha)             d;
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

% Inertia matrices for each link
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

% Define q and dq
q = [theta1; theta2; theta3; theta4; theta5];
dq = [dtheta1; dtheta2; dtheta3; dtheta4; dtheta5];
ddq = [ddtheta1; ddtheta2; ddtheta3; ddtheta4; ddtheta5];

% Compute Coriolis and centrifugal matrix
C = sym(zeros(5));
for k = 1:5
    for j = 1:5
        for i = 1:5
            C(k, j) = C(k, j) + 0.5 * (diff(M(k, j), q(i)) + diff(M(k, i), q(j)) - diff(M(i, j), q(k))) * dq(i);
        end
    end
end

% External forces and torques
F_ext = [external_forces, external_torques];

% Total torque calculation
tau = M * ddq + C * dq + G;

% Convert symbolic expression to numeric function
tau_func = matlabFunction(tau, 'Vars', {theta1, theta2, theta3, theta4, theta5, ...
                                        dtheta1, dtheta2, dtheta3, dtheta4, dtheta5, ...
                                        ddtheta1, ddtheta2, ddtheta3, ddtheta4, ddtheta5});

% Numerical evaluation
num_steps = 10;  % Number of steps for joint angles
theta_vals = linspace(-pi, pi, num_steps);
dtheta_vals = linspace(-2, 2, num_steps);  % Example range for joint velocities
ddtheta_vals = linspace(-2, 2, num_steps);  % Example range for joint accelerations

max_torque = zeros(5, 1);
num_combinations = numel(theta_vals)^5;
configurations = zeros(num_combinations, 5 + 5);  % Initialize configurations matrix

% Check for existing parallel pool and delete it if necessary
pool = gcp('nocreate');
if ~isempty(pool)
    delete(pool);
end

% Parallel pool setup
parpool('local');

parfor idx = 1:num_combinations
    % Convert linear index to subscripts
    [t1_idx, t2_idx, t3_idx, t4_idx, t5_idx] = ind2sub([num_steps, num_steps, num_steps, num_steps, num_steps], idx);
    t1 = theta_vals(t1_idx);
    t2 = theta_vals(t2_idx);
    t3 = theta_vals(t3_idx);
    t4 = theta_vals(t4_idx);
    t5 = theta_vals(t5_idx);

    % Randomly select velocities and accelerations for dynamic analysis
    dt1 = dtheta_vals(randi([1 num_steps]));
    dt2 = dtheta_vals(randi([1 num_steps]));
    dt3 = dtheta_vals(randi([1 num_steps]));
    dt4 = dtheta_vals(randi([1 num_steps]));
    dt5 = dtheta_vals(randi([1 num_steps]));

    ddt1 = ddtheta_vals(randi([1 num_steps]));
    ddt2 = ddtheta_vals(randi([1 num_steps]));
    ddt3 = ddtheta_vals(randi([1 num_steps]));
    ddt4 = ddtheta_vals(randi([1 num_steps]));
    ddt5 = ddtheta_vals(randi([1 num_steps]));

    torques = tau_func(t1, t2, t3, t4, t5, dt1, dt2, dt3, dt4, dt5, ddt1, ddt2, ddt3, ddt4, ddt5);
    
    max_torque = max(max_torque, abs(torques));
    configurations(idx, :) = [t1, t2, t3, t4, t5, torques'];
end

delete(gcp);  % Shut down the parallel pool

% Plot the maximum torques
figure;
bar(max_torque);
xlabel('Joints');
ylabel('Torque (Nm)');
title('Maximum Dynamic Torques on Each Joint');
grid on;

% Print maximum torques
disp('Maximum Dynamic Torques for given values:');
disp(max_torque);
