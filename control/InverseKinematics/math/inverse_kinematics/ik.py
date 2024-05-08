import numpy as np


def inverse_kinematics(x_d, y_d, z_d, l1=0.03, l2=0.25, l3=0.28, d1=0.05):
    # Calculate theta1
    theta1 = np.degrees(np.arctan2(y_d, x_d))

    # Project target onto plane adjusted for the first link and base height
    x_prime = np.sqrt(x_d**2 + y_d**2) - l1
    y_prime = z_d - d1

    # Use the law of cosines to solve for theta2 and theta3
    c2 = (x_prime**2 + y_prime**2 - l2**2 - l3**2) / (2 * l2 * l3)
    theta3 = np.degrees(np.arccos(c2))

    k1 = l2 + l3 * c2
    k2 = l3 * np.sqrt(1 - c2**2)
    theta2 = np.degrees(np.arctan2(y_prime, x_prime) - np.arctan2(k2, k1))

    return theta1, theta2, theta3


# Assuming you want to calculate for some end-effector position (x_d, y_d, z_d)
x_d = 0  # Desired x position
y_d = 0  # Desired y position
z_d = 0  # Desired z position

# Calculate the inverse kinematics
theta1, theta2, theta3 = inverse_kinematics(x_d, y_d, z_d)
print("Theta 1:", theta1, "degrees")
print("Theta 2:", theta2, "degrees")
print("Theta 3:", theta3, "degrees")
