import pandas as pd
import numpy as np


def check_data_ranges(df, column_prefix, expected_min, expected_max):
    """Check if data in specific columns are within expected range."""
    for column in df.columns:
        if column.startswith(column_prefix):
            if not (df[column].between(expected_min, expected_max).all()):
                print(f"Data out of expected range in column: {column}")
                return False
    return True


def verify_output_files(joint_data_file, end_effector_file):
    # Load data
    joint_data = pd.read_csv(joint_data_file)
    end_effector_data = pd.read_csv(end_effector_file)

    # Check data integrity by ensuring no NaN values
    if joint_data.isnull().values.any() or end_effector_data.isnull().values.any():
        print("Data integrity check failed: NaN values found.")
        return False

    # Check joint angles in radians are within the typical range [-pi, pi]
    if not check_data_ranges(joint_data, "Angle (Rad)", -np.pi, np.pi):
        return False

    # Check joint angles in degrees are within the typical range [-180, 180]
    if not check_data_ranges(joint_data, "Angle (Deg)", -180, 180):
        return False

    # Check if velocities and efforts are within a reasonable expected range
    if not check_data_ranges(
        joint_data, "Vel", -100, 100
    ):  # Arbitrary range, adjust as necessary
        return False
    if not check_data_ranges(
        joint_data, "Eff", -500, 500
    ):  # Arbitrary range, adjust as necessary
        return False

    # Check end-effector positions (assuming they should be within the workspace boundaries)
    if not check_data_ranges(
        end_effector_data, "X", -10, 10
    ):  # Workspace boundary range, adjust as necessary
        return False
    if not check_data_ranges(
        end_effector_data, "Y", -10, 10
    ):  # Workspace boundary range, adjust as necessary
        return False
    if not check_data_ranges(
        end_effector_data, "Z", -10, 10
    ):  # Workspace boundary range, adjust as necessary
        return False

    print("All checks passed.")
    return True


# Call the function with file paths
verify_output_files("robot_joint_data.csv", "robot_end_effector_positions.csv")
