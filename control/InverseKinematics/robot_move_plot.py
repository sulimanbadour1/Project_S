import pybullet as p
import pybullet_data
import time
import pandas as pd


def load_robot(urdf_path):
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -10)
    robot_id = p.loadURDF(urdf_path, useFixedBase=True)
    return robot_id


def set_joint_positions(robot_id, joint_angles):
    for i, angle in enumerate(joint_angles):
        p.resetJointState(robot_id, i, angle)


def visualize_robot_with_trajectory(urdf_path, joint_data_file, end_effector_file):
    robot_id = load_robot(urdf_path)
    joint_data = pd.read_csv(joint_data_file)
    end_effector_data = pd.read_csv(end_effector_file)

    # Strip whitespace from column names
    joint_data.columns = joint_data.columns.str.strip()
    end_effector_data.columns = end_effector_data.columns.str.strip()

    print("Corrected Joint Data Columns:", joint_data.columns)
    print("Corrected End Effector Data Columns:", end_effector_data.columns)

    prev_position = None

    for index, row in joint_data.iterrows():
        joint_angles = row.filter(
            regex="Rad"
        ).tolist()  # Extract only the columns with joint angles in radians
        set_joint_positions(robot_id, joint_angles)

        # Ensure index is valid and end_effector_data has rows
        if index < len(end_effector_data):
            end_effector_position = end_effector_data.loc[index, ["X", "Y", "Z"]].values
            if prev_position is not None:
                p.addUserDebugLine(
                    prev_position,
                    end_effector_position,
                    lineColorRGB=[1, 0, 0],
                    lineWidth=2.5,
                )
            prev_position = end_effector_position

        time.sleep(0.1)  # Adjust sleep time as needed for the speed of visualization

    p.disconnect()


if __name__ == "__main__":
    urdf_path = "urdfs/s.urdf"  # Update the URDF path
    joint_data_file = (
        "robot_joint_data.csv"  # Update the output file path for joint data
    )
    end_effector_file = "robot_end_effector_positions.csv"  # Update the output file path for end-effector positions
    visualize_robot_with_trajectory(urdf_path, joint_data_file, end_effector_file)
