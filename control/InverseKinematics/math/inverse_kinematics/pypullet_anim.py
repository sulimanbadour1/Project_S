import pybullet as p
import pybullet_data
import time
import numpy as np


def setup_simulation():
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -10)
    plane_id = p.loadURDF("plane.urdf")


def load_robot(urdf_path):
    robot_id = p.loadURDF(urdf_path, useFixedBase=True)
    return robot_id


def compute_forward_kinematics(robot_id, joint_angles):
    # Manually set each joint position
    for i, angle in enumerate(joint_angles):
        p.resetJointState(robot_id, i, angle)
    # Use getLinkState to get the position and orientation of the end-effector
    end_effector_info = p.getLinkState(robot_id, p.getNumJoints(robot_id) - 1)
    position = end_effector_info[4]  # World position of the URDF link frame
    orientation = end_effector_info[5]  # World orientation of the URDF link frame
    return position, orientation


def perform_inverse_kinematics(robot_id, end_effector_pos, end_effector_ori):
    # Calculate joint positions needed to achieve the desired end-effector position and orientation
    joint_angles = p.calculateInverseKinematics(
        robot_id, p.getNumJoints(robot_id) - 1, end_effector_pos, end_effector_ori
    )
    return joint_angles


def main():
    setup_simulation()
    urdf_path = "urdfs/s.urdf"  # Update this path to your URDF file
    robot_id = load_robot(urdf_path)

    # Define desired joint angles for forward kinematics
    joint_angles = [0.0, 0, -1.57, 1.57, -1.57]  # Example joint angles in radians
    position, orientation = compute_forward_kinematics(robot_id, joint_angles)
    print("Forward Kinematics - Position:", position)
    print("Forward Kinematics - Orientation:", orientation)

    # Define a desired position and orientation for inverse kinematics
    desired_position = (
        position  # Use the position from forward kinematics as an example
    )
    desired_orientation = (
        orientation  # Use the orientation from forward kinematics as an example
    )
    calculated_angles = perform_inverse_kinematics(
        robot_id, desired_position, desired_orientation
    )
    print("Inverse Kinematics - Calculated Joint Angles:", calculated_angles)

    # Set robot to the calculated joint positions
    for joint_index, joint_position in enumerate(calculated_angles):
        p.setJointMotorControl2(
            robot_id, joint_index, p.POSITION_CONTROL, targetPosition=joint_position
        )

    # Run the simulation for a bit to see the result
    for _ in range(3000):
        p.stepSimulation()
        time.sleep(1.0 / 240.0)

    p.disconnect()


if __name__ == "__main__":
    main()
