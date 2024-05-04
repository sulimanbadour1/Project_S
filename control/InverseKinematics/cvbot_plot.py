import pybullet as p
import time
import pybullet_data
import numpy as np
import matplotlib.pyplot as plt  # Optional, for plotting


def load_robot_and_object(urdf_path, object_position):
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -10)
    robot_id = p.loadURDF(urdf_path, useFixedBase=True)
    print(f"Robot loaded with ID: {robot_id}")

    # Load a simple object (e.g., a box) directly under the robot
    visual_shape_id = p.createVisualShape(
        shapeType=p.GEOM_BOX, halfExtents=[0.1, 0.1, 0.1]
    )
    collision_shape_id = p.createCollisionShape(
        shapeType=p.GEOM_BOX, halfExtents=[0.1, 0.1, 0.1]
    )
    object_id = p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=collision_shape_id,
        baseVisualShapeIndex=visual_shape_id,
        basePosition=object_position,
    )
    print(f"Object loaded with ID: {object_id}")
    return robot_id, object_id


def setup_joint_control(robot_id, initial_positions):
    num_joints = p.getNumJoints(robot_id)
    joint_params = []
    for joint in range(num_joints):
        joint_info = p.getJointInfo(robot_id, joint)
        if joint_info[2] != p.JOINT_FIXED:
            joint_name = joint_info[1].decode("utf-8")
            param_id = p.addUserDebugParameter(
                joint_name, joint_info[8], joint_info[9], initial_positions[joint]
            )
            p.resetJointState(robot_id, joint, initial_positions[joint])
            joint_params.append((joint, param_id))
    return joint_params


def attach_camera_to_link(robot_id, link_id, target_position):
    """
    Attach camera to a robot link and dynamically adjust its position and orientation.
    """
    com_p, com_o, _, _, _, _ = p.getLinkState(robot_id, link_id)
    rot_matrix = p.getMatrixFromQuaternion(com_o)
    camera_position = com_p
    camera_target_position = target_position
    up_vector = [rot_matrix[2], rot_matrix[5], rot_matrix[8]]
    view_matrix = p.computeViewMatrix(
        camera_position, camera_target_position, up_vector
    )
    aspect = 960 / 720
    fov = 60
    near = 0.1
    far = 100
    projection_matrix = p.computeProjectionMatrixFOV(
        fov=fov, aspect=aspect, nearVal=near, farVal=far
    )
    return view_matrix, projection_matrix


def capture_camera_data(view_matrix, projection_matrix):
    width, height, rgbImg, depthImg, segImg = p.getCameraImage(
        width=960,
        height=720,
        viewMatrix=view_matrix,
        projectionMatrix=projection_matrix,
    )
    print("Captured images.")


def simulate_and_capture(
    robot_id, object_id, joint_params, target_position, output_file
):
    end_effector_positions = []
    joint_angles = []
    last_link_id = p.getNumJoints(robot_id) - 1
    try:
        while True:
            p.stepSimulation()
            for joint, param_id in joint_params:
                param_value = p.readUserDebugParameter(param_id)
                p.setJointMotorControl2(
                    robot_id, joint, p.POSITION_CONTROL, targetPosition=param_value
                )

            # Capture end-effector position
            end_effector_pos, _ = p.getLinkState(robot_id, last_link_id)[:2]
            end_effector_positions.append(end_effector_pos)

            # Capture joint angles
            current_joint_angles = [
                p.getJointState(robot_id, joint)[0] for joint, _ in joint_params
            ]
            joint_angles.append(current_joint_angles)

            time.sleep(1.0 / 240.0)

            # Write data to file
            output_file.write(
                f"{end_effector_pos[0]}, {end_effector_pos[1]}, {end_effector_pos[2]}, "
            )
            output_file.write(", ".join(str(angle) for angle in current_joint_angles))
            output_file.write("\n")

            # Debug prints
            print(f"End-effector position: {end_effector_pos}")
            print(f"Joint angles: {current_joint_angles}")

    except KeyboardInterrupt:
        print("Simulation stopped by user.")

    return end_effector_positions, joint_angles


def main():
    urdf_path = "urdfs/s.urdf"
    initial_positions = [-3.142, 1.571, 0.959, 0.529, 1.422]
    object_position = [0, 0, -0.58]  # Position under the robot
    robot_id, object_id = load_robot_and_object(urdf_path, object_position)
    joint_params = setup_joint_control(robot_id, initial_positions)

    # Open output file
    with open("robot_data.txt", "w") as output_file:
        # Simulate robot movement and capture end-effector positions and joint angles
        end_effector_positions, joint_angles = simulate_and_capture(
            robot_id, object_id, joint_params, object_position, output_file
        )
        print("End-effector positions:")
        print(end_effector_positions)

        # Extract path from end-effector positions
        path = np.array(end_effector_positions)

        # Output path
        print("Path points:")
        for i, point in enumerate(path):
            print(f"Point {i + 1}: {point}")

        # Plot path (optional, requires matplotlib)
        plt.figure()
        plt.plot(path[:, 0], path[:, 1], "-o")
        plt.title("Robot End-Effector Path")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.grid(True)
        plt.show()


if __name__ == "__main__":
    main()
