import pybullet as p
import time
import pybullet_data


def load_robot_and_object(urdf_path, object_position):
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
    print("Camera activated.")


def simulate(robot_id, joint_params, target_position, enable_camera=False):
    last_link_id = p.getNumJoints(robot_id) - 1
    recording = []
    try:
        while True:
            p.stepSimulation()
            joint_positions = []
            for joint, param_id in joint_params:
                param_value = p.readUserDebugParameter(param_id)
                p.setJointMotorControl2(
                    robot_id, joint, p.POSITION_CONTROL, targetPosition=param_value
                )
                joint_positions.append(param_value)
            recording.append(joint_positions)

            if enable_camera:
                view_matrix, projection_matrix = attach_camera_to_link(
                    robot_id, last_link_id, target_position
                )
                capture_camera_data(view_matrix, projection_matrix)

            time.sleep(1.0 / 240.0)
    except KeyboardInterrupt:
        print("Simulation stopped by user.")
        return recording
    finally:
        if p.isConnected():
            p.disconnect()


def save_positions_to_file(positions, filename="recorded_positions.txt"):
    with open(filename, "w") as file:
        for pos in positions:
            file.write(",".join(map(str, pos)) + "\n")
    print(f"Positions saved to {filename}")


def main():
    p.connect(p.GUI)

    urdf_path = "urdfs/s.urdf"
    initial_positions = [-3.142, 1.571, 0.876, 0.711, 0.976]
    object_position = [0, 0, -0.58]  # Position under the robot
    robot_id, _ = load_robot_and_object(urdf_path, object_position)
    joint_params = setup_joint_control(robot_id, initial_positions)

    enable_camera = input("Enable camera? (y/n): ").lower() == "y"
    if enable_camera:
        print("Camera activated.")

    target_position = object_position
    recording = simulate(robot_id, joint_params, target_position, enable_camera)
    print("Simulation completed.")
    save_positions_to_file(recording)


if __name__ == "__main__":
    main()
