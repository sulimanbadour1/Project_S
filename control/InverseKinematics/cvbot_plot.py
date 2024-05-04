import pybullet as p
import time
import pybullet_data


def load_robot(urdf_path):
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -10)
    robot_id = p.loadURDF(urdf_path, useFixedBase=True)
    print(f"Robot loaded with ID: {robot_id}")
    return robot_id


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


def setup_camera():
    camera_distance = 2.0
    camera_yaw = 50
    camera_pitch = -35
    camera_target_position = (0, 0, 0.5)
    view_matrix = p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=camera_target_position,
        distance=camera_distance,
        yaw=camera_yaw,
        pitch=camera_pitch,
        roll=0,
        upAxisIndex=2,
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


def simulate(robot_id, joint_params, view_matrix, projection_matrix):
    print("Use the sliders to control the robot joints.")
    try:
        while True:
            p.stepSimulation()
            for joint, param_id in joint_params:
                param_value = p.readUserDebugParameter(param_id)
                p.setJointMotorControl2(
                    robot_id, joint, p.POSITION_CONTROL, targetPosition=param_value
                )

            # Capture camera data - optional, you can remove this if you only want position data
            capture_camera_data(view_matrix, projection_matrix)

            # Get the state of the last link
            last_link_index = p.getNumJoints(robot_id) - 1
            last_link_state = p.getLinkState(robot_id, last_link_index)
            position_of_last_link = last_link_state[0]
            print(f"Position of last link (x, y, z): {position_of_last_link}")

            time.sleep(1.0 / 240.0)
    except KeyboardInterrupt:
        print("Simulation stopped by user.")


def main():
    urdf_path = "urdfs/s.urdf"
    initial_positions = [-3.142, 1.571, 0.876, 0.711, 0.976]
    robot_id = load_robot(urdf_path)
    joint_params = setup_joint_control(robot_id, initial_positions)
    view_matrix, projection_matrix = setup_camera()
    try:
        simulate(robot_id, joint_params, view_matrix, projection_matrix)
    finally:
        p.disconnect()
        print("Disconnected from the physics server.")


if __name__ == "__main__":
    main()
