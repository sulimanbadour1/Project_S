import pybullet as p
import time
import pybullet_data


def load_robot_and_object(urdf_path, object_position):
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -10)
    robot_id = p.loadURDF(urdf_path, useFixedBase=True)
    print(f"Robot loaded with ID: {robot_id}")

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
    robot_id,
    object_id,
    joint_params,
    output_file_joints,
    output_file_positions,
    step_interval=1.0 / 240.0,
    target_position=[0, 0, -0.58],
):
    end_effector_positions = []
    joint_angles = []
    joint_velocities = []
    joint_efforts = []
    last_link_id = p.getNumJoints(robot_id) - 1
    start_time = time.time()

    try:
        while True:
            p.stepSimulation()
            current_time = time.time() - start_time

            for joint, param_id in joint_params:
                param_value = p.readUserDebugParameter(param_id)
                p.setJointMotorControl2(
                    robot_id, joint, p.POSITION_CONTROL, targetPosition=param_value
                )

            view_matrix, projection_matrix = attach_camera_to_link(
                robot_id, last_link_id, target_position
            )
            capture_camera_data(view_matrix, projection_matrix)
            # Capture end-effector position
            end_effector_pos, _ = p.getLinkState(robot_id, last_link_id)[:2]
            end_effector_positions.append(end_effector_pos)

            # Capture joint states (angles, velocities, efforts)
            current_joint_states = [
                p.getJointState(robot_id, joint) for joint, _ in joint_params
            ]
            current_joint_angles = [state[0] for state in current_joint_states]
            current_joint_velocities = [state[1] for state in current_joint_states]
            current_joint_efforts = [state[3] for state in current_joint_states]

            joint_angles.append(current_joint_angles)
            joint_velocities.append(current_joint_velocities)
            joint_efforts.append(current_joint_efforts)

            time.sleep(step_interval)

            # Write end-effector positions to file
            output_file_positions.write(
                f"{current_time}, {end_effector_pos[0]}, {end_effector_pos[1]}, {end_effector_pos[2]}\n"
            )

            # Write joint angles, velocities, and efforts to file
            output_file_joints.write(
                f"{current_time}, "
                + ", ".join(str(angle) for angle in current_joint_angles)
                + ", "
                + ", ".join(str(vel) for vel in current_joint_velocities)
                + ", "
                + ", ".join(str(eff) for eff in current_joint_efforts)
                + "\n"
            )

            # Debug prints
            print(
                f"Time: {current_time}s, End-effector position: {end_effector_pos}, Joint angles: {current_joint_angles}, Joint velocities: {current_joint_velocities}, Joint efforts: {current_joint_efforts}"
            )

    except KeyboardInterrupt:
        print("Simulation stopped by user.")

    return end_effector_positions, joint_angles, joint_velocities, joint_efforts


def main():
    urdf_path = "urdfs/s.urdf"
    initial_positions = [-3.142, 1.571, 0.959, 0.529, 1.422]
    object_position = [0, 0, -0.58]  # Position under the robot
    robot_id, object_id = load_robot_and_object(urdf_path, object_position)
    joint_params = setup_joint_control(robot_id, initial_positions)

    with open("robot_joint_data.csv", "w") as output_file_joints, open(
        "robot_end_effector_positions.csv", "w"
    ) as output_file_positions:
        output_file_joints.write(
            "Time, Joint 1 Angle, Joint 2 Angle, Joint 3 Angle, Joint 4 Angle, Joint 5 Angle, Joint 1 Vel, Joint 2 Vel, Joint 3 Vel, Joint 4 Vel, Joint 5 Vel, Joint 1 Eff, Joint 2 Eff, Joint 3 Eff, Joint 4 Eff, Joint 5 Eff\n"
        )
        output_file_positions.write("Time, X, Y, Z\n")
        simulate_and_capture(
            robot_id,
            object_id,
            joint_params,
            output_file_joints,
            output_file_positions,
            step_interval=1.0 / 240.0,
        )


if __name__ == "__main__":
    main()
