import pybullet as p
import time
import pybullet_data


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


def simulate_and_capture(
    robot_id, object_id, joint_params, output_file_joints, output_file_positions
):
    end_effector_positions = []
    joint_angles = []
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

            # Capture end-effector position
            end_effector_pos, _ = p.getLinkState(robot_id, last_link_id)[:2]
            end_effector_positions.append(end_effector_pos)

            # Capture joint angles
            current_joint_angles = [
                p.getJointState(robot_id, joint)[0] for joint, _ in joint_params
            ]
            joint_angles.append(current_joint_angles)

            time.sleep(1.0 / 240.0)

            # Write end-effector positions to file
            output_file_positions.write(
                f"{current_time}, {end_effector_pos[0]}, {end_effector_pos[1]}, {end_effector_pos[2]}\n"
            )

            # Write joint angles to file
            output_file_joints.write(
                f"{current_time}, "
                + ", ".join(str(angle) for angle in current_joint_angles)
                + "\n"
            )

            # Debug prints
            print(
                f"Time: {current_time}s, End-effector position: {end_effector_pos}, Joint angles: {current_joint_angles}"
            )

    except KeyboardInterrupt:
        print("Simulation stopped by user.")

    return end_effector_positions, joint_angles


def main():
    urdf_path = "urdfs/s.urdf"
    initial_positions = [-3.142, 1.571, 0.959, 0.529, 1.422]
    object_position = [0, 0, -0.58]  # Position under the robot
    robot_id, object_id = load_robot_and_object(urdf_path, object_position)
    joint_params = setup_joint_control(robot_id, initial_positions)

    with open("robot_joint_angles.csv", "w") as output_file_joints, open(
        "robot_end_effector_positions.csv", "w"
    ) as output_file_positions:
        output_file_joints.write("Time, Joint 1, Joint 2, Joint 3, Joint 4, Joint 5\n")
        output_file_positions.write("Time, X, Y, Z\n")
        simulate_and_capture(
            robot_id, object_id, joint_params, output_file_joints, output_file_positions
        )


if __name__ == "__main__":
    main()
