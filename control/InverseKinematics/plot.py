import pybullet as p
import time
import pybullet_data


def load_robot(urdf_path):
    p.connect(p.GUI)
    p.setAdditionalSearchPath(
        pybullet_data.getDataPath()
    )  # Optional, for loading default models
    p.setGravity(0, 0, -10)  # Set gravity to simulate Earth's gravity
    robot_id = p.loadURDF(urdf_path, useFixedBase=True)
    print(f"Robot loaded with ID: {robot_id}")
    return robot_id


def setup_joint_control(robot_id):
    num_joints = p.getNumJoints(robot_id)
    joint_params = []
    for joint in range(num_joints):
        joint_info = p.getJointInfo(robot_id, joint)
        if joint_info[2] != p.JOINT_FIXED:
            joint_name = joint_info[1].decode("utf-8")
            param_id = p.addUserDebugParameter(
                joint_name, joint_info[8], joint_info[9], 0
            )
            joint_params.append((joint, param_id))
    return joint_params


def simulate(robot_id, joint_params):
    print("Use the sliders to control the robot joints.")
    try:
        while True:
            p.stepSimulation()
            for joint, param_id in joint_params:
                param_value = p.readUserDebugParameter(param_id)
                p.setJointMotorControl2(
                    robot_id, joint, p.POSITION_CONTROL, targetPosition=param_value
                )
            time.sleep(1.0 / 240.0)  # Simulation time step
    except KeyboardInterrupt:
        print("Simulation stopped by user.")


def main():
    urdf_path = "s.urdf"
    robot_id = load_robot(urdf_path)
    joint_params = setup_joint_control(robot_id)
    try:
        simulate(robot_id, joint_params)
    finally:
        p.disconnect()
        print("Disconnected from the physics server.")


if __name__ == "__main__":
    main()
