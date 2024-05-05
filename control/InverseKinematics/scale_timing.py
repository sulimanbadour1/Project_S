import csv


def read_data(file_path, columns):
    data = []
    with open(file_path, "r") as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        for row in reader:
            entry = [float(val) for val in row]
            data.append(entry if columns is None else [entry[i] for i in columns])
    return data


def write_data(file_path, data, header):
    with open(file_path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(data)


def calculate_simulation_time(data):
    return data[-1][0] - data[0][0]


def apply_timing(data, time_scale_factor):
    for entry in data:
        entry[0] *= time_scale_factor


def main():
    # Read the data from the output files
    end_effector_positions = read_data("robot_end_effector_positions.csv", None)
    joint_data = read_data("robot_joint_data.csv", [0, *range(1, 16)])

    # Calculate the simulation time
    simulation_time = calculate_simulation_time(end_effector_positions)
    print(f"Simulation time: {simulation_time} seconds")

    # Prompt the user for desired timing
    while True:
        try:
            desired_timing = float(input("Enter the desired timing in seconds: "))
            break
        except ValueError:
            print("Please enter a valid number.")

    # Calculate the time scale factor
    time_scale_factor = desired_timing / simulation_time

    # Apply desired timing
    apply_timing(end_effector_positions, time_scale_factor)
    apply_timing(joint_data, time_scale_factor)

    # Write the scaled data to existing files by modifying the time column
    write_data(
        "robot_end_effector_positions_scaled.csv",
        end_effector_positions,
        ["Time", "X", "Y", "Z"],
    )
    write_data(
        "robot_joint_data_scaled.csv",
        joint_data,
        [
            "Time",
            "Joint 1 Angle",
            "Joint 2 Angle",
            "Joint 3 Angle",
            "Joint 4 Angle",
            "Joint 5 Angle",
            "Joint 1 Vel",
            "Joint 2 Vel",
            "Joint 3 Vel",
            "Joint 4 Vel",
            "Joint 5 Vel",
            "Joint 1 Eff",
            "Joint 2 Eff",
            "Joint 3 Eff",
            "Joint 4 Eff",
            "Joint 5 Eff",
        ],
    )

    print("Scaled files have been created.")


if __name__ == "__main__":
    main()
