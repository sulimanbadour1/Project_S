import csv


def read_data(file_path, columns=None):
    data = []
    with open(file_path, "r") as file:
        reader = csv.reader(file)
        header = next(reader)  # Read and store header for debugging
        print("Header:", header)  # Debug print to check the actual headers
        for row in reader:
            entry = [float(val) for val in row]
            # Safeguard against out of range indices
            filtered_entry = (
                [entry[i] for i in columns if i < len(entry)]
                if columns is not None
                else entry
            )
            data.append(filtered_entry)
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
    # Adjusted columns index after checking the CSV structure
    columns_for_joint_data = [
        0,
        *range(1, 31),
    ]  # Update this based on the CSV file structure

    # Read the data from the output files
    end_effector_positions = read_data("robot_end_effector_positions.csv")
    joint_data = read_data("robot_joint_data.csv", columns_for_joint_data)

    # Calculate the simulation time
    simulation_time = calculate_simulation_time(end_effector_positions)
    print(f"Simulation time: {simulation_time} seconds")

    # Prompt the user for desired timing
    desired_timing = float(input("Enter the desired timing in seconds: "))

    # Calculate the time scale factor
    time_scale_factor = desired_timing / simulation_time

    # Apply desired timing
    apply_timing(end_effector_positions, time_scale_factor)
    apply_timing(joint_data, time_scale_factor)

    # Write the scaled data to new files
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
            "Joint 1 Angle (Rad)",
            "Joint 2 Angle (Rad)",
            "Joint 3 Angle (Rad)",
            "Joint 4 Angle (Rad)",
            "Joint 5 Angle (Rad)",
            "Joint 1 Angle (Deg)",
            "Joint 2 Angle (Deg)",
            "Joint 3 Angle (Deg)",
            "Joint 4 Angle (Deg)",
            "Joint 5 Angle (Deg)",
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
