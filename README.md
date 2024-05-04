# Project_S (Camera control system for desktop 3d printers)


# 1- Develop Camera Path Mapping Algorithm

## Overview
This project involves the development of a camera path mapping algorithm to create a detailed simulation of camera movements around a 3D model. The simulation is designed to help in visualizing camera coverage for monitoring or inspection purposes, ensuring complete 360-degree visibility of the object being examined.

## Features
- **2D and 3D Visualization**: Utilizes `matplotlib` for dynamic visual representation of camera paths around the 3D model.
- **Interactive Animation**: Includes adjustable parameters for animation speed using sliders, enhancing user interactivity.
- **Camera Path Mapping**: Calculates optimal camera points based on the geometry of the 3D model, providing focused surveillance or inspection capabilities.

## Dependencies
Ensure you have the following Python packages installed:
- `numpy`
- `matplotlib`
- `trimesh`
- `pyglet<2`
- `PyQt5`

You can install these packages via pip:
```pip install numpy matplotlib trimesh pyglet<2 PyQt5```

## File Structure
- `project_s.py` : Contains the main Python script for generating the camera paths and animating them.
- `project_y.py` : Contains the main Python script for generating the camera paths with **timestamps** you can select the time for each cycle.
- `load_display.py` : loads the stl file and generate the camera points.
- `plotting_with_time.py`: plot the path with the time.
- `plotting.py` : plot the path without time.


## Usage
To run this script, make sure you have a 3D model file in STL format. Update the model_path in the script to point to your model file location.
- **Setting up the model path:** `model_path = "path_to_your_model/model_file.stl"`
- **Running the script:** `python project_s.py`
- **Interact with the simulation:**
Use the slider at the bottom of the 3D animation window to adjust the animation speed.
Watch both 2D and 3D animations to understand the camera's coverage.


## Output
- The script will visualize the camera paths in 2D and 3D.
- Camera points are written to camera_points.txt in the current directory, which can be used for further processing or control systems.

## Demo 

| Camera Movement 3D | PyQt5 Demo `project_s.py` |
|--------------------|------------|
| ![Camera Movement 3D](demos/camera_movement_3d.gif) | ![PyQt5 Demo](demos/demo.JPG) |

|PyQt5 Demo `project_y.py`|
|--------------------|
| ![PyQt5 Demo](demos/demo_timing.JPG) |

| Plotting with Time | Plotting without Time |
|---------------------|-----------------------|
| ![Plotting with Time](demos/time.JPG) | ![Plotting without Time](demos/no_time.JPG) |



# 2- Control 
## Real-time Robot Arm Movement Visualization + IK

- `plot_CAMBOT.py`
This Python script uses the PyBullet physics engine to simulate interactions between a robot and an object in a 3D environment. It demonstrates the loading of a robot from a URDF file, setting up joint controls, and capturing and logging various simulation data.

## Dependencies

- **PyBullet**: A Python module used for physics simulations in robotics, games, and visual effects.

## Setup and Execution

### Initial Setup

Import necessary modules and define the main functions used for loading the robot and object, setting up joint control, and running the simulation.

```python
import pybullet as p
import time
import pybullet_data
```


### Function: load_robot_and_object
This function initializes the PyBullet simulation environment, loads the robot from a URDF file, and creates a simple box-shaped object at a specified position.

### Parameters:

- urdf_path: String, path to the robot's URDF file.
- object_position: List of three floats, specifying the object's position in the simulation environment.

Returns:

- Tuple containing identifiers for the loaded robot and object.



### Function: setup_joint_control
Sets up control for the robot's joints based on initial positions.

## Parameters:

- robot_id: Integer, the identifier of the robot.
 -initial_positions: List of floats, initial positions for the robot's movable joints.

Returns:

- List of tuples, each containing a joint index and its corresponding debug parameter ID.