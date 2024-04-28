# Project_S ( Camera control system for desktop 3d printers)


# Develop Camera Path Mapping Algorithm

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

You can install these packages via pip:
```pip install numpy matplotlib trimesh```

## File Structure
`project_s.py` : Contains the main Python script for generating the camera paths and animating them.

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
<img src="camera_movement_3d.mp4" alt="demo">
