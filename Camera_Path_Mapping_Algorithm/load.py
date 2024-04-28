import trimesh
import numpy as np


def calculate_edge_angles(mesh):
    # Get the unique edges and corresponding vertices
    edges = mesh.edges_unique
    edge_vectors = mesh.vertices[edges[:, 0]] - mesh.vertices[edges[:, 1]]
    edge_lengths = np.linalg.norm(edge_vectors, axis=1)
    edge_unit_vectors = edge_vectors / edge_lengths[:, None]

    # Find the angles between all connected edges
    angle_list = []
    for edge, vector in zip(edges, edge_unit_vectors):
        connected_edges = np.concatenate(
            (np.where(edges[:, 0] == edge[1])[0], np.where(edges[:, 1] == edge[0])[0])
        )
        connected_vectors = edge_unit_vectors[connected_edges]
        cos_angles = np.clip(
            np.einsum(
                "ij,ij->i", np.full_like(connected_vectors, vector), connected_vectors
            ),
            -1.0,
            1.0,
        )
        angles = np.arccos(cos_angles)
        angle_list.append(np.degrees(angles))

    return edges, angle_list


def analyze_model(mesh):
    # Calculate sharp edges
    edges, angle_list = calculate_edge_angles(mesh)
    sharp_edges = [
        edge
        for edge, angles in zip(edges, angle_list)
        if any(angle > 30 for angle in angles)
    ]
    print(f"Number of sharp edges: {len(sharp_edges)}")

    # Assuming other analyses are correct or would be adjusted similarly if needed.

    # Visualization of critical areas
    mesh.show()


def load_and_analyze_model(file_path):
    # Load the 3D model from file
    mesh = trimesh.load(file_path)
    print(f"Is the mesh watertight? {mesh.is_watertight}")

    # Analyze the model
    analyze_model(mesh)


# Load and analyze the model
model_path = "logo3d.stl"
load_and_analyze_model(model_path)
