"""Triangle Meshes to Point Clouds"""
import numpy as np


def sample_point_cloud(vertices, faces, n_points):
    """
    Sample n_points uniformly from the mesh represented by vertices and faces
    :param vertices: Nx3 numpy array of mesh vertices
    :param faces: Mx3 numpy array of mesh faces
    :param n_points: number of points to be sampled
    :return: sampled points, a numpy array of shape (n_points, 3)

    """
    n_points_per_face = n_points // len(faces)
    remaining_points = n_points % len(faces)
    points = np.empty((0, 3))
    for idx, face in enumerate(faces):
        n_per_face = n_points_per_face
        if idx < remaining_points:
            n_per_face += 1
        for _ in range(n_per_face):
            r_1 = np.random.rand(0,1)
            r_2 = np.random.rand(0,1)
            u = 1 - np.sqrt(r_1)
            v = np.sqrt(r_1) * (1 - r_2)
            w  = np.sqrt(r_1) * r_2
            new_point = u * vertices[face[0]] + v * vertices[face[1]] + w * vertices[face[2]]
            points = np.vstack([points, new_point])
    # ###############
