"""SDF to Occupancy Grid"""
import numpy as np


def occupancy_grid(sdf_function, resolution):
    """
    Create an occupancy grid at the specified resolution given the implicit representation.
    :param sdf_function: A function that takes in a point (x, y, z) and returns the sdf at the given point.
    Points may be provides as vectors, i.e. x, y, z can be scalars or 1D numpy arrays, such that (x[0], y[0], z[0])
    is the first point, (x[1], y[1], z[1]) is the second point, and so on
    :param resolution: Resolution of the occupancy grid
    :return: An occupancy grid of specified resolution (i.e. an array of dim (resolution, resolution, resolution) with value 0 outside the shape and 1 inside.
    """

    # ###############
    x = np.linspace(-1, 1, resolution)
    y = np.linspace(-1, 1, resolution)
    z = np.linspace(-1, 1, resolution)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    # Reshape the grid to a list of points
    points = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T  # Shape: (resolution^3, 3)

    # Apply sdf_function to all points in a batch
    sdf_values = np.array([sdf_function(p[0], p[1], p[2]) for p in points])

    # Reshape sdf_values back to a 3D grid and create occupancy grid
    sdf_grid = sdf_values.reshape((resolution, resolution, resolution))
    occupancy_grid = (sdf_grid < 0).astype(int)

    return occupancy_grid


    # ###############
