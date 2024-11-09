"""Creating an SDF grid"""
import numpy as np


def sdf_grid(sdf_function, resolution):
    """
    Create an occupancy grid at the specified resolution given the implicit representation.
    :param sdf_function: A function that takes in a point (x, y, z) and returns the sdf at the given point.
    Points may be provides as vectors, i.e. x, y, z can be scalars or 1D numpy arrays, such that (x[0], y[0], z[0])
    is the first point, (x[1], y[1], z[1]) is the second point, and so on
    :param resolution: Resolution of the occupancy grid
    :return: An SDF grid of specified resolution (i.e. an array of dim (resolution, resolution, resolution) with positive values outside the shape and negative values inside.
    """

    # ###############
    # TODO: Implement
    x = np.linspace(-1, 1, resolution)
    y = np.linspace(-1, 1, resolution)
    z = np.linspace(-1, 1, resolution)
    
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')

    # Stack the coordinates together into a single array (each row is a point in 3D space)
    points = np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T  # Shape: (resolution^3, 3)

    # Extract the x, y, z components for the batch call
    x_batch = points[:, 0]
    y_batch = points[:, 1]
    z_batch = points[:, 2]

    # Use the sdf_function to get the signed distance for each point
    sdf_values = sdf_function(x_batch, y_batch, z_batch)

    # Reshape the sdf values back to a 3D grid with shape (resolution, resolution, resolution)
    sdf_grid = sdf_values.reshape((resolution, resolution, resolution))

    return sdf_grid
