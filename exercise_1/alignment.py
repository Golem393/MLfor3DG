""" Procrustes Aligment for point clouds """
import numpy as np
from pathlib import Path


def procrustes_align(pc_x, pc_y):
    """
    calculate the rigid transform to go from point cloud pc_x to point cloud pc_y, assuming points are corresponding
    :param pc_x: Nx3 input point cloud
    :param pc_y: Nx3 target point cloud, corresponding to pc_x locations
    :return: rotation (3, 3) and translation (3,) needed to go from pc_x to pc_y
    """
    R = np.zeros((3, 3), dtype=np.float32)
    t = np.zeros((3,), dtype=np.float32)

    # TODO: Your implementation starts here ###############
    # 1. get centered pc_x and centered pc_y
    # 2. create X and Y both of shape 3XN by reshaping centered pc_x, centered pc_y
    # 3. estimate rotation
    # 4. estimate translation
    # R and t should now contain the rotation (shape 3x3) and translation (shape 3,)
    # TODO: Your implementation ends here ###############

    # 1. Get centered pc_x and centered pc_y
    center_x = get_centeriod(pc_x)  # Should be shape (3,)
    center_y = get_centeriod(pc_y)  # Should be shape (3,)
    centered_pc_x = pc_x - center_x
    centered_pc_y = pc_y - center_y

    # 2. Reshape centered_pc_x and centered_pc_y to 3xN
    X = centered_pc_x.T  # Shape 3xN
    Y = centered_pc_y.T  # Shape 3xN

    # 3. Estimate rotation using SVD
    H = np.matmul(X, Y.T)  # Cross-covariance matrix
    U, S, VT = np.linalg.svd(H)
    R = np.matmul(VT.T, U.T)

    # 4. estimate translation
    t = center_y - np.matmul(R, center_x)

    t_broadcast = np.broadcast_to(t[:, np.newaxis], (3, pc_x.shape[0]))
    print('Procrustes Aligment Loss: ', np.abs((np.matmul(R, pc_x.T) + t_broadcast) - pc_y.T).mean())

    return R, t


def load_correspondences():
    """
    loads correspondences between meshes from disk
    """

    load_obj_as_np = lambda path: np.array(list(map(lambda x: list(map(float, x.split(' ')[1:4])), path.read_text().splitlines())))
    path_x = (Path(__file__).parent / "resources" / "points_input.obj").absolute()
    path_y = (Path(__file__).parent / "resources" / "points_target.obj").absolute()
    return load_obj_as_np(path_x), load_obj_as_np(path_y)

def get_centeriod(pc):
    """
    get the centeriod of a point cloud
    :param pc: Nx3 point cloud
    :return: 3D centeriod
    """
    return np.mean(pc, axis=0)