"""Export to disk"""


def export_mesh_to_obj(path, vertices, faces):
    """
    exports mesh as OBJ
    :param path: output path for the OBJ file
    :param vertices: Nx3 vertices
    :param faces: Mx3 faces
    :return: None
    """
    with open(path, "w") as file:
        for v in vertices:
            file.write(f"v {v[0]:.1f} {v[1]:.1f} {v[2]:.1f}\n")
        for f in faces:
            file.write(f"f {f[0]} {f[1]} {f[2]}\n")
    # write vertices starting with "v "
    # write faces starting with "f "
    # ###############


def export_pointcloud_to_obj(path, pointcloud):
    """
    export pointcloud as OBJ
    :param path: output path for the OBJ file
    :param pointcloud: Nx3 points
    :return: None
    """
    with open(path, "w") as file:
        for p in pointcloud:
            file.write(f"v {p[0]:.1f} {p[1]:.1f} {p[2]:.1f}\n")
    # ###############
