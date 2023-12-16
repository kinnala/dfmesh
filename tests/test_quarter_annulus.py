import numpy as np
from helpers import assert_norm_equality

import dfmesh


def test_quarter_annulus():
    h = 0.05
    disk0 = dfmesh.Circle([0.0, 0.0], 0.25)
    disk1 = dfmesh.Circle([0.0, 0.0], 1.0)
    diff0 = dfmesh.Difference(disk1, disk0)

    rect = dfmesh.Rectangle(0.0, 1.0, 0.0, 1.0)
    quarter = dfmesh.Intersection([diff0, rect])

    points, cells = dfmesh.generate(
        quarter,
        lambda x: h + 0.1 * np.abs(disk0.dist(x)),
        tol=1.0e-10,
        max_steps=100,
    )

    ref_norms = [7.7455372708027483e01, 6.5770003813066431e00, 1.0000000000000000e00]
    assert_norm_equality(points.flatten(), ref_norms, 1.0e-2)
    return points, cells


if __name__ == "__main__":
    import meshio

    points, cells = test_quarter_annulus()
    meshio.Mesh(points, {"triangle": cells}).write("out.vtk")
