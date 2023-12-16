import numpy as np
from helpers import assert_norm_equality, save

import dfmesh


def test_boundary_step():
    geo = dfmesh.Rectangle(-2.0, +2.0, -1.0, +1.0)

    # Check boundary steps
    out = geo.boundary_step([0.1, 0.0])
    assert np.all(np.abs(out - [2.0, 0.0]) < 1.0e-10)
    out = geo.boundary_step([0.0, 0.1])
    assert np.all(np.abs(out - [0.0, 1.0]) < 1.0e-10)
    out = geo.boundary_step([-0.1, 0.0])
    assert np.all(np.abs(out - [-2.0, 0.0]) < 1.0e-10)
    out = geo.boundary_step([0.0, -0.1])
    assert np.all(np.abs(out - [0.0, -1.0]) < 1.0e-10)

    out = geo.boundary_step([2.1, 0.037])
    assert np.all(np.abs(out - [2.0, 0.037]) < 1.0e-10)
    out = geo.boundary_step([0.037, 1.1])
    assert np.all(np.abs(out - [0.037, 1.0]) < 1.0e-10)
    out = geo.boundary_step([-2.1, 0.037])
    assert np.all(np.abs(out - [-2.0, 0.037]) < 1.0e-10)
    out = geo.boundary_step([0.037, -1.1])
    assert np.all(np.abs(out - [0.037, -1.0]) < 1.0e-10)

    out = geo.boundary_step([2.1, 1.1])
    assert np.all(np.abs(out - [2.0, 1.0]) < 1.0e-10)
    out = geo.boundary_step([-2.1, 1.1])
    assert np.all(np.abs(out - [-2.0, 1.0]) < 1.0e-10)
    out = geo.boundary_step([2.1, -1.1])
    assert np.all(np.abs(out - [2.0, -1.0]) < 1.0e-10)
    out = geo.boundary_step([-2.1, -1.1])
    assert np.all(np.abs(out - [-2.0, -1.0]) < 1.0e-10)


def test_rectangle(show=False):
    geo = dfmesh.Rectangle(-1.0, +2.0, -1.0, +1.0)
    X, cells = dfmesh.generate(geo, 0.1, show=show, max_steps=100)

    ref_norms = [9.7172325705673779e02, 3.1615286239175994e01, 2.0000000000000000e00]
    assert_norm_equality(X.flatten(), ref_norms, 1.0e-10)
    return X, cells


def test_duplicate_points(show=False):
    # https://github.com/nschloe/dfmesh/issues/66
    # geo = dfmesh.Rectangle(0.0, 1.8, 0.0, 0.41)

    # points, cells = dfmesh.generate(geo, 0.2, tol=2e-2, show=show)

    # is_part_of_cell = np.zeros(len(points), dtype=bool)
    # is_part_of_cell[cells.flat] = True
    # assert np.all(is_part_of_cell)

    geo = dfmesh.Rectangle(0.0, 1.4, 0.0, 0.41)
    points, cells = dfmesh.generate(geo, 0.025, tol=1e-5, show=show, max_steps=1)
    is_part_of_cell = np.zeros(len(points), dtype=bool)
    is_part_of_cell[cells.flat] = True
    assert np.all(is_part_of_cell)


if __name__ == "__main__":
    # test_duplicate_points(show=True)
    X, cells = test_rectangle(show=False)
    save("rectangle.png", X, cells)
