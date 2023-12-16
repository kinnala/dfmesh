import numpy as np
from helpers import assert_norm_equality

import dfmesh


def test(show=False):
    geo = dfmesh.Polygon(
        [
            [0.0, 0.0],
            [1.1, 0.0],
            [1.2, 0.5],
            [0.7, 0.6],
            [2.0, 1.0],
            [1.0, 2.0],
            [0.5, 1.5],
        ]
    )
    # geo.show()
    X, cells = dfmesh.generate(geo, 0.1, show=show, max_steps=100)

    ref_norms = [4.1426056822140765e02, 2.1830112296142847e01, 2.0000000000000000e00]
    assert_norm_equality(X.flatten(), ref_norms, 1.0e-5)
    return X, cells


def test_boundary_step2(plot=False):
    geo = dfmesh.Polygon(
        [
            [0.0, 0.0],
            [1.1, 0.0],
            [1.2, 0.5],
            [0.7, 0.6],
            [2.0, 1.0],
            [1.0, 2.0],
            [0.5, 1.5],
        ]
    )
    np.random.seed(0)
    pts = np.random.uniform(-2.0, 2.0, (2, 100))
    pts = geo.boundary_step(pts)
    if plot:
        geo.plot()
        import matplotlib.pyplot as plt

        plt.plot(pts[0], pts[1], "xk")
        plt.show()
    dist = geo.dist(pts)
    assert np.all(np.abs(dist) < 1.0e-12)


if __name__ == "__main__":
    # from helpers import save
    # X, cells = test(show=False)
    # save("polygon.svg", X, cells)
    test_boundary_step2(plot=True)
