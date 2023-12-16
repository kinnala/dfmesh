from helpers import assert_norm_equality

import dfmesh


def test_pacman(show=False):
    geo = dfmesh.Difference(
        dfmesh.Circle([0.0, 0.0], 1.0),
        dfmesh.Polygon([[0.0, 0.0], [1.5, 0.4], [1.5, -0.4]]),
    )
    X, cells = dfmesh.generate(geo, 0.1, show=show, tol=1.0e-5, max_steps=100)

    ref_norms = [3.0173012692535394e02, 1.3565685453257570e01, 9.9999999999884770e-01]
    assert_norm_equality(X.flatten(), ref_norms, 1.0e-10)
    return X, cells


if __name__ == "__main__":
    X, cells = test_pacman(show=True)
    # from helpers import save
    # save("pacman.png", X, cells)
