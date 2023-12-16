from helpers import assert_norm_equality, save

import dfmesh


def test(show=False):
    # should both work
    geo = dfmesh.Rectangle(-1.0, +2.0, -1.0, +1.0) * 2.0
    geo = 2.0 * dfmesh.Rectangle(-1.0, +2.0, -1.0, +1.0)

    X, cells = dfmesh.generate(geo, 0.1, show=show, tol=1.0e-5, max_steps=100)

    ref_norms = [7.6829959173892494e03, 1.2466061090733828e02, 4.0000000000000000e00]
    assert_norm_equality(X.flatten(), ref_norms, 1.0e-7)
    return X, cells


if __name__ == "__main__":
    X, cells = test(show=False)
    save("scaling.png", X, cells)
