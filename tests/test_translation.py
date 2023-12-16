from helpers import assert_norm_equality

import dfmesh


def test(show=False):
    # should both work
    geo = [1.0, 1.0] + dfmesh.Rectangle(-1.0, +2.0, -1.0, +1.0)
    geo = dfmesh.Rectangle(-1.0, +2.0, -1.0, +1.0) + [1.0, 1.0]

    X, _ = dfmesh.generate(geo, 0.1, show=show, max_steps=100)

    ref_norms = [1.7524999999999998e03, 5.5612899955332637e01, 3.0000000000000000e00]
    assert_norm_equality(X.flatten(), ref_norms, 1.0e-7)


if __name__ == "__main__":
    test(show=False)
