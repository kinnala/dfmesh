import numpy as np
from helpers import assert_norm_equality, save

import dfmesh


def test(show=False):
    geo = dfmesh.Rotation(dfmesh.Rectangle(-1.0, +2.0, -1.0, +1.0), 0.1 * np.pi)
    X, cells = dfmesh.generate(geo, 0.1, show=show, tol=1.0e-10, max_steps=100)

    ref_norms = [9.4730152857365385e02, 3.1160562530932285e01, 2.2111300269652543e00]
    assert_norm_equality(X.flatten(), ref_norms, 1.0e-10)
    return X, cells


if __name__ == "__main__":
    X, cells = test(show=False)
    save("rotation.png", X, cells)
