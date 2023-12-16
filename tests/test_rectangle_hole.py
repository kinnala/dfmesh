from helpers import assert_norm_equality

import dfmesh


def test_rectangle_hole(show=False):
    geo = dfmesh.Difference(
        dfmesh.Rectangle(60, 330, 380, 650), dfmesh.Rectangle(143, 245, 440, 543)
    )

    X, cells = dfmesh.generate(
        geo, 20, tol=1.0e-5, show=show, flip_tol=1.0e-10, max_steps=100
    )

    ref_norms = [1.2931633675576400e05, 7.6377328985582844e03, 6.5000000000000000e02]
    assert_norm_equality(X.flatten(), ref_norms, 1.0e-10)


def test_rectangle_hole2(show=False):
    geo = dfmesh.Difference(
        dfmesh.Rectangle(0.0, 5.0, 0.0, 5.0),
        dfmesh.Polygon([[1, 1], [4, 1], [4, 4], [1, 4]]),
    )
    X, cells = dfmesh.generate(geo, 1.0, show=show, tol=1.0e-3, max_steps=100)

    ref_norms = [1.3990406144096474e02, 2.2917592510234346e01, 5.0000000000000000e00]
    assert_norm_equality(X.flatten(), ref_norms, 1.0e-2)


if __name__ == "__main__":
    test_rectangle_hole2(show=True)
