import dfmesh


def test_show():
    # geo = dfmesh.Circle([0.0, 0.0], 1.0)
    geo = dfmesh.Rectangle(-1.0, +1.0, -1.0, +1.0)
    geo.show()


if __name__ == "__main__":
    test_show()
