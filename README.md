# dfmesh

A mesh generator inspired by [distmesh](http://persson.berkeley.edu/distmesh/).

## Installation

```
pip install git+https://github.com/kinnala/dfmesh.git
```

## Examples

### Basic usage

```python
import dfmesh as df
geo = df.Circle([0., 0.], 1.)
p, t = df.triangulate(geo, 0.1)  # edge length
```

### Primitives

```python
import dfmesh as df
geo = df.Circle([0., 0.], 1.)
geo = df.Rectangle(-1., 2., -1., 1.)
geo = df.Polygon([[0.0, 0.0],
                  [1.1, 0.0],
                  [1.2, 0.5],
                  [0.7, 0.6],
                  [2.0, 1.0],
                  [1.0, 2.0],
                  [0.5, 1.5]])
geo = df.HalfSpace([1., 1.])  # normal vector (1, 1)
```

### Set operations

```python
import dfmesh as df
geo = df.Circle([-.5, 0.], 1.) - df.Circle([.5, 0.], 1.)
geo = df.Circle([-.5, 0.], 1.) + df.Circle([.5, 0.], 1.)
geo = df.Circle([0., -.5], 1.) & df.Circle([0., .5], 1.)  # intersection
```

### Rotate, translate, scale

```python
import dfmesh as df
geo = df.Rotation(df.Rectangle(-1., 2., -1., 1.), 3.1415 / 4.)
geo = [1., 1.] + df.Rectangle(-1., 2., -1., 1.)
geo = 2. * df.Rectangle(-1., 2., -1., 1.)
```

### Nonuniform elements

```python
import dfmesh as df
geo = df.Rectangle(-1., 2., -1., 1.)
path = df.Path([[.4, .6], [.6, .4]])
p, t = df.triangulate(geo, lambda x: .03 + .1 * path.dist(x))
```

## License

This is a fork of dmsh 0.2.19 and meshplex 0.17.1 (c) Nico Schlömer which
are both GPLv3:

- [Debian package python3-dmsh](https://packages.debian.org/bookworm/python3-dmsh)
- [Debian package python3-meshplex](https://packages.debian.org/bookworm/python3-meshplex)
- [Ubuntu package python-dmsh](https://launchpad.net/ubuntu/+source/python-dmsh)
- [Zenodo entry of dmsh 0.2.19](https://zenodo.org/records/6053014)
- [Pypi entry of dmsh 0.2.18](https://web.archive.org/web/20211204014637/https://pypi.org/project/dmsh/)

The later versions of dmsh use a proprietary license.
