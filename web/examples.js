// Default examples. Each one is plain Python run by dfmesh_bridge.py: it must
// define `geo` and may define `h`, `options` and `view`.
const EXAMPLES = [
  {
    name: "Circle",
    code: `# Unit disk with uniform elements
geo = df.Circle([0.0, 0.0], 1.0)

h = 0.1  # target edge length
`,
  },
  {
    name: "Pac-Man (difference)",
    code: `# Set difference: a disk minus a triangle
geo = df.Circle([0.0, 0.0], 1.0) - df.Polygon([[0.0, 0.0], [1.5, 0.4], [1.5, -0.4]])

h = 0.08
options = {"max_steps": 300}
`,
  },
  {
    name: "Plate with holes",
    code: `# A rectangular plate with three circular holes
geo = df.Rectangle(-2.0, 2.0, -1.0, 1.0)
for x in [-1.2, 0.0, 1.2]:
    geo = geo - df.Circle([x, 0.0], 0.4)

h = 0.1
`,
  },
  {
    name: "Union and intersection",
    code: `# Two overlapping disks joined, then cut by a third
a = df.Circle([-0.5, 0.0], 1.0)
b = df.Circle([0.5, 0.0], 1.0)
c = df.Circle([0.0, -1.4], 1.6)

geo = (a + b) & c

h = 0.1
`,
  },
  {
    name: "Polygon",
    code: `# A non-convex polygon (corners become feature points)
geo = df.Polygon([
    [0.0, 0.0],
    [1.1, 0.0],
    [1.2, 0.5],
    [0.7, 0.6],
    [2.0, 1.0],
    [1.0, 2.0],
    [0.5, 1.5],
])

h = 0.07
`,
  },
  {
    name: "Rotate, translate, scale",
    code: `# Transformations: build a shape, then rotate, scale and translate it.
# (dfmesh supports set operations on primitives and rotations, so do the
# set operations first and the scaling/translation last.)
shape = df.Rectangle(-1.0, 1.0, -0.4, 0.4) - df.Circle([0.5, 0.0], 0.2)
geo = [0.5, 0.25] + 0.8 * df.Rotation(shape, math.pi / 6)

h = 0.06
options = {"max_steps": 300}
`,
  },
  {
    name: "Graded: refine near a line",
    code: `# Nonuniform elements: h is a function of position
geo = df.Rectangle(0.0, 1.0, 0.0, 1.0)
crack = df.Path([[0.4, 0.6], [0.6, 0.4]])

def h(x):
    # x has shape (2, N)
    return 0.02 + 0.15 * crack.dist(x)

options = {"tol": 1e-6, "max_steps": 200}
`,
  },
  {
    name: "Quarter annulus (graded)",
    code: `# Intersection of an annulus and a square, refined toward the inner arc
inner = df.Circle([0.0, 0.0], 0.25)
outer = df.Circle([0.0, 0.0], 1.0)
geo = (outer - inner) & df.Rectangle(0.0, 1.0, 0.0, 1.0)

h = lambda x: 0.03 + 0.15 * np.abs(inner.dist(x))

options = {"max_steps": 200}
`,
  },
  {
    name: "Half space cut",
    code: `# A half space is unbounded; intersect it with a disk to get a domain
cut = df.HalfSpace(np.sqrt(0.5) * np.array([1.0, 1.0]))
geo = cut & df.Circle([0.0, 0.0], 1.0)

h = 0.08
`,
  },
];
