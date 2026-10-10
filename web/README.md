# dfmesh in the browser

A static web app that runs dfmesh with [Pyodide](https://pyodide.org/)
(CPython, numpy and scipy compiled to WebAssembly). Nothing is installed and
no server-side Python is needed.

- Write the geometry in Python in the side panel, or draw circles, rectangles
  and polygons on the canvas. Each drawn shape is appended to the code as a
  set operation (`geo = geo + ...`, `geo - ...`, `geo & ...`), so the code is
  always the full description of the geometry and mesh.
- The signed distance field of `geo` (or the target edge size `h`) is
  plotted with contour lines; the zero contour is the domain boundary.
  It updates as you type.
- **Generate mesh** (Ctrl+Enter) runs `dfmesh.triangulate` and draws the
  result, with node and triangle counts and element quality
  (radius ratio, 1 = equilateral).
- **Copy script** copies a standalone Python script; **Download mesh** saves
  the points and triangles as JSON.
- The example menu has a few default geometries.

## Running locally

The app loads the dfmesh sources from `../src/dfmesh/`, so serve the
repository root (not just this folder) and open `/web/`:

```sh
python -m http.server 8000
# open http://localhost:8000/web/
```

The first load downloads Pyodide, numpy and scipy (a few tens of MB) from
cdn.jsdelivr.net; later loads come from the browser cache. Meshing runs in a
Web Worker, so the page stays responsive; **Stop** restarts Python if a run
takes too long.

To host it, publish the repository root as a static site (e.g. GitHub Pages
from the branch root) and open `<site>/web/`.

URL parameters:

- `?pyodide=<url>` loads Pyodide from another location (must contain the
  full distribution, including the numpy and scipy wheels).
- `?src=<url>` loads the dfmesh sources from another location.

## Files

- `index.html`, `style.css`, `app.js`: the UI (canvas rendering, drawing
  tools, code panel).
- `worker.js`: starts Pyodide, loads numpy, scipy and the dfmesh sources.
- `dfmesh_bridge.py`: runs the user code and returns the field / mesh as
  JSON. It is plain Python, so it is also tested under CPython.
- `examples.js`: the default examples.
- `tests/`: `test_bridge.py` (pytest, runs every example under CPython) and
  `e2e.cjs` (Playwright, drives the UI in headless Chromium with a native
  Python backend in place of Pyodide; see the comments in the file).
