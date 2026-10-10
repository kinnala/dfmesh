"""Glue between the browser UI and dfmesh.

The same module runs inside Pyodide (see worker.js) and under regular CPython
(see web/tests/), so it must not depend on anything browser specific.

User code is plain Python that must define ``geo`` (a dfmesh geometry).  It
may also define ``h`` (target edge size, a float or a callable taking an
array of shape (2, N)), ``options`` (extra keyword arguments passed to
``dfmesh.triangulate``) and ``view`` (``[x0, x1, y0, y1]``, the plotting
window, needed e.g. for unbounded geometries such as half spaces).
"""

import contextlib
import io
import json
import math
import time

import numpy as np

import dfmesh


def _run_user_code(code):
    ns = {"df": dfmesh, "dfmesh": dfmesh, "np": np, "math": math}
    exec(compile(code, "<geometry>", "exec"), ns)
    if "geo" not in ns:
        raise ValueError("The code must define a variable named `geo`.")
    if not isinstance(ns["geo"], dfmesh.Geometry):
        raise TypeError("`geo` must be a dfmesh geometry, e.g. df.Circle(...).")
    return ns


def _view(ns, pad=0.15):
    if "view" in ns:
        x0, x1, y0, y1 = [float(v) for v in ns["view"]]
    else:
        x0, x1, y0, y1 = [float(v) for v in ns["geo"].bounding_box]
        if not all(math.isfinite(v) for v in (x0, x1, y0, y1)):
            raise ValueError(
                "The geometry is unbounded; define `view = [x0, x1, y0, y1]`."
            )
        w, h = x1 - x0, y1 - y0
        m = pad * max(w, h)
        x0, x1, y0, y1 = x0 - m, x1 + m, y0 - m, y1 + m
    if not (x1 > x0 and y1 > y0):
        raise ValueError(f"Invalid view {[x0, x1, y0, y1]}.")
    return [x0, x1, y0, y1]


def _respond(fn):
    """Run fn(), capturing stdout, and return a JSON string."""
    out = io.StringIO()
    t0 = time.perf_counter()
    try:
        with contextlib.redirect_stdout(out):
            result = fn()
        result["ok"] = True
    except Exception as e:  # report every user error back to the UI
        import traceback

        tb = traceback.format_exc()
        # drop the frames belonging to this module
        lines = tb.splitlines()
        keep = [i for i, ln in enumerate(lines) if '"<geometry>"' in ln]
        if keep:
            tb = "\n".join(lines[keep[0]:])
        result = {"ok": False, "error": f"{type(e).__name__}: {e}", "traceback": tb}
    result["elapsed"] = time.perf_counter() - t0
    result["log"] = out.getvalue()
    return json.dumps(result)


def _edge_size(ns, xy):
    h = ns.get("h", 0.1)
    if callable(h):
        return np.asarray(h(xy), dtype=float)
    return np.full(xy.shape[1], float(h))


def field(code, resolution=200, kind="dist"):
    """Evaluate a field on a grid covering the view.

    kind="dist" gives the signed distance of ``geo``; kind="h" gives the
    target edge size ``h`` (inside the domain only).
    """

    def run():
        ns = _run_user_code(code)
        x0, x1, y0, y1 = _view(ns)
        w, h = x1 - x0, y1 - y0
        if w >= h:
            nx, ny = resolution, max(2, round(resolution * h / w))
        else:
            nx, ny = max(2, round(resolution * w / h)), resolution
        X, Y = np.meshgrid(np.linspace(x0, x1, nx), np.linspace(y0, y1, ny))
        # evaluate on a flat (2, N) array; not every geometry handles (2, ny, nx)
        xy = np.array([X.ravel(), Y.ravel()])
        d = np.asarray(ns["geo"].dist(xy), dtype=float)
        if kind == "h":
            d = np.where(d <= 0.0, _edge_size(ns, xy), np.nan)
        Z = d.reshape(X.shape)
        Z = np.where(np.isfinite(Z), Z, np.nan)
        return {
            "kind": kind,
            "view": [x0, x1, y0, y1],
            "nx": nx,
            "ny": ny,
            # row-major, row 0 is y = y0
            "values": [None if math.isnan(v) else v for v in Z.ravel().tolist()],
        }

    return _respond(run)


def _quality(p, t):
    """Radius ratio 2 r_in / r_circ per triangle (1 = equilateral)."""
    a = np.linalg.norm(p[:, t[1]] - p[:, t[2]], axis=0)
    b = np.linalg.norm(p[:, t[2]] - p[:, t[0]], axis=0)
    c = np.linalg.norm(p[:, t[0]] - p[:, t[1]], axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        q = (b + c - a) * (c + a - b) * (a + b - c) / (a * b * c)
    return np.nan_to_num(q)


def mesh(code):
    """Run dfmesh.triangulate on ``geo`` with edge size ``h``."""

    def run():
        ns = _run_user_code(code)
        h = ns.get("h", 0.1)
        options = dict(ns.get("options", {}))
        options.pop("show", None)
        p, t = dfmesh.triangulate(ns["geo"], h, **options)
        q = _quality(p, t)
        return {
            "view": _view(ns),
            "points": p.T.tolist(),
            "triangles": t.T.astype(int).tolist(),
            "stats": {
                "nodes": int(p.shape[1]),
                "triangles": int(t.shape[1]),
                "q_min": float(q.min()) if q.size else 0.0,
                "q_mean": float(q.mean()) if q.size else 0.0,
            },
        }

    return _respond(run)
