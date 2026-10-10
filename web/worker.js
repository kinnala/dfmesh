// Web worker that runs Pyodide, numpy, scipy and dfmesh off the UI thread.
//
// Messages in:  {id, cmd: "field" | "mesh", code, resolution?, kind?}
// Messages out: {type: "status", text}  while starting up
//               {type: "ready", versions}
//               {type: "result", id, cmd, payload}  (payload is a JSON string
//                from dfmesh_bridge.py)
//               {type: "fatal", error}  if start-up fails

const params = new URLSearchParams(self.location.search);
const PYODIDE_URL =
  params.get("pyodide") || "https://cdn.jsdelivr.net/pyodide/v0.29.3/full/";
// Location of the dfmesh package, relative to this file: ../src/dfmesh/ when
// the repository root is served, dfmesh/ in the GitHub Pages build (see
// .github/workflows/pages.yml). ?src=<url> overrides both.
const DFMESH_SRC_CANDIDATES = params.get("src")
  ? [params.get("src")]
  : ["../src/dfmesh/", "dfmesh/"];

// The files of src/dfmesh. Keep in sync with the package
// (web/tests/test_bridge.py checks this).
const DFMESH_FILES = [
  "__about__.py",
  "__init__.py",
  "_helpers.py",
  "_mesh.py",
  "_mesh_tri.py",
  "main.py",
  "geometry/__init__.py",
  "geometry/circle.py",
  "geometry/ellipse.py",
  "geometry/geometry.py",
  "geometry/halfspace.py",
  "geometry/path.py",
  "geometry/polygon.py",
  "geometry/rectangle.py",
  "geometry/rotation.py",
  "geometry/pypathlib/__init__.py",
  "geometry/pypathlib/closed_path.py",
  "geometry/pypathlib/helpers.py",
  "geometry/pypathlib/path.py",
];

const SITE = "/home/pyodide/dfmesh_site";

let pyodide = null;
let bridge = null;
const queue = [];

function status(text) {
  self.postMessage({ type: "status", text });
}

async function fetchText(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Could not load ${url} (HTTP ${res.status})`);
  return res.text();
}

async function findDfmeshSource() {
  for (const src of DFMESH_SRC_CANDIDATES) {
    try {
      const res = await fetch(src + "__init__.py", { method: "HEAD" });
      if (res.ok) return src;
    } catch (e) {
      // try the next location
    }
  }
  throw new Error(
    `Could not find the dfmesh sources (tried ${DFMESH_SRC_CANDIDATES.join(", ")}). ` +
      "On GitHub Pages, make sure files starting with an underscore are published " +
      "(add a .nojekyll file).",
  );
}

async function start() {
  status("Loading Pyodide…");
  importScripts(PYODIDE_URL + "pyodide.js");
  pyodide = await loadPyodide({ indexURL: PYODIDE_URL });

  status("Loading numpy and scipy…");
  const loadErrors = [];
  await pyodide.loadPackage(["numpy", "scipy"], {
    errorCallback: (m) => loadErrors.push(m),
  });
  if (!pyodide.loadedPackages.numpy || !pyodide.loadedPackages.scipy) {
    throw new Error(
      `Could not load numpy/scipy from ${PYODIDE_URL}\n${loadErrors.join("\n")}`,
    );
  }

  status("Loading dfmesh…");
  const src = await findDfmeshSource();
  const sources = await Promise.all(DFMESH_FILES.map((f) => fetchText(src + f)));
  DFMESH_FILES.forEach((f, i) => {
    const path = `${SITE}/dfmesh/${f}`;
    pyodide.FS.mkdirTree(path.slice(0, path.lastIndexOf("/")));
    pyodide.FS.writeFile(path, sources[i]);
  });
  pyodide.FS.writeFile(`${SITE}/dfmesh_bridge.py`, await fetchText("dfmesh_bridge.py"));

  pyodide.runPython(`
import sys
sys.path.insert(0, "${SITE}")
import dfmesh_bridge
`);
  bridge = pyodide.pyimport("dfmesh_bridge");
  const versions = JSON.parse(
    pyodide.runPython(`
import json, sys, numpy, scipy, dfmesh
json.dumps({"python": sys.version.split()[0], "numpy": numpy.__version__,
            "scipy": scipy.__version__, "dfmesh": dfmesh.__version__})
`),
  );
  self.postMessage({ type: "ready", versions: { pyodide: pyodide.version, ...versions } });
  while (queue.length) handle(queue.shift());
}

function handle(msg) {
  let payload;
  if (msg.cmd === "field") payload = bridge.field(msg.code, msg.resolution || 200, msg.kind || "dist");
  else if (msg.cmd === "mesh") payload = bridge.mesh(msg.code);
  else payload = JSON.stringify({ ok: false, error: `Unknown command ${msg.cmd}` });
  self.postMessage({ type: "result", id: msg.id, cmd: msg.cmd, payload });
}

self.onmessage = (e) => {
  if (bridge) handle(e.data);
  else queue.push(e.data);
};

start().catch((err) => {
  self.postMessage({ type: "fatal", error: String(err && err.message ? err.message : err) });
});
