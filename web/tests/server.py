"""Static file server for the web app, plus a native-Python stand-in for Pyodide.

Usage (from the repository root):

    python web/tests/server.py [port]

Then open http://localhost:8000/web/ to use the app with Pyodide as usual.

For the end-to-end test (web/tests/e2e.mjs), POST /api/field and /api/mesh
run dfmesh_bridge.py under regular CPython, so the UI can be exercised without
downloading Pyodide's numpy and scipy (web/tests/fake_worker.js forwards to
these endpoints).  Optionally serves a local Pyodide copy at /pyodide/ when
the PYODIDE_DIR environment variable points to one.
"""

import functools
import http.server
import json
import os
import sys
import threading

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path[:0] = [os.path.join(ROOT, "src"), os.path.join(ROOT, "web")]

import dfmesh_bridge  # noqa: E402

LOCK = threading.Lock()
PYODIDE_DIR = os.environ.get("PYODIDE_DIR")


class Handler(http.server.SimpleHTTPRequestHandler):
    def translate_path(self, path):
        if PYODIDE_DIR and path.startswith("/pyodide/"):
            rel = path[len("/pyodide/"):].split("?", 1)[0]
            return os.path.join(PYODIDE_DIR, rel)
        return super().translate_path(path)

    def do_POST(self):
        cmd = self.path.rsplit("/", 1)[-1]
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        with LOCK:
            if cmd == "field":
                out = dfmesh_bridge.field(
                    body["code"], body.get("resolution", 200), body.get("kind", "dist")
                )
            elif cmd == "mesh":
                out = dfmesh_bridge.mesh(body["code"])
            else:
                self.send_error(404)
                return
        data = out.encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, *args):
        pass


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    handler = functools.partial(Handler, directory=ROOT)
    Handler.extensions_map = {
        **http.server.SimpleHTTPRequestHandler.extensions_map,
        ".wasm": "application/wasm",
        ".mjs": "text/javascript",
        ".js": "text/javascript",
    }
    print(f"Serving {ROOT} on http://localhost:{port}/web/", flush=True)
    http.server.ThreadingHTTPServer(("127.0.0.1", port), handler).serve_forever()
