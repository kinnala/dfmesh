// Test double for worker.js: speaks the same message protocol, but runs
// dfmesh_bridge.py under native CPython via web/tests/server.py instead of
// Pyodide. Used by e2e.mjs only.
self.postMessage({ type: "status", text: "Loading (test backend)…" });
setTimeout(() => {
  self.postMessage({
    type: "ready",
    versions: { pyodide: "test", python: "native", numpy: "native", scipy: "native", dfmesh: "src" },
  });
}, 50);
self.onmessage = async (e) => {
  const msg = e.data;
  const res = await fetch(`/api/${msg.cmd}`, { method: "POST", body: JSON.stringify(msg) });
  self.postMessage({ type: "result", id: msg.id, cmd: msg.cmd, payload: await res.text() });
};
