// End-to-end check of the web app in headless Chromium (Playwright).
//
//   node web/tests/e2e.cjs [screenshot-dir]
//
// Part 1 loads the real worker.js. Set PYODIDE_DIR to a local Pyodide
// distribution (e.g. the "pyodide" npm package) to boot from it; it checks
// that Python starts and then reports how far start-up got. The npm package
// has no numpy/scipy, so with it start-up is expected to stop there.
//
// Part 2 swaps worker.js for fake_worker.js, which runs dfmesh_bridge.py under
// native CPython through server.py, and drives the UI: every example, drawing
// shapes, and error display.
const { chromium } = require("playwright");
const { spawn } = require("child_process");
const path = require("path");
const fs = require("fs");

const ROOT = path.resolve(__dirname, "..", "..");
const SHOTS = process.argv[2] || path.join(__dirname, "screenshots");
const PORT = 8000 + Math.floor(Math.random() * 1000);
const BASE = `http://127.0.0.1:${PORT}`;

let failures = 0;
function check(cond, msg) {
  console.log(`${cond ? "ok  " : "FAIL"} ${msg}`);
  if (!cond) failures++;
}

async function startServer() {
  const proc = spawn("python3", [path.join(__dirname, "server.py"), String(PORT)], {
    cwd: ROOT,
    env: process.env,
    stdio: ["ignore", "pipe", "inherit"],
  });
  await new Promise((resolve) => proc.stdout.once("data", resolve));
  return proc;
}

async function realWorker(browser) {
  if (!process.env.PYODIDE_DIR) {
    console.log("skip real worker (PYODIDE_DIR not set)");
    return;
  }
  const page = await browser.newPage();
  const requested = [];
  page.on("request", (r) => requested.push(r.url()));
  const statuses = [];
  await page.exposeFunction("__status", (s) => statuses.push(s));
  await page.addInitScript(() => {
    new MutationObserver(() => {
      const el = document.getElementById("status");
      if (el) window.__status(el.textContent);
    }).observe(document, { subtree: true, childList: true, characterData: true });
  });
  await page.goto(`${BASE}/web/?pyodide=/pyodide/`);
  await page.waitForFunction(
    () => /Ready|Could not start/.test(document.getElementById("status").textContent),
    null,
    { timeout: 120000 },
  );
  const status = await page.textContent("#status");
  const log = await page.textContent("#log");
  console.log(`  status sequence: ${[...new Set(statuses)].join(" -> ")}`);
  console.log(`  final log: ${log.trim().slice(0, 300)}`);
  check(statuses.some((s) => s.includes("Loading numpy and scipy")), "Pyodide core booted in a worker");
  const wheels = requested.filter((u) => /\/(numpy|scipy)-[^/]*\.whl/.test(u));
  console.log(`  wheel requests: ${wheels.map((u) => u.split("/").pop()).join(", ") || "none"}`);
  if (status.startsWith("Ready")) {
    check(true, "real Pyodide stack is ready");
  } else {
    check(log.includes("Start-up failed"), "start-up failure is reported in the UI");
  }
  await page.close();
}

async function waitMesh(page) {
  await page.waitForFunction(
    () => /Nodes/.test(document.getElementById("stats").textContent) ||
      /failed/.test(document.getElementById("status").textContent),
    null,
    { timeout: 60000 },
  );
  return page.textContent("#stats");
}

async function fakeWorker(browser) {
  const page = await browser.newPage({ viewport: { width: 1400, height: 860 } });
  const errors = [];
  page.on("pageerror", (e) => errors.push(String(e)));
  page.on("console", (m) => m.type() === "error" && errors.push(m.text()));
  await page.route("**/web/worker.js*", (route) =>
    route.fulfill({ path: path.join(__dirname, "fake_worker.js"), contentType: "text/javascript" }),
  );
  await page.goto(`${BASE}/web/`);

  let stats = await waitMesh(page);
  check(stats.includes("Nodes"), `first example meshed on load (${stats.replace(/\s+/g, " ")})`);
  await page.screenshot({ path: path.join(SHOTS, "01-circle.png") });

  const names = await page.$$eval("#example option", (os) => os.map((o) => o.textContent));
  for (let i = 0; i < names.length; i++) {
    await page.selectOption("#example", String(i));
    stats = await waitMesh(page);
    const hint = await page.isVisible("#hint");
    check(stats.includes("Nodes") && !hint, `example "${names[i]}": ${stats.replace(/\s+/g, " ")}`);
    await page.screenshot({ path: path.join(SHOTS, `ex-${String(i).padStart(2, "0")}.png`) });
  }

  // Edge size field
  await page.selectOption("#example", String(names.indexOf("Graded: refine near a line")));
  await waitMesh(page);
  await page.selectOption("#field-kind", "h");
  await page.waitForFunction(() => /edge size/.test(document.getElementById("legend-label").textContent), null, { timeout: 5000 });
  check(true, "edge size field can be plotted");
  await page.screenshot({ path: path.join(SHOTS, "05-edge-size.png") });
  await page.selectOption("#field-kind", "dist");

  // Drawing: start from the plate, subtract a circle and add a polygon
  await page.selectOption("#example", String(names.indexOf("Plate with holes")));
  await waitMesh(page);
  const box = await page.locator("#canvas").boundingBox();
  const cx = box.x + box.width * 0.5, cy = box.y + box.height * 0.6;

  await page.click('[data-op="-"]');
  await page.click('[data-tool="circle"]');
  await page.mouse.move(cx, cy - 120);
  await page.mouse.down();
  await page.mouse.move(cx + 30, cy - 120, { steps: 5 });
  await page.mouse.up();
  let code = await page.inputValue("#code");
  check(/geo = geo - df\.Circle\(\[/.test(code), "drawing a circle appends `geo = geo - df.Circle(...)`");

  await page.click('[data-op="+"]');
  await page.click('[data-tool="polygon"]');
  for (const [dx, dy] of [[-300, -200], [-200, -200], [-250, -280]]) {
    await page.mouse.click(cx + dx, cy + dy);
  }
  await page.mouse.click(cx - 300, cy - 200); // click the first corner to close
  code = await page.inputValue("#code");
  check(/geo = geo \+ df\.Polygon\(\[\[/.test(code), "drawing a polygon appends `geo = geo + df.Polygon(...)`");

  await page.click('[data-op="&"]');
  await page.click('[data-tool="rect"]');
  await page.mouse.move(box.x + 40, box.y + 70);
  await page.mouse.down();
  await page.mouse.move(box.x + box.width - 40, box.y + box.height - 40, { steps: 5 });
  await page.mouse.up();
  code = await page.inputValue("#code");
  check(/geo = geo & df\.Rectangle\(/.test(code), "drawing a rectangle appends `geo = geo & df.Rectangle(...)`");
  console.log("  code after drawing:\n" + code.split("\n").filter((l) => l.startsWith("geo")).map((l) => "    " + l).join("\n"));

  await page.waitForTimeout(800); // debounced field update
  await page.screenshot({ path: path.join(SHOTS, "02-drawn-field.png") });
  await page.click("#mesh");
  await page.waitForFunction(() => /Meshed|failed/.test(document.getElementById("status").textContent), null, { timeout: 60000 });
  check((await page.textContent("#status")).startsWith("Meshed"), "drawn geometry meshes");
  await page.screenshot({ path: path.join(SHOTS, "03-drawn-mesh.png") });

  // Undo removes the last drawn shape
  await page.click("#undo");
  code = await page.inputValue("#code");
  check(!/df\.Rectangle\(-?\d+\.\d+, /.test(code.split("# drawn").pop()), "undo removes the last drawn shape");

  // Errors are shown on the canvas
  await page.fill("#code", "geo = df.Circle([0, 0], ");
  await page.waitForSelector("#hint:not([hidden])", { timeout: 5000 });
  check(/SyntaxError/.test(await page.textContent("#hint")), "syntax errors are shown");
  await page.fill("#code", "h = 0.1\nview = [-1, 1, -1, 1]\n");
  await page.waitForFunction(() => /Draw a shape/.test(document.getElementById("hint").textContent), null, { timeout: 5000 });
  check(true, "empty geometry shows the drawing hint");
  await page.click('[data-tool="circle"]');
  await page.mouse.move(cx, cy);
  await page.mouse.down();
  await page.mouse.move(cx + 80, cy, { steps: 5 });
  await page.mouse.up();
  code = await page.inputValue("#code");
  check(/\ngeo = df\.Circle\(/.test(code), "first drawn shape becomes `geo = df.Circle(...)`");
  await page.waitForFunction(() => document.getElementById("hint").hidden, null, { timeout: 5000 });
  check(true, "field appears after drawing on an empty canvas");

  check(errors.length === 0, `no page errors${errors.length ? ": " + errors.join("; ") : ""}`);
  await page.close();

  const mobile = await browser.newPage({ viewport: { width: 390, height: 844 } });
  await mobile.route("**/web/worker.js*", (route) =>
    route.fulfill({ path: path.join(__dirname, "fake_worker.js"), contentType: "text/javascript" }),
  );
  await mobile.goto(`${BASE}/web/`);
  await waitMesh(mobile);
  const overflow = await mobile.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
  check(overflow <= 0, `no horizontal scroll at phone width (overflow ${overflow}px)`);
  await mobile.screenshot({ path: path.join(SHOTS, "04-mobile.png"), fullPage: true });
  await mobile.close();
}

(async () => {
  fs.mkdirSync(SHOTS, { recursive: true });
  const server = await startServer();
  const browser = await chromium.launch();
  try {
    await realWorker(browser);
    await fakeWorker(browser);
  } finally {
    await browser.close();
    server.kill();
  }
  console.log(failures ? `${failures} check(s) failed` : "all checks passed");
  process.exit(failures ? 1 : 0);
})();
