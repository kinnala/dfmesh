"""Run the web app's examples through dfmesh_bridge under CPython."""

import json
import pathlib
import re
import sys

import pytest

WEB = pathlib.Path(__file__).resolve().parents[1]
SRC = WEB.parent / "src"
sys.path[:0] = [str(SRC), str(WEB)]

import dfmesh_bridge  # noqa: E402

EXAMPLES = re.findall(
    r'name: "([^"]+)",\s*code: `([^`]*)`', (WEB / "examples.js").read_text()
)


def test_examples_found():
    assert len(EXAMPLES) >= 5


@pytest.mark.parametrize("name,code", EXAMPLES, ids=[e[0] for e in EXAMPLES])
def test_example(name, code):
    for kind in ["dist", "h"]:
        f = json.loads(dfmesh_bridge.field(code, 60, kind))
        assert f["ok"], f.get("traceback")
        assert len(f["values"]) == f["nx"] * f["ny"]
        assert any(v is not None and v < 0 for v in f["values"]) or kind == "h"
    m = json.loads(dfmesh_bridge.mesh(code))
    assert m["ok"], m.get("traceback")
    assert m["stats"]["triangles"] > 0
    assert m["stats"]["q_min"] > 0.5


def test_errors_are_reported():
    r = json.loads(dfmesh_bridge.field("h = 0.1"))
    assert not r["ok"] and "geo" in r["error"]
    r = json.loads(dfmesh_bridge.mesh("geo = df.Circle([0, 0], "))
    assert not r["ok"] and r["error"].startswith("SyntaxError")
    r = json.loads(dfmesh_bridge.field("geo = df.HalfSpace(np.array([1.0, 0.0]))"))
    assert not r["ok"] and "view" in r["error"]


def test_worker_file_list_matches_package():
    listed = re.search(
        r"const DFMESH_FILES = \[(.*?)\];", (WEB / "worker.js").read_text(), re.S
    ).group(1)
    listed = sorted(re.findall(r'"([^"]+)"', listed))
    package = sorted(
        p.relative_to(SRC / "dfmesh").as_posix()
        for p in (SRC / "dfmesh").rglob("*.py")
    )
    assert listed == package
