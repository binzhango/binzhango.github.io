#!/usr/bin/env python3
import json
import sys
from pathlib import Path


def fail(message: str) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(1)


def main() -> None:
    if len(sys.argv) != 2:
        fail("usage: check_scene.py path/to/scene.json")
    path = Path(sys.argv[1])
    try:
        scene = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        fail(str(exc))

    canvas = scene.get("canvas", {})
    width, height = canvas.get("width", 0), canvas.get("height", 0)
    if width < 800 or height < 450:
        fail("canvas must be at least 800x450")
    nodes = scene.get("nodes", [])
    ids = [node.get("id") for node in nodes]
    if not 3 <= len(nodes) <= 9:
        fail("use 3–9 primary nodes")
    if None in ids or len(ids) != len(set(ids)):
        fail("every node needs a unique id")
    valid_accents = {"blue", "violet", "green", "neutral"}
    for node in nodes:
        if node.get("accent", "neutral") not in valid_accents:
            fail(f"node {node['id']} has an unsupported accent")
        for key in ("x", "y", "w", "h"):
            if not isinstance(node.get(key), (int, float)):
                fail(f"node {node['id']} needs numeric {key}")
        if node["x"] < 48 or node["y"] < 48 or node["x"] + node["w"] > width - 48 or node["y"] + node["h"] > height - 48:
            fail(f"node {node['id']} violates the 48px canvas safe area")
        if len(node.get("title", "")) > 28 or len(node.get("detail", "")) > 52:
            fail(f"node {node['id']} copy is too long")
    known = set(ids)
    for edge in scene.get("edges", []):
        if edge.get("from") not in known or edge.get("to") not in known:
            fail("every edge endpoint must reference a node id")
    for node_id in scene.get("sequence", ids):
        if node_id not in known:
            fail(f"sequence references unknown node {node_id}")
    print(f"OK: {len(nodes)} nodes, {len(scene.get('edges', []))} edges, {width}x{height}")


if __name__ == "__main__":
    main()

