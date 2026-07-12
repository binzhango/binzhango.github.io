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

    layout = scene.get("layout", {})
    mode = layout.get("mode", "manual")
    density = layout.get("density", "overview")
    if mode not in {"manual", "auto", "flow", "tree"}:
        fail("layout.mode must be manual, auto, flow, or tree")
    if density not in {"overview", "reference"}:
        fail("layout.density must be overview or reference")
    if layout.get("direction", "TB" if mode == "tree" else "LR") not in {"LR", "TB"}:
        fail("layout.direction must be LR or TB")

    nodes = scene.get("nodes", [])
    ids = [node.get("id") for node in nodes]
    max_nodes = 30 if density == "reference" else 12
    if not 3 <= len(nodes) <= max_nodes:
        fail(f"use 3–{max_nodes} primary nodes for {density} density")
    if None in ids or len(ids) != len(set(ids)):
        fail("every node needs a unique id")

    valid_accents = {"blue", "violet", "green", "neutral"}
    for node in nodes:
        if node.get("accent", "neutral") not in valid_accents:
            fail(f"node {node['id']} has an unsupported accent")
        if len(node.get("title", "")) > 28 or len(node.get("detail", "")) > 52:
            fail(f"node {node['id']} copy is too long")
        if mode == "manual":
            for key in ("x", "y", "w", "h"):
                if not isinstance(node.get(key), (int, float)):
                    fail(f"manual-layout node {node['id']} needs numeric {key}")
            if node["x"] < 48 or node["y"] < 48 or node["x"] + node["w"] > width - 48 or node["y"] + node["h"] > height - 48:
                fail(f"node {node['id']} violates the 48px canvas safe area")

    known = set(ids)
    incoming = {node_id: 0 for node_id in ids}
    outgoing = {node_id: [] for node_id in ids}
    for edge in scene.get("edges", []):
        source, target = edge.get("from"), edge.get("to")
        if source not in known or target not in known:
            fail("every edge endpoint must reference a node id")
        incoming[target] += 1
        outgoing[source].append(target)

    if mode == "tree":
        roots = [node_id for node_id, count in incoming.items() if count == 0]
        if len(roots) != 1:
            fail("tree layout requires exactly one root")
        if any(count > 1 for count in incoming.values()):
            fail("tree layout does not allow nodes with multiple parents")
        if len(scene.get("edges", [])) != len(nodes) - 1:
            fail("tree layout requires exactly nodes-1 edges")
        visited = set()

        def visit(node_id):
            if node_id in visited:
                fail("tree layout contains a cycle")
            visited.add(node_id)
            for child_id in outgoing[node_id]:
                visit(child_id)

        visit(roots[0])
        if visited != known:
            fail("tree layout must connect every node to the root")

    for node_id in scene.get("sequence", ids):
        if node_id not in known:
            fail(f"sequence references unknown node {node_id}")

    formats = scene.get("exports", {}).get("formats", ["mp4", "gif"])
    if not formats or any(fmt not in {"mp4", "webp", "gif"} for fmt in formats):
        fail("exports.formats must contain mp4, webp, and/or gif")

    resolved = "tree" if mode == "auto" and any(len(children) > 1 for children in outgoing.values()) else ("flow" if mode == "auto" else mode)
    print(f"OK: {len(nodes)} nodes, {len(scene.get('edges', []))} edges, {width}x{height}, layout={resolved}, exports={','.join(formats)}")


if __name__ == "__main__":
    main()
