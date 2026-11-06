"""Author the code-verified two-path comparison as editable draw.io cells."""
from pathlib import Path
from xml.etree import ElementTree as E
import html
import json

ROOT = Path(__file__).resolve().parent
OUT = ROOT.parents[2] / "public/diagrams/hindsight-jev"
nodes = [
 ("scope", "Request + user/project scope", "Application selects the memory bank", 30, 20, 620, 80, "base"),
 ("recall", "Hindsight recalls once", "Fact IDs + text form the shared evidence snapshot", 30, 155, 620, 80, "base"),
 ("preflight", "Jev preflight", "Proceed, clarify, or review · hosted prediction", 30, 290, 620, 80, "model"),
 ("worker", "Local worker drafts an answer", "Same question, instructions and memory evidence", 30, 435, 620, 80, "base"),
 ("review", "Jev reviews the draft", "Same snapshot + draft · hosted prediction", 30, 570, 620, 80, "model"),
 ("policy", "Python applies the gate", "Proceed with confidence ≥ 0.60 → accepted answer<br>Otherwise → clarify/review; draft kept separate", 30, 705, 620, 100, "policy"),
]
edges = [
 ("scope", "recall", "question", "", []),
 ("recall", "preflight", "shared history", "", []),
 ("preflight", "worker", "proceed + confidence ≥ 0.60", "", []),
 ("worker", "review", "draft + history", "", []),
 ("review", "policy", "decision", "", []),
]
for theme in ("light", "dark"):
    dark = theme == "dark"
    bg = "#101b2b" if dark else "#f8fafc"
    fg = "#e5edf8" if dark else "#192c43"
    stroke = "#71849e" if dark else "#74849b"
    colors = {"base": "#1c2b41" if dark else "#eaf0f7", "model": "#143d49" if dark else "#e0f3f4", "policy": "#372e4f" if dark else "#f0eafa"}
    mx = E.Element("mxfile", host="app.diagrams.net")
    d = E.SubElement(mx, "diagram", name="Memory-informed answer decisions", id="jev-comparison")
    g = E.SubElement(d, "mxGraphModel", page="1", pageWidth="900", pageHeight="1140", background=bg)
    r = E.SubElement(g, "root")
    E.SubElement(r, "mxCell", id="0")
    E.SubElement(r, "mxCell", id="1", parent="0")
    for id, title, detail, x, y, w, h, tone in nodes:
        value = f"<b>{html.escape(title)}</b>"
        if detail:
            value += '<br><span style="font-size:16px">' + detail.replace(' > ', ' &gt; ') + '</span>'
        fill = colors.get(tone, "none")
        border = "none" if tone == "label" else stroke
        c = E.SubElement(r, "mxCell", id=id, value=value, vertex="1", parent="1", style=f"rounded=0;whiteSpace=wrap;html=1;fontFamily=Helvetica;fontSize=20;fontColor={fg};fillColor={fill};strokeColor={border};spacing=12;")
        E.SubElement(c, "mxGeometry", x=str(x), y=str(y), width=str(w), height=str(h), attrib={"as": "geometry"})
    for i, (a, b, label, style, points) in enumerate(edges):
        c = E.SubElement(r, "mxCell", id=f"e{i}", source=a, target=b, value=label, edge="1", parent="1", style=f"edgeStyle=orthogonalEdgeStyle;rounded=0;html=1;endArrow=block;endFill=1;strokeWidth=1.8;strokeColor={stroke};fontColor={fg};fontSize=15;labelBackgroundColor={bg};" + style)
        geometry = E.SubElement(c, "mxGeometry", relative="1", attrib={"as": "geometry"})
        if points:
            array = E.SubElement(geometry, "Array", attrib={"as": "points"})
            for x, y in points:
                E.SubElement(array, "mxPoint", x=str(x), y=str(y))
    stem = "evidence-flow" + (".dark" if dark else "")
    xml = E.tostring(mx, encoding="unicode")
    for destination in (ROOT, OUT):
        (destination / (stem + ".drawio")).write_text(xml)
(ROOT / "manifest.json").write_text(json.dumps([{"slug": "evidence-flow", "title": "Hindsight evidence shared by Jev and the local worker", "nodes": nodes, "edges": edges}], indent=2) + "\n")

# A native narrow layout keeps the figure readable without horizontal panning.
mobile_nodes = [(id, title, detail, 20, 20+i*165, 340, 110 if id != "policy" else 135, tone)
                for i,(id,title,detail,x,y,w,h,tone) in enumerate(nodes)]
for theme in ("light", "dark"):
    suffix = ".dark" if theme == "dark" else ""
    document = E.fromstring((ROOT / ("evidence-flow"+suffix+".drawio")).read_text())
    by_id = {n[0]: n for n in mobile_nodes}
    for cell in document.iter("mxCell"):
        if cell.get("id") in by_id:
            n = by_id[cell.get("id")]
            geometry = cell.find("mxGeometry")
            for key,value in zip(("x","y","width","height"),n[3:7]):
                geometry.set(key,str(value))
            cell.set("style", cell.get("style").replace("fontSize=20", "fontSize=18"))
    for destination in (ROOT, OUT):
        (destination / ("evidence-flow.mobile"+suffix+".drawio")).write_text(E.tostring(document,encoding="unicode"))
manifest = json.loads((ROOT/"manifest.json").read_text())
manifest.append({"slug":"evidence-flow.mobile","title":"Memory-informed answer decisions, narrow layout","nodes":mobile_nodes,"edges":edges})
(ROOT/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
