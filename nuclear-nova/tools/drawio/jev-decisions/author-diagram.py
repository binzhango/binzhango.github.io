"""Author the code-verified two-path comparison as editable draw.io cells."""
from pathlib import Path
from xml.etree import ElementTree as E
import html
import json

ROOT = Path(__file__).resolve().parent
OUT = ROOT.parents[2] / "public/diagrams/jev-decisions"
nodes = [
    ("input", "One task · same starting facts", "Same local LLM, tools, policy knowledge and generation settings", 30, 30, 820, 85, "base"),
    ("normal_label", "NORMAL AGENT", "", 30, 145, 380, 42, "label"),
    ("jev_label", "JEV-ASSISTED AGENT", "", 470, 145, 380, 42, "label"),
    ("normal_llm", "Local LLM chooses", "Selects workflow and proposes the action<br>within its generation", 30, 215, 380, 185, "base"),
    ("triage", "Jev triage → prompt hint", "Category + confidence select advisory guidance", 470, 215, 380, 80, "model"),
    ("jev_llm", "Same local LLM proposes", "All tools remain available; LLM chooses action", 470, 325, 380, 75, "base"),
    ("normal_policy", "Python: hard-policy checks", "Ownership, amount limits, duplicate refunds", 30, 455, 380, 80, "policy"),
    ("jev_policy", "Same hard-policy checks", "Deny invalid refunds before Jev risk", 470, 455, 380, 80, "policy"),
    ("risk", "Jev assesses eligible refund", "Proposed arguments + verified evidence", 470, 585, 380, 80, "model"),
    ("normal_review", "Python: review when required", "Amount > $25 or LLM requests review<br>Recheck edited arguments", 30, 715, 380, 100, "policy"),
    ("jev_review", "Python: review when required", "Same rules + risk ≥ threshold or unavailable<br>Recheck edited arguments", 470, 715, 380, 100, "policy"),
    ("normal_result", "Simulated tool → LLM response", "Own order data and checkpoints", 30, 875, 380, 85, "base"),
    ("jev_result", "Simulated tool → LLM response", "Independent order data and checkpoints", 470, 875, 380, 85, "base"),
    ("compare", "Compare the two recorded runs", "Actual effects · response quality · calls · timing", 30, 1020, 820, 80, "base"),
]
edges = [
    ("input", "normal_llm", "", "exitX=0.23;exitY=1;entryX=0.5;entryY=0;", [(220, 130), (12, 130), (12, 195), (220, 195)]),
    ("input", "triage", "", "exitX=0.77;exitY=1;entryX=0.5;entryY=0;", [(660, 130), (868, 130), (868, 195), (660, 195)]),
    ("triage", "jev_llm", "", "", []),
    ("normal_llm", "normal_policy", "refund proposal", "", []),
    ("jev_llm", "jev_policy", "refund proposal", "", []),
    ("normal_policy", "normal_review", "eligible", "", []),
    ("jev_policy", "risk", "eligible", "", []),
    ("risk", "jev_review", "risk score", "", []),
    ("normal_review", "normal_result", "allowed / approved", "", []),
    ("jev_review", "jev_result", "allowed / approved", "", []),
    ("normal_result", "compare", "", "exitX=0.5;exitY=1;entryX=0.23;entryY=0;dashed=1;", []),
    ("jev_result", "compare", "", "exitX=0.5;exitY=1;entryX=0.77;entryY=0;dashed=1;", []),
]
for theme in ("light", "dark"):
    dark = theme == "dark"
    bg = "#101b2b" if dark else "#f8fafc"
    fg = "#e5edf8" if dark else "#192c43"
    stroke = "#71849e" if dark else "#74849b"
    colors = {"base": "#1c2b41" if dark else "#eaf0f7", "model": "#143d49" if dark else "#e0f3f4", "policy": "#372e4f" if dark else "#f0eafa"}
    mx = E.Element("mxfile", host="app.diagrams.net")
    d = E.SubElement(mx, "diagram", name="Normal vs Jev-assisted agent", id="jev-comparison")
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
    stem = "refund-path" + (".dark" if dark else "")
    xml = E.tostring(mx, encoding="unicode")
    for destination in (ROOT, OUT):
        (destination / (stem + ".drawio")).write_text(xml)
(ROOT / "manifest.json").write_text(json.dumps([{"slug": "refund-path", "title": "Normal LLM agent versus Jev-assisted agent on the same task", "nodes": nodes, "edges": edges}], indent=2) + "\n")
