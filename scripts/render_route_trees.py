"""Render continuous synthesis route trees from AutoPlanner route JSON.

This renderer creates Graphviz DOT graphs where molecule nodes are RDKit PNG
depictions and edges are retrosynthetic steps.  It is intended for expert
reports and presentation decks, where a route should read as one connected
path/tree rather than as isolated one-step panels.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import subprocess
from pathlib import Path
from typing import Any

from rdkit import Chem, RDLogger
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D


RDLogger.DisableLog("rdApp.*")


def main() -> None:
    parser = argparse.ArgumentParser(description="Render top route trees as Graphviz SVG/PDF figures.")
    parser.add_argument("--input", required=True, help="AutoPlanner route JSON with a routes array.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--formats", default="svg,pdf", help="Comma-separated Graphviz output formats.")
    parser.add_argument("--mol-width", type=int, default=260)
    parser.add_argument("--mol-height", type=int, default=170)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_dir = output_dir / "molecules"
    image_dir.mkdir(exist_ok=True)
    doc = json.loads(input_path.read_text(encoding="utf-8"))
    routes = [route for route in doc.get("routes") or [] if isinstance(route, dict)]
    target = str(doc.get("target") or doc.get("target_smiles") or "")
    formats = [fmt.strip() for fmt in args.formats.split(",") if fmt.strip()]

    rendered = []
    for idx, route in enumerate(routes[: max(0, args.top_k)], start=1):
        stem = f"route_tree_{idx:02d}"
        dot_path = output_dir / f"{stem}.dot"
        dot = build_route_dot(
            route,
            route_number=idx,
            target_smiles=target,
            image_dir=image_dir,
            mol_width=args.mol_width,
            mol_height=args.mol_height,
        )
        dot_path.write_text(dot, encoding="utf-8")
        outputs = {}
        for fmt in formats:
            out_path = output_dir / f"{stem}.{fmt}"
            subprocess.run(["dot", f"-T{fmt}", str(dot_path), "-o", str(out_path)], check=True)
            outputs[fmt] = out_path.name
        rendered.append({"rank": idx, "dot": dot_path.name, **outputs})

    index_path = output_dir / "index.html"
    index_path.write_text(render_index(rendered, source=input_path), encoding="utf-8")
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps({"source": str(input_path), "figures": rendered}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps({"output_dir": str(output_dir), "index": str(index_path), "figures": rendered}, indent=2))


def build_route_dot(
    route: dict[str, Any],
    *,
    route_number: int,
    target_smiles: str,
    image_dir: Path,
    mol_width: int,
    mol_height: int,
) -> str:
    steps = [step for step in route.get("steps") or [] if isinstance(step, dict)]
    nodes: dict[str, dict[str, Any]] = {}
    edges: list[tuple[str, str, str]] = []
    terminal_set = set(_terminal_reactants(route))
    target = target_smiles or (steps[0].get("product") if steps else "")
    if target:
        _ensure_node(nodes, str(target), role="target", terminal=False)

    for index, step in enumerate(steps, start=1):
        product = str(step.get("product") or _reaction_product(step) or "")
        if product:
            _ensure_node(nodes, product, role="intermediate", terminal=product in terminal_set)
        reactants = _step_reactants(step)
        for ridx, reactant in enumerate(reactants, start=1):
            _ensure_node(nodes, reactant, role="terminal" if reactant in terminal_set else "intermediate", terminal=reactant in terminal_set)
            edges.append((product, reactant, _edge_label(step, index, ridx)))

    for terminal in terminal_set:
        _ensure_node(nodes, terminal, role="terminal", terminal=True)

    title = _route_title(route, route_number)
    lines = [
        "digraph RouteTree {",
        '  graph [rankdir=LR, bgcolor="white", pad="0.35", nodesep="0.62", ranksep="0.95", splines=spline, outputorder=edgesfirst];',
        '  node [shape=plain, fontname="Arial"];',
        '  edge [fontname="Arial", fontsize=9, color="#333333", arrowsize=0.7, penwidth=1.2];',
        f'  labelloc="t";',
        f'  label=<{_html_text(title)}>;',
    ]

    node_ids: dict[str, str] = {}
    for idx, (smiles, meta) in enumerate(nodes.items()):
        node_id = f"n{idx}"
        node_ids[smiles] = node_id
        image = molecule_png(smiles, image_dir=image_dir, width=mol_width, height=mol_height)
        label = molecule_node_label(
            smiles,
            image=image,
            role=str(meta.get("role") or ""),
            width=mol_width,
            height=mol_height,
        )
        lines.append(f"  {node_id} [label=<{label}>];")
    for product, reactant, label in edges:
        if product not in node_ids or reactant not in node_ids:
            continue
        lines.append(f'  {node_ids[product]} -> {node_ids[reactant]} [label="{_dot_escape(label)}"];')

    terminal_ids = [node_ids[smi] for smi, meta in nodes.items() if meta.get("terminal") and smi in node_ids]
    if terminal_ids:
        lines.append("  { rank=same; " + "; ".join(terminal_ids) + "; }")
    lines.append("}")
    return "\n".join(lines) + "\n"


def molecule_png(smiles: str, *, image_dir: Path, width: int, height: int) -> Path:
    digest = hashlib.sha1(str(smiles).encode("utf-8")).hexdigest()[:16]
    path = image_dir / f"{digest}.png"
    if path.exists():
        return path
    mol = Chem.MolFromSmiles(str(smiles or ""))
    if mol is None:
        _placeholder_png(path, width, height)
        return path
    rdDepictor.Compute2DCoords(mol)
    drawer = rdMolDraw2D.MolDraw2DCairo(int(width), int(height))
    opts = drawer.drawOptions()
    opts.padding = 0.06
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    path.write_bytes(drawer.GetDrawingText())
    return path


def _placeholder_png(path: Path, width: int, height: int) -> None:
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (int(width), int(height)), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((1, 1, int(width) - 2, int(height) - 2), outline=(210, 210, 210))
    draw.text((12, int(height) // 2 - 8), "invalid SMILES", fill=(100, 100, 100))
    image.save(path)


def molecule_node_label(smiles: str, *, image: Path, role: str, width: int, height: int) -> str:
    border = "#111111" if role == "target" else "#5f8f3d" if role == "terminal" else "#bdbdbd"
    title = "TARGET" if role == "target" else "TERMINAL" if role == "terminal" else "INTERMEDIATE"
    caption = _short_smiles(smiles)
    return f'''<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="4" COLOR="{border}">
<TR><TD BGCOLOR="{_role_color(role)}"><FONT POINT-SIZE="10"><B>{title}</B></FONT></TD></TR>
<TR><TD FIXEDSIZE="TRUE" WIDTH="{width}" HEIGHT="{height}"><IMG SRC="{_dot_html_escape(str(image))}"/></TD></TR>
<TR><TD><FONT POINT-SIZE="8">{_html_text(caption)}</FONT></TD></TR>
</TABLE>'''


def _route_title(route: dict[str, Any], route_number: int) -> str:
    audit = route.get("product_audit") or {}
    parts = [f"Route {route_number:02d}", f"{route.get('n_steps') or len(route.get('steps') or [])} steps"]
    if route.get("score") is not None:
        parts.append(f"score {float(route.get('score')):.3g}")
    if audit.get("route_class"):
        parts.append(str(audit.get("route_class")))
    tags = audit.get("tags") or []
    if tags:
        parts.append(", ".join(str(tag) for tag in tags[:3]))
    issues = audit.get("issues") or []
    if issues:
        parts.append("issues: " + ", ".join(str(issue) for issue in issues[:2]))
    return " | ".join(parts)


def _edge_label(step: dict[str, Any], index: int, reactant_index: int) -> str:
    bits = [f"S{index}.{reactant_index}"]
    reaction_type = step.get("reaction_type")
    if reaction_type:
        bits.append(str(reaction_type))
    ec = step.get("ec")
    enzyme = (step.get("enzyme_ec_annotations") or [{}])[0]
    if not ec:
        ec = enzyme.get("ec_number")
    if ec:
        bits.append(f"EC {ec}")
    condition = (step.get("condition_predictions") or [{}])[0]
    if condition.get("Temperature") is not None:
        try:
            bits.append(f"{float(condition.get('Temperature')):.0f} C")
        except (TypeError, ValueError):
            pass
    return " | ".join(bits[:4])


def _terminal_reactants(route: dict[str, Any]) -> list[str]:
    audit = route.get("product_audit") or {}
    profile = audit.get("terminal_profile") or {}
    metrics = route.get("metrics") or {}
    values = profile.get("terminal_reactants") or metrics.get("terminal_reactants") or []
    return [str(smi) for smi in values if smi]


def _step_reactants(step: dict[str, Any]) -> list[str]:
    values = []
    if step.get("main_reactant"):
        values.append(str(step.get("main_reactant")))
    values.extend(str(smi) for smi in step.get("aux_reactants") or [] if smi)
    if not values:
        rxn = str(step.get("reaction_smiles") or "")
        if ">>" in rxn:
            lhs, _rhs = rxn.split(">>", 1)
            values = [part for part in lhs.split(".") if part]
    return _dedupe(values)


def _reaction_product(step: dict[str, Any]) -> str:
    rxn = str(step.get("reaction_smiles") or "")
    if ">>" not in rxn:
        return ""
    _lhs, rhs = rxn.split(">>", 1)
    return rhs.split(".")[0] if rhs else ""


def _ensure_node(nodes: dict[str, dict[str, Any]], smiles: str, *, role: str, terminal: bool) -> None:
    if not smiles:
        return
    if smiles not in nodes:
        nodes[smiles] = {"role": role, "terminal": terminal}
        return
    if role == "target":
        nodes[smiles]["role"] = "target"
    elif terminal and nodes[smiles].get("role") != "target":
        nodes[smiles]["role"] = "terminal"
    nodes[smiles]["terminal"] = bool(nodes[smiles].get("terminal") or terminal)


def _dedupe(values: list[str]) -> list[str]:
    out = []
    seen = set()
    for value in values:
        if value and value not in seen:
            out.append(value)
            seen.add(value)
    return out


def _role_color(role: str) -> str:
    if role == "target":
        return "#f0f0f0"
    if role == "terminal":
        return "#edf6e7"
    return "#ffffff"


def _short_smiles(smiles: str, limit: int = 44) -> str:
    text = str(smiles or "")
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def render_index(rows: list[dict[str, Any]], *, source: Path) -> str:
    blocks = []
    for row in rows:
        svg = row.get("svg")
        pdf = row.get("pdf")
        links = []
        if svg:
            links.append(f'<a href="{html.escape(svg)}">SVG</a>')
        if pdf:
            links.append(f'<a href="{html.escape(pdf)}">PDF</a>')
        embed = f'<object data="{html.escape(svg)}" type="image/svg+xml"></object>' if svg else ""
        blocks.append(
            f'''<section>
<h2>Route {row["rank"]:02d}</h2>
<p>{' | '.join(links)}</p>
{embed}
</section>'''
        )
    return f'''<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Continuous Route Trees</title>
<style>
body {{ font-family: Arial, Helvetica, sans-serif; margin: 28px; color: #111; background: #fff; }}
h1 {{ font-size: 24px; margin-bottom: 4px; }}
h2 {{ font-size: 18px; margin: 28px 0 4px; }}
p {{ color: #555; }}
object {{ width: 100%; min-height: 720px; border: 1px solid #ddd; }}
section {{ break-after: page; }}
</style>
</head>
<body>
<h1>Continuous Route Trees</h1>
<p>Source: {html.escape(str(source))}</p>
{''.join(blocks)}
</body>
</html>
'''


def _html_text(text: str) -> str:
    return html.escape(str(text or ""), quote=False)


def _dot_html_escape(text: str) -> str:
    return html.escape(str(text or ""), quote=True)


def _dot_escape(text: str) -> str:
    return str(text or "").replace("\\", "\\\\").replace('"', '\\"')


if __name__ == "__main__":
    main()
