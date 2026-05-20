"""Render route figures from AutoPlanner route JSON.

The output is intentionally static SVG plus an HTML index so the figures can be
used outside the web UI in reports or manuscripts.
"""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any

from rdkit import Chem, RDLogger
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D


RDLogger.DisableLog("rdApp.*")


def main() -> None:
    parser = argparse.ArgumentParser(description="Render top route figures as publication-style SVG panels.")
    parser.add_argument("--input", required=True, help="AutoPlanner route JSON containing a routes array.")
    parser.add_argument("--output-dir", required=True, help="Directory for generated SVG/HTML files.")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=24)
    parser.add_argument("--route-offset", type=int, default=1, help="Displayed route numbering offset.")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    doc = json.loads(input_path.read_text(encoding="utf-8"))
    routes = [route for route in doc.get("routes") or [] if isinstance(route, dict)]
    target = str(doc.get("target") or doc.get("target_smiles") or "")
    rows = []
    for idx, route in enumerate(routes[: max(0, args.top_k)]):
        display_rank = idx + int(args.route_offset)
        svg_name = f"route_{display_rank:02d}.svg"
        svg_path = output_dir / svg_name
        svg = render_route_svg(route, route_number=display_rank, target_smiles=target, max_steps=args.max_steps)
        svg_path.write_text(svg, encoding="utf-8")
        rows.append({"rank": display_rank, "path": svg_name, "route": route})

    index_path = output_dir / "index.html"
    index_path.write_text(render_index(rows, source=input_path), encoding="utf-8")
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "source": str(input_path),
                "output_dir": str(output_dir),
                "n_routes": len(rows),
                "figures": [{"rank": row["rank"], "svg": row["path"]} for row in rows],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"output_dir": str(output_dir), "index": str(index_path), "n_routes": len(rows)}, indent=2))


def render_route_svg(
    route: dict[str, Any],
    *,
    route_number: int,
    target_smiles: str = "",
    max_steps: int = 24,
) -> str:
    steps = [step for step in route.get("steps") or [] if isinstance(step, dict)]
    steps = steps[: max(1, max_steps)]
    target = target_smiles or (steps[0].get("product") if steps else "")
    route_class = ((route.get("product_audit") or {}).get("route_class") or route.get("route_class") or "")
    issues = (route.get("product_audit") or {}).get("issues") or []
    tags = (route.get("product_audit") or {}).get("tags") or []
    score = route.get("score")
    n_steps = route.get("n_steps") or len(steps)
    title = f"Route {route_number}: {n_steps} steps"
    subtitle_bits = []
    if route_class:
        subtitle_bits.append(str(route_class))
    if score is not None:
        subtitle_bits.append(f"score {float(score):.3g}")
    if tags:
        subtitle_bits.append(", ".join(str(tag) for tag in tags[:4]))
    if issues:
        subtitle_bits.append("issues: " + ", ".join(str(issue) for issue in issues[:3]))
    subtitle = " | ".join(subtitle_bits)

    terminal_smiles = _terminal_reactants(route)
    terminal_count = max(1, len(terminal_smiles))
    step_count = max(1, len(steps))
    mol_w = 230
    mol_h = 150
    gap_x = 34
    gap_y = 52
    header_h = 76
    row_h = mol_h + 108
    width = max(1040, terminal_count * (mol_w + gap_x) + 88)
    height = header_h + (step_count + 1) * row_h + 36

    chunks = [
        _svg_header(width, height),
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff"/>',
        f'<text x="36" y="34" class="title">{_esc(title)}</text>',
        f'<text x="36" y="58" class="subtitle">{_esc(subtitle)}</text>',
    ]

    y = header_h
    chunks.append(f'<text x="36" y="{y + 18}" class="section">Target</text>')
    chunks.append(_mol_panel(target, x=36, y=y + 28, width=mol_w + 80, height=mol_h + 28, label="target"))
    y += row_h
    chunks.append(f'<line x1="36" y1="{y - 24}" x2="{width - 36}" y2="{y - 24}" class="rule"/>')

    for step_index, step in enumerate(steps, start=1):
        product = str(step.get("product") or _reaction_product(step) or "")
        reactants = _step_reactants(step)
        step_label = _step_label(step, step_index)
        chunks.append(f'<text x="36" y="{y + 18}" class="section">{_esc(step_label)}</text>')
        chunks.append(_mol_panel(product, x=36, y=y + 30, width=mol_w, height=mol_h, label="product"))
        arrow_x1 = 36 + mol_w + 14
        arrow_x2 = arrow_x1 + 64
        arrow_y = y + 30 + mol_h / 2
        chunks.append(_arrow(arrow_x1, arrow_y, arrow_x2, arrow_y))
        chunks.append(_step_meta(step, x=arrow_x1 - 10, y=arrow_y + 24, max_width=100))
        x = arrow_x2 + 22
        if not reactants:
            chunks.append(f'<text x="{x}" y="{y + 104}" class="placeholder">no reactants recorded</text>')
        for ridx, smi in enumerate(reactants):
            chunks.append(_mol_panel(smi, x=x + ridx * (mol_w + gap_x), y=y + 30, width=mol_w, height=mol_h, label=f"r{ridx + 1}"))
            if ridx < len(reactants) - 1:
                plus_x = x + (ridx + 1) * (mol_w + gap_x) - gap_x / 2
                chunks.append(f'<text x="{plus_x}" y="{y + 115}" class="plus">+</text>')
        y += row_h

    chunks.append(f'<line x1="36" y1="{y - 24}" x2="{width - 36}" y2="{y - 24}" class="rule"/>')
    chunks.append(f'<text x="36" y="{y + 18}" class="section">Terminal materials</text>')
    x0 = 36
    for idx, smi in enumerate(terminal_smiles):
        x = x0 + idx * (mol_w + gap_x)
        chunks.append(_mol_panel(smi, x=x, y=y + 30, width=mol_w, height=mol_h, label=f"T{idx + 1}"))
    chunks.append("</svg>")
    return "\n".join(chunks)


def _svg_header(width: int, height: int) -> str:
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<defs>
  <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
    <polygon points="0 0, 10 3.5, 0 7" fill="#222"/>
  </marker>
  <style>
    .title {{ font: 700 22px Arial, Helvetica, sans-serif; fill: #111; }}
    .subtitle {{ font: 12px Arial, Helvetica, sans-serif; fill: #555; }}
    .section {{ font: 700 13px Arial, Helvetica, sans-serif; fill: #222; }}
    .caption {{ font: 10px Arial, Helvetica, sans-serif; fill: #444; }}
    .placeholder {{ font: 12px Arial, Helvetica, sans-serif; fill: #777; }}
    .plus {{ font: 22px Arial, Helvetica, sans-serif; fill: #222; text-anchor: middle; }}
    .panel {{ fill: #fff; stroke: #d7d7d7; stroke-width: 1; }}
    .arrow {{ stroke: #222; stroke-width: 1.8; marker-end: url(#arrowhead); }}
    .rule {{ stroke: #eeeeee; stroke-width: 1; }}
  </style>
</defs>'''


def _mol_panel(smiles: str, *, x: float, y: float, width: int, height: int, label: str) -> str:
    mol_svg = molecule_svg(smiles, width=width, height=height)
    caption = _short_label(smiles)
    return "\n".join(
        [
            f'<rect class="panel" x="{x}" y="{y}" width="{width}" height="{height + 30}" rx="3"/>',
            f'<g transform="translate({x},{y})">{mol_svg}</g>',
            f'<text x="{x + 8}" y="{y + height + 18}" class="caption">{_esc(label)}: {_esc(caption)}</text>',
        ]
    )


def molecule_svg(smiles: str, *, width: int, height: int) -> str:
    mol = Chem.MolFromSmiles(str(smiles or ""))
    if mol is None:
        return f'<text x="8" y="{height / 2:.1f}" class="placeholder">invalid SMILES</text>'
    rdDepictor.Compute2DCoords(mol)
    drawer = rdMolDraw2D.MolDraw2DSVG(int(width), int(height))
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    body = svg
    svg_start = body.find("<svg")
    start = body.find(">", svg_start) if svg_start >= 0 else body.find(">")
    end = body.rfind("</svg>")
    if start >= 0 and end >= 0:
        body = body[start + 1 : end]
    return body


def _arrow(x1: float, y1: float, x2: float, y2: float) -> str:
    return f'<line class="arrow" x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}"/>'


def _step_meta(step: dict[str, Any], *, x: float, y: float, max_width: int) -> str:
    bits = []
    reaction_type = step.get("reaction_type")
    if reaction_type:
        bits.append(str(reaction_type))
    ec = step.get("ec")
    enzyme = (step.get("enzyme_ec_annotations") or [{}])[0]
    if not ec:
        ec = enzyme.get("ec_number")
    if ec:
        conf = enzyme.get("confidence")
        if conf is not None:
            bits.append(f"EC {ec} ({float(conf):.2f})")
        else:
            bits.append(f"EC {ec}")
    condition = (step.get("condition_predictions") or [{}])[0]
    if condition.get("Temperature") is not None:
        try:
            bits.append(f"{float(condition.get('Temperature')):.0f} C")
        except (TypeError, ValueError):
            pass
    label = " | ".join(bits[:3])
    return f'<text x="{x}" y="{y}" class="caption">{_esc(label[:max_width])}</text>'


def _step_label(step: dict[str, Any], step_index: int) -> str:
    source = step.get("source") or ""
    return f"Step {step_index}  {source}".strip()


def _terminal_reactants(route: dict[str, Any]) -> list[str]:
    metrics = route.get("metrics") or {}
    audit = route.get("product_audit") or {}
    terminal_profile = audit.get("terminal_profile") or {}
    values = terminal_profile.get("terminal_reactants") or metrics.get("terminal_reactants") or []
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
    seen = set()
    out = []
    for smi in values:
        if smi not in seen:
            out.append(smi)
            seen.add(smi)
    return out


def _reaction_product(step: dict[str, Any]) -> str:
    rxn = str(step.get("reaction_smiles") or "")
    if ">>" not in rxn:
        return ""
    _lhs, rhs = rxn.split(">>", 1)
    return rhs.split(".")[0] if rhs else ""


def _short_label(smiles: str, limit: int = 38) -> str:
    text = str(smiles or "")
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def render_index(rows: list[dict[str, Any]], *, source: Path) -> str:
    cards = []
    for row in rows:
        route = row["route"]
        audit = route.get("product_audit") or {}
        meta = " | ".join(
            part
            for part in [
                f"steps {route.get('n_steps') or len(route.get('steps') or [])}",
                str(audit.get("route_class") or ""),
                f"score {float(route.get('score')):.3g}" if route.get("score") is not None else "",
            ]
            if part
        )
        cards.append(
            f'''<section>
  <h2>Route {row["rank"]:02d}</h2>
  <p>{html.escape(meta)}</p>
  <object data="{html.escape(row["path"])}" type="image/svg+xml"></object>
</section>'''
        )
    return f'''<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Route Figures</title>
  <style>
    body {{ margin: 32px; font-family: Arial, Helvetica, sans-serif; color: #111; background: #fff; }}
    h1 {{ font-size: 24px; margin: 0 0 4px; }}
    h2 {{ font-size: 18px; margin: 28px 0 4px; }}
    p {{ color: #555; margin: 0 0 10px; }}
    object {{ width: 100%; border: 1px solid #ddd; }}
    section {{ break-after: page; }}
  </style>
</head>
<body>
  <h1>Route Figures</h1>
  <p>Source: {html.escape(str(source))}</p>
  {''.join(cards)}
</body>
</html>
'''


def _esc(value: Any) -> str:
    return html.escape(str(value or ""), quote=True)


if __name__ == "__main__":
    main()
