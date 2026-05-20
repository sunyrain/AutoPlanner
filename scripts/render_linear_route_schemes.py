"""Render publication-style continuous linear route schemes.

This renderer follows the common synthesis-scheme layout: main-chain molecules
are placed left-to-right, reaction conditions are written above/below the
reaction arrow, and auxiliary reactants/reagents are drawn near the arrow.
"""
from __future__ import annotations

import argparse
import html
import json
import math
import subprocess
from pathlib import Path
from typing import Any

from rdkit import Chem, RDLogger
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D


RDLogger.DisableLog("rdApp.*")


def main() -> None:
    parser = argparse.ArgumentParser(description="Render top routes as continuous synthesis-scheme SVG/PDF figures.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--formats", default="svg,pdf")
    parser.add_argument("--mol-width", type=int, default=250)
    parser.add_argument("--mol-height", type=int, default=165)
    parser.add_argument("--steps-per-row", type=int, default=4)
    parser.add_argument("--aux-mode", choices=["mini", "text", "none"], default="mini")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    doc = json.loads(input_path.read_text(encoding="utf-8"))
    routes = [route for route in doc.get("routes") or [] if isinstance(route, dict)]
    target = str(doc.get("target") or doc.get("target_smiles") or "")
    formats = [fmt.strip() for fmt in args.formats.split(",") if fmt.strip()]
    rows = []
    for idx, route in enumerate(routes[: max(0, args.top_k)], start=1):
        svg_name = f"scheme_route_{idx:02d}.svg"
        svg_path = output_dir / svg_name
        svg = render_scheme_svg(
            route,
            route_number=idx,
            target_smiles=target,
            mol_width=args.mol_width,
            mol_height=args.mol_height,
            steps_per_row=args.steps_per_row,
            aux_mode=args.aux_mode,
        )
        svg_path.write_text(svg, encoding="utf-8")
        row = {"rank": idx, "svg": svg_name}
        if "pdf" in formats:
            pdf_name = f"scheme_route_{idx:02d}.pdf"
            subprocess.run(["rsvg-convert", "-f", "pdf", "-o", str(output_dir / pdf_name), str(svg_path)], check=True)
            row["pdf"] = pdf_name
        rows.append(row)
    (output_dir / "index.html").write_text(render_index(rows, source=input_path), encoding="utf-8")
    (output_dir / "manifest.json").write_text(json.dumps({"source": str(input_path), "figures": rows}, indent=2), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "index": str(output_dir / "index.html"), "figures": rows}, indent=2))


def render_scheme_svg(
    route: dict[str, Any],
    *,
    route_number: int,
    target_smiles: str,
    mol_width: int = 250,
    mol_height: int = 165,
    steps_per_row: int = 4,
    aux_mode: str = "mini",
) -> str:
    steps = [step for step in route.get("steps") or [] if isinstance(step, dict)]
    route_graph = _route_graph(steps)
    retro_chain, retro_step_infos = _retro_main_path(steps, target_smiles=target_smiles, graph=route_graph)
    chain = list(reversed(retro_chain))
    step_infos = list(reversed(retro_step_infos))
    condition_audit = (route.get("product_audit") or {}).get("condition_audit") or {}
    condition_by_step = _condition_audit_by_step(condition_audit)
    if not chain:
        chain = [target_smiles or ""]
    n_steps = max(0, len(chain) - 1)
    steps_per_row = max(1, int(steps_per_row))
    rows = max(1, math.ceil(max(1, n_steps) / steps_per_row))
    margin_x = 38
    header_h = 76
    row_h = 310
    arrow_w = 190
    row_capacity = min(steps_per_row, max(1, n_steps))
    width = margin_x * 2 + (row_capacity + 1) * mol_width + row_capacity * arrow_w
    height = header_h + rows * row_h + 52
    title = _title(route, route_number)
    parts = [_svg_header(width, height), f'<rect width="{width}" height="{height}" fill="#ffffff"/>']
    parts.append(f'<text x="{margin_x}" y="34" class="title">{_esc(title)}</text>')
    subtitle = _subtitle(route)
    if subtitle:
        parts.append(f'<text x="{margin_x}" y="58" class="subtitle">{_esc(subtitle)}</text>')

    for row_index in range(rows):
        start_step = row_index * steps_per_row
        end_step = min(n_steps, start_step + steps_per_row)
        y = header_h + row_index * row_h
        for col, step_idx in enumerate(range(start_step, end_step + 1)):
            x = margin_x + col * (mol_width + arrow_w)
            mol = chain[step_idx]
            role = "starting" if step_idx == 0 else "target" if step_idx == len(chain) - 1 else "intermediate"
            label = "Starting material" if step_idx == 0 else "Target" if role == "target" else f"I{step_idx}"
            parts.append(_mol_panel(mol, x=x, y=y + 74, width=mol_width, height=mol_height, label=label, role=role))
            if step_idx < end_step:
                arrow_x1 = x + mol_width + 18
                arrow_x2 = arrow_x1 + arrow_w - 36
                arrow_y = y + 74 + mol_height / 2
                info = step_infos[step_idx] if step_idx < len(step_infos) else {}
                step = info.get("step") or {}
                condition_row = condition_by_step.get(_step_original_index(step, steps))
                parts.append(_scheme_arrow(step, arrow_x1, arrow_y, arrow_x2, arrow_y, condition_row=condition_row))
                aux = list(info.get("side_reactants") or _aux_reactants(step))
                if aux_mode == "text" and aux:
                    parts.append(_aux_text(aux, x=(arrow_x1 + arrow_x2) / 2, y=arrow_y - 43))
                elif aux_mode == "mini":
                    aux = _display_aux_reactants(aux, step)
                    parts.append(
                        _aux_or_branch_structures(
                            aux,
                            graph=route_graph,
                            main_chain=set(chain),
                            x=(arrow_x1 + arrow_x2) / 2,
                            y=arrow_y - 112,
                        )
                    )

    footer = _condition_footer(condition_audit)
    if footer:
        parts.append(f'<text x="{margin_x}" y="{height - 18}" class="footnote">{_esc(footer)}</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def _route_graph(steps: list[dict[str, Any]]) -> dict[str, Any]:
    product_to_step: dict[str, dict[str, Any]] = {}
    for step in steps:
        product = str(step.get("product") or _reaction_product(step) or "")
        if product and product not in product_to_step:
            product_to_step[product] = step
    return {"product_to_step": product_to_step, "products": set(product_to_step)}


def _condition_audit_by_step(condition_audit: dict[str, Any]) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    for row in condition_audit.get("steps") or []:
        if not isinstance(row, dict):
            continue
        try:
            rows[int(row.get("step_index"))] = row
        except (TypeError, ValueError):
            continue
    return rows


def _step_original_index(step: dict[str, Any], steps: list[dict[str, Any]]) -> int:
    for idx, candidate in enumerate(steps, start=1):
        if candidate is step:
            return idx
    return int(step.get("index") or 0) + 1 if step.get("index") is not None else 0


def _condition_risk_marker(risk: str) -> str:
    if risk == "high":
        return " !"
    if risk == "warn":
        return " ?"
    return ""


def _condition_footer(condition_audit: dict[str, Any]) -> str:
    if not condition_audit:
        return ""
    risk = str(condition_audit.get("route_risk") or "ok")
    if risk == "ok":
        return "Conditions are model-predicted per-step hypotheses."
    high = int(condition_audit.get("high_risk_step_count") or 0)
    warn = int(condition_audit.get("warning_step_count") or 0)
    span = condition_audit.get("temperature_span_c")
    bits = ["!/? mark condition-audit warnings; conditions are model-predicted per-step hypotheses"]
    if high:
        bits.append(f"high-risk steps {high}")
    if warn:
        bits.append(f"warning steps {warn}")
    if span is not None:
        bits.append(f"T span {float(span):.0f} °C")
    return "; ".join(bits) + "."


def _retro_main_path(
    steps: list[dict[str, Any]],
    *,
    target_smiles: str,
    graph: dict[str, Any],
) -> tuple[list[str], list[dict[str, Any]]]:
    product_to_step: dict[str, dict[str, Any]] = graph["product_to_step"]
    products: set[str] = graph["products"]
    current = str(steps[0].get("product") or _reaction_product(steps[0]) or target_smiles or "") if steps else str(target_smiles or "")
    chain = [current] if current else []
    infos: list[dict[str, Any]] = []
    seen = set()
    while current and current in product_to_step and current not in seen:
        seen.add(current)
        step = product_to_step[current]
        reactants = _step_reactants(step)
        if not reactants:
            break
        chosen = _choose_main_path_reactant(reactants, products=products)
        if not chosen:
            break
        infos.append(
            {
                "step": step,
                "product": current,
                "chosen_reactant": chosen,
                "side_reactants": [smi for smi in reactants if smi != chosen],
            }
        )
        chain.append(chosen)
        if chosen not in product_to_step:
            break
        current = chosen
    return _dedupe_consecutive(chain), infos


def _step_reactants(step: dict[str, Any]) -> list[str]:
    values = []
    if step.get("main_reactant"):
        values.append(str(step.get("main_reactant")))
    values.extend(str(smi) for smi in step.get("aux_reactants") or [] if smi)
    if not values:
        values = _rxn_lhs_parts(str(step.get("reaction_smiles") or ""))
    return _dedupe(values)


def _choose_main_path_reactant(reactants: list[str], *, products: set[str]) -> str:
    internal = [smi for smi in reactants if smi in products]
    if internal:
        return max(internal, key=_heavy_atoms)
    return max(reactants, key=_heavy_atoms) if reactants else ""


def _heavy_atoms(smiles: str) -> int:
    mol = Chem.MolFromSmiles(str(smiles or ""))
    if mol is None:
        return 0
    return sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() != "H")


def _scheme_arrow(
    step: dict[str, Any],
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    *,
    condition_row: dict[str, Any] | None = None,
) -> str:
    top = _condition_top(step, condition_row=condition_row)
    bottom = _condition_bottom(step, condition_row=condition_row)
    risk = str((condition_row or {}).get("risk") or "ok")
    marker = _condition_risk_marker(risk)
    css = "cond-risk-high" if risk == "high" else "cond-risk-warn" if risk == "warn" else "cond"
    bottom_css = "cond2-risk-high" if risk == "high" else "cond2-risk-warn" if risk == "warn" else "cond2"
    top_text = top + marker if top else top
    bottom_text = bottom + marker if not top and bottom else bottom
    return "\n".join(
        [
            f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" class="arrow"/>',
            f'<text x="{(x1 + x2) / 2}" y="{y1 - 19}" class="{css}" text-anchor="middle">{_esc(top_text)}</text>',
            f'<text x="{(x1 + x2) / 2}" y="{y1 + 26}" class="{bottom_css}" text-anchor="middle">{_esc(bottom_text)}</text>',
        ]
    )


def _condition_top(step: dict[str, Any], *, condition_row: dict[str, Any] | None = None) -> str:
    bits = []
    cond = (step.get("condition_predictions") or [{}])[0]
    reagent = _cond_value(cond, "Reagent", "reagent")
    catalyst = _cond_value(cond, "Catalyst", "catalyst")
    if reagent:
        bits.append(_condition_label(str(reagent), context="reagent"))
    if catalyst:
        bits.append(_catalyst_label(str(catalyst)))
    text = "; ".join(bits[:2])
    return _short(text, 52)


def _condition_bottom(step: dict[str, Any], *, condition_row: dict[str, Any] | None = None) -> str:
    cond = (step.get("condition_predictions") or [{}])[0]
    bits = []
    solvent = _cond_value(cond, "Solvent", "solvent")
    temp = _cond_value(cond, "Temperature", "temperature")
    if solvent:
        bits.append(_condition_label(str(solvent), context="solvent"))
    if temp is not None and temp != "":
        try:
            bits.append(f"{float(temp):.0f} \N{DEGREE SIGN}C")
        except (TypeError, ValueError):
            bits.append(str(temp))
    return _short("; ".join(bits[:3]), 52)


def _condition_label(text: str, *, context: str = "generic") -> str:
    aliases = {
        "[OH-].[Na+]": "NaOH",
        "[Na+].[OH-]": "NaOH",
        "O.[OH-].[Na+]": "aq. NaOH",
        "O=C([O-])[O-].[Na+]": "Na2CO3",
        "O=C([O-])[O-].[Cs+]": "Cs2CO3",
        "O=C([O-])[O-].[K+]": "K2CO3",
        "O=C(O)C(F)(F)F": "TFA",
        "O=P(Cl)(Cl)Cl": "POCl3",
        "O.O=P(Cl)(Cl)Cl": "POCl3/H2O",
        "O=P(Cl)(Cl)Cl.[OH-].[Na+]": "POCl3/NaOH",
        "CC(C)[N-]C(C)C.[Li+]": "LDA",
        "[Li]CCCC.CC(C)NC(C)C": "n-BuLi/i-Pr2NH",
        "C[Si](C)(C)[N-][Si](C)(C)C.[Li+]": "LiHMDS",
        "CC(C)C[AlH4]CC(C)C": "DIBAL-H",
        "[H-].[Na+]": "NaH",
        "[NaH]": "NaH",
        "[H][N-][H].[Na+]": "NaNH2",
        "CCB(CC)OC.[BH4-].[Na+]": "NaBH4/EtOBEt2",
        "CCB(CC)CC.[BH4-].[Na+]": "NaBH4/Et3B",
        "[Mg++].[Cl-].[Cl-].O=C(N1C=CN=C1)N1C=CN=C1": "CDI/MgCl2",
        "O=C(N1C=CN=C1)N1C=CN=C1": "CDI",
        "CC(=O)OI1(OC(C)=O)(OC(C)=O)OC(=O)c2ccccc21": "DMP",
        "O=[N+]([O-])c1ccccc1.Cl[AlH3](Cl)Cl": "AlCl3/nitrobenzene",
        "[Al+3].[Cl-].[Cl-].[Cl-]": "AlCl3",
        "c1ccncc1": "pyridine",
        "CN(C)c1ccncc1": "DMAP",
        "C=O.CC[O-].[Na+]": "HCHO/NaOEt",
        "C1CCOC1": "THF",
        "C1CCOC1.C1CCOC1": "THF",
        "C1CCOC1.O": "THF/H2O",
        "C1CCOC1.CO": "THF/MeOH",
        "COCCOC.O": "DME/H2O",
        "Cc1ccccc1": "toluene",
        "Cc1ccccc1.ClCCl": "toluene/DCE",
        "ClCCl": "DCE",
        "ClC(Cl)Cl": "CHCl3",
        "ClC1=CC=CC=C1": "chlorobenzene",
        "CC#N": "MeCN",
        "CC#N.O": "MeCN/H2O",
        "CN(C)C=O": "DMF",
        "CS(C)=O": "DMSO",
        "O": "H2O",
        "CO": "MeOH",
        "CCO": "EtOH",
    }
    if text in aliases:
        return aliases[text]
    if "." in text:
        labels = [aliases.get(part, _formula_or_short_smiles(part)) for part in text.split(".") if part]
        labels = _dedupe(labels)
        sep = "/" if context == "solvent" else ", "
        return sep.join(labels[:3])
    return _formula_or_short_smiles(text)


def _catalyst_label(text: str) -> str:
    aliases = {
        "O=[Mn]=O": "MnO2",
    }
    if text in aliases:
        return aliases[text]
    if "Pd" in text:
        return "Pd catalyst"
    return "cat. " + _condition_label(text, context="catalyst")


def _aux_text(aux: list[str], *, x: float, y: float) -> str:
    label = " + ".join(_reagent_label(smi) for smi in aux[:3])
    if len(aux) > 3:
        label += f" + {len(aux) - 3} more"
    return f'<text x="{x}" y="{y}" class="reagent" text-anchor="middle">{_esc(_short(label, 52))}</text>'


def _aux_mini_structures(aux: list[str], *, x: float, y: float) -> str:
    shown = aux[:3]
    if not shown:
        return ""
    w = 66 if len(shown) >= 3 else 76
    h = 48
    gap = 6
    total = len(shown) * w + (len(shown) - 1) * gap
    x0 = x - total / 2
    chunks = []
    for idx, smi in enumerate(shown):
        chunks.append(_mol_panel(smi, x=x0 + idx * (w + gap), y=y, width=w, height=h, label="", role="aux"))
        if idx < len(shown) - 1:
            plus_x = x0 + (idx + 1) * w + idx * gap + gap / 2
            chunks.append(f'<text x="{plus_x}" y="{y + h / 2 + 4}" class="mini-plus" text-anchor="middle">+</text>')
    return "\n".join(chunks)


def _aux_or_branch_structures(
    aux: list[str],
    *,
    graph: dict[str, Any],
    main_chain: set[str],
    x: float,
    y: float,
) -> str:
    products: set[str] = graph["products"]
    branches = [smi for smi in aux if smi in products and smi not in main_chain]
    if branches:
        return _branch_mini_scheme(branches[0], graph=graph, x=x, y=y)
    return _aux_mini_structures(aux, x=x, y=y)


def _display_aux_reactants(aux: list[str], step: dict[str, Any]) -> list[str]:
    condition_parts = _condition_component_smiles(step)
    shown = []
    for smi in aux:
        if _is_tiny_inorganic(smi):
            continue
        if smi in condition_parts and _heavy_atoms(smi) <= 3 and _carbon_count(smi) == 0:
            continue
        shown.append(smi)
    return shown


def _condition_component_smiles(step: dict[str, Any]) -> set[str]:
    cond = (step.get("condition_predictions") or [{}])[0]
    parts: set[str] = set()
    if not isinstance(cond, dict):
        return parts
    for key in ("Reagent", "reagent", "Catalyst", "catalyst"):
        value = cond.get(key)
        if not value:
            continue
        text = str(value)
        parts.add(text)
        parts.update(part for part in text.split(".") if part)
    return parts


def _is_tiny_inorganic(smiles: str) -> bool:
    mol = Chem.MolFromSmiles(str(smiles or ""))
    if mol is None:
        return False
    heavy = 0
    carbon = 0
    for atom in mol.GetAtoms():
        if atom.GetSymbol() == "H":
            continue
        heavy += 1
        if atom.GetSymbol() == "C":
            carbon += 1
    return heavy <= 1 and carbon == 0


def _carbon_count(smiles: str) -> int:
    mol = Chem.MolFromSmiles(str(smiles or ""))
    if mol is None:
        return 0
    return sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == "C")


def _branch_mini_scheme(smiles: str, *, graph: dict[str, Any], x: float, y: float) -> str:
    product_to_step: dict[str, dict[str, Any]] = graph["product_to_step"]
    branch_chain_retro, _infos = _retro_main_path(
        [product_to_step[smiles]],
        target_smiles=smiles,
        graph=graph,
    )
    branch_chain = list(reversed(branch_chain_retro))
    if len(branch_chain) > 4:
        branch_chain = [branch_chain[0], "...", branch_chain[-1]]
    w = 58
    h = 42
    arrow_w = 24
    total = len(branch_chain) * w + max(0, len(branch_chain) - 1) * arrow_w
    x0 = x - total / 2
    chunks = []
    for idx, smi in enumerate(branch_chain):
        px = x0 + idx * (w + arrow_w)
        if smi == "...":
            chunks.append(f'<text x="{px + w / 2}" y="{y + h / 2 + 4}" class="mini-plus" text-anchor="middle">...</text>')
        else:
            chunks.append(_mol_panel(smi, x=px, y=y, width=w, height=h, label="", role="aux"))
        if idx < len(branch_chain) - 1:
            ax1 = px + w + 3
            ax2 = ax1 + arrow_w - 6
            ay = y + h / 2
            chunks.append(f'<line x1="{ax1}" y1="{ay}" x2="{ax2}" y2="{ay}" class="mini-arrow"/>')
    return "\n".join(chunks)


def _reagent_label(smiles: str) -> str:
    aliases = {
        "[OH-]": "OH-",
        "O": "H2O",
        "CO": "MeOH",
        "CCO": "EtOH",
        "CI": "MeI",
        "CN(C)C=O": "DMF",
        "O=P(Cl)(Cl)Cl": "POCl3",
        "CS(=O)(=O)Cl": "MsCl",
        "N=C(N)N": "guanidine",
        "O=C([O-])[O-]": "carbonate",
        "CC(=O)Cl": "AcCl",
        "CC(=O)OC(C)(C)C": "tert-butyl acetate",
        "COC(=O)CC(=O)OC": "dimethyl malonate",
        "OB(O)c1ccc(F)cc1": "4-F-PhB(OH)2",
        "CCOC(=O)C=P(c1ccccc1)(c1ccccc1)c1ccccc1": "Ph3P=CHCO2Et",
        "CCOC(=O)CP(=O)(OCC)OCC": "(EtO)2P(O)CH2CO2Et",
    }
    if smiles in aliases:
        return aliases[smiles]
    return _formula_or_short_smiles(smiles)


def _aux_reactants(step: dict[str, Any]) -> list[str]:
    aux = [str(smi) for smi in step.get("aux_reactants") or [] if smi]
    if aux:
        return aux
    parts = _rxn_lhs_parts(str(step.get("reaction_smiles") or ""))
    main = str(step.get("main_reactant") or "")
    return [smi for smi in parts if smi != main][0:2]


def _mol_panel(smiles: str, *, x: float, y: float, width: int, height: int, label: str, role: str) -> str:
    mol = molecule_svg(smiles, width=width, height=height)
    label_row = f'<text x="{x + width / 2}" y="{y - 8}" class="mol-label" text-anchor="middle">{_esc(label)}</text>' if label else ""
    return "\n".join([label_row, f'<g transform="translate({x},{y})">{mol}</g>'])


def molecule_svg(smiles: str, *, width: int, height: int) -> str:
    mol = Chem.MolFromSmiles(str(smiles or ""))
    if mol is None:
        return f'<text x="8" y="{height / 2}" class="caption">invalid SMILES</text>'
    rdDepictor.Compute2DCoords(mol)
    drawer = rdMolDraw2D.MolDraw2DSVG(int(width), int(height))
    opts = drawer.drawOptions()
    opts.padding = 0.06
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    start = svg.find("<svg")
    start = svg.find(">", start) if start >= 0 else svg.find(">")
    end = svg.rfind("</svg>")
    if start >= 0 and end >= 0:
        return svg[start + 1 : end]
    return svg


def _continuation_marker(x: float, y: float, smiles: str) -> str:
    return f'<text x="{x}" y="{y}" class="continuation">continued from previous row: {_esc(_short(smiles, 52))}</text>'


def _down_continuation(x: float, y: float) -> str:
    return "\n".join(
        [
            f'<line x1="{x}" y1="{y}" x2="{x}" y2="{y + 42}" class="dash"/>',
            f'<text x="{x + 8}" y="{y + 28}" class="continuation">continue</text>',
        ]
    )


def _svg_header(width: int, height: int) -> str:
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<defs>
  <marker id="arrowhead" markerWidth="11" markerHeight="8" refX="10" refY="4" orient="auto">
    <polygon points="0 0, 11 4, 0 8" fill="#111"/>
  </marker>
  <style>
    text {{ font-family: Arial, Helvetica, "Noto Sans CJK SC", sans-serif; }}
    .title {{ font-size: 24px; font-weight: 700; fill: #111; }}
    .subtitle {{ font-size: 13px; fill: #555; }}
    .cond {{ font-size: 11px; fill: #111; }}
    .cond-risk-warn {{ font-size: 11px; fill: #8a5a00; font-weight: 700; }}
    .cond-risk-high {{ font-size: 11px; fill: #a40000; font-weight: 700; }}
    .cond2 {{ font-size: 10px; fill: #555; }}
    .cond2-risk-warn {{ font-size: 10px; fill: #8a5a00; font-weight: 700; }}
    .cond2-risk-high {{ font-size: 10px; fill: #a40000; font-weight: 700; }}
    .reagent {{ font-size: 10px; fill: #111; }}
    .mini-plus {{ font-size: 10px; fill: #333; }}
    .caption {{ font-size: 8px; fill: #555; }}
    .mol-label {{ font-size: 10px; font-weight: 700; fill: #333; }}
    .continuation {{ font-size: 10px; fill: #666; }}
    .footnote {{ font-size: 10px; fill: #555; }}
    .arrow {{ stroke: #111; stroke-width: 1.8; marker-end: url(#arrowhead); }}
    .mini-arrow {{ stroke: #333; stroke-width: 1.0; marker-end: url(#arrowhead); }}
    .dash {{ stroke: #777; stroke-width: 1.2; stroke-dasharray: 4 4; }}
  </style>
</defs>'''


def _title(route: dict[str, Any], route_number: int) -> str:
    return f"Route {route_number:02d} ({route.get('n_steps') or len(route.get('steps') or [])} steps)"


def _subtitle(route: dict[str, Any]) -> str:
    return ""


def _reaction_product(step: dict[str, Any]) -> str:
    rxn = str(step.get("reaction_smiles") or "")
    if ">>" not in rxn:
        return ""
    _lhs, rhs = rxn.split(">>", 1)
    return rhs.split(".")[0] if rhs else ""


def _rxn_lhs_parts(rxn: str) -> list[str]:
    if ">>" not in rxn:
        return []
    lhs, _rhs = rxn.split(">>", 1)
    return [part for part in lhs.split(".") if part]


def _cond_value(row: dict[str, Any], *keys: str) -> Any:
    if not isinstance(row, dict):
        return None
    for key in keys:
        if key in row and row.get(key) not in {None, ""}:
            return row.get(key)
    return None


def _dedupe_consecutive(values: list[str]) -> list[str]:
    out = []
    for value in values:
        if value and (not out or out[-1] != value):
            out.append(value)
    return out


def _dedupe(values: list[str]) -> list[str]:
    out = []
    seen = set()
    for value in values:
        if value and value not in seen:
            out.append(value)
            seen.add(value)
    return out


def _short(text: str, limit: int) -> str:
    text = str(text or "")
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _formula_or_short_smiles(smiles: str) -> str:
    mol = Chem.MolFromSmiles(str(smiles or ""))
    if mol is None:
        return _short(smiles, 24)
    formula = _mol_formula(mol)
    if formula and len(str(smiles)) > 22:
        return formula
    return _short(smiles, 24)


def _mol_formula(mol: Chem.Mol) -> str:
    counts: dict[str, int] = {}
    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol()
        counts[symbol] = counts.get(symbol, 0) + 1
    order = ["C", "H", "N", "O", "F", "Cl", "Br", "I", "S", "P", "B"]
    parts = []
    for element in order:
        value = counts.pop(element, 0)
        if value:
            parts.append(element if value == 1 else f"{element}{value}")
    for element in sorted(counts):
        value = counts[element]
        parts.append(element if value == 1 else f"{element}{value}")
    return "".join(parts)


def _esc(value: Any) -> str:
    return html.escape(str(value or ""), quote=True)


def render_index(rows: list[dict[str, Any]], *, source: Path) -> str:
    blocks = []
    for row in rows:
        links = [f'<a href="{html.escape(row["svg"])}">SVG</a>']
        if row.get("pdf"):
            links.append(f'<a href="{html.escape(row["pdf"])}">PDF</a>')
        blocks.append(
            f'''<section>
<h2>Route {row["rank"]:02d}</h2>
<p>{' | '.join(links)}</p>
<object data="{html.escape(row["svg"])}" type="image/svg+xml"></object>
</section>'''
        )
    return f'''<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Linear Route Schemes</title>
<style>
body {{ font-family: Arial, Helvetica, "Noto Sans CJK SC", sans-serif; margin: 28px; color: #111; background: #fff; }}
h1 {{ font-size: 24px; margin-bottom: 4px; }}
h2 {{ font-size: 18px; margin: 28px 0 4px; }}
p {{ color: #555; }}
object {{ width: 100%; min-height: 760px; border: 1px solid #ddd; }}
section {{ break-after: page; }}
</style>
</head>
<body>
<h1>Linear Route Schemes</h1>
<p>Source: {html.escape(str(source))}</p>
{''.join(blocks)}
</body>
</html>
'''


if __name__ == "__main__":
    main()
