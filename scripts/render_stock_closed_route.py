"""Render a stock-closed target-solve route as an auditable scientific figure.

The renderer deliberately keeps stock closure, deterministic ReactionJSON replay,
and reaction credibility as separate visual channels.  It is intended for route
inspection and manuscript figures, not as evidence that a route is executable.
"""

from __future__ import annotations

import argparse
import json
import textwrap
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.patches import FancyBboxPatch
from PIL import Image
from rdkit import Chem
from rdkit.Chem import Draw, rdMolDescriptors


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 8,
        "axes.linewidth": 0.8,
    }
)


NAVY = "#17324D"
SLATE = "#5B6B7F"
LIGHT_SLATE = "#D5DEE8"
GREEN = "#27855B"
GREEN_FILL = "#EAF6EF"
AMBER = "#B86B16"
AMBER_FILL = "#FFF4E5"
RED = "#B43A3A"
RED_FILL = "#FCEBEC"
WHITE = "#FFFFFF"


def _canonical_smiles(value: Any) -> str:
    text = str(value or "").strip()
    molecule = Chem.MolFromSmiles(text)
    if molecule is None:
        return text
    return Chem.MolToSmiles(molecule, isomericSmiles=True)


def _formula(smiles: str) -> str:
    molecule = Chem.MolFromSmiles(smiles)
    return rdMolDescriptors.CalcMolFormula(molecule) if molecule is not None else ""


def _molecule_image(smiles: str, *, size: tuple[int, int] = (390, 245)) -> Image.Image:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return Image.new("RGBA", size, WHITE)
    image = Draw.MolToImage(molecule, size=size, kekulize=True)
    return image.convert("RGBA")


def _walk(value: Any) -> Iterable[Any]:
    yield value
    if isinstance(value, Mapping):
        for item in value.values():
            yield from _walk(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk(item)


def _critic_status(report: Mapping[str, Any]) -> str:
    priorities = {"reject": 3, "uncertain": 2, "accept": 1}
    statuses: list[str] = []
    for item in _walk(report):
        if not isinstance(item, Mapping):
            continue
        status = str(item.get("overall_assessment") or "").lower()
        if status in priorities:
            statuses.append(status)
    return max(statuses, key=lambda item: priorities[item]) if statuses else "not assessed"


def _select_route(report: Mapping[str, Any], route_index: int) -> dict[str, Any]:
    gates = report.get("gates") or {}
    routes = [dict(row) for row in gates.get("routes") or [] if isinstance(row, Mapping)]
    stock_closed = [row for row in routes if row.get("stock_closed") is True and row.get("steps")]
    candidates = stock_closed or [row for row in routes if row.get("steps")]
    if not candidates:
        raise ValueError("report contains no materialized route with steps")
    if route_index < 0 or route_index >= len(candidates):
        raise IndexError(f"route index {route_index} outside 0..{len(candidates) - 1}")
    return candidates[route_index]


def _route_tree(route: Mapping[str, Any]) -> tuple[str, dict[str, dict[str, Any]]]:
    steps: dict[str, dict[str, Any]] = {}
    precursors: set[str] = set()
    for raw in route.get("steps") or []:
        step = dict(raw)
        product = _canonical_smiles(step.get("product_smiles"))
        if not product:
            continue
        step["product_smiles"] = product
        step["precursor_smiles"] = [
            _canonical_smiles(value) for value in step.get("precursor_smiles") or [] if value
        ]
        steps[product] = step
        precursors.update(step["precursor_smiles"])
    roots = [product for product in steps if product not in precursors]
    if len(roots) != 1:
        raise ValueError(f"expected one target root, found {len(roots)}")
    return roots[0], steps


def _layout_tree(
    root: str, steps: Mapping[str, Mapping[str, Any]]
) -> tuple[dict[str, tuple[float, int]], dict[str, tuple[float, int]], list[str]]:
    molecule_positions: dict[str, tuple[float, int]] = {}
    reaction_positions: dict[str, tuple[float, int]] = {}
    leaf_order: list[str] = []
    active: set[str] = set()

    def visit(smiles: str, depth: int) -> float:
        if smiles in active:
            raise ValueError("route contains a cycle")
        if smiles in molecule_positions:
            return molecule_positions[smiles][0]
        active.add(smiles)
        step = steps.get(smiles)
        children = list(step.get("precursor_smiles") or []) if step else []
        if not children:
            x = float(len(leaf_order))
            leaf_order.append(smiles)
        else:
            child_x = [visit(child, depth + 1) for child in children]
            x = sum(child_x) / len(child_x)
            reaction_positions[smiles] = (x, depth)
        molecule_positions[smiles] = (x, depth)
        active.remove(smiles)
        return x

    visit(root, 0)
    return molecule_positions, reaction_positions, leaf_order


def _draw_molecule_card(
    ax: plt.Axes,
    *,
    x: float,
    y: float,
    smiles: str,
    label: str,
    is_target: bool,
    is_stock: bool,
) -> None:
    width = 2.58
    height = 1.52
    edge = GREEN if is_stock else (NAVY if is_target else LIGHT_SLATE)
    fill = GREEN_FILL if is_stock else WHITE
    card = FancyBboxPatch(
        (x - width / 2, y - height / 2),
        width,
        height,
        boxstyle="round,pad=0.045,rounding_size=0.12",
        linewidth=1.8 if is_target or is_stock else 1.2,
        edgecolor=edge,
        facecolor=fill,
        zorder=2,
    )
    ax.add_patch(card)
    image = _molecule_image(smiles)
    artist = AnnotationBbox(
        OffsetImage(image, zoom=0.33),
        (x, y + 0.07),
        frameon=False,
        box_alignment=(0.5, 0.5),
        zorder=3,
    )
    ax.add_artist(artist)
    role = "TARGET" if is_target else ("STOCK LEAF" if is_stock else "INTERMEDIATE")
    role_color = GREEN if is_stock else NAVY
    ax.text(
        x - width / 2 + 0.10,
        y + height / 2 - 0.17,
        f"{label}  {role}",
        ha="left",
        va="top",
        color=role_color,
        fontsize=7.4,
        fontweight="bold",
        zorder=4,
    )
    ax.text(
        x,
        y - height / 2 + 0.12,
        _formula(smiles),
        ha="center",
        va="bottom",
        color=SLATE,
        fontsize=7.0,
        zorder=4,
    )


def _draw_reaction_card(
    ax: plt.Axes,
    *,
    x: float,
    y: float,
    step: Mapping[str, Any],
    ordinal: int,
) -> None:
    audit = step.get("reactionjson_audit") or {}
    replayed = audit.get("accepted") is True
    conditions = len(step.get("condition_predictions") or [])
    title = str(step.get("transformation_hypothesis") or "Reaction hypothesis")
    lines = textwrap.wrap(title, width=31)[:2]
    status = f"host replay {'pass' if replayed else 'open'}  |  conditions {conditions}  |  proof open"
    width = 2.68
    height = 0.86
    card = FancyBboxPatch(
        (x - width / 2, y - height / 2),
        width,
        height,
        boxstyle="round,pad=0.045,rounding_size=0.09",
        linewidth=1.25,
        edgecolor=AMBER,
        facecolor=AMBER_FILL,
        zorder=5,
    )
    ax.add_patch(card)
    ax.text(
        x - width / 2 + 0.10,
        y + height / 2 - 0.12,
        f"S{ordinal}",
        ha="left",
        va="top",
        color=AMBER,
        fontsize=7.3,
        fontweight="bold",
        zorder=6,
    )
    ax.text(
        x - width / 2 + 0.43,
        y + height / 2 - 0.12,
        "\n".join(lines),
        ha="left",
        va="top",
        color=NAVY,
        fontsize=6.7,
        fontweight="semibold",
        zorder=6,
    )
    ax.text(
        x,
        y - height / 2 + 0.11,
        status,
        ha="center",
        va="bottom",
        color=SLATE,
        fontsize=5.8,
        zorder=6,
    )


def render(report_path: Path, output_prefix: Path, route_index: int = 0) -> dict[str, Any]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    route = _select_route(report, route_index)
    root, steps = _route_tree(route)
    molecule_positions, _, leaves = _layout_tree(root, steps)

    x_gap = 3.05
    y_gap = 2.75
    max_depth = max(depth for _, depth in molecule_positions.values())
    leaf_count = max(1, len(leaves))
    figure_width = max(12.0, 2.9 * leaf_count + 2.0)
    figure_height = max(9.0, 2.1 * (max_depth + 1) + 2.2)
    fig, ax = plt.subplots(figsize=(figure_width, figure_height), constrained_layout=False)
    fig.patch.set_facecolor(WHITE)
    ax.set_facecolor(WHITE)

    ordinal_by_product = {
        _canonical_smiles(step.get("product_smiles")): index
        for index, step in enumerate(route.get("steps") or [], start=1)
    }

    # Draw links first so that molecule and reaction cards remain visually dominant.
    for product, step in steps.items():
        product_x, depth = molecule_positions[product]
        px = product_x * x_gap
        py = -depth * y_gap
        ry = py - y_gap / 2
        ax.annotate(
            "",
            xy=(px, ry + 0.44),
            xytext=(px, py - 0.78),
            arrowprops=dict(arrowstyle="-|>", color=SLATE, lw=1.15, mutation_scale=9),
            zorder=1,
        )
        for precursor in step.get("precursor_smiles") or []:
            child_x, child_depth = molecule_positions[precursor]
            cx = child_x * x_gap
            cy = -child_depth * y_gap
            ax.annotate(
                "",
                xy=(cx, cy + 0.80),
                xytext=(px, ry - 0.44),
                arrowprops=dict(
                    arrowstyle="-|>",
                    color=SLATE,
                    lw=1.15,
                    mutation_scale=9,
                    connectionstyle="arc3,rad=0.0",
                ),
                zorder=1,
            )

    molecule_order = sorted(molecule_positions, key=lambda item: (molecule_positions[item][1], molecule_positions[item][0]))
    labels = {smiles: f"M{index}" for index, smiles in enumerate(molecule_order)}
    for smiles, (x_index, depth) in molecule_positions.items():
        _draw_molecule_card(
            ax,
            x=x_index * x_gap,
            y=-depth * y_gap,
            smiles=smiles,
            label=labels[smiles],
            is_target=smiles == root,
            is_stock=smiles in leaves,
        )
    for product, step in steps.items():
        x_index, depth = molecule_positions[product]
        _draw_reaction_card(
            ax,
            x=x_index * x_gap,
            y=-depth * y_gap - y_gap / 2,
            step=step,
            ordinal=ordinal_by_product.get(product, 0),
        )

    paper = report.get("paper_equivalent") or {}
    paper_solved = paper.get("paper_equivalent_solved") is True
    route_validated = route.get("reaction_validated") is True
    critic = _critic_status(report)
    title = "Traversiadiene retrosynthesis | stock-closed hypothesis route"
    subtitle = (
        f"Paper-equivalent B4: {'SOLVED' if paper_solved else 'OPEN'}   |   "
        f"Host reaction validation: {'PASS' if route_validated else 'FAIL'}   |   "
        f"Route-level Critic: {critic.upper()}   |   {len(steps)} steps, {len(leaves)} stock leaves"
    )
    fig.suptitle(title, x=0.055, y=0.982, ha="left", fontsize=16, fontweight="bold", color=NAVY)
    fig.text(0.055, 0.952, subtitle, ha="left", va="top", fontsize=8.7, color=SLATE)
    fig.text(
        0.055,
        0.927,
        "Green = exact frozen-stock leaf. Amber = deterministic graph replay only; reaction proof remains open. "
        "Arrows point in the retrosynthetic direction.",
        ha="left",
        va="top",
        fontsize=7.7,
        color=SLATE,
    )
    if critic == "reject":
        fig.text(
            0.945,
            0.973,
            "CRITIC REJECT",
            ha="right",
            va="top",
            fontsize=8.0,
            color=RED,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.35", facecolor=RED_FILL, edgecolor=RED, linewidth=1.0),
        )

    x_min = -1.65
    x_max = max(1.0, (leaf_count - 1) * x_gap + 1.65)
    y_min = -max_depth * y_gap - 1.1
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, 1.08)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    fig.subplots_adjust(left=0.025, right=0.975, top=0.90, bottom=0.025)

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    paths = {
        "png": output_prefix.with_suffix(".png"),
        "svg": output_prefix.with_suffix(".svg"),
        "pdf": output_prefix.with_suffix(".pdf"),
        "tiff": output_prefix.with_suffix(".tiff"),
        "source": output_prefix.with_name(f"{output_prefix.name}.source.json"),
    }
    fig.savefig(paths["png"], dpi=300, bbox_inches="tight", facecolor=WHITE)
    fig.savefig(paths["svg"], bbox_inches="tight", facecolor=WHITE)
    fig.savefig(paths["pdf"], bbox_inches="tight", facecolor=WHITE)
    fig.savefig(
        paths["tiff"],
        dpi=600,
        bbox_inches="tight",
        facecolor=WHITE,
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(fig)

    source = {
        "schema_version": "stock_closed_route_figure_source.v1",
        "input_report": str(report_path.resolve()),
        "run_id": report.get("run_id"),
        "route_index": route_index,
        "route_family_id": route.get("route_family_id"),
        "skeleton_id": route.get("skeleton_id"),
        "paper_equivalent_solved": paper_solved,
        "reaction_validated": route_validated,
        "critic_status": critic,
        "stock_closed": route.get("stock_closed") is True,
        "target_smiles": root,
        "steps": list(route.get("steps") or []),
        "terminal_stock_leaf_smiles": leaves,
        "figure_contract": {
            "core_conclusion": "The route closes the frozen stock boundary but remains a rejected reaction hypothesis.",
            "archetype": "schematic-led composite",
            "backend": "python",
            "status_channels_are_independent": True,
        },
    }
    paths["source"].write_text(
        json.dumps(source, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8"
    )
    return {key: str(value.resolve()) for key, value in paths.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path, help="target-only-solve-report.json")
    parser.add_argument("output_prefix", type=Path, help="output path without extension")
    parser.add_argument("--route-index", type=int, default=0)
    args = parser.parse_args()
    print(json.dumps(render(args.report, args.output_prefix, args.route_index), indent=2))


if __name__ == "__main__":
    main()
