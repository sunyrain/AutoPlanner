"""Render auditable molecule examples from recent-total-synthesis candidates.

The script reads candidate JSONL records instead of embedding structures.  The
result is a review aid only: labels preserve the non-admission status and the
visual/PubChem relation reported by the dataset projection.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont
from rdkit import Chem
from rdkit.Chem import Draw, rdMolDescriptors


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = REPO_ROOT / "benchmarks" / "recent_total_synthesis"
DEFAULT_OUTPUT = REPO_ROOT / "output" / "benchmark_examples"
DEFAULT_TARGETS = (
    "scytonemin",
    "Plasmodiophorol A",
    "Calothrixin B",
    "Bufogargarizin B",
)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _load_candidates(dataset_dir: Path) -> dict[str, dict[str, Any]]:
    paths = (
        dataset_dir / "visual_structure_candidates.jsonl",
        dataset_dir
        / "curation_candidates"
        / "p1_scope"
        / "visual-structure-candidates.jsonl",
    )
    rows = [row for path in paths for row in _load_jsonl(path)]
    return {str(row["target_name"]): row for row in rows}


def _font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    candidates = (
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/calibrib.ttf" if bold else "C:/Windows/Fonts/calibri.ttf",
    )
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def _molecule_metrics(molecule: Chem.Mol) -> str:
    heavy_atoms = molecule.GetNumHeavyAtoms()
    rings = rdMolDescriptors.CalcNumRings(molecule)
    stereocentres = len(Chem.FindMolChiralCenters(molecule, includeUnassigned=True))
    formula = rdMolDescriptors.CalcMolFormula(molecule)
    return (
        f"{formula}  |  {heavy_atoms} heavy atoms  |  "
        f"{rings} rings  |  {stereocentres} stereocentres"
    )


def _safe_stem(value: str) -> str:
    return "".join(char.lower() if char.isalnum() else "-" for char in value).strip("-")


def _draw_structure(smiles: str, size: tuple[int, int]) -> Image.Image:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"RDKit could not parse candidate SMILES: {smiles}")
    Chem.rdDepictor.Compute2DCoords(molecule)
    return Draw.MolToImage(
        molecule,
        size=size,
        kekulize=True,
        fitImage=True,
        options=None,
    ).convert("RGB")


def _render_card(
    row: dict[str, Any],
    *,
    smiles: str,
    source_label: str,
    size: tuple[int, int] = (1040, 760),
) -> Image.Image:
    width, height = size
    image = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(image)
    title_font = _font(31, bold=True)
    body_font = _font(20)
    small_font = _font(17)
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"RDKit could not parse candidate SMILES: {smiles}")

    draw.text((34, 26), str(row["target_name"]), fill="#17324D", font=title_font)
    draw.text(
        (34, 70),
        f"DOI {row['doi']}  |  {source_label}",
        fill="#5B6B7F",
        font=body_font,
    )
    structure = _draw_structure(smiles, (width - 52, height - 205))
    image.paste(structure, (26, 102))
    draw.line((26, height - 92, width - 26, height - 92), fill="#D5DEE8", width=2)
    draw.text((34, height - 78), _molecule_metrics(molecule), fill="#17324D", font=small_font)
    draw.text(
        (34, height - 49),
        (
            f"candidate={row['visual_status']}  |  "
            f"cross-source relation={row['visual_pubchem_relation']}  |  NOT ADMITTED"
        ),
        fill="#9A5B13",
        font=small_font,
    )
    return image


def _fit_tile(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    result = image.copy()
    result.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, "white")
    x = (size[0] - result.width) // 2
    y = (size[1] - result.height) // 2
    canvas.paste(result, (x, y))
    return canvas


def render(
    dataset_dir: Path,
    output_dir: Path,
    target_names: list[str],
) -> dict[str, Any]:
    candidates = _load_candidates(dataset_dir)
    missing = [name for name in target_names if name not in candidates]
    if missing:
        raise KeyError(f"unknown target name(s): {', '.join(missing)}")

    output_dir.mkdir(parents=True, exist_ok=True)
    rendered: list[tuple[str, Image.Image]] = []
    records: list[dict[str, Any]] = []
    for name in target_names:
        row = candidates[name]
        visual_smiles = str(row.get("visual_canonical_isomeric_smiles") or "").strip()
        if not visual_smiles:
            raise ValueError(f"{name} has no RDKit-valid visual candidate")
        card = _render_card(row, smiles=visual_smiles, source_label="paper-image transcription")
        path = output_dir / f"{_safe_stem(name)}.png"
        card.save(path, dpi=(300, 300))
        rendered.append((name, card))
        records.append(
            {
                "target_name": name,
                "doi": row["doi"],
                "target_slot_id": row["target_slot_id"],
                "visual_status": row["visual_status"],
                "visual_pubchem_relation": row["visual_pubchem_relation"],
                "source_locator": row["source_locator"],
                "visual_smiles": visual_smiles,
                "output": str(path.relative_to(REPO_ROOT)),
            }
        )

        pubchem = list(row.get("pubchem_canonical_isomeric_smiles") or [])
        if row.get("visual_pubchem_relation") == "connectivity_match_stereo_difference" and pubchem:
            comparison = _render_card(
                row,
                smiles=str(pubchem[0]),
                source_label="PubChem name-resolution candidate",
            )
            comparison_path = output_dir / f"{_safe_stem(name)}-pubchem-comparison.png"
            comparison.save(comparison_path, dpi=(300, 300))
            rendered.append((f"{name} (PubChem)", comparison))
            records[-1]["pubchem_comparison_smiles"] = str(pubchem[0])
            records[-1]["pubchem_comparison_output"] = str(
                comparison_path.relative_to(REPO_ROOT)
            )

    tile_size = (780, 570)
    columns = 2
    rows = (len(rendered) + columns - 1) // columns
    grid = Image.new("RGB", (columns * tile_size[0], rows * tile_size[1]), "white")
    for index, (_, card) in enumerate(rendered):
        x = (index % columns) * tile_size[0]
        y = (index // columns) * tile_size[1]
        grid.paste(_fit_tile(card, tile_size), (x, y))
    grid_path = output_dir / "recent-total-synthesis-examples.png"
    grid.save(grid_path, dpi=(240, 240))

    manifest = {
        "schema_version": "recent_total_synthesis_example_render.v1",
        "claim_boundary": (
            "Review render of non-admitted structure candidates; not benchmark truth."
        ),
        "dataset_dir": str(dataset_dir.resolve()),
        "grid_output": str(grid_path.relative_to(REPO_ROOT)),
        "records": records,
    }
    manifest_path = output_dir / "recent-total-synthesis-examples.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return {
        "grid": str(grid_path.resolve()),
        "manifest": str(manifest_path.resolve()),
        "targets": len(target_names),
        "panels": len(rendered),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--target", action="append", dest="targets")
    args = parser.parse_args()
    targets = args.targets or list(DEFAULT_TARGETS)
    print(json.dumps(render(args.dataset_dir, args.output_dir, targets), indent=2))


if __name__ == "__main__":
    main()
