"""Convert external open datasets into vNext StepEncoder rows.

Supported local files:
* ECREACT CSV
* Enzymatic retrosynthesis JSON train/val rows
* USPTO-50K tab file
* Rhea release tarball with reaction SMILES and EC TSVs
* ReactZyme Zenodo zip; sequence-pair splits are summarized because product
  direction is not explicit in the downloaded positive-pair files.
* RetroRules template CSVs; summarized as template assets, not step labels.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import tarfile
import time
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from rdkit import Chem, RDLogger

from cascade_planner.vnext.features import stable_id, write_jsonl


RDLogger.DisableLog("rdApp.*")


def build_external_step_pairs(
    *,
    output_dir: Path,
    ecreact_csv: Path | None = None,
    enzymatic_json: Iterable[Path] = (),
    uspto_tab: Path | None = None,
    rhea_tar: Path | None = None,
    reactzyme_zip: Path | None = None,
    retrorules_templates: Iterable[Path] = (),
    max_rows_per_source: int | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {}

    if ecreact_csv and ecreact_csv.exists():
        ecreact = list(load_ecreact(ecreact_csv, max_rows=max_rows_per_source))
        rows.extend(ecreact)
        summaries["ecreact"] = {"rows": len(ecreact)}

    enz_rows: list[dict[str, Any]] = []
    for path in enzymatic_json:
        if path.exists():
            enz_rows.extend(load_enzymatic_retro_json(path, max_rows=max_rows_per_source))
    rows.extend(enz_rows)
    if enz_rows:
        summaries["enzymatic_retro_data"] = {"rows": len(enz_rows)}

    if uspto_tab and uspto_tab.exists():
        uspto = list(load_uspto_tab(uspto_tab, max_rows=max_rows_per_source))
        rows.extend(uspto)
        summaries["uspto50k_tab"] = {"rows": len(uspto)}

    if rhea_tar and rhea_tar.exists():
        rhea = list(load_rhea_tar(rhea_tar, max_rows=max_rows_per_source))
        rows.extend(rhea)
        summaries["rhea"] = {"rows": len(rhea)}

    rows = dedupe_step_pairs(rows)
    step_path = output_dir / "external_step_pairs.jsonl"
    write_jsonl(step_path, rows)

    if reactzyme_zip and reactzyme_zip.exists():
        summaries["reactzyme"] = summarize_reactzyme_zip(reactzyme_zip)

    template_summaries = []
    for path in retrorules_templates:
        if path.exists():
            template_summaries.append(summarize_template_csv(path))
    if template_summaries:
        summaries["retrorules_templates"] = template_summaries

    manifest = {
        "schema_version": "external_step_pairs.v1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "output_dir": str(output_dir),
        "files": {"external_step_pairs": str(step_path), "manifest": str(output_dir / "manifest.json"), "report": str(output_dir / "report.md")},
        "counts": {"external_step_pairs": len(rows)},
        "quality": quality_summary(rows),
        "summaries": summaries,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "report.md").write_text(report_markdown(manifest), encoding="utf-8")
    return manifest


def load_ecreact(path: Path, *, max_rows: int | None = None) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        for idx, row in enumerate(csv.DictReader(fh)):
            if max_rows is not None and idx >= max_rows:
                break
            rxn = _normalize_reaction_ec_arrow(row.get("rxn_smiles") or "")
            parsed = _parse_reaction_parts(rxn)
            if not parsed:
                continue
            yield step_pair(
                source="ecreact",
                idx=idx,
                rxn=parsed[0],
                ec=row.get("ec") or "",
                reaction_type="enzyme_reaction",
                evidence={"dataset_source": row.get("source") or ""},
                weight=2.0,
                parsed=parsed,
            )


def load_enzymatic_retro_json(path: Path, *, max_rows: int | None = None) -> Iterable[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    for idx, row in enumerate(data):
        if max_rows is not None and idx >= max_rows:
            break
        reactants = row.get("reactants") or ""
        product = row.get("product") or ""
        rxn = f"{reactants}>>{product}"
        parsed = _parse_reaction_parts(rxn)
        if not parsed:
            continue
        yield step_pair(
            source="enzymatic_retro_data",
            idx=stable_id(path.name, idx),
            rxn=parsed[0],
            ec=row.get("ec") or "",
            reaction_type="enzyme_retro_reaction",
            evidence={"source_file": str(path)},
            weight=2.0,
            parsed=parsed,
        )


def load_uspto_tab(path: Path, *, max_rows: int | None = None) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        for idx, row in enumerate(csv.DictReader(fh, delimiter="\t")):
            if max_rows is not None and idx >= max_rows:
                break
            rxn = f"{row.get('reactant') or ''}>>{row.get('product') or ''}"
            parsed = _parse_reaction_parts(rxn)
            if not parsed:
                continue
            yield step_pair(
                source="uspto50k",
                idx=idx,
                rxn=parsed[0],
                ec="",
                reaction_type=f"uspto_class_{row.get('category') or ''}",
                evidence={"category": row.get("category") or ""},
                weight=1.5,
                parsed=parsed,
            )


def load_rhea_tar(path: Path, *, max_rows: int | None = None) -> Iterable[dict[str, Any]]:
    ec_by_rhea: dict[str, set[str]] = {}
    with tarfile.open(path) as tar:
        ec_member = tar.extractfile("140/tsv/rhea2ec.tsv")
        if ec_member:
            text = io.TextIOWrapper(ec_member, encoding="utf-8")
            for row in csv.DictReader(text, delimiter="\t"):
                rhea_id = row.get("RHEA_ID") or row.get("MASTER_ID") or ""
                ec = row.get("ID") or ""
                if rhea_id and ec:
                    ec_by_rhea.setdefault(rhea_id, set()).add(ec)
        member = tar.extractfile("140/tsv/rhea-reaction-smiles.tsv")
        if not member:
            return
        text = io.TextIOWrapper(member, encoding="utf-8")
        for idx, line in enumerate(text):
            if max_rows is not None and idx >= max_rows:
                break
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            rhea_id, rxn = parts[0], parts[1]
            parsed = _parse_reaction_parts(rxn)
            if not parsed:
                continue
            ecs = sorted(ec_by_rhea.get(rhea_id) or [])
            yield step_pair(
                source="rhea",
                idx=rhea_id,
                rxn=parsed[0],
                ec=ecs[0] if ecs else "",
                reaction_type="rhea_reaction",
                evidence={"rhea_id": rhea_id, "ecs": ecs},
                weight=1.8,
                parsed=parsed,
            )


def summarize_reactzyme_zip(path: Path) -> dict[str, Any]:
    summary: dict[str, Any] = {"zip": str(path)}
    with zipfile.ZipFile(path) as zf:
        summary["files"] = {info.filename: info.file_size for info in zf.infolist()}
        for name in ("reaction_smi_split.zip", "enzyme_smi_split.zip", "time_split.zip"):
            if name not in summary["files"]:
                continue
            with zf.open(name) as fh:
                blob = fh.read()
            with zipfile.ZipFile(io.BytesIO(blob)) as inner:
                summary[name] = {info.filename: info.file_size for info in inner.infolist()}
    return summary


def summarize_template_csv(path: Path) -> dict[str, Any]:
    opener = gzip.open if str(path).endswith(".gz") else open
    rows = valid = with_ec = 0
    datasets = Counter()
    with opener(path, "rt", encoding="utf-8", errors="replace", newline="") as fh:
        for row in csv.DictReader(fh):
            rows += 1
            valid += str(row.get("VALID") or "").lower() == "true"
            with_ec += bool(row.get("ECS"))
            datasets.update([row.get("DATASETS") or ""])
    return {"path": str(path), "rows": rows, "valid": valid, "with_ec": with_ec, "datasets": dict(datasets)}


def step_pair(
    *,
    source: str,
    idx: Any,
    rxn: str,
    ec: str,
    reaction_type: str,
    evidence: dict[str, Any],
    weight: float,
    parsed: tuple[str, list[str], list[str]] | None = None,
) -> dict[str, Any]:
    parsed = parsed or _parse_reaction_parts(rxn)
    if parsed is None:
        raise ValueError(f"invalid reaction SMILES: {rxn}")
    canonical_rxn, reactants, products = parsed
    product = max(products, key=_heavy_atoms, default="")
    main = max(reactants, key=_heavy_atoms, default="")
    aux = [smi for smi in reactants if smi != main]
    candidate = {
        "main_reactant": main,
        "aux_reactants": aux,
        "source": source,
        "score": 1.0,
        "type": reaction_type,
        "reaction_type": reaction_type,
        "rxn_smiles": canonical_rxn,
        "ec": ec,
        "evidence": evidence,
    }
    return {
        "step_id": stable_id("external", source, idx, canonical_rxn),
        "group_id": stable_id("external_group", source, product),
        "route_id": "",
        "target_smiles": product,
        "product": product,
        "reactants": reactants,
        "reaction_smiles": canonical_rxn,
        "reaction_type": reaction_type,
        "ec": ec,
        "source": source,
        "rank": 1,
        "label": 1.0,
        "label_type": "external_curated_step",
        "weight": weight,
        "gt_available": True,
        "exact_gt_reaction": True,
        "exact_gt_reactants": True,
        "selected_exact": True,
        "candidate": candidate,
    }


def dedupe_step_pairs(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        key = row.get("reaction_smiles") or row.get("step_id")
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def quality_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "sources": dict(Counter(row.get("source") for row in rows)),
        "with_ec": sum(1 for row in rows if row.get("ec")),
        "reaction_types": dict(Counter(row.get("reaction_type") for row in rows).most_common(20)),
    }


def report_markdown(manifest: dict[str, Any]) -> str:
    q = manifest.get("quality") or {}
    lines = [
        "# External Step Pair Import",
        "",
        f"- rows: `{manifest.get('counts', {}).get('external_step_pairs', 0)}`",
        f"- with EC: `{q.get('with_ec', 0)}`",
        f"- sources: `{q.get('sources', {})}`",
        "",
        "Template CSV files are summarized as candidate-generation assets, not direct step-pair labels.",
        "",
    ]
    return "\n".join(lines)


def _normalize_reaction_ec_arrow(rxn: str) -> str:
    if "|" in rxn and ">>" in rxn:
        lhs, rhs = rxn.split(">>", 1)
        lhs = lhs.split("|", 1)[0]
        return f"{lhs}>>{rhs}"
    return rxn


def _valid_reaction(rxn: str) -> bool:
    return _parse_reaction_parts(rxn) is not None


def _parse_reaction_parts(rxn: str) -> tuple[str, list[str], list[str]] | None:
    if not rxn or ">>" not in rxn:
        return None
    lhs, rhs = rxn.split(">>", 1)
    reactants = _canonical_parts(lhs)
    products = _canonical_parts(rhs)
    if not reactants or not products:
        return None
    return f"{'.'.join(reactants)}>>{'.'.join(products)}", reactants, products


def _canonical_parts(side: str) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for part in side.split("."):
        smi = part.strip()
        if not smi or "*" in smi:
            continue
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        can = Chem.MolToSmiles(mol)
        if can in seen:
            continue
        seen.add(can)
        out.append(can)
    return out


def _largest_fragment(side: str) -> str:
    return max(_canonical_parts(side), key=_heavy_atoms, default="")


def _heavy_atoms(smiles: str) -> int:
    mol = Chem.MolFromSmiles(smiles or "")
    return mol.GetNumHeavyAtoms() if mol is not None else 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Build external vNext step-pair rows from downloaded datasets")
    parser.add_argument("--output-dir", default="results/shared/external_step_pairs/current")
    parser.add_argument("--ecreact", default="data_external/ecreact/ecreact-1.0.csv")
    parser.add_argument("--enzymatic-json", action="append", default=["data_external/enzymatic_retro_data/train.json", "data_external/enzymatic_retro_data/val.json"])
    parser.add_argument("--uspto-tab", default="data/uspto50k.tab")
    parser.add_argument("--rhea-tar", default="data_external/rhea/140.tar.bz2")
    parser.add_argument("--reactzyme-zip", default="data_external/reactzyme/13635807.zip")
    parser.add_argument("--template", action="append", default=[
        "data_external/retrorules/templates_metanetx.csv.gz",
        "data_external/retrorules/templates_rhea.csv.gz",
        "data_external/retrorules/templates_uspto.csv.gz",
    ])
    parser.add_argument("--max-rows-per-source", type=int, default=None)
    args = parser.parse_args()
    manifest = build_external_step_pairs(
        output_dir=Path(args.output_dir),
        ecreact_csv=Path(args.ecreact) if args.ecreact else None,
        enzymatic_json=[Path(p) for p in args.enzymatic_json or []],
        uspto_tab=Path(args.uspto_tab) if args.uspto_tab else None,
        rhea_tar=Path(args.rhea_tar) if args.rhea_tar else None,
        reactzyme_zip=Path(args.reactzyme_zip) if args.reactzyme_zip else None,
        retrorules_templates=[Path(p) for p in args.template or []],
        max_rows_per_source=args.max_rows_per_source,
    )
    print(json.dumps({"counts": manifest["counts"], "quality": manifest["quality"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
