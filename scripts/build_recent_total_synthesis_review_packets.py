#!/usr/bin/env python3
"""Build compact, non-admitting review packets for synthesis experts.

The packets are navigation and authoring aids.  They link to the immutable local
source cache, expose all automated candidates as unverified, and write editable
submission templates.  They never update the human review ledger.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import html
import json
import os
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import quote


SUBMISSION_SCHEMA = "recent_total_synthesis_review_submission.v1"
PACKET_SCHEMA = "recent_total_synthesis_review_packets.v1"
STRUCTURED_ROUTE_SCHEMA = "recent_total_synthesis_structured_route_candidate.v1"
READY_VISUAL_STATUSES = {
    "exact_source_structure_candidate",
    "partial_stereo_candidate",
}


STYLE = """
:root { color-scheme: light; font-family: Inter, Arial, sans-serif; }
body { max-width: 1120px; margin: 24px auto; padding: 0 20px 56px; color: #17202a; }
h1 { font-size: 1.65rem; margin-bottom: .35rem; }
h2 { margin-top: 1.6rem; border-bottom: 1px solid #dce3ea; padding-bottom: .3rem; }
h3 { margin: 1rem 0 .4rem; }
.meta, .notice { background: #f4f7fa; border-left: 4px solid #557799; padding: 10px 14px; }
.warning { background: #fff4df; border-left-color: #c47a00; }
.grid { display: grid; grid-template-columns: repeat(auto-fit,minmax(310px,1fr)); gap: 14px; }
.card { border: 1px solid #dce3ea; border-radius: 7px; padding: 12px 14px; }
img, object { max-width: 100%; max-height: 640px; }
code { overflow-wrap: anywhere; }
pre { white-space: pre-wrap; background: #f7f7f7; padding: 10px; border-radius: 5px; }
table { border-collapse: collapse; width: 100%; }
th, td { text-align: left; vertical-align: top; border-bottom: 1px solid #e3e7eb; padding: 7px; }
details { margin: 8px 0; }
.tag { display: inline-block; background: #e8eef4; border-radius: 12px; padding: 2px 8px; margin-right: 4px; }
.route-step { border-left: 4px solid #557799; margin: 10px 0; padding: 8px 12px; background: #f8fafc; }
.compound-grid { display: grid; grid-template-columns: repeat(auto-fit,minmax(220px,1fr)); gap: 10px; }
.compound-card { border: 1px solid #dce3ea; border-radius: 6px; padding: 8px; }
.compound-card object { height: 190px; width: 100%; }
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir",
        default="benchmarks/recent_total_synthesis",
        help="Repository-relative benchmark directory.",
    )
    parser.add_argument(
        "--output-dir",
        default="output/recent_total_synthesis_review_packets",
        help="Repository-relative output directory.",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verified_artifact(
    artifact: dict[str, Any],
    *,
    repo_root: Path,
) -> Path:
    raw_path = str(artifact.get("cache_path") or artifact.get("source_artifact_path") or "")
    expected = str(artifact.get("sha256") or artifact.get("source_artifact_sha256") or "")
    relative = Path(raw_path)
    if not raw_path or relative.is_absolute() or not expected:
        raise RuntimeError("review_packet_source_binding_incomplete")
    resolved = (repo_root / relative).resolve()
    try:
        resolved.relative_to(repo_root.resolve())
    except ValueError as exc:
        raise RuntimeError("review_packet_source_outside_repository") from exc
    if not resolved.is_file():
        raise RuntimeError(f"review_packet_source_missing:{relative.as_posix()}")
    if sha256(resolved).lower() != expected.lower():
        raise RuntimeError(f"review_packet_source_hash_mismatch:{relative.as_posix()}")
    return resolved


def href(target: Path, *, page_dir: Path) -> str:
    relative = Path(os.path.relpath(target, start=page_dir)).as_posix()
    return quote(relative, safe="/._-~")


def locator_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def artifact_label(kind: str) -> str:
    labels = {
        "authorized_publisher_fulltext_html": "正文 HTML",
        "authorized_publisher_fulltext_xml": "正文 XML",
        "authorized_publisher_structured_text": "结构化正文",
        "authorized_publisher_main_pdf": "正文 PDF",
        "repository_fulltext_xml": "仓储正文 XML",
        "repository_main_pdf": "仓储正文 PDF",
        "supporting_information": "Supporting Information",
    }
    return labels.get(kind, kind)


def render_candidate_svg(
    smiles: str,
    output_path: Path,
    legend: str,
    *,
    width: int = 620,
    height: int = 390,
) -> None:
    from rdkit import Chem
    from rdkit.Chem.Draw import rdMolDraw2D

    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise RuntimeError("review_packet_candidate_smiles_invalid")
    drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
    rdMolDraw2D.PrepareAndDrawMolecule(drawer, molecule, legend=legend)
    drawer.FinishDrawing()
    output_path.write_text(drawer.GetDrawingText(), encoding="utf-8")


def validate_structured_route_candidate(
    value: dict[str, Any],
    *,
    candidate_path: Path,
    repo_root: Path,
    targets: dict[str, dict[str, Any]],
    visual_by_target: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Validate one model/visual route transcription without granting admission."""

    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors

    if value.get("schema_version") != STRUCTURED_ROUTE_SCHEMA:
        raise RuntimeError(f"structured_route_schema_invalid:{candidate_path.name}")
    if value.get("admission_authority") is not False:
        raise RuntimeError(f"structured_route_must_be_nonadmitting:{candidate_path.name}")
    target_id = str(value.get("target_slot_id") or "")
    target = targets.get(target_id)
    if target is None:
        raise RuntimeError(f"structured_route_target_unknown:{target_id}")
    if str(value.get("paper_id") or "") != str(target.get("paper_id") or ""):
        raise RuntimeError(f"structured_route_paper_mismatch:{target_id}")
    if str(value.get("source_doi") or "").lower() != str(target.get("doi") or "").lower():
        raise RuntimeError(f"structured_route_doi_mismatch:{target_id}")
    if value.get("reference_scope") != "ordered_route":
        raise RuntimeError(f"structured_route_scope_invalid:{target_id}")

    sources = list(value.get("source_artifacts") or [])
    if not sources:
        raise RuntimeError(f"structured_route_sources_missing:{target_id}")
    for source in sources:
        verified_artifact(dict(source), repo_root=repo_root)

    compounds = [dict(row) for row in value.get("compounds") or []]
    if not compounds:
        raise RuntimeError(f"structured_route_compounds_missing:{target_id}")
    compound_by_id: dict[str, dict[str, Any]] = {}
    for compound in compounds:
        compound_id = str(compound.get("compound_id") or "")
        label = str(compound.get("label") or "")
        if not compound_id or not label or compound_id in compound_by_id:
            raise RuntimeError(f"structured_route_compound_identity_invalid:{target_id}")
        structures = []
        if compound.get("smiles"):
            structures.append((label, str(compound["smiles"]), compound.get("molecular_formula")))
        for variant in compound.get("structure_variants") or []:
            structures.append(
                (
                    str(variant.get("label") or ""),
                    str(variant.get("smiles") or ""),
                    variant.get("molecular_formula") or compound.get("molecular_formula"),
                )
            )
        if not structures:
            raise RuntimeError(f"structured_route_compound_structure_missing:{compound_id}")
        for structure_label, smiles, expected_formula in structures:
            molecule = Chem.MolFromSmiles(smiles)
            if molecule is None or not structure_label:
                raise RuntimeError(f"structured_route_smiles_invalid:{compound_id}")
            formula = rdMolDescriptors.CalcMolFormula(molecule)
            if expected_formula and formula != str(expected_formula):
                raise RuntimeError(
                    f"structured_route_formula_mismatch:{compound_id}:{formula}"
                )
        compound_by_id[compound_id] = compound

    steps = [dict(row) for row in value.get("steps") or []]
    if not steps or [row.get("order") for row in steps] != list(range(1, len(steps) + 1)):
        raise RuntimeError(f"structured_route_step_order_invalid:{target_id}")
    step_ids: set[str] = set()
    produced: set[str] = set()
    initially_available = {
        compound_id
        for compound_id, compound in compound_by_id.items()
        if compound.get("role") in {"starting_material", "reagent"}
    }
    for step in steps:
        step_id = str(step.get("step_id") or "")
        precursor_ids = [str(item) for item in step.get("precursor_compound_ids") or []]
        product_id = str(step.get("product_compound_id") or "")
        if (
            not step_id
            or step_id in step_ids
            or not precursor_ids
            or product_id not in compound_by_id
            or not str(step.get("transformation_class") or "")
            or not str(step.get("strategic_role") or "")
            or not step.get("source_locator")
        ):
            raise RuntimeError(f"structured_route_step_invalid:{step_id or target_id}")
        for precursor_id in precursor_ids:
            if precursor_id not in compound_by_id:
                raise RuntimeError(f"structured_route_precursor_unknown:{precursor_id}")
            if precursor_id not in initially_available and precursor_id not in produced:
                raise RuntimeError(f"structured_route_precursor_not_yet_produced:{precursor_id}")
        if product_id in produced:
            raise RuntimeError(f"structured_route_product_repeated:{product_id}")
        expected_labels = [compound_by_id[item]["label"] for item in precursor_ids]
        if list(step.get("precursor_labels") or []) != expected_labels:
            raise RuntimeError(f"structured_route_precursor_labels_mismatch:{step_id}")
        if str(step.get("product_label") or "") != str(compound_by_id[product_id]["label"]):
            raise RuntimeError(f"structured_route_product_label_mismatch:{step_id}")
        step_ids.add(step_id)
        produced.add(product_id)

    target_compound_id = str(value.get("target_compound_id") or "")
    if target_compound_id != str(steps[-1].get("product_compound_id") or ""):
        raise RuntimeError(f"structured_route_final_product_mismatch:{target_id}")
    target_compound = compound_by_id.get(target_compound_id, {})
    target_smiles = str(target_compound.get("smiles") or "")
    visual_smiles = str(
        visual_by_target.get(target_id, {}).get("visual_canonical_isomeric_smiles") or ""
    )
    target_molecule = Chem.MolFromSmiles(target_smiles)
    visual_molecule = Chem.MolFromSmiles(visual_smiles)
    if (
        target_molecule is None
        or visual_molecule is None
        or Chem.MolToSmiles(target_molecule, isomericSmiles=True)
        != Chem.MolToSmiles(visual_molecule, isomericSmiles=True)
    ):
        raise RuntimeError(f"structured_route_target_structure_mismatch:{target_id}")
    if not list(value.get("strategic_events") or []):
        raise RuntimeError(f"structured_route_strategic_events_missing:{target_id}")

    normalized = dict(value)
    normalized["candidate_path"] = candidate_path.relative_to(repo_root).as_posix()
    return normalized


def load_structured_route_candidates(
    *,
    dataset_dir: Path,
    repo_root: Path,
    targets: dict[str, dict[str, Any]],
    visual_by_target: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    root = dataset_dir / "curation_candidates" / "structured_routes"
    result: dict[str, dict[str, Any]] = {}
    if not root.exists():
        return result
    for candidate_path in sorted(root.glob("*.json")):
        candidate = validate_structured_route_candidate(
            json.loads(candidate_path.read_text(encoding="utf-8")),
            candidate_path=candidate_path,
            repo_root=repo_root,
            targets=targets,
            visual_by_target=visual_by_target,
        )
        target_id = str(candidate["target_slot_id"])
        if target_id in result:
            raise RuntimeError(f"structured_route_candidate_duplicate:{target_id}")
        result[target_id] = candidate
    return result


def source_audit(
    *,
    repo_root: Path,
    primary_targets: list[dict[str, Any]],
    p1_targets: list[dict[str, Any]],
    receipts: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    cohort_ids = {
        str(row["paper_id"])
        for row in primary_targets
        if row.get("slot_class") == "primary"
    } | {
        str(row["paper_id"])
        for row in p1_targets
        if row.get("slot_class") == "primary_candidate"
    }
    completeness = Counter()
    artifact_kinds = Counter()
    acquired = 0
    checked = 0
    checked_bytes = 0
    missing_packages: list[dict[str, Any]] = []
    for paper_id in sorted(cohort_ids):
        receipt = receipts.get(paper_id, {})
        is_acquired = bool(receipt.get("source_package_acquired"))
        acquired += int(is_acquired)
        completeness[str(receipt.get("source_package_completeness") or "none")] += 1
        if not is_acquired:
            missing_packages.append(
                {
                    "paper_id": paper_id,
                    "doi": receipt.get("doi", ""),
                    "status": receipt.get("status", "missing_receipt"),
                    "errors": list(receipt.get("errors") or []),
                }
            )
        for artifact in receipt.get("artifacts") or []:
            path = verified_artifact(dict(artifact), repo_root=repo_root)
            checked += 1
            checked_bytes += path.stat().st_size
            artifact_kinds[str(artifact.get("artifact_kind") or "unknown")] += 1
    return {
        "candidate_papers": len(cohort_ids),
        "source_packages_acquired": acquired,
        "source_packages_missing": len(cohort_ids) - acquired,
        "package_completeness": dict(sorted(completeness.items())),
        "artifact_kind_counts": dict(sorted(artifact_kinds.items())),
        "verified_artifact_files": checked,
        "verified_artifact_bytes": checked_bytes,
        "missing_packages": missing_packages,
        "file_integrity": "all_receipt_artifacts_present_and_sha256_matched",
    }


def source_rows(
    receipt: dict[str, Any],
    *,
    repo_root: Path,
    page_dir: Path,
) -> str:
    rows = []
    for artifact in receipt.get("artifacts") or []:
        path = verified_artifact(dict(artifact), repo_root=repo_root)
        kind = str(artifact.get("artifact_kind") or "unknown")
        rows.append(
            "<tr>"
            f"<td>{html.escape(artifact_label(kind))}</td>"
            f"<td><a href=\"{href(path, page_dir=page_dir)}\">打开本地文件</a></td>"
            f"<td><code>{html.escape(str(artifact.get('sha256') or ''))}</code></td>"
            "</tr>"
        )
    return "\n".join(rows) or '<tr><td colspan="3">无可用本地附件</td></tr>'


def page_shell(title: str, body: str) -> str:
    return (
        "<!doctype html><html lang=\"zh-CN\"><head><meta charset=\"utf-8\">"
        f"<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\"><title>{html.escape(title)}</title>"
        f"<style>{STYLE}</style></head><body>{body}</body></html>\n"
    )


def source_binding_from_passage(passage: dict[str, Any]) -> dict[str, Any]:
    return {
        "source_artifact_path": str(passage.get("source_artifact_path") or ""),
        "source_artifact_sha256": str(passage.get("source_artifact_sha256") or ""),
        "source_locator": passage.get("source_locator") or "",
    }


def unique_bindings(passages: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for passage in passages:
        binding = source_binding_from_passage(passage)
        # One source declaration per immutable file is enough. Step/event locators
        # remain on the chemistry records, so repeating a HTML file for every
        # paragraph only makes the expert form longer without adding provenance.
        key = "|".join(
            [binding["source_artifact_path"], binding["source_artifact_sha256"]]
        )
        if key not in seen and binding["source_artifact_path"]:
            seen.add(key)
            result.append(binding)
    return result


def structured_route_record(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "source_doi": candidate["source_doi"],
        "reference_scope": "ordered_route",
        "source_artifacts": list(candidate["source_artifacts"]),
        "steps": list(candidate["steps"]),
        "strategic_events": list(candidate["strategic_events"]),
    }


def structured_route_html(
    candidate: dict[str, Any] | None,
    *,
    packet_dir: Path,
    repo_root: Path,
) -> str:
    if not candidate:
        return (
            '<div class="notice">尚未形成结构化有序路线；下方段落仍只是定位线索。</div>'
        )
    compound_dir = packet_dir / "route-compounds"
    compound_dir.mkdir(parents=True, exist_ok=True)
    compound_cards = []
    for compound in candidate.get("compounds") or []:
        compound_id = str(compound["compound_id"])
        variants = list(compound.get("structure_variants") or [])
        structures = variants or [compound]
        drawings = []
        for index, structure in enumerate(structures, start=1):
            suffix = f"-{index}" if len(structures) > 1 else ""
            svg_path = compound_dir / f"{compound_id}{suffix}.svg"
            legend = str(structure.get("label") or compound.get("label") or compound_id)
            render_candidate_svg(
                str(structure["smiles"]),
                svg_path,
                legend,
                width=360,
                height=220,
            )
            drawings.append(
                f'<object data="{href(svg_path, page_dir=packet_dir)}" '
                'type="image/svg+xml"></object>'
                f'<code>{html.escape(str(structure["smiles"]))}</code>'
            )
        compound_cards.append(
            '<section class="compound-card">'
            f'<strong>{html.escape(str(compound["label"]))}</strong>'
            f'<span class="tag">{html.escape(str(compound.get("role") or ""))}</span>'
            f'{"".join(drawings)}</section>'
        )

    step_cards = []
    for step in candidate.get("steps") or []:
        precursors = " + ".join(str(item) for item in step.get("precursor_labels") or [])
        conditions = "; ".join(
            f"{key}: {value}"
            for key, value in (step.get("conditions") or {}).items()
            if value not in (None, "", [], {})
        )
        yield_text = (
            f"{step['yield_percent']}%"
            if step.get("yield_percent") is not None
            else "未报告"
        )
        step_cards.append(
            '<section class="route-step">'
            f'<strong>Step {int(step["order"])} · {html.escape(precursors)} → '
            f'{html.escape(str(step["product_label"]))}</strong><br>'
            f'{html.escape(str(step["transformation_class"]))} · 收率 {html.escape(yield_text)}<br>'
            f'<span>{html.escape(conditions)}</span><br>'
            f'<em>{html.escape(str(step["strategic_role"]))}</em><br>'
            f'<small>{html.escape(locator_text(step["source_locator"]))}</small>'
            '</section>'
        )
    candidate_path = repo_root / str(candidate["candidate_path"])
    summary = dict(candidate.get("route_summary") or {})
    return f"""
<div class="notice warning"><strong>结构化路线候选，仍不具 admission 权限。</strong>
已通过来源哈希、RDKit、目标一致性、步骤顺序和前体连续性检查；专家仍须逐图逐步确认。</div>
<p><span class="tag">{len(candidate.get('steps') or [])} steps</span>
<span class="tag">{html.escape(str(summary.get('route_variant') or 'ordered route'))}</span>
<a href="{href(candidate_path, page_dir=packet_dir)}">打开结构化候选 JSON</a></p>
{''.join(step_cards)}
<h3>化合物结构候选</h3><div class="compound-grid">{''.join(compound_cards)}</div>
"""


def paper_submission_template(
    *,
    paper: dict[str, Any],
    target_ids: list[str],
    receipt: dict[str, Any],
) -> dict[str, Any]:
    artifacts = list(receipt.get("artifacts") or [])
    evidence = []
    if artifacts:
        artifact = artifacts[0]
        evidence.append(
            {
                "source_artifact_path": artifact["cache_path"],
                "source_artifact_sha256": artifact["sha256"],
                "source_locator": "TO_FILL: exact page, scheme, section, or paragraph",
            }
        )
    return {
        "schema_version": SUBMISSION_SCHEMA,
        "packet_type": "paper_scope",
        "packet_id": f"paper-scope--{paper['paper_id']}--v1",
        "paper_id": paper["paper_id"],
        "doi": paper["doi"],
        "reviewer": {
            "reviewer_id": "",
            "reviewed_at": "",
            "attestation": False,
        },
        "decision": "not_reviewed",
        "target_slot_ids": target_ids,
        "evidence_locators": evidence,
        "reviewer_notes": "",
    }


def target_submission_template(
    *,
    target: dict[str, Any],
    visual: dict[str, Any],
    route: dict[str, Any],
    structured_route: dict[str, Any] | None,
) -> dict[str, Any]:
    passages = list(route.get("evidence_passages") or [])
    source_image = dict(visual.get("source_image") or {})
    event_locator = passages[0].get("source_locator") if passages else ""
    return {
        "schema_version": SUBMISSION_SCHEMA,
        "packet_type": "target_truth",
        "packet_id": f"target-truth--{target['target_slot_id']}--v1",
        "target_slot_id": target["target_slot_id"],
        "paper_id": target["paper_id"],
        "doi": target["doi"],
        "reviewer": {
            "reviewer_id": "",
            "reviewed_at": "",
            "attestation": False,
        },
        "structure_review": {
            "decision": "not_reviewed",
            "record": {
                "isomeric_smiles": visual.get("visual_canonical_isomeric_smiles", ""),
                "source_doi": target["doi"],
                "source_artifact_path": source_image.get("source_artifact_path", ""),
                "source_artifact_sha256": source_image.get("source_artifact_sha256", ""),
                "source_locator": visual.get("source_locator", ""),
                "identity_confirmed": False,
                "relative_stereochemistry_confirmed": False,
                "absolute_stereochemistry_status": "not_reviewed",
            },
            "reviewer_notes": "",
        },
        "route_review": {
            "decision": "not_reviewed",
            "record": structured_route_record(structured_route)
            if structured_route
            else {
                    "source_doi": target["doi"],
                    "reference_scope": "strategic_key_step",
                    "source_artifacts": unique_bindings(passages),
                    "steps": [],
                    "strategic_events": [
                        {
                            "event_id": "event-1",
                            "description": "",
                            "transformation_class": "",
                            "source_locator": event_locator,
                        }
                    ],
                },
            "reviewer_notes": "",
        },
    }


def build_paper_packet(
    *,
    output_root: Path,
    repo_root: Path,
    paper: dict[str, Any],
    targets: list[dict[str, Any]],
    receipt: dict[str, Any],
    visual_by_target: dict[str, dict[str, Any]],
    route_by_target: dict[str, dict[str, Any]],
    structured_route_by_target: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    packet_id = f"paper-scope--{paper['paper_id']}--v1"
    packet_dir = output_root / "paper_scope" / str(paper["paper_id"])
    packet_dir.mkdir(parents=True, exist_ok=True)
    target_ids = sorted(str(row["target_slot_id"]) for row in targets)
    submission_path = packet_dir / "submission.json"
    write_json(
        submission_path,
        paper_submission_template(
            paper=paper,
            target_ids=target_ids,
            receipt=receipt,
        ),
    )
    target_rows = []
    for row in sorted(targets, key=lambda item: str(item["target_name"])):
        target_id = str(row["target_slot_id"])
        visual = visual_by_target.get(target_id, {})
        route = route_by_target.get(target_id, {})
        visual_status = str(visual.get("visual_status") or "unresolved")
        smiles = str(visual.get("visual_canonical_isomeric_smiles") or "")
        route_passages = len(route.get("evidence_passages") or [])
        structured_status = "已形成有序候选" if target_id in structured_route_by_target else "—"
        is_ready = (
            visual_status in READY_VISUAL_STATUSES
            and (visual.get("rdkit_validation") or {}).get("status") == "roundtrip_valid"
            and route_passages > 0
        )
        if is_ready:
            batch = (
                "A_exact_source"
                if visual_status == "exact_source_structure_candidate"
                else "B_stereo_resolution"
            )
            target_page = output_root / "target_truth" / batch / target_id / "index.html"
            review_link = f'<a href="{href(target_page, page_dir=packet_dir)}">打开目标复核页</a>'
        else:
            review_link = "尚未形成可提交目标包"
        status_label = {
            "exact_source_structure_candidate": "候选较完整",
            "partial_stereo_candidate": "需裁决立体化学",
            "unresolved": "结构未解析",
        }.get(visual_status, visual_status)
        target_rows.append(
            "<tr>"
            f"<td><strong>{html.escape(str(row['target_name']))}</strong><br><code>{html.escape(target_id)}</code></td>"
            f"<td>{html.escape(status_label)}</td>"
            f"<td><code>{html.escape(smiles) if smiles else '—'}</code></td>"
            f"<td>{route_passages}</td><td>{structured_status}</td><td>{review_link}</td>"
            "</tr>"
        )
    body = f"""
<p><a href="../../index.html">← 返回总目录</a></p>
<h1>论文范围复核</h1>
<div class="notice warning"><strong>非接纳材料。</strong>自动筛选和候选 target 只用于定位；请以正文/SI 为准。</div>
<h2>{html.escape(str(paper['title']))}</h2>
<div class="meta"><div><strong>期刊：</strong>{html.escape(str(paper.get('journal') or ''))}</div>
<div><strong>日期：</strong>{html.escape(str(paper.get('publication_date') or ''))}</div>
<div><strong>DOI：</strong><a href="{html.escape(str(paper.get('source_url') or ''))}">{html.escape(str(paper['doi']))}</a></div>
<div><strong>paper_id：</strong><code>{html.escape(str(paper['paper_id']))}</code></div></div>
<h2>需要确认</h2>
<ol><li>是否报告了本数据集范围内的完成合成；</li><li>应归为 primary、conditional、control 或 exclude；</li>
<li>下列最终目标是否完整、是否误含中间体或类似物；</li><li>在 submission.json 中给出精确页码/图式/段落。</li></ol>
<h2>目标、SMILES 与后续复核入口</h2>
<p>本表仅帮助浏览。SMILES 和状态均是自动候选，不参与本页的论文范围结论。</p>
<table><thead><tr><th>目标</th><th>结构状态</th><th>候选 isomeric SMILES</th><th>路线段落</th><th>结构化路线</th><th>目标级页面</th></tr></thead>
<tbody>{''.join(target_rows)}</tbody></table>
<h2>来源附件</h2><table><thead><tr><th>类型</th><th>文件</th><th>SHA-256</th></tr></thead><tbody>
{source_rows(receipt, repo_root=repo_root, page_dir=packet_dir)}</tbody></table>
<h2>提交</h2><p>只编辑 <a href="submission.json"><code>submission.json</code></a>。完成后在仓库根目录运行：</p>
<pre>python scripts/validate_recent_total_synthesis_review_submission.py --submission "{submission_path.relative_to(repo_root).as_posix()}"</pre>
"""
    (packet_dir / "index.html").write_text(
        page_shell(f"论文复核：{paper['title']}", body), encoding="utf-8"
    )
    return {
        "packet_id": packet_id,
        "packet_type": "paper_scope",
        "paper_id": paper["paper_id"],
        "doi": paper["doi"],
        "index_path": (packet_dir / "index.html").relative_to(repo_root).as_posix(),
        "submission_path": submission_path.relative_to(repo_root).as_posix(),
        "admission_authority": False,
    }


def build_target_packet(
    *,
    output_root: Path,
    repo_root: Path,
    paper: dict[str, Any],
    target: dict[str, Any],
    visual: dict[str, Any],
    route: dict[str, Any],
    structured_route: dict[str, Any] | None,
    receipt: dict[str, Any],
) -> dict[str, Any]:
    status = str(visual["visual_status"])
    batch = "A_exact_source" if status == "exact_source_structure_candidate" else "B_stereo_resolution"
    packet_id = f"target-truth--{target['target_slot_id']}--v1"
    packet_dir = output_root / "target_truth" / batch / str(target["target_slot_id"])
    packet_dir.mkdir(parents=True, exist_ok=True)
    smiles = str(visual.get("visual_canonical_isomeric_smiles") or "")
    render_candidate_svg(smiles, packet_dir / "candidate.svg", str(target["target_name"]))
    submission_path = packet_dir / "submission.json"
    write_json(
        submission_path,
        target_submission_template(
            target=target,
            visual=visual,
            route=route,
            structured_route=structured_route,
        ),
    )
    source_image = dict(visual.get("source_image") or {})
    image_path = verified_artifact(
        {
            "source_artifact_path": source_image.get("image_path"),
            "source_artifact_sha256": source_image.get("image_sha256"),
        },
        repo_root=repo_root,
    )
    passage_html = []
    for index, passage in enumerate(route.get("evidence_passages") or [], start=1):
        passage_source = verified_artifact(dict(passage), repo_root=repo_root)
        passage_html.append(
            f"<details><summary>路线线索 {index} · {html.escape(locator_text(passage.get('source_locator')))}</summary>"
            f"<p><a href=\"{href(passage_source, page_dir=packet_dir)}\">打开证据文件</a></p>"
            f"<blockquote>{html.escape(str(passage.get('verbatim_text') or ''))}</blockquote></details>"
        )
    status_cn = "结构候选较完整" if batch.startswith("A_") else "必须裁决未解析的立体化学"
    body = f"""
<p><a href="../../../index.html">← 返回总目录</a></p>
<h1>{html.escape(str(target['target_name']))}</h1>
<div class="notice warning"><strong>非接纳候选。</strong>模型转录和 RDKit round-trip 均不能替代专家核对。</div>
<p><span class="tag">{html.escape(status_cn)}</span><span class="tag">{html.escape(str(target['target_slot_id']))}</span></p>
<div class="grid"><section class="card"><h2>论文原图</h2>
<p>{html.escape(str(visual.get('source_locator') or ''))}</p>
<a href="{href(image_path, page_dir=packet_dir)}"><img src="{href(image_path, page_dir=packet_dir)}" alt="论文来源页"></a></section>
<section class="card"><h2>待核对结构候选</h2><object data="candidate.svg" type="image/svg+xml"></object>
<p><code>{html.escape(smiles)}</code></p><p>{html.escape(str(visual.get('transcription_note') or ''))}</p></section></div>
<h2>论文信息</h2><div class="meta"><strong>{html.escape(str(paper['title']))}</strong><br>
{html.escape(str(paper.get('journal') or ''))} · {html.escape(str(paper.get('publication_date') or ''))} ·
<a href="{html.escape(str(paper.get('source_url') or ''))}">{html.escape(str(paper['doi']))}</a></div>
<h2>结构化有序路线</h2>
{structured_route_html(structured_route, packet_dir=packet_dir, repo_root=repo_root)}
<h2>自动抽取的路线线索</h2><p>这些段落只帮助定位。路线步骤、化合物身份和战略事件必须回到图式/SI 核验。</p>
{''.join(passage_html)}
<h2>全部来源附件</h2><table><thead><tr><th>类型</th><th>文件</th><th>SHA-256</th></tr></thead><tbody>
{source_rows(receipt, repo_root=repo_root, page_dir=packet_dir)}</tbody></table>
<h2>提交</h2><ol><li>只编辑 <a href="submission.json"><code>submission.json</code></a>；不改候选文件和人工总账。</li>
<li>结构和路线可分开提交；未审部分保持 <code>not_reviewed</code>。</li>
<li><code>accept</code> 时必须给出来源一致的结构/立体化学，或可复核的路线/关键步骤。</li></ol>
<pre>python scripts/validate_recent_total_synthesis_review_submission.py --submission "{submission_path.relative_to(repo_root).as_posix()}"</pre>
"""
    (packet_dir / "index.html").write_text(
        page_shell(f"目标复核：{target['target_name']}", body), encoding="utf-8"
    )
    return {
        "packet_id": packet_id,
        "packet_type": "target_truth",
        "review_batch": batch,
        "paper_id": paper["paper_id"],
        "target_slot_id": target["target_slot_id"],
        "target_name": target["target_name"],
        "visual_status": status,
        "structured_route_candidate": bool(structured_route),
        "index_path": (packet_dir / "index.html").relative_to(repo_root).as_posix(),
        "submission_path": submission_path.relative_to(repo_root).as_posix(),
        "admission_authority": False,
    }


def build_packets(*, repo_root: Path, dataset_dir: Path, output_root: Path) -> dict[str, Any]:
    papers = {str(row["paper_id"]): row for row in read_jsonl(dataset_dir / "papers.jsonl")}
    targets = read_jsonl(dataset_dir / "target_slots.jsonl")
    target_by_id = {str(row["target_slot_id"]): row for row in targets}
    p1_targets = read_jsonl(
        dataset_dir / "curation_candidates" / "p1_scope" / "candidate-target-slots.jsonl"
    )
    visual_by_target = {
        str(row["target_slot_id"]): row
        for row in read_jsonl(dataset_dir / "visual_structure_candidates.jsonl")
    }
    route_by_target = {
        str(row["target_slot_id"]): row
        for row in read_jsonl(dataset_dir / "route_evidence_candidates.jsonl")
    }
    structured_route_by_target = load_structured_route_candidates(
        dataset_dir=dataset_dir,
        repo_root=repo_root,
        targets=target_by_id,
        visual_by_target=visual_by_target,
    )
    receipt_rows = read_jsonl(dataset_dir / "source_package_receipts.jsonl") + read_jsonl(
        dataset_dir / "p1_source_package_receipts.jsonl"
    )
    receipts = {str(row["paper_id"]): row for row in receipt_rows}
    audit = source_audit(
        repo_root=repo_root,
        primary_targets=targets,
        p1_targets=p1_targets,
        receipts=receipts,
    )
    ready_targets = []
    for target in targets:
        if target.get("slot_class") != "primary":
            continue
        target_id = str(target["target_slot_id"])
        visual = visual_by_target.get(target_id, {})
        route = route_by_target.get(target_id, {})
        if (
            visual.get("visual_status") in READY_VISUAL_STATUSES
            and (visual.get("rdkit_validation") or {}).get("status") == "roundtrip_valid"
            and bool(route.get("evidence_passages"))
        ):
            ready_targets.append(target)
    ready_targets.sort(
        key=lambda row: (
            visual_by_target[str(row["target_slot_id"])]["visual_status"]
            != "exact_source_structure_candidate",
            str(row["target_name"]),
        )
    )
    packet_paper_ids = {str(row["paper_id"]) for row in ready_targets}
    targets_by_paper: dict[str, list[dict[str, Any]]] = {}
    for target in targets:
        if target.get("slot_class") == "primary":
            targets_by_paper.setdefault(str(target["paper_id"]), []).append(target)

    packets: list[dict[str, Any]] = []
    for paper_id in sorted(packet_paper_ids):
        packets.append(
            build_paper_packet(
                output_root=output_root,
                repo_root=repo_root,
                paper=papers[paper_id],
                targets=targets_by_paper[paper_id],
                receipt=receipts[paper_id],
                visual_by_target=visual_by_target,
                route_by_target=route_by_target,
                structured_route_by_target=structured_route_by_target,
            )
        )
    for target in ready_targets:
        target_id = str(target["target_slot_id"])
        paper_id = str(target["paper_id"])
        packets.append(
            build_target_packet(
                output_root=output_root,
                repo_root=repo_root,
                paper=papers[paper_id],
                target=target,
                visual=visual_by_target[target_id],
                route=route_by_target[target_id],
                structured_route=structured_route_by_target.get(target_id),
                receipt=receipts[paper_id],
            )
        )

    counts = Counter(str(row.get("review_batch") or row["packet_type"]) for row in packets)
    manifest = {
        "schema_version": PACKET_SCHEMA,
        "admission_authority": False,
        "claim_boundary": (
            "Packets are navigation and authoring aids. Only validated submissions merged "
            "into curation_inputs/review_decisions.json participate in dual-review admission."
        ),
        "source_audit": audit,
        "packet_counts": dict(sorted(counts.items())),
        "structured_route_candidates": len(structured_route_by_target),
        "packets": packets,
    }
    write_json(output_root / "packet-manifest.json", manifest)
    paper_links = "".join(
        f"<li><a href=\"{href(repo_root / row['index_path'], page_dir=output_root)}\">{html.escape(str(row['doi']))}</a></li>"
        for row in packets
        if row["packet_type"] == "paper_scope"
    )
    target_links = "".join(
        f"<li><a href=\"{href(repo_root / row['index_path'], page_dir=output_root)}\">{html.escape(str(row['target_name']))}</a>"
        f" · {html.escape(str(row['review_batch']))}</li>"
        for row in packets
        if row["packet_type"] == "target_truth"
    )
    body = f"""
<h1>近期全合成数据集 · 专家复核包</h1>
<div class="notice warning"><strong>所有内容均为候选，不具 admission 权限。</strong>专家只编辑并返回各包的 submission.json。</div>
<h2>工作顺序</h2><ol><li>先完成同一论文的范围和 target 枚举；</li><li>再核对目标结构、立体化学与路线证据；</li>
<li>本地校验通过后将 JSON 交给数据管理员；</li><li>管理员用 <code>--merge</code> 原子合并，第二位独立专家重复复核。</li></ol>
<p>详细标准见 <a href="{href(dataset_dir / 'REVIEW_PROTOCOL.md', page_dir=output_root)}">REVIEW_PROTOCOL.md</a>。</p>
<h2>来源状态</h2><p>{audit['source_packages_acquired']}/{audit['candidate_papers']} 篇有本地来源包；
{audit['verified_artifact_files']} 个 receipt 附件已逐文件通过 SHA-256。缺失 {audit['source_packages_missing']} 篇。</p>
<p>结构化有序路线候选：{len(structured_route_by_target)}。这些候选仍需两位专家独立确认。</p>
<h2>论文范围复核（{counts['paper_scope']}）</h2><ul>{paper_links}</ul>
<h2>目标真值复核（{counts['A_exact_source'] + counts['B_stereo_resolution']}）</h2><ul>{target_links}</ul>
"""
    (output_root / "index.html").write_text(
        page_shell("近期全合成专家复核包", body), encoding="utf-8"
    )
    return manifest


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    dataset_dir = (repo_root / args.dataset_dir).resolve()
    output_root = (repo_root / args.output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = build_packets(
        repo_root=repo_root,
        dataset_dir=dataset_dir,
        output_root=output_root,
    )
    print(json.dumps({"output": str(output_root), **manifest["packet_counts"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
