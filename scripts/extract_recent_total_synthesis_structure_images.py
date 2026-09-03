#!/usr/bin/env python3
"""Create source-bound target-structure candidates from exact-paper images.

One visual call covers one paper and all of its primary targets. Results remain
non-admitting review leads: an exact source locator, RDKit round trip, and later
independent stereochemical review are still required before planner exposure.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cascade_planner.harness.visual_literature_chain_agent import (
    _parse_json_object,
    _run_visual_json_prompt,
)


PROMPT_VERSION = "recent_total_synthesis_structure_visual.v3"
PRIMARY_TARGET_SLOT_CLASSES = {"primary", "primary_candidate"}
IMAGE_ARTIFACT_KINDS = {
    "repository_main_pdf",
    "open_access_main_pdf",
    "authorized_publisher_main_pdf",
    "supporting_information",
}
MAIN_PDF_KINDS = {
    "repository_main_pdf",
    "open_access_main_pdf",
    "authorized_publisher_main_pdf",
}
ROUTE_TERMS = (
    "scheme",
    "synthesis",
    "synthetic route",
    "completion",
    "natural products",
    "retrosynthesis",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target-slots",
        type=Path,
        default=Path("benchmarks/recent_total_synthesis/target_slots.jsonl"),
    )
    parser.add_argument(
        "--source-receipts",
        type=Path,
        default=Path("benchmarks/recent_total_synthesis/source_package_receipts.jsonl"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("tmp/recent-total-synthesis-visual-structure-candidates"),
    )
    parser.add_argument("--doi", action="append", default=[])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-images", type=int, default=8)
    parser.add_argument("--model", default="gpt-5.6-sol")
    parser.add_argument("--reasoning-effort", default="high")
    parser.add_argument("--timeout-s", type=float, default=600.0)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalized_words(value: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", value.casefold())


def _target_on_page(text: str, target_name: str) -> bool:
    haystack = _normalized_words(text)
    needle = _normalized_words(target_name)
    if not needle:
        return False
    width = len(needle)
    return any(
        haystack[index : index + width] == needle for index in range(len(haystack) - width + 1)
    )


def _pdf_page_index(path: Path, targets: list[str]) -> list[dict[str, Any]]:
    try:
        import fitz  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("pymupdf_unavailable") from exc
    document = fitz.open(str(path))
    rows: list[dict[str, Any]] = []
    references_started = False
    try:
        for index in range(document.page_count):
            page = document[index]
            text = page.get_text("text") or ""
            lowered = text.casefold()
            if re.search(r"\breferences\b", text, flags=re.IGNORECASE):
                references_started = True
            rows.append(
                {
                    "page_number": index + 1,
                    "target_names": [name for name in targets if _target_on_page(text, name)],
                    "route_signal_count": sum(lowered.count(term) for term in ROUTE_TERMS),
                    "text_characters": len(text),
                    "drawing_count": len(page.get_drawings()),
                    "embedded_image_count": len(page.get_images(full=True)),
                    "reference_section": references_started,
                }
            )
    finally:
        document.close()
    return rows


def _select_pdf_pages(
    artifacts: list[dict[str, Any]], targets: list[str], *, max_images: int
) -> list[tuple[dict[str, Any], int]]:
    pdf_artifacts = [
        artifact
        for artifact in artifacts
        if Path(str(artifact.get("absolute_path") or "")).suffix.casefold() == ".pdf"
    ]
    main_artifacts = [
        artifact
        for artifact in pdf_artifacts
        if str(artifact.get("artifact_kind") or "") in MAIN_PDF_KINDS
    ]
    # The main paper is the authoritative and compact visual source. Scanning a
    # large SI when a main PDF is available mostly selects spectra and wastes a
    # visual call; SI remains the fallback for papers without a main PDF.
    candidate_artifacts = main_artifacts or pdf_artifacts
    indexed: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for artifact in candidate_artifacts:
        path = Path(str(artifact.get("absolute_path") or ""))
        for page in _pdf_page_index(path, targets):
            indexed.append((artifact, page))

    body_or_graphical = [
        item
        for item in indexed
        if not item[1]["reference_section"]
        or int(item[1]["drawing_count"]) >= 20
        or int(item[1]["embedded_image_count"]) > 0
    ]
    if body_or_graphical:
        indexed = body_or_graphical

    def page_rank(item: tuple[dict[str, Any], dict[str, Any]]) -> tuple[int, ...]:
        _artifact, page = item
        drawings = int(page["drawing_count"])
        images = int(page["embedded_image_count"])
        has_chemical_graphics = drawings >= 20 or images > 0
        return (
            int(not page["reference_section"]),
            int(bool(page["target_names"]) and has_chemical_graphics),
            int(has_chemical_graphics),
            len(page["target_names"]),
            int(page["page_number"] == 1),
            min(drawings, 2000) + min(images, 20) * 100,
            min(int(page["route_signal_count"]), 50),
            -int(page["page_number"]),
        )

    ranked = sorted(indexed, key=page_rank, reverse=True)[:max_images]
    ranked.sort(
        key=lambda item: (
            str(item[0].get("cache_path") or ""),
            int(item[1]["page_number"]),
        )
    )
    return [(artifact, int(page["page_number"])) for artifact, page in ranked]


def _render_pdf_page(
    source: Path,
    page_number: int,
    output: Path,
) -> str:
    import fitz  # type: ignore

    document = fitz.open(str(source))
    try:
        page = document[page_number - 1]
        pixmap = page.get_pixmap(
            matrix=fitz.Matrix(2.5, 2.5),
            alpha=False,
        )
        pixmap.save(str(output))
        return "full page"
    finally:
        document.close()


def _article_images(paper_root: Path) -> list[Path]:
    images = paper_root / "article" / "images"
    if not images.is_dir():
        return []
    return sorted(
        path
        for path in images.iterdir()
        if path.is_file() and path.suffix.casefold() in {".png", ".jpg", ".jpeg", ".webp"}
    )


def _materialize_images(
    *,
    paper_root: Path,
    artifacts: list[dict[str, Any]],
    targets: list[str],
    output_dir: Path,
    max_images: int,
) -> list[dict[str, Any]]:
    images_dir = output_dir / "source-images"
    images_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for index, (artifact, page_number) in enumerate(
        _select_pdf_pages(artifacts, targets, max_images=max_images), start=1
    ):
        source = Path(str(artifact["absolute_path"]))
        output = images_dir / f"page-{index:02d}.png"
        crop_locator = _render_pdf_page(
            source,
            page_number,
            output,
        )
        rows.append(
            {
                "image_path": str(output),
                "source_artifact_path": artifact["cache_path"],
                "source_artifact_sha256": artifact["sha256"],
                "source_locator": f"PDF page {page_number}; {crop_locator}",
                "image_sha256": _sha256_file(output),
            }
        )
    remaining = max_images - len(rows)
    if remaining > 0:
        for source in _article_images(paper_root)[:remaining]:
            output = images_dir / f"figure-{len(rows) + 1:02d}{source.suffix.lower()}"
            shutil.copy2(source, output)
            rows.append(
                {
                    "image_path": str(output),
                    "source_artifact_path": str(source),
                    "source_artifact_sha256": _sha256_file(source),
                    "source_locator": f"publisher HTML figure {source.name}",
                    "image_sha256": _sha256_file(output),
                }
            )
    return rows


def _prompt(*, targets: list[str], image_rows: list[dict[str, Any]]) -> str:
    context = {
        "schema_version": PROMPT_VERSION,
        "requested_targets": targets,
        "image_locators": [
            {
                "image_name": Path(row["image_path"]).name,
                "source_locator": row["source_locator"],
            }
            for row in image_rows
        ],
    }
    return "\n".join(
        [
            "Act as a source-structure transcription assistant for a total-synthesis benchmark.",
            "Use direct visual reasoning only. Do not call shell commands, tools, Python, RDKit, file listing, web search, or any external validator.",
            "Use only the attached exact-paper pages or figures. For every requested target, locate the explicitly labeled chemical drawing and transcribe that visible structure to an isomeric SMILES. Do not answer from memory, the target name, or an inferred analogue.",
            "Inspect all attached pages before deciding. Preserve every stereobond explicitly shown. If the drawing is absent, too small, ambiguous, or does not determine absolute stereochemistry, return unresolved or partial_stereo_candidate instead of guessing. A family scheme does not authorize copying a neighboring analogue.",
            "Return one JSON object only: schema_version='recent_total_synthesis_visual_structure_report.v1', and targets as a list in the requested order. Each row has target_name, status ('exact_source_structure_candidate', 'partial_stereo_candidate', or 'unresolved'), isomeric_smiles, source_image_name, source_locator, and a one-sentence transcription_note. Use an empty SMILES for unresolved rows. Do not provide routes, conditions, literature commentary, or markdown.",
            "VisualStructureExtractionInput:",
            json.dumps(context, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
        ]
    )


def _rdkit_validation(smiles: str) -> dict[str, Any]:
    if not smiles:
        return {"status": "missing", "canonical_isomeric_smiles": ""}
    from rdkit import Chem

    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return {"status": "invalid", "canonical_isomeric_smiles": ""}
    return {
        "status": "roundtrip_valid",
        "canonical_isomeric_smiles": Chem.MolToSmiles(
            molecule, canonical=True, isomericSmiles=True
        ),
    }


def _validated_candidates(parsed: dict[str, Any], targets: list[str]) -> list[dict[str, Any]]:
    rows = parsed.get("targets") or []
    by_name = {
        str(row.get("target_name") or ""): dict(row) for row in rows if isinstance(row, dict)
    }
    result: list[dict[str, Any]] = []
    for target in targets:
        raw = by_name.get(target, {})
        smiles = str(raw.get("isomeric_smiles") or "").strip()
        validation = _rdkit_validation(smiles)
        status = str(raw.get("status") or "unresolved")
        if status != "unresolved" and validation["status"] != "roundtrip_valid":
            status = "invalid_model_smiles"
        result.append(
            {
                "target_name": target,
                "status": status,
                "reported_isomeric_smiles": smiles,
                "rdkit_validation": validation,
                "source_image_name": str(raw.get("source_image_name") or ""),
                "source_locator": str(raw.get("source_locator") or ""),
                "transcription_note": str(raw.get("transcription_note") or ""),
                "admission_authority": False,
            }
        )
    return result


def _input_sha256(
    *, targets: list[str], images: list[dict[str, Any]], model: str, effort: str
) -> str:
    payload = {
        "prompt_version": PROMPT_VERSION,
        "targets": targets,
        "image_sha256": [row["image_sha256"] for row in images],
        "model": model,
        "reasoning_effort": effort,
    }
    return _sha256_bytes(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def main() -> int:
    args = parse_args()
    if args.workers < 1 or args.max_images < 1:
        raise ValueError("workers and max-images must be positive")
    repo_root = Path(__file__).resolve().parents[1]
    output_root = (repo_root / args.output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    slots = [
        row
        for row in _read_jsonl((repo_root / args.target_slots).resolve())
        if row.get("slot_class") in PRIMARY_TARGET_SLOT_CLASSES and row.get("target_name")
    ]
    receipts = {
        str(row["paper_id"]): row
        for row in _read_jsonl((repo_root / args.source_receipts).resolve())
    }
    selected_dois = {str(value).casefold() for value in args.doi}
    groups: list[tuple[str, list[dict[str, Any]]]] = []
    for paper_id in dict.fromkeys(str(row["paper_id"]) for row in slots):
        group = [row for row in slots if str(row["paper_id"]) == paper_id]
        if selected_dois and str(group[0]["doi"]).casefold() not in selected_dois:
            continue
        groups.append((paper_id, group))
    if args.limit > 0:
        groups = groups[: args.limit]

    def run_one(item: tuple[str, list[dict[str, Any]]]) -> dict[str, Any]:
        paper_id, group = item
        receipt = receipts.get(paper_id, {})
        paper_output = output_root / paper_id
        paper_output.mkdir(parents=True, exist_ok=True)
        targets = [str(row["target_name"]) for row in group]
        artifacts: list[dict[str, Any]] = []
        for raw in receipt.get("artifacts") or []:
            if str(raw.get("artifact_kind") or "") not in IMAGE_ARTIFACT_KINDS:
                continue
            path = repo_root / str(raw.get("cache_path") or "")
            if path.is_file():
                artifacts.append({**dict(raw), "absolute_path": str(path)})
        paper_root = repo_root / "tmp" / "authorized-literature-source-cache" / paper_id
        images = _materialize_images(
            paper_root=paper_root,
            artifacts=artifacts,
            targets=targets,
            output_dir=paper_output,
            max_images=args.max_images,
        )
        digest = _input_sha256(
            targets=targets,
            images=images,
            model=args.model,
            effort=args.reasoning_effort,
        )
        result_path = paper_output / "visual-structure-result.json"
        if result_path.is_file() and not args.force:
            existing = json.loads(result_path.read_text(encoding="utf-8"))
            reusable = existing.get("status") == "no_visual_source" or (
                existing.get("status") == "completed"
                and existing.get("attempt_status") == "completed"
            )
            if existing.get("input_sha256") == digest and reusable:
                return existing
        if not images:
            result = {
                "schema_version": "recent_total_synthesis_visual_structure_attempt.v1",
                "paper_id": paper_id,
                "doi": group[0]["doi"],
                "input_sha256": digest,
                "status": "no_visual_source",
                "model_invocations": 0,
                "targets": [
                    {
                        "target_name": target,
                        "status": "unresolved",
                        "reported_isomeric_smiles": "",
                        "rdkit_validation": {
                            "status": "missing",
                            "canonical_isomeric_smiles": "",
                        },
                        "admission_authority": False,
                    }
                    for target in targets
                ],
                "source_images": [],
            }
            result_path.write_text(
                json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            return result
        if args.prepare_only:
            return {
                "schema_version": "recent_total_synthesis_visual_structure_attempt.v1",
                "paper_id": paper_id,
                "doi": group[0]["doi"],
                "input_sha256": digest,
                "status": "images_prepared",
                "model_invocations": 0,
                "targets": [
                    {
                        "target_name": target,
                        "status": "unresolved",
                        "rdkit_validation": {"status": "missing"},
                    }
                    for target in targets
                ],
                "source_images": images,
                "admission_authority": False,
            }
        attempt = _run_visual_json_prompt(
            executable=shutil.which("codex"),
            api_key="",
            base_url="",
            model=args.model,
            output_dir=paper_output,
            image_paths=[Path(row["image_path"]) for row in images],
            prompt=_prompt(targets=targets, image_rows=images),
            timeout_s=args.timeout_s,
            prompt_filename="visual-structure-prompt.txt",
            event_log_filename="visual-structure-events.jsonl",
            stderr_log_filename="visual-structure-stderr.log",
            last_message_filename="visual-structure-last-message.txt",
            ambient_auth=True,
            reasoning_effort=args.reasoning_effort,
        )
        parsed = _parse_json_object(str(attempt.get("raw_last_message") or ""))
        attempt_status = str(attempt.get("status") or "")
        if attempt_status == "completed" and isinstance(parsed, dict):
            result_status = "completed"
        elif attempt_status != "completed":
            result_status = f"model_attempt_{attempt_status or 'failed'}"
        else:
            result_status = "model_output_unparseable"
        result = {
            "schema_version": "recent_total_synthesis_visual_structure_attempt.v1",
            "paper_id": paper_id,
            "doi": group[0]["doi"],
            "input_sha256": digest,
            "status": result_status,
            "model": args.model,
            "reasoning_effort": args.reasoning_effort,
            "model_invocations": 1,
            "usage": dict(attempt.get("usage") or {}),
            "attempt_status": attempt_status,
            "attempt_returncode": int(attempt.get("returncode") or 0),
            "targets": (
                _validated_candidates(parsed, targets)
                if isinstance(parsed, dict)
                else _validated_candidates({}, targets)
            ),
            "source_images": images,
            "admission_authority": False,
        }
        result_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return result

    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run_one, item): item for item in groups}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(
                json.dumps(
                    {
                        "doi": result["doi"],
                        "status": result["status"],
                        "resolved_candidates": sum(
                            row.get("rdkit_validation", {}).get("status") == "roundtrip_valid"
                            for row in result["targets"]
                        ),
                        "target_count": len(result["targets"]),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    summary = {
        "schema_version": "recent_total_synthesis_visual_structure_batch.v1",
        "paper_attempts": len(results),
        "model_invocations": sum(int(row.get("model_invocations") or 0) for row in results),
        "target_rows": sum(len(row.get("targets") or []) for row in results),
        "rdkit_valid_candidates": sum(
            candidate.get("rdkit_validation", {}).get("status") == "roundtrip_valid"
            for row in results
            for candidate in row.get("targets") or []
        ),
        "admission_authority": False,
    }
    (output_root / "batch-summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
