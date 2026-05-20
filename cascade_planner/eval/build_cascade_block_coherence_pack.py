"""Build self-supervised CascadeBlock coherence packs.

This pack uses real adjacent v4 cascade steps as positives and synthetic
corruptions as negatives. It is meant to test whether cascade block coherence
can be learned before wiring any scorer into ChemEnzy search.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter
from pathlib import Path
from typing import Any


BLOCK_PACK_SCHEMA_VERSION = "cascade_block_coherence_pack.v1"


def build_cascade_block_coherence_pack(
    *,
    program_manifest: Path,
    output_dir: Path,
    seed: int = 42,
) -> dict[str, Any]:
    started = time.monotonic()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(program_manifest.read_text(encoding="utf-8"))
    program_paths = {split: Path((manifest.get("outputs") or {})[split]) for split in ("train", "val", "test")}
    outputs = {
        "train": output_dir / "cascade_block_coherence_train.jsonl",
        "val": output_dir / "cascade_block_coherence_val.jsonl",
        "test": output_dir / "cascade_block_coherence_test.jsonl",
        "manifest": output_dir / "cascade_block_coherence_manifest.json",
        "report": output_dir / "cascade_block_coherence_report.md",
    }
    rows_by_split = {}
    for split, path in program_paths.items():
        programs = _read_jsonl(path)
        rows_by_split[split] = _examples_for_split(programs, split=split, seed=seed)
        _write_jsonl(outputs[split], rows_by_split[split])
    result = {
        "schema_version": BLOCK_PACK_SCHEMA_VERSION,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "metadata": {
            "program_manifest": str(program_manifest),
            "output_dir": str(output_dir),
            "seed": seed,
            "elapsed_s": round(time.monotonic() - started, 3),
        },
        "counts": {
            split: _row_counts(rows)
            for split, rows in rows_by_split.items()
        },
        "outputs": {key: str(path) for key, path in outputs.items()},
    }
    outputs["manifest"].write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    outputs["report"].write_text(_markdown(result), encoding="utf-8")
    return result


def _examples_for_split(programs: list[dict[str, Any]], *, split: str, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(f"{seed}:{split}")
    positives = []
    all_steps = []
    for program in programs:
        steps = program.get("steps") or []
        for step in steps:
            all_steps.append((program, step))
        for index, (left, right) in enumerate(zip(steps, steps[1:])):
            positives.append(_block_row(program, left, right, label=1, example_type="positive_adjacent", anchor_index=index))
    if not positives:
        return []
    rows = []
    for pos in positives:
        rows.append(pos)
        left = pos["left_step"]
        right = pos["right_step"]
        program_id = pos.get("program_id")
        rows.append(
            _block_row(
                pos,
                right,
                left,
                label=0,
                example_type="order_shuffle",
                anchor_index=pos.get("anchor_index"),
                positive_block_id=pos.get("block_id"),
                source_is_block_row=True,
            )
        )
        splice_right = _sample_step(all_steps, rng, exclude_program_id=program_id)
        rows.append(
            _block_row(
                pos,
                left,
                splice_right,
                label=0,
                example_type="cross_route_splice",
                anchor_index=pos.get("anchor_index"),
                positive_block_id=pos.get("block_id"),
                source_is_block_row=True,
            )
        )
        same_transform = _sample_step(
            all_steps,
            rng,
            exclude_program_id=program_id,
            transform=right.get("transformation_superclass"),
        )
        if same_transform:
            rows.append(
                _block_row(
                    pos,
                    left,
                    same_transform,
                    label=0,
                    example_type="same_transform_wrong_partner",
                    anchor_index=pos.get("anchor_index"),
                    positive_block_id=pos.get("block_id"),
                    source_is_block_row=True,
                )
            )
        donor = _sample_step(all_steps, rng, exclude_program_id=program_id)
        rows.append(
            _block_row(
                pos,
                left,
                _metadata_corrupted_right(right, donor),
                label=0,
                example_type="metadata_shuffle",
                anchor_index=pos.get("anchor_index"),
                positive_block_id=pos.get("block_id"),
                source_is_block_row=True,
            )
        )
        if left.get("intermediate_isolated") is False:
            rows.append(
                _block_row(
                    pos,
                    left,
                    _terminal_shortcut_step(left),
                    label=0,
                    example_type="hidden_intermediate_shortcut",
                    anchor_index=pos.get("anchor_index"),
                    positive_block_id=pos.get("block_id"),
                    source_is_block_row=True,
                )
            )
    return rows


def _block_row(
    program_or_row: dict[str, Any],
    left: dict[str, Any],
    right: dict[str, Any],
    *,
    label: int,
    example_type: str,
    anchor_index: int | None,
    positive_block_id: str | None = None,
    source_is_block_row: bool = False,
) -> dict[str, Any]:
    source = program_or_row
    program_id = source.get("program_id")
    doi = source.get("doi")
    cascade_id = source.get("cascade_id")
    route_domain = source.get("cascade_type") or source.get("route_domain")
    compatibility = source.get("compatibility") or {}
    block_id = (
        f"{program_id}::{anchor_index}::{example_type}::{left.get('transition_id')}::{right.get('transition_id')}"
    )
    if source_is_block_row:
        compatibility = {
            "compatibility_label": source.get("compatibility_label"),
            "evidence_strength": source.get("compatibility_evidence_strength"),
            "issue_types": source.get("compatibility_issue_types") or [],
            "mitigation_strategies": source.get("compatibility_mitigation_strategies") or [],
        }
    return {
        "block_id": block_id,
        "positive_block_id": positive_block_id or block_id,
        "program_id": program_id,
        "doi": doi,
        "cascade_id": cascade_id,
        "target_smiles": source.get("target_smiles"),
        "route_domain": route_domain or "unknown",
        "anchor_index": anchor_index,
        "label": int(label),
        "example_type": example_type,
        "compatibility_label": compatibility.get("compatibility_label") or "",
        "compatibility_evidence_strength": compatibility.get("evidence_strength") or "",
        "compatibility_issue_types": compatibility.get("issue_types") or [],
        "compatibility_mitigation_strategies": compatibility.get("mitigation_strategies") or [],
        "left_step": _compact_step(left),
        "right_step": _compact_step(right),
    }


def _compact_step(step: dict[str, Any]) -> dict[str, Any]:
    keep = [
        "transition_id",
        "step_id",
        "step_index",
        "step_pos",
        "remaining_steps",
        "rxn_smiles",
        "product_smiles",
        "reactants",
        "main_reactant",
        "transformation_name",
        "transformation_superclass",
        "previous_transformation_superclass",
        "next_transformation_superclass",
        "step_mode",
        "pairwise_mode",
        "intermediate_isolated",
        "step_role",
        "condition_tokens",
        "catalyst_classes",
        "ec1_values",
        "enzyme_families",
        "cofactors",
        "metal_identities",
        "pseudo_terminal_stock",
        "metadata_corrupted",
    ]
    return {key: step.get(key) for key in keep if step.get(key) not in (None, "", [])}


def _sample_step(
    all_steps: list[tuple[dict[str, Any], dict[str, Any]]],
    rng: random.Random,
    *,
    exclude_program_id: str | None,
    transform: str | None = None,
) -> dict[str, Any]:
    candidates = [
        step
        for program, step in all_steps
        if program.get("program_id") != exclude_program_id
        and (not transform or step.get("transformation_superclass") == transform)
    ]
    if not candidates:
        candidates = [step for program, step in all_steps if program.get("program_id") != exclude_program_id]
    return dict(rng.choice(candidates))


def _metadata_corrupted_right(right: dict[str, Any], donor: dict[str, Any]) -> dict[str, Any]:
    out = dict(right)
    for key in (
        "transformation_name",
        "transformation_superclass",
        "step_mode",
        "pairwise_mode",
        "intermediate_isolated",
        "condition_tokens",
        "catalyst_classes",
        "ec1_values",
        "enzyme_families",
        "cofactors",
        "metal_identities",
    ):
        out[key] = donor.get(key)
    out["metadata_corrupted"] = True
    out["transition_id"] = f"{right.get('transition_id')}::metadata_from::{donor.get('transition_id')}"
    return out


def _terminal_shortcut_step(left: dict[str, Any]) -> dict[str, Any]:
    return {
        "transition_id": f"{left.get('transition_id')}::terminal_stock",
        "step_id": "terminal_stock",
        "step_pos": int(left.get("step_pos") or 0) + 1,
        "remaining_steps": 0,
        "rxn_smiles": "",
        "product_smiles": left.get("product_smiles"),
        "reactants": [left.get("product_smiles")] if left.get("product_smiles") else [],
        "main_reactant": left.get("product_smiles"),
        "transformation_name": "terminal stock shortcut",
        "transformation_superclass": "terminal_stock",
        "step_mode": "terminal",
        "pairwise_mode": "terminal",
        "intermediate_isolated": True,
        "condition_tokens": [],
        "catalyst_classes": ["stock"],
        "pseudo_terminal_stock": True,
    }


def _row_counts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "rows": len(rows),
        "positive_rows": sum(1 for row in rows if row.get("label") == 1),
        "negative_rows": sum(1 for row in rows if row.get("label") == 0),
        "example_type_counts": dict(Counter(row.get("example_type") for row in rows)),
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _markdown(result: dict[str, Any]) -> str:
    lines = [
        "# CascadeBlock Coherence Pack",
        "",
        f"- schema: `{result.get('schema_version')}`",
        "",
        "## Counts",
        "",
        "```json",
        json.dumps(result.get("counts") or {}, indent=2, ensure_ascii=False),
        "```",
        "",
        "## Outputs",
        "",
    ]
    for key, path in (result.get("outputs") or {}).items():
        lines.append(f"- {key}: `{path}`")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="Build self-supervised CascadeBlock coherence pack")
    ap.add_argument("--program-manifest", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    result = build_cascade_block_coherence_pack(
        program_manifest=Path(args.program_manifest),
        output_dir=Path(args.output_dir),
        seed=args.seed,
    )
    print(json.dumps({"counts": result["counts"], "outputs": result["outputs"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
