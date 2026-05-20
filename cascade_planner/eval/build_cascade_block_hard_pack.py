"""Build a harder CascadeBlock coherence pack.

Compared with build_cascade_block_coherence_pack, this builder removes explicit
artifact flags, aligns step positions for corrupted blocks, and uses nearest
wrong partners as hard negatives. It is an audit pack: high performance here is
better evidence that block coherence is more than artifact detection.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter
from pathlib import Path
from typing import Any

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem

from cascade_planner.eval.build_cascade_block_coherence_pack import (
    BLOCK_PACK_SCHEMA_VERSION,
    _block_row,
    _read_jsonl,
    _row_counts,
    _write_jsonl,
)


HARD_BLOCK_PACK_SCHEMA_VERSION = "cascade_block_coherence_hard_pack.v1"


def build_cascade_block_hard_pack(
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
        "train": output_dir / "cascade_block_hard_train.jsonl",
        "val": output_dir / "cascade_block_hard_val.jsonl",
        "test": output_dir / "cascade_block_hard_test.jsonl",
        "manifest": output_dir / "cascade_block_hard_manifest.json",
        "report": output_dir / "cascade_block_hard_report.md",
    }
    rows_by_split = {}
    for split, path in program_paths.items():
        programs = _read_jsonl(path)
        rows_by_split[split] = _examples_for_split(programs, split=split, seed=seed)
        _write_jsonl(outputs[split], rows_by_split[split])
    result = {
        "schema_version": HARD_BLOCK_PACK_SCHEMA_VERSION,
        "base_schema_version": BLOCK_PACK_SCHEMA_VERSION,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "metadata": {
            "program_manifest": str(program_manifest),
            "output_dir": str(output_dir),
            "seed": seed,
            "elapsed_s": round(time.monotonic() - started, 3),
        },
        "counts": {split: _row_counts(rows) for split, rows in rows_by_split.items()},
        "outputs": {key: str(path) for key, path in outputs.items()},
    }
    outputs["manifest"].write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    outputs["report"].write_text(_markdown(result), encoding="utf-8")
    return result


def _examples_for_split(programs: list[dict[str, Any]], *, split: str, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(f"{seed}:hard:{split}")
    all_steps = [(program, step) for program in programs for step in (program.get("steps") or [])]
    step_index = _step_fp_index(all_steps)
    positives = []
    for program in programs:
        steps = program.get("steps") or []
        for index, (left, right) in enumerate(zip(steps, steps[1:])):
            positives.append((program, index, left, right))
    rows = []
    for program, index, left, right in positives:
        rows.append(_block_row(program, _clean_step(left), _clean_step(right), label=1, example_type="positive_adjacent", anchor_index=index))
        left_anchor = _clean_step(left)
        actual_right = _clean_step(right)
        right_pos = int(left.get("step_pos") or 0) + 1
        right_remaining = max(0, int(left.get("remaining_steps") or 0) - 1)

        swapped_left = _align_step(_clean_step(right), int(left.get("step_pos") or 0), int(left.get("remaining_steps") or 0))
        swapped_right = _align_step(_clean_step(left), right_pos, right_remaining)
        rows.append(
            _negative_row(
                program,
                swapped_left,
                swapped_right,
                index=index,
                example_type="order_shuffle_aligned",
                positive_block_id=rows[-1]["block_id"],
            )
        )

        near_any = _nearest_wrong_step(step_index, left_anchor, exclude_program_id=program.get("program_id"))
        rows.append(
            _negative_row(
                program,
                left_anchor,
                _align_step(_clean_step(near_any), right_pos, right_remaining),
                index=index,
                example_type="near_splice",
                positive_block_id=rows[-2]["block_id"],
            )
        )

        near_same_transform = _nearest_wrong_step(
            step_index,
            left_anchor,
            exclude_program_id=program.get("program_id"),
            transform=right.get("transformation_superclass"),
        )
        rows.append(
            _negative_row(
                program,
                left_anchor,
                _align_step(_clean_step(near_same_transform), right_pos, right_remaining),
                index=index,
                example_type="same_transform_near_splice",
                positive_block_id=rows[-3]["block_id"],
            )
        )

        donor = _nearest_wrong_step(step_index, left_anchor, exclude_program_id=program.get("program_id"))
        rows.append(
            _negative_row(
                program,
                left_anchor,
                _align_step(_metadata_from(actual_right, donor), right_pos, right_remaining),
                index=index,
                example_type="metadata_shuffle_hard",
                positive_block_id=rows[-4]["block_id"],
            )
        )

        rows.append(
            _negative_row(
                program,
                left_anchor,
                _align_step(_catalyst_condition_from(actual_right, donor), right_pos, right_remaining),
                index=index,
                example_type="catalyst_condition_shuffle",
                positive_block_id=rows[-5]["block_id"],
            )
        )
    return rows


def _negative_row(
    program: dict[str, Any],
    left: dict[str, Any],
    right: dict[str, Any],
    *,
    index: int,
    example_type: str,
    positive_block_id: str,
) -> dict[str, Any]:
    return _block_row(
        program,
        left,
        right,
        label=0,
        example_type=example_type,
        anchor_index=index,
        positive_block_id=positive_block_id,
    )


def _clean_step(step: dict[str, Any]) -> dict[str, Any]:
    out = dict(step)
    out.pop("metadata_corrupted", None)
    out.pop("pseudo_terminal_stock", None)
    return out


def _align_step(step: dict[str, Any], step_pos: int, remaining_steps: int) -> dict[str, Any]:
    out = _clean_step(step)
    out["step_pos"] = int(step_pos)
    out["remaining_steps"] = int(remaining_steps)
    return out


def _metadata_from(step: dict[str, Any], donor: dict[str, Any]) -> dict[str, Any]:
    out = _clean_step(step)
    donor = _clean_step(donor)
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
    out["transition_id"] = f"{step.get('transition_id')}::hard_meta::{donor.get('transition_id')}"
    return out


def _catalyst_condition_from(step: dict[str, Any], donor: dict[str, Any]) -> dict[str, Any]:
    out = _clean_step(step)
    donor = _clean_step(donor)
    for key in (
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
    out["transition_id"] = f"{step.get('transition_id')}::hard_condition::{donor.get('transition_id')}"
    return out


def _nearest_wrong_step(
    step_index: dict[str, Any],
    left: dict[str, Any],
    *,
    exclude_program_id: str | None,
    transform: str | None = None,
) -> dict[str, Any]:
    left_fp = _fp(str(left.get("product_smiles") or ""))
    buckets = [str(transform or ""), ""]
    for bucket_name in buckets:
        bucket = step_index.get(bucket_name) or {}
        fps = bucket.get("fps") or []
        items = bucket.get("items") or []
        if not fps or left_fp is None:
            continue
        scores = DataStructs.BulkTanimotoSimilarity(left_fp, fps)
        ranked = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)
        for idx, _score in ranked:
            program_id, step = items[idx]
            if program_id != exclude_program_id:
                return dict(step)
    for program_id, step in step_index.get("all_items") or []:
        if program_id != exclude_program_id:
            return dict(step)
    return {}


def _step_fp_index(all_steps: list[tuple[dict[str, Any], dict[str, Any]]]) -> dict[str, Any]:
    index: dict[str, Any] = {"all_items": []}
    for program, step in all_steps:
        program_id = program.get("program_id")
        fp = _fp(str(step.get("main_reactant") or ""))
        index["all_items"].append((program_id, step))
        if fp is None:
            continue
        for key in ("", str(step.get("transformation_superclass") or "")):
            bucket = index.setdefault(key, {"fps": [], "items": []})
            bucket["fps"].append(fp)
            bucket["items"].append((program_id, step))
    return index


def _fp(smiles: str):
    mol = Chem.MolFromSmiles(str(smiles or ""))
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)


def _sim(left: Any, right: Any) -> float:
    if left is None or right is None:
        return 0.0
    return float(DataStructs.TanimotoSimilarity(left, right))


def _markdown(result: dict[str, Any]) -> str:
    lines = [
        "# CascadeBlock Hard Coherence Pack",
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
    RDLogger.DisableLog("rdApp.*")
    ap = argparse.ArgumentParser(description="Build hard self-supervised CascadeBlock coherence pack")
    ap.add_argument("--program-manifest", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    result = build_cascade_block_hard_pack(
        program_manifest=Path(args.program_manifest),
        output_dir=Path(args.output_dir),
        seed=args.seed,
    )
    print(json.dumps({"counts": result["counts"], "outputs": result["outputs"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
