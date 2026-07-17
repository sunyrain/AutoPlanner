"""Freeze a target-only PaRoutes stress panel and evaluator-only references.

The checked-in manifest contains only opaque target labels and canonical
SMILES.  Reference reactions, source indices, and route-depth strata are
written to a separate evaluator pack and must never be passed to a planner.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from rdkit import Chem

from cascade_planner.application.blind_benchmark_contract import (
    BLIND_CASE_SCHEMA,
    BLIND_MANIFEST_SCHEMA,
    canonical_smiles,
    load_blind_manifest,
)


SCHEMA_VERSION = "classic_multistep_benchmark_pack.v1"
PROTOCOL_SCHEMA_VERSION = "classic_multistep_benchmark_protocol.v1"
DEFAULT_SEED = "autoplanner-paroutes-multistep20-v1"
DEFAULT_SPLITS = ("n1", "n5")


@dataclass(frozen=True, slots=True)
class DepthStratum:
    name: str
    minimum: int
    maximum: int
    quota_per_split: int = 2

    def contains(self, value: int) -> bool:
        return self.minimum <= int(value) <= self.maximum

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "minimum_longest_linear_depth": self.minimum,
            "maximum_longest_linear_depth": self.maximum,
            "quota_per_split": self.quota_per_split,
        }


DEFAULT_STRATA = (
    DepthStratum("llr_3", 3, 3),
    DepthStratum("llr_4", 4, 4),
    DepthStratum("llr_5_6", 5, 6),
    DepthStratum("llr_7_8", 7, 8),
    DepthStratum("llr_9_10", 9, 10),
)


def build_classic_multistep_benchmark(
    *,
    paroutes_root: Path,
    manifest_output: Path,
    reference_output: Path,
    protocol_output: Path | None = None,
    search_benchmark_output_dir: Path | None = None,
    leakage_audit_output: Path | None = None,
    seed: str = DEFAULT_SEED,
    splits: Iterable[str] = DEFAULT_SPLITS,
    strata: Iterable[DepthStratum] = DEFAULT_STRATA,
) -> dict[str, Any]:
    """Build one deterministic 20-target panel without planner-side answers."""

    root = Path(paroutes_root).resolve()
    resolved_splits = tuple(str(value).strip().lower() for value in splits)
    resolved_strata = tuple(strata)
    if not resolved_splits or not resolved_strata:
        raise ValueError("benchmark_splits_and_strata_required")

    source_files: dict[str, dict[str, Any]] = {}
    candidates: dict[str, list[dict[str, Any]]] = {}
    for split in resolved_splits:
        targets_path = root / f"targets_{split}.txt"
        references_path = root / f"ref_routes_{split}.json"
        targets = _read_targets(targets_path)
        references = _read_reference_routes(references_path)
        if len(targets) != len(references):
            raise ValueError(f"paroutes_target_reference_count_mismatch:{split}")
        candidates[split] = [
            _candidate(
                split=split,
                index=index,
                target=target,
                reference=reference,
            )
            for index, (target, reference) in enumerate(zip(targets, references, strict=True))
        ]
        source_files[split] = {
            "targets_path": str(targets_path),
            "targets_sha256": _file_digest(targets_path),
            "references_path": str(references_path),
            "references_sha256": _file_digest(references_path),
            "available_target_count": len(targets),
        }

    selected: list[dict[str, Any]] = []
    seen_targets: set[str] = set()
    for split in resolved_splits:
        for stratum in resolved_strata:
            eligible = [
                row
                for row in candidates[split]
                if row["target_smiles"] not in seen_targets
                and stratum.contains(row["reference_metrics"]["longest_linear_depth"])
            ]
            ranked = sorted(
                eligible,
                key=lambda row: _selection_key(
                    seed=seed,
                    split=split,
                    source_index=row["source_index"],
                    target_smiles=row["target_smiles"],
                ),
            )
            if len(ranked) < stratum.quota_per_split:
                raise ValueError(
                    f"paroutes_stratum_quota_unavailable:{split}:{stratum.name}:{len(ranked)}"
                )
            for row in ranked[: stratum.quota_per_split]:
                selected.append({**row, "depth_stratum": stratum.name})
                seen_targets.add(row["target_smiles"])

    cases: list[dict[str, Any]] = []
    references: list[dict[str, Any]] = []
    for ordinal, row in enumerate(selected, start=1):
        identity = hashlib.sha256(
            f"{seed}\0{row['split']}\0{row['source_index']}\0{row['target_smiles']}".encode("utf-8")
        ).hexdigest()
        case_id = f"classic-ms-{ordinal:02d}-{identity[:10]}"
        target_name = f"opaque benchmark target {ordinal:02d}"
        cases.append(
            {
                "schema_version": BLIND_CASE_SCHEMA,
                "case_id": case_id,
                "target_name": target_name,
                "target_smiles": row["target_smiles"],
                "acceptance": {
                    "minimum_complete_routes": 1,
                    "minimum_edge_proof_level": 2,
                    "minimum_independent_source_groups": 1,
                    "minimum_planning_route_steps": 0,
                    "stock_boundary": "benchmark_search",
                },
                "budget": {
                    "max_model_invocations": 3,
                    "max_total_input_tokens": 100_000,
                    "max_total_output_tokens": 24_000,
                    "max_total_wall_time_s": 1_200,
                    "max_accepted_expansions": 96,
                    "max_attempt_runs": 192,
                    "max_prompt_context_bytes": 160_000,
                },
            }
        )
        references.append(
            {
                "case_id": case_id,
                "target_name": target_name,
                "target_smiles": row["target_smiles"],
                "split": row["split"],
                "source_index": row["source_index"],
                "depth_stratum": row["depth_stratum"],
                "reference_metrics": row["reference_metrics"],
                "gt_route": row["gt_route"],
            }
        )

    manifest = {
        "schema_version": BLIND_MANIFEST_SCHEMA,
        "cases": cases,
    }
    _write_json(manifest_output, manifest)
    loaded_cases = load_blind_manifest(manifest_output)
    if len(loaded_cases) != len(cases):
        raise RuntimeError("written_blind_manifest_failed_round_trip")

    manifest_sha256 = _file_digest(manifest_output)
    reference_pack = {
        "schema_version": SCHEMA_VERSION,
        "metadata": {
            "seed": seed,
            "target_count": len(cases),
            "splits": list(resolved_splits),
            "depth_strata": [row.to_dict() for row in resolved_strata],
            "manifest_path": str(Path(manifest_output).resolve()),
            "manifest_sha256": manifest_sha256,
            "planner_input_contract": "target_name_and_canonical_smiles_only",
            "reference_material_is_evaluator_only": True,
            "source_files": source_files,
        },
        "cases": references,
    }
    _write_json(reference_output, reference_pack)
    search_benchmarks: dict[str, str] = {}
    if search_benchmark_output_dir is not None:
        search_root = Path(search_benchmark_output_dir)
        for split in resolved_splits:
            output = search_root / f"paroutes_{split}_multistep.json"
            _write_json(
                output,
                {
                    "metadata": {
                        "schema_version": SCHEMA_VERSION,
                        "split": split,
                        "reference_material_is_evaluator_only": True,
                    },
                    "targets": [
                        {
                            "target_smiles": row["target_smiles"],
                            "cascade_id": row["case_id"],
                            "split": row["split"],
                            "route_domain": "all_chemical",
                            "depth": row["reference_metrics"]["longest_linear_depth"],
                            "reference_depth": row["reference_metrics"]["reaction_count"],
                            "gt_route": row["gt_route"],
                        }
                        for row in references
                        if row["split"] == split
                    ],
                },
            )
            search_benchmarks[split] = str(output.resolve())

    protocol = {
        "schema_version": PROTOCOL_SCHEMA_VERSION,
        "benchmark": "PaRoutes set-n1 and set-n5 deterministic stress mini-panel",
        "seed": seed,
        "target_count": len(cases),
        "split_target_counts": {
            split: sum(row["split"] == split for row in references) for split in resolved_splits
        },
        "depth_strata": [row.to_dict() for row in resolved_strata],
        "manifest_sha256": manifest_sha256,
        "selection": {
            "ranking": "sha256(seed, split, source_index, canonical_smiles)",
            "cross_split_target_deduplication": True,
            "reference_depth_used_only_for_sampling": True,
            "route_length_is_not_an_acceptance_or_ranking_reward": True,
        },
        "blindness": {
            "planner_receives_reference_routes": False,
            "planner_receives_source_indices": False,
            "planner_receives_depth_strata": False,
            "planner_receives_target_names_with_identity": False,
        },
        "metrics": {
            "reported_independently": [
                "route_present",
                "host_reaction_validated",
                "official_benchmark_stock_closed",
                "reference_reaction_and_leaf_overlap",
                "exact_source_grade",
                "condition_completeness",
                "resource_envelope",
            ],
            "shorter_valid_route_is_allowed": True,
            "missing_evidence_does_not_erase_a_structural_route": True,
        },
        "provenance": {
            "paper_doi": "10.1039/D2DD00015F",
            "zenodo_record": "https://zenodo.org/records/6275421",
            "license_note": "See the upstream PaRoutes record and repository.",
        },
    }
    if protocol_output is not None:
        _write_json(protocol_output, protocol)
    leakage_audit_pack = ""
    if leakage_audit_output is not None:
        leakage = {
            "schema_version": "blind_leakage_audit_pack.v1",
            "manifest_sha256": manifest_sha256,
            "cases": {
                str(row["case_id"]): {
                    "target_synonyms": [],
                    "target_synonym_not_applicable_reason": (
                        "opaque PaRoutes dataset identity has no public target name"
                    ),
                    "key_intermediate_smiles": _key_intermediate_smiles(
                        row.get("gt_route") or [],
                        target_smiles=str(row.get("target_smiles") or ""),
                    ),
                }
                for row in references
            },
            "semantics": {
                "evaluator_only": True,
                "never_passed_to_planner_subprocess": True,
                "contains_reference_route_derived_intermediates": True,
                "must_remain_outside_the_tracked_repository": True,
            },
        }
        leakage["content_sha256"] = _json_digest(leakage)
        _write_json(leakage_audit_output, leakage)
        leakage_audit_pack = str(Path(leakage_audit_output).resolve())

    return {
        "schema_version": SCHEMA_VERSION,
        "manifest": str(Path(manifest_output).resolve()),
        "reference_pack": str(Path(reference_output).resolve()),
        "search_benchmarks": search_benchmarks,
        "protocol": (str(Path(protocol_output).resolve()) if protocol_output is not None else ""),
        "manifest_sha256": manifest_sha256,
        "leakage_audit_pack": leakage_audit_pack,
        "target_count": len(cases),
        "split_target_counts": protocol["split_target_counts"],
        "stratum_counts": {
            stratum.name: sum(row["depth_stratum"] == stratum.name for row in references)
            for stratum in resolved_strata
        },
    }


def reference_route_metrics(route: Mapping[str, Any]) -> dict[str, Any]:
    reaction_count = 0
    longest_linear_depth = 0
    leaf_count = 0
    stock_leaf_count = 0
    convergent_reaction_count = 0

    def walk(value: Any, depth: int) -> None:
        nonlocal reaction_count
        nonlocal longest_linear_depth
        nonlocal leaf_count
        nonlocal stock_leaf_count
        nonlocal convergent_reaction_count
        if not isinstance(value, Mapping):
            return
        node_type = str(value.get("type") or "")
        next_depth = depth + (1 if node_type == "reaction" else 0)
        if node_type == "reaction":
            reaction_count += 1
            molecule_children = [
                child
                for child in value.get("children") or []
                if isinstance(child, Mapping) and child.get("type") == "mol"
            ]
            synthesized_children = [
                child
                for child in molecule_children
                if any(
                    isinstance(grandchild, Mapping) and grandchild.get("type") == "reaction"
                    for grandchild in child.get("children") or []
                )
            ]
            if len(synthesized_children) > 1:
                convergent_reaction_count += 1
        longest_linear_depth = max(longest_linear_depth, next_depth)
        children = [child for child in value.get("children") or [] if isinstance(child, Mapping)]
        if node_type == "mol" and not children:
            leaf_count += 1
            stock_leaf_count += value.get("in_stock") is True
        for child in children:
            walk(child, next_depth)

    walk(route, 0)
    return {
        "reaction_count": reaction_count,
        "longest_linear_depth": longest_linear_depth,
        "leaf_count": leaf_count,
        "stock_leaf_count": stock_leaf_count,
        "reference_stock_closed": bool(leaf_count and leaf_count == stock_leaf_count),
        "convergent_reaction_count": convergent_reaction_count,
        "convergent": convergent_reaction_count > 0,
    }


def _candidate(
    *,
    split: str,
    index: int,
    target: str,
    reference: Mapping[str, Any],
) -> dict[str, Any]:
    canonical_target = canonical_smiles(target)
    canonical_reference = canonical_smiles(reference.get("smiles"))
    if not canonical_target or not canonical_reference:
        raise ValueError(f"paroutes_target_invalid:{split}:{index}")
    if canonical_target != canonical_reference:
        raise ValueError(f"paroutes_target_reference_identity_mismatch:{split}:{index}")
    metrics = reference_route_metrics(reference)
    if not metrics["reference_stock_closed"]:
        raise ValueError(f"paroutes_reference_not_stock_closed:{split}:{index}")
    return {
        "split": split,
        "source_index": index,
        "target_smiles": canonical_target,
        "reference_metrics": metrics,
        "gt_route": _reference_reactions(reference),
    }


def _reference_reactions(route: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def walk(value: Any) -> None:
        if not isinstance(value, Mapping):
            return
        if value.get("type") == "reaction":
            metadata = dict(value.get("metadata") or {})
            reaction = _clean_reaction(metadata.get("smiles") or metadata.get("rsmi"))
            if reaction:
                rows.append(
                    {
                        "rxn_smiles": reaction,
                        "transformation": "uspto_clean_reference",
                        "step_role": "external_paroutes_reference",
                    }
                )
        for child in value.get("children") or []:
            walk(child)

    walk(route)
    return rows


def _clean_reaction(value: Any) -> str:
    text = str(value or "").strip()
    if ">>" not in text and text.count(">") == 2:
        reactants, _agents, product = text.split(">", 2)
        text = f"{reactants}>>{product}"
    if ">>" not in text:
        return ""
    reactants, product = text.split(">>", 1)
    if not reactants.strip() or not product.strip():
        return ""
    return f"{reactants.strip()}>>{product.strip()}"


def _key_intermediate_smiles(
    reactions: Iterable[Mapping[str, Any]], *, target_smiles: str
) -> list[str]:
    target = _unmapped_smiles(target_smiles)
    values: set[str] = set()
    for row in reactions:
        reaction = str(dict(row).get("rxn_smiles") or "")
        if ">>" not in reaction:
            continue
        _reactants, products = reaction.split(">>", 1)
        for component in products.split("."):
            canonical = _unmapped_smiles(component)
            if canonical and canonical != target:
                values.add(canonical)
    return sorted(values)


def _unmapped_smiles(value: Any) -> str:
    molecule = Chem.MolFromSmiles(str(value or "").strip())
    if molecule is None:
        return ""
    for atom in molecule.GetAtoms():
        atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)


def _selection_key(*, seed: str, split: str, source_index: int, target_smiles: str) -> str:
    return hashlib.sha256(
        f"{seed}\0{split}\0{source_index}\0{target_smiles}".encode("utf-8")
    ).hexdigest()


def _read_targets(path: Path) -> list[str]:
    try:
        rows = [row.strip() for row in path.read_text(encoding="utf-8").splitlines()]
    except (OSError, UnicodeDecodeError) as exc:
        raise ValueError(f"paroutes_targets_unreadable:{path}") from exc
    return [row for row in rows if row]


def _read_reference_routes(path: Path) -> list[dict[str, Any]]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"paroutes_references_unreadable:{path}") from exc
    if not isinstance(value, list) or not all(isinstance(row, Mapping) for row in value):
        raise ValueError(f"paroutes_references_invalid:{path}")
    return [dict(row) for row in value]


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_digest(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paroutes-root", default="data/benchmarks/paroutes", type=Path)
    parser.add_argument("--manifest-output", required=True, type=Path)
    parser.add_argument("--reference-output", required=True, type=Path)
    parser.add_argument("--protocol-output", type=Path)
    parser.add_argument("--search-benchmark-output-dir", type=Path)
    parser.add_argument("--leakage-audit-output", type=Path)
    parser.add_argument("--seed", default=DEFAULT_SEED)
    args = parser.parse_args(argv)
    result = build_classic_multistep_benchmark(
        paroutes_root=args.paroutes_root,
        manifest_output=args.manifest_output,
        reference_output=args.reference_output,
        protocol_output=args.protocol_output,
        search_benchmark_output_dir=args.search_benchmark_output_dir,
        leakage_audit_output=args.leakage_audit_output,
        seed=args.seed,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
