"""Audit transform-conditioned v4 retro-template upstream proposals.

This is a proposal-coverage diagnostic, not a deployed planner.  It asks:
given a ChemEnzy downstream connector, can retro-templates extracted from v4
train steps generate upstream reactants that recover held-out connected
cascade blocks better than nearest-neighbor retrieval?
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem

from cascade_planner.cascade_search.v4_product_value import canonical_smiles
from cascade_planner.eval.audit_nonoracle_provider_bridge import (
    _cache_index,
    _downstream_candidates,
    _fp_similarity,
    _heavy_atoms,
    _label_downstream_candidates,
    _read_json,
    _reference_connection_similarity,
    _reference_examples,
    _reference_summary,
    _target_list,
)
from cascade_planner.eval.audit_provider_routepool_oracle import _fp, _load_reference_blocks, _transition_fp


SCHEMA_VERSION = "v4_template_upstream_bridge_audit.v1"


def audit_v4_template_upstream_bridge(
    *,
    program_manifest: Path,
    chem_enzy_cache: Path,
    output_json: Path,
    atommap_cache: Path,
    route_recovery_json: Path | None = None,
    split: str = "test",
    only_routed: bool = True,
    max_targets: int | None = None,
    max_chem_candidates: int = 100,
    main_reactant_only: bool = True,
    min_connector_heavy_atoms: int = 6,
    connected_ref_similarity: float = 0.55,
    analog_similarity: float = 0.55,
    max_templates_per_transform: int = 50,
    max_templates_per_connector: int = 80,
    max_outcomes_per_template: int = 3,
    generalize: int = 1,
    oracle_transform_filter: bool = False,
    transform_policy_scores_jsonl: Path | None = None,
    transform_policy: str = "none",
    transform_top_m: int = 5,
    map_missing_limit: int = 0,
    save_all_proposals: bool = False,
    top_ks: tuple[int, ...] = (1, 3, 5, 10, 20, 50),
) -> dict[str, Any]:
    started = time.monotonic()
    cache = _read_json(chem_enzy_cache)
    cache_index = _cache_index(cache)
    refs = _load_reference_blocks(program_manifest, split=split)
    refs_by_target: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for ref in refs:
        if connected_ref_similarity > 0.0 and _reference_connection_similarity(ref) < float(connected_ref_similarity):
            continue
        target = canonical_smiles(str(ref.get("target_smiles") or "")) or str(ref.get("target_smiles") or "")
        if target:
            refs_by_target[target].append(ref)

    targets = _target_list(
        refs_by_target=refs_by_target,
        route_recovery_json=route_recovery_json,
        only_routed=only_routed,
        cache_index=cache_index,
    )
    if max_targets is not None:
        targets = targets[: max(0, int(max_targets))]

    template_bank = _build_template_bank(
        program_manifest=program_manifest,
        atommap_cache=atommap_cache,
        max_templates_per_transform=max_templates_per_transform,
        map_missing_limit=map_missing_limit,
    )
    all_templates = _all_templates(template_bank)
    transform_policy_map = _load_transform_policy_scores(
        transform_policy_scores_jsonl,
        policy=transform_policy,
        top_m=transform_top_m,
    )

    target_rows = []
    for target in targets:
        refs_for_target = refs_by_target.get(target) or []
        downstream_candidates = _downstream_candidates(
            cache_index,
            target,
            max_chem_candidates=max_chem_candidates,
            main_reactant_only=main_reactant_only,
            min_connector_heavy_atoms=min_connector_heavy_atoms,
        )
        downstream_label = _label_downstream_candidates(downstream_candidates, refs_for_target, analog_similarity=analog_similarity)
        proposal_rows = _build_template_rows(
            target=target,
            downstream_candidates=downstream_candidates,
            refs_for_target=refs_for_target,
            template_bank=template_bank,
            all_templates=all_templates,
            max_templates_per_connector=max_templates_per_connector,
            max_outcomes_per_template=max_outcomes_per_template,
            generalize=generalize,
            analog_similarity=analog_similarity,
            oracle_transform_filter=oracle_transform_filter,
            transform_policy_map=transform_policy_map,
        )
        proposal_rows.sort(key=_proposal_sort_key)
        for idx, row in enumerate(proposal_rows, start=1):
            row["proposal_rank"] = idx
        target_rows.append(
            {
                "target_smiles": target,
                "reference_blocks": len(refs_for_target),
                "chem_enzy_candidate_rows": len(_candidate_rows(cache_index, target)[:max_chem_candidates]),
                "chem_enzy_downstream_connectors": len(downstream_candidates),
                "downstream_analog_any": bool(downstream_label.get("analog_any")),
                "downstream_analog_top_rank": downstream_label.get("best_analog_rank"),
                "downstream_best_similarity": downstream_label.get("best_similarity"),
                "template_proposals": len(proposal_rows),
                "template_fire_any": bool(proposal_rows),
                "analog_template_any": any(row.get("analog_hit") for row in proposal_rows),
                "pair_and_analog_template_any": any(row.get("pair_and_analog") for row in proposal_rows),
                "best_analog_template_rank": _best_rank(proposal_rows, "analog_hit"),
                "best_pair_and_analog_template_rank": _best_rank(proposal_rows, "pair_and_analog"),
                "topk": _topk_hits(proposal_rows, top_ks=top_ks),
                "examples": proposal_rows[:10],
                "proposals": proposal_rows if save_all_proposals else None,
                "reference_examples": _reference_examples(refs_for_target),
            }
        )

    report = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "metadata": {
            "program_manifest": str(program_manifest),
            "chem_enzy_cache": str(chem_enzy_cache),
            "atommap_cache": str(atommap_cache),
            "route_recovery_json": str(route_recovery_json) if route_recovery_json else None,
            "split": split,
            "only_routed": bool(only_routed),
            "max_targets": max_targets,
            "max_chem_candidates": max_chem_candidates,
            "main_reactant_only": bool(main_reactant_only),
            "min_connector_heavy_atoms": min_connector_heavy_atoms,
            "connected_ref_similarity": connected_ref_similarity,
            "analog_similarity": analog_similarity,
            "max_templates_per_transform": max_templates_per_transform,
            "max_templates_per_connector": max_templates_per_connector,
            "max_outcomes_per_template": max_outcomes_per_template,
            "generalize": generalize,
            "oracle_transform_filter": bool(oracle_transform_filter),
            "transform_policy_scores_jsonl": str(transform_policy_scores_jsonl) if transform_policy_scores_jsonl else None,
            "transform_policy": transform_policy,
            "transform_top_m": transform_top_m,
            "map_missing_limit": map_missing_limit,
            "save_all_proposals": bool(save_all_proposals),
            "top_ks": list(top_ks),
            "elapsed_s": round(time.monotonic() - started, 3),
            "contract": (
                "Templates are extracted only from v4 train. Held-out references "
                "are used only for labels, except oracle_transform_filter which "
                "is an explicit non-deployable upper-bound diagnostic."
            ),
        },
        "template_bank_summary": _template_summary(template_bank),
        "reference_summary": _reference_summary(refs_by_target),
        "summary": _summary(target_rows, top_ks=top_ks),
        "targets": target_rows,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    output_json.with_suffix(".md").write_text(_markdown(report), encoding="utf-8")
    return report


def _build_template_bank(
    *,
    program_manifest: Path,
    atommap_cache: Path,
    max_templates_per_transform: int,
    map_missing_limit: int,
) -> dict[str, list[dict[str, Any]]]:
    from rdchiral.template_extractor import extract_from_reaction

    manifest = json.loads(program_manifest.read_text(encoding="utf-8"))
    train_path = Path((manifest.get("outputs") or {})["train"])
    atom_cache: dict[str, str | None] = {}
    if atommap_cache.exists():
        atom_cache = json.loads(atommap_cache.read_text(encoding="utf-8"))
    train_programs = _read_jsonl(train_path)
    missing = []
    raw_records = []
    for program in train_programs:
        steps = [step for step in program.get("steps") or [] if isinstance(step, dict)]
        for step_idx, step in enumerate(steps):
            rxn = str(step.get("rxn_smiles") or "")
            if not rxn:
                continue
            if rxn not in atom_cache:
                missing.append(rxn)
            raw_records.append((program, step, steps[step_idx + 1] if step_idx + 1 < len(steps) else None))
    if missing and map_missing_limit > 0:
        _map_missing_reactions(missing[: int(map_missing_limit)], atom_cache)
        atommap_cache.parent.mkdir(parents=True, exist_ok=True)
        atommap_cache.write_text(json.dumps(atom_cache), encoding="utf-8")

    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for program, step, downstream in raw_records:
        rxn = str(step.get("rxn_smiles") or "")
        mapped = atom_cache.get(rxn)
        if not mapped or ">>" not in mapped:
            continue
        try:
            lhs, rhs = mapped.split(">>", 1)
            result = extract_from_reaction({"_id": str(step.get("transition_id") or ""), "reactants": lhs, "products": rhs})
        except Exception:
            continue
        template = (result or {}).get("reaction_smarts")
        if not template:
            continue
        transform = _norm_transform(step.get("transformation_superclass"))
        downstream_transform = _norm_transform((downstream or {}).get("transformation_superclass"))
        pair = f"{transform}->{downstream_transform}" if downstream else f"{transform}->"
        key = template
        row = grouped[pair].get(key)
        if row is None:
            grouped[pair][key] = {
                "template": template,
                "transform_pair": pair,
                "upstream_transform": transform,
                "downstream_transform": downstream_transform,
                "count": 0,
                "examples": [],
            }
            row = grouped[pair][key]
        row["count"] += 1
        if len(row["examples"]) < 3:
            row["examples"].append(
                {
                    "program_id": program.get("program_id"),
                    "transition_id": step.get("transition_id"),
                    "rxn_smiles": rxn,
                    "product_smiles": step.get("product_smiles"),
                    "main_reactant": step.get("main_reactant"),
                }
            )
    out: dict[str, list[dict[str, Any]]] = {}
    for pair, rows_by_template in grouped.items():
        rows = sorted(rows_by_template.values(), key=lambda row: (-int(row.get("count") or 0), row.get("template") or ""))
        out[pair] = rows[: max(0, int(max_templates_per_transform))]
    return out


def _all_templates(template_bank: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    seen = set()
    rows = []
    for templates in template_bank.values():
        for row in templates:
            key = row.get("template")
            if key in seen:
                continue
            seen.add(key)
            rows.append(row)
    rows.sort(key=lambda row: (-int(row.get("count") or 0), row.get("transform_pair") or "", row.get("template") or ""))
    return rows


def _build_template_rows(
    *,
    target: str,
    downstream_candidates: list[dict[str, Any]],
    refs_for_target: list[dict[str, Any]],
    template_bank: dict[str, list[dict[str, Any]]],
    all_templates: list[dict[str, Any]],
    max_templates_per_connector: int,
    max_outcomes_per_template: int,
    generalize: int,
    analog_similarity: float,
    oracle_transform_filter: bool,
    transform_policy_map: dict[tuple[str, str, str], list[str]] | None,
) -> list[dict[str, Any]]:
    rows = []
    for cand in downstream_candidates:
        connector = str(cand.get("connector") or "")
        if not connector:
            continue
        templates = _templates_for_candidate(
            refs_for_target=refs_for_target,
            template_bank=template_bank,
            all_templates=all_templates,
            max_templates_per_connector=max_templates_per_connector,
            oracle_transform_filter=oracle_transform_filter,
            selected_transform_pairs=_selected_transform_pairs(
                transform_policy_map=transform_policy_map,
                target=target,
                connector=connector,
                downstream_rank=cand.get("downstream_rank"),
            ),
        )
        for template_rank, template_row in enumerate(templates, start=1):
            mapped_outcomes = _mapped_template_outcomes(
                str(template_row.get("template") or ""),
                connector,
                generalize=generalize,
            )
            outcomes = _apply_template(
                str(template_row.get("template") or ""),
                connector,
                max_outcomes=max_outcomes_per_template,
                generalize=generalize,
            )
            for outcome_rank, reactants in enumerate(outcomes, start=1):
                main_reactant = _main_reactant_from_outcome(reactants)
                labels = _best_reference_label(
                    target=target,
                    connector=connector,
                    product=connector,
                    main_reactant=main_reactant,
                    transform_pair=template_row.get("transform_pair"),
                    refs_for_target=refs_for_target,
                    analog_similarity=analog_similarity,
                )
                applicability = _template_applicability_features(
                    connector=connector,
                    reactants=reactants,
                    main_reactant=main_reactant,
                    template_row=template_row,
                )
                reaction_center = mapped_outcomes.get(tuple(sorted(reactants))) or {}
                rows.append(
                    {
                        "proposal_rank": None,
                        "proposal_score": round(float(_proposal_score(cand, template_row, template_rank, outcome_rank)), 6),
                        "downstream_rank": cand.get("downstream_rank"),
                        "connector": connector,
                        "connector_heavy_atoms": cand.get("connector_heavy_atoms"),
                        "connector_is_main_reactant": cand.get("is_main_reactant"),
                        "template_rank": template_rank,
                        "outcome_rank": outcome_rank,
                        "template_count": template_row.get("count"),
                        "template_transform_pair": template_row.get("transform_pair"),
                        "template": template_row.get("template"),
                        "reactants": list(reactants),
                        "main_reactant": main_reactant,
                        "upstream_similarity": labels.get("upstream_similarity"),
                        "downstream_similarity": labels.get("downstream_similarity"),
                        "best_reference_block_id": labels.get("reference_block_id"),
                        "reference_transform_pair": labels.get("reference_transform_pair"),
                        "pair_hit": labels.get("pair_hit"),
                        "analog_hit": labels.get("analog_hit"),
                        "pair_and_analog": labels.get("pair_and_analog"),
                        **applicability,
                        **reaction_center,
                    }
                )
    return _dedupe_rows(rows)


def _template_applicability_features(
    *,
    connector: str,
    reactants: tuple[str, ...],
    main_reactant: str,
    template_row: dict[str, Any],
) -> dict[str, Any]:
    connector_atoms = max(0, _heavy_atoms(connector))
    main_atoms = max(0, _heavy_atoms(main_reactant))
    reactant_atoms = [_heavy_atoms(value) for value in reactants]
    total_reactant_atoms = sum(max(0, value) for value in reactant_atoms)
    connector_fp = _fp(connector)
    main_fp = _fp(main_reactant)
    outcome_transition_fp = _transition_fp(connector, main_reactant)
    transition_sims = []
    product_sims = []
    main_sims = []
    for example in template_row.get("examples") or []:
        if not isinstance(example, dict):
            continue
        example_product = example.get("product_smiles")
        example_main = example.get("main_reactant")
        transition_sims.append(_fp_similarity(outcome_transition_fp, _transition_fp(example_product, example_main)))
        product_sims.append(_fp_similarity(connector_fp, _fp(example_product)))
        main_sims.append(_fp_similarity(main_fp, _fp(example_main)))
    transition_sims.sort(reverse=True)
    product_sims.sort(reverse=True)
    main_sims.sort(reverse=True)
    return {
        "app_connector_main_similarity": round(float(_fp_similarity(connector_fp, main_fp)), 6),
        "app_main_heavy_atoms_norm": round(min(1.0, float(main_atoms) / 40.0), 6),
        "app_heavy_atom_delta_norm": round(float(main_atoms - connector_atoms) / max(connector_atoms, 1), 6),
        "app_abs_heavy_atom_delta_norm": round(abs(float(main_atoms - connector_atoms)) / max(connector_atoms, 1), 6),
        "app_total_reactant_atoms_norm": round(min(2.0, float(total_reactant_atoms) / max(connector_atoms, 1)), 6),
        "app_reactant_count": len(reactants),
        "app_largest_reactant_fraction": round(float(max(reactant_atoms) if reactant_atoms else 0) / max(total_reactant_atoms, 1), 6),
        "app_template_example_best_transition_sim": round(float(transition_sims[0]), 6) if transition_sims else 0.0,
        "app_template_example_mean_top3_transition_sim": round(float(sum(transition_sims[:3]) / min(3, len(transition_sims))), 6) if transition_sims else 0.0,
        "app_template_example_best_product_sim": round(float(product_sims[0]), 6) if product_sims else 0.0,
        "app_template_example_best_main_sim": round(float(main_sims[0]), 6) if main_sims else 0.0,
    }


def _templates_for_candidate(
    *,
    refs_for_target: list[dict[str, Any]],
    template_bank: dict[str, list[dict[str, Any]]],
    all_templates: list[dict[str, Any]],
    max_templates_per_connector: int,
    oracle_transform_filter: bool,
    selected_transform_pairs: list[str] | None = None,
) -> list[dict[str, Any]]:
    if selected_transform_pairs is not None:
        rows = []
        for pair in selected_transform_pairs:
            for row in template_bank.get(str(pair or ""), []):
                rows.append(row)
        return rows[: max(0, int(max_templates_per_connector))]
    if oracle_transform_filter:
        rows = []
        seen = set()
        for ref in refs_for_target:
            for row in template_bank.get(str(ref.get("transform_pair") or ""), []):
                key = row.get("template")
                if key in seen:
                    continue
                seen.add(key)
                rows.append(row)
        return rows[: max(0, int(max_templates_per_connector))]
    return all_templates[: max(0, int(max_templates_per_connector))]


def _load_transform_policy_scores(path: Path | None, *, policy: str, top_m: int) -> dict[tuple[str, str, str], list[str]] | None:
    if path is None or str(policy or "none") == "none":
        return None
    policy = str(policy or "").lower()
    if policy not in {"selector", "frequency"}:
        raise ValueError(f"Unsupported transform policy: {policy}")
    grouped: dict[tuple[str, str, str], list[tuple[str, float, int]]] = defaultdict(list)
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            key = _policy_key(row.get("target_smiles"), row.get("connector"), row.get("downstream_rank"))
            pair = str(row.get("transform_pair") or "")
            if not pair:
                continue
            rank = int(row.get("frequency_rank") or 10**9)
            if policy == "selector":
                score = float(row.get("selector_score") or 0.0)
            else:
                score = float(row.get("baseline_frequency_score") or (1.0 / max(1, rank) ** 0.5))
            grouped[key].append((pair, score, rank))
    out: dict[tuple[str, str, str], list[str]] = {}
    for key, rows in grouped.items():
        rows.sort(key=lambda item: (-float(item[1]), int(item[2]), item[0]))
        selected = []
        seen = set()
        for pair, _, _ in rows:
            if pair in seen:
                continue
            seen.add(pair)
            selected.append(pair)
            if len(selected) >= max(0, int(top_m)):
                break
        out[key] = selected
    return out


def _selected_transform_pairs(
    *,
    transform_policy_map: dict[tuple[str, str, str], list[str]] | None,
    target: str,
    connector: str,
    downstream_rank: Any,
) -> list[str] | None:
    if transform_policy_map is None:
        return None
    return transform_policy_map.get(_policy_key(target, connector, downstream_rank), [])


def _policy_key(target: Any, connector: Any, downstream_rank: Any) -> tuple[str, str, str]:
    return (str(target or ""), str(connector or ""), str(downstream_rank or ""))


def _apply_template(template: str, product: str, *, max_outcomes: int, generalize: int) -> list[tuple[str, ...]]:
    from cascade_planner.expand.enz_template import apply_template_to_product

    outcomes = []
    seen = set()
    for outcome in apply_template_to_product(template, product, max_outcomes=max_outcomes, generalize=generalize):
        parts = tuple(sorted(str(value) for value in outcome if value))
        if parts and parts not in seen:
            seen.add(parts)
            outcomes.append(parts)
    return outcomes


def _mapped_template_outcomes(template: str, product: str, *, generalize: int) -> dict[tuple[str, ...], dict[str, Any]]:
    from rdchiral.main import rdchiralRunText

    from cascade_planner.expand.enz_template import canon_set, generalize_template

    if generalize:
        template = generalize_template(template, generalize)
    try:
        payload = rdchiralRunText(template, product, return_mapped=True)
    except Exception:
        return {}
    if not isinstance(payload, tuple) or len(payload) != 2:
        return {}
    outcomes, mapped = payload
    if not isinstance(mapped, dict):
        return {}
    out: dict[tuple[str, ...], dict[str, Any]] = {}
    for outcome in outcomes or []:
        try:
            parts = tuple(sorted(str(value) for value in canon_set(str(outcome)) if value))
        except Exception:
            continue
        mapped_row = mapped.get(outcome)
        if not parts or not mapped_row:
            continue
        mapped_smiles = mapped_row[0] if isinstance(mapped_row, tuple) and mapped_row else None
        matched_atoms = mapped_row[1] if isinstance(mapped_row, tuple) and len(mapped_row) > 1 else ()
        out[parts] = _mapped_outcome_features(mapped_smiles, matched_atoms)
    return out


def _mapped_outcome_features(mapped_smiles: Any, matched_atoms: Any) -> dict[str, Any]:
    mol = Chem.MolFromSmiles(str(mapped_smiles or ""))
    if mol is None:
        return {}
    map_nums = [int(atom.GetAtomMapNum()) for atom in mol.GetAtoms() if atom.GetAtomMapNum()]
    inherited = [value for value in map_nums if value < 900]
    new_atoms = [value for value in map_nums if value >= 900]
    matched = {int(value) for value in matched_atoms or [] if isinstance(value, int) or str(value).isdigit()}
    inherited_set = set(inherited)
    denom = max(1, len(map_nums))
    return {
        "rc_mapped_atom_count": len(map_nums),
        "rc_inherited_atom_count": len(inherited),
        "rc_new_atom_count": len(new_atoms),
        "rc_new_atom_fraction": round(float(len(new_atoms)) / denom, 6),
        "rc_inherited_atom_fraction": round(float(len(inherited)) / denom, 6),
        "rc_template_matched_atom_count": len(matched),
        "rc_template_matched_fraction": round(float(len(matched)) / max(1, len(inherited_set)), 6),
    }


def _main_reactant_from_outcome(reactants: tuple[str, ...]) -> str:
    best = ""
    best_atoms = -1
    for reactant in reactants:
        atoms = _heavy_atoms(reactant)
        if atoms > best_atoms:
            best = reactant
            best_atoms = atoms
    return best


def _best_reference_label(
    *,
    target: str,
    connector: str,
    product: str,
    main_reactant: str,
    transform_pair: Any,
    refs_for_target: list[dict[str, Any]],
    analog_similarity: float,
) -> dict[str, Any]:
    upstream_fp = _transition_fp(product, main_reactant)
    downstream_fp = _transition_fp(target, connector)
    best = {
        "score": -1.0,
        "upstream_similarity": 0.0,
        "downstream_similarity": 0.0,
        "reference_block_id": None,
        "reference_transform_pair": None,
        "pair_hit": False,
        "analog_hit": False,
        "pair_and_analog": False,
    }
    for ref in refs_for_target:
        upstream_sim = _fp_similarity(upstream_fp, ref.get("upstream_fp"))
        ref_downstream_fp = _transition_fp(ref.get("downstream_product"), ref.get("downstream_main_reactant"))
        downstream_sim = _fp_similarity(downstream_fp, ref_downstream_fp)
        analog = bool(upstream_sim >= analog_similarity and downstream_sim >= analog_similarity)
        pair = str(transform_pair or "").lower() == str(ref.get("transform_pair") or "").lower()
        score = float(upstream_sim + downstream_sim + (1.0 if pair else 0.0) + (1.0 if analog else 0.0))
        if score > float(best["score"]):
            best = {
                "score": score,
                "upstream_similarity": round(float(upstream_sim), 6),
                "downstream_similarity": round(float(downstream_sim), 6),
                "reference_block_id": ref.get("block_id"),
                "reference_transform_pair": ref.get("transform_pair"),
                "pair_hit": bool(pair),
                "analog_hit": bool(analog),
                "pair_and_analog": bool(pair and analog),
            }
    best.pop("score", None)
    return best


def _proposal_score(cand: dict[str, Any], template_row: dict[str, Any], template_rank: int, outcome_rank: int) -> float:
    rank_score = 1.0 / max(1, int(cand.get("downstream_rank") or 10**6)) ** 0.5
    template_score = 1.0 / max(1, int(template_rank)) ** 0.5
    outcome_score = 1.0 / max(1, int(outcome_rank))
    count_score = min(1.0, float(template_row.get("count") or 0) / 10.0)
    size_score = min(1.0, float(cand.get("connector_heavy_atoms") or 0) / 20.0)
    return 0.25 * rank_score + 0.30 * template_score + 0.15 * outcome_score + 0.15 * count_score + 0.15 * size_score


def _proposal_sort_key(row: dict[str, Any]) -> tuple[float, int, int, int]:
    return (
        -float(row.get("proposal_score") or 0.0),
        int(row.get("downstream_rank") or 10**9),
        int(row.get("template_rank") or 10**9),
        int(row.get("outcome_rank") or 10**9),
    )


def _dedupe_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    out = []
    for row in sorted(rows, key=_proposal_sort_key):
        key = (row.get("connector"), tuple(row.get("reactants") or ()), row.get("template_transform_pair"))
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def _best_rank(rows: list[dict[str, Any]], label_key: str) -> int | None:
    ranks = [int(row.get("proposal_rank") or 10**9) for row in rows if row.get(label_key)]
    return min(ranks) if ranks else None


def _topk_hits(rows: list[dict[str, Any]], *, top_ks: tuple[int, ...]) -> dict[str, dict[str, bool]]:
    out = {}
    for k in top_ks:
        top = rows[: max(0, int(k))]
        out[str(k)] = {
            "template_fire": bool(top),
            "analog_template": any(row.get("analog_hit") for row in top),
            "pair_and_analog_template": any(row.get("pair_and_analog") for row in top),
        }
    return out


def _summary(rows: list[dict[str, Any]], *, top_ks: tuple[int, ...]) -> dict[str, Any]:
    denom = max(len(rows), 1)
    out: dict[str, Any] = {
        "targets": len(rows),
        "reference_blocks": sum(int(row.get("reference_blocks") or 0) for row in rows),
        "chem_enzy_candidate_rows": sum(int(row.get("chem_enzy_candidate_rows") or 0) for row in rows),
        "chem_enzy_downstream_connectors": sum(int(row.get("chem_enzy_downstream_connectors") or 0) for row in rows),
        "template_proposals": sum(int(row.get("template_proposals") or 0) for row in rows),
        "targets_downstream_analog_any": sum(1 for row in rows if row.get("downstream_analog_any")),
        "targets_template_fire_any": sum(1 for row in rows if row.get("template_fire_any")),
        "targets_analog_template_any": sum(1 for row in rows if row.get("analog_template_any")),
        "targets_pair_and_analog_template_any": sum(1 for row in rows if row.get("pair_and_analog_template_any")),
        "downstream_analog_any_rate": round(sum(1 for row in rows if row.get("downstream_analog_any")) / denom, 6),
        "template_fire_any_rate": round(sum(1 for row in rows if row.get("template_fire_any")) / denom, 6),
        "analog_template_any_rate": round(sum(1 for row in rows if row.get("analog_template_any")) / denom, 6),
        "pair_and_analog_template_any_rate": round(sum(1 for row in rows if row.get("pair_and_analog_template_any")) / denom, 6),
    }
    for k in top_ks:
        key = str(k)
        out[f"template_fire_at_{k}"] = round(sum(1 for row in rows if (row.get("topk") or {}).get(key, {}).get("template_fire")) / denom, 6)
        out[f"analog_template_at_{k}"] = round(sum(1 for row in rows if (row.get("topk") or {}).get(key, {}).get("analog_template")) / denom, 6)
        out[f"pair_and_analog_template_at_{k}"] = round(
            sum(1 for row in rows if (row.get("topk") or {}).get(key, {}).get("pair_and_analog_template")) / denom,
            6,
        )
    return out


def _template_summary(template_bank: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    rows = [row for values in template_bank.values() for row in values]
    return {
        "transform_pairs": len(template_bank),
        "templates_kept": len(rows),
        "top_transform_pairs": dict(Counter(row.get("transform_pair") for row in rows).most_common(20)),
        "top_template_counts": [int(row.get("count") or 0) for row in sorted(rows, key=lambda item: -int(item.get("count") or 0))[:20]],
    }


def _candidate_rows(cache_index: dict[str, list[dict[str, Any]]], target: str) -> list[dict[str, Any]]:
    canonical = canonical_smiles(str(target or "")) or str(target or "")
    return cache_index.get(canonical) or []


def _map_missing_reactions(rxns: list[str], cache: dict[str, str | None]) -> None:
    from rxnmapper import RXNMapper

    mapper = RXNMapper()
    for rxn in rxns:
        if rxn in cache:
            continue
        try:
            mapped = mapper.get_attention_guided_atom_maps([rxn])[0].get("mapped_rxn")
        except Exception:
            mapped = None
        cache[rxn] = mapped


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                payload = json.loads(line)
                if isinstance(payload, dict):
                    rows.append(payload)
    return rows


def _norm_transform(value: Any) -> str:
    text = str(value or "unknown").strip().lower()
    return text or "unknown"


def _parse_top_ks(value: str) -> tuple[int, ...]:
    return tuple(sorted({int(item.strip()) for item in str(value or "").split(",") if item.strip()})) or (1, 3, 5, 10, 20, 50)


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# v4 Template Upstream Bridge Audit",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    for key, value in (report.get("summary") or {}).items():
        lines.append(f"| `{key}` | `{value}` |")
    lines.extend(
        [
            "",
            "## Template Bank",
            "",
            "```json",
            json.dumps(report.get("template_bank_summary") or {}, indent=2, ensure_ascii=False),
            "```",
            "",
            "## Examples",
            "",
            "```json",
            json.dumps((report.get("targets") or [])[:8], indent=2, ensure_ascii=False)[:12000],
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    RDLogger.DisableLog("rdApp.*")
    parser = argparse.ArgumentParser(description="Audit v4 train retro-template upstream proposals for ChemEnzy connectors")
    parser.add_argument("--program-manifest", default="results/shared/cascadebench_v2_20260516/program_pack/cascade_program_pack_manifest.json")
    parser.add_argument("--chem-enzy-cache", default="results/shared/cascadebench_strict_20260516/cache/chem_enzy_onestep_top100.json")
    parser.add_argument("--atommap-cache", default="results/shared/v4_atommap_cache.json")
    parser.add_argument("--route-recovery-json")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--all-reference-targets", action="store_true")
    parser.add_argument("--max-targets", type=int)
    parser.add_argument("--max-chem-candidates", type=int, default=100)
    parser.add_argument("--include-all-reactants", action="store_true")
    parser.add_argument("--min-connector-heavy-atoms", type=int, default=6)
    parser.add_argument("--connected-ref-similarity", type=float, default=0.55)
    parser.add_argument("--analog-similarity", type=float, default=0.55)
    parser.add_argument("--max-templates-per-transform", type=int, default=50)
    parser.add_argument("--max-templates-per-connector", type=int, default=80)
    parser.add_argument("--max-outcomes-per-template", type=int, default=3)
    parser.add_argument("--generalize", type=int, default=1)
    parser.add_argument("--oracle-transform-filter", action="store_true")
    parser.add_argument("--transform-policy-scores-jsonl")
    parser.add_argument("--transform-policy", choices=("none", "selector", "frequency"), default="none")
    parser.add_argument("--transform-top-m", type=int, default=5)
    parser.add_argument("--map-missing-limit", type=int, default=0)
    parser.add_argument("--save-all-proposals", action="store_true")
    parser.add_argument("--top-ks", default="1,3,5,10,20,50")
    args = parser.parse_args()
    report = audit_v4_template_upstream_bridge(
        program_manifest=Path(args.program_manifest),
        chem_enzy_cache=Path(args.chem_enzy_cache),
        atommap_cache=Path(args.atommap_cache),
        route_recovery_json=Path(args.route_recovery_json) if args.route_recovery_json else None,
        output_json=Path(args.output_json),
        split=args.split,
        only_routed=not args.all_reference_targets,
        max_targets=args.max_targets,
        max_chem_candidates=args.max_chem_candidates,
        main_reactant_only=not args.include_all_reactants,
        min_connector_heavy_atoms=args.min_connector_heavy_atoms,
        connected_ref_similarity=args.connected_ref_similarity,
        analog_similarity=args.analog_similarity,
        max_templates_per_transform=args.max_templates_per_transform,
        max_templates_per_connector=args.max_templates_per_connector,
        max_outcomes_per_template=args.max_outcomes_per_template,
        generalize=args.generalize,
        oracle_transform_filter=args.oracle_transform_filter,
        transform_policy_scores_jsonl=Path(args.transform_policy_scores_jsonl) if args.transform_policy_scores_jsonl else None,
        transform_policy=args.transform_policy,
        transform_top_m=args.transform_top_m,
        map_missing_limit=args.map_missing_limit,
        save_all_proposals=args.save_all_proposals,
        top_ks=_parse_top_ks(args.top_ks),
    )
    print(json.dumps({"summary": report["summary"], "output_json": args.output_json}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
