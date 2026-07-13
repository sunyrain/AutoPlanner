"""Executable, fail-closed guidance for ChemEnzy one-step candidates.

The strategic policy deliberately does not contain reaction SMILES.  This
module consumes only molecule/subgoal hints and changes the ordering of
candidates already proposed by ChemEnzy.  It also removes a narrow set of
structurally impossible candidates before they enter MolStar.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import math
from typing import Any

from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

from cascade_planner.routes.admission import (
    RetrosyntheticAdmissionPolicy,
    audit_retrosynthetic_candidate,
    retrosynthetic_admission_record,
)

GUIDANCE_SCHEMA = "chem_enzy_guidance.v1"
GUIDANCE_STATS_SCHEMA = "chem_enzy_guidance.stats.v1"


@dataclass(frozen=True)
class ChemEnzyGuidanceConfig:
    enabled: bool = False
    policy_id: str = ""
    preferred_smiles: tuple[str, ...] = ()
    preferred_precursor_sets: tuple[tuple[str, ...], ...] = ()
    anchor_smiles: tuple[str, ...] = ()
    terminal_blacklist: tuple[str, ...] = ()
    exact_set_cost_reward: float = 2.4
    exact_component_cost_reward: float = 1.25
    similarity_cost_reward: float = 0.65
    similarity_threshold: float = 0.55
    terminal_blacklist_cost_penalty: float = 3.0
    preferred_reaction_classes: tuple[str, ...] = ()
    preferred_retrons: tuple[str, ...] = ()
    reaction_class_cost_reward: float = 0.20
    retron_cost_reward: float = 0.15
    hard_filter_element_inventory: bool = True
    max_tolerated_missing_heavy_atoms: int = 3
    hard_filter_large_atom_jump: bool = True
    large_atom_jump_threshold: int = 15
    hard_filter_self_loop: bool = True

    @classmethod
    def from_flags(cls, search_flags: dict[str, Any] | None) -> "ChemEnzyGuidanceConfig":
        flags = dict(search_flags or {})
        raw = flags.get("chem_enzy_guidance")
        if not isinstance(raw, dict):
            return cls(enabled=False)
        return cls(
            enabled=bool(raw.get("enabled", True)),
            policy_id=str(raw.get("policy_id") or ""),
            preferred_smiles=_canonical_smiles_tuple(raw.get("preferred_smiles") or []),
            preferred_precursor_sets=_canonical_precursor_sets(raw.get("preferred_precursor_sets") or []),
            anchor_smiles=_canonical_smiles_tuple(raw.get("anchor_smiles") or []),
            terminal_blacklist=_canonical_smiles_tuple(raw.get("terminal_blacklist") or []),
            exact_set_cost_reward=_float(raw.get("exact_set_cost_reward"), cls.exact_set_cost_reward, lo=0.0),
            exact_component_cost_reward=_float(
                raw.get("exact_component_cost_reward"), cls.exact_component_cost_reward, lo=0.0
            ),
            similarity_cost_reward=_float(raw.get("similarity_cost_reward"), cls.similarity_cost_reward, lo=0.0),
            similarity_threshold=_float(raw.get("similarity_threshold"), cls.similarity_threshold, lo=0.0, hi=1.0),
            terminal_blacklist_cost_penalty=_float(
                raw.get("terminal_blacklist_cost_penalty"), cls.terminal_blacklist_cost_penalty, lo=0.0
            ),
            preferred_reaction_classes=_bounded_hint_tuple(
                raw.get("preferred_reaction_classes") or []
            ),
            preferred_retrons=_bounded_hint_tuple(
                raw.get("preferred_retrons")
                or raw.get("derived_retrons")
                or raw.get("retron_hints")
                or []
            ),
            reaction_class_cost_reward=_float(
                raw.get("reaction_class_cost_reward"),
                cls.reaction_class_cost_reward,
                lo=0.0,
                hi=0.5,
            ),
            retron_cost_reward=_float(
                raw.get("retron_cost_reward"),
                cls.retron_cost_reward,
                lo=0.0,
                hi=0.5,
            ),
            hard_filter_element_inventory=bool(raw.get("hard_filter_element_inventory", True)),
            max_tolerated_missing_heavy_atoms=_int(
                raw.get("max_tolerated_missing_heavy_atoms"), cls.max_tolerated_missing_heavy_atoms, lo=0
            ),
            hard_filter_large_atom_jump=bool(raw.get("hard_filter_large_atom_jump", True)),
            large_atom_jump_threshold=_int(raw.get("large_atom_jump_threshold"), cls.large_atom_jump_threshold, lo=8),
            hard_filter_self_loop=bool(raw.get("hard_filter_self_loop", True)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": GUIDANCE_SCHEMA,
            "enabled": self.enabled,
            "policy_id": self.policy_id,
            "preferred_smiles": list(self.preferred_smiles),
            "preferred_precursor_sets": [list(item) for item in self.preferred_precursor_sets],
            "anchor_smiles": list(self.anchor_smiles),
            "terminal_blacklist": list(self.terminal_blacklist),
            "exact_set_cost_reward": self.exact_set_cost_reward,
            "exact_component_cost_reward": self.exact_component_cost_reward,
            "similarity_cost_reward": self.similarity_cost_reward,
            "similarity_threshold": self.similarity_threshold,
            "terminal_blacklist_cost_penalty": self.terminal_blacklist_cost_penalty,
            "preferred_reaction_classes": list(self.preferred_reaction_classes),
            "preferred_retrons": list(self.preferred_retrons),
            "reaction_class_cost_reward": self.reaction_class_cost_reward,
            "retron_cost_reward": self.retron_cost_reward,
            "hard_filter_element_inventory": self.hard_filter_element_inventory,
            "max_tolerated_missing_heavy_atoms": self.max_tolerated_missing_heavy_atoms,
            "hard_filter_large_atom_jump": self.hard_filter_large_atom_jump,
            "large_atom_jump_threshold": self.large_atom_jump_threshold,
            "hard_filter_self_loop": self.hard_filter_self_loop,
            "raw_reaction_injection": False,
        }


@dataclass
class ChemEnzyGuidanceState:
    config: ChemEnzyGuidanceConfig
    target_smiles: str = ""
    calls: int = 0
    candidates_seen: int = 0
    candidates_kept: int = 0
    candidates_rejected: int = 0
    hint_comparisons: int = 0
    exact_set_matches: int = 0
    exact_component_matches: int = 0
    similarity_matches: int = 0
    terminal_blacklist_matches: int = 0
    terminal_stock_exclusions_requested: int = 0
    terminal_stock_exclusions_removed: int = 0
    score_adjusted: int = 0
    reranked_calls: int = 0
    rejected_by_reason: Counter[str] = field(default_factory=Counter)
    matched_hint_smiles: set[str] = field(default_factory=set)
    reaction_prior_candidates_with_metadata: int = 0
    reaction_class_comparisons: int = 0
    retron_comparisons: int = 0
    reaction_class_matches: int = 0
    retron_matches: int = 0
    reaction_prior_score_adjusted: int = 0
    matched_reaction_classes: set[str] = field(default_factory=set)
    matched_retrons: set[str] = field(default_factory=set)
    rejected_candidate_audits: list[dict[str, Any]] = field(default_factory=list)

    def reset_for_target(self, target_smiles: str) -> None:
        config = self.config
        exclusions_requested = self.terminal_stock_exclusions_requested
        exclusions_removed = self.terminal_stock_exclusions_removed
        self.__dict__.clear()
        self.__dict__.update(ChemEnzyGuidanceState(config=config, target_smiles=target_smiles).__dict__)
        self.terminal_stock_exclusions_requested = exclusions_requested
        self.terminal_stock_exclusions_removed = exclusions_removed

    def to_dict(self) -> dict[str, Any]:
        requested_hint_count = len(set(self.config.preferred_smiles) | set(self.config.anchor_smiles))
        requested_reaction_prior_count = len(self.config.preferred_reaction_classes) + len(
            self.config.preferred_retrons
        )
        return {
            "schema_version": GUIDANCE_STATS_SCHEMA,
            "enabled": bool(self.config.enabled),
            "policy_id": self.config.policy_id,
            "target_smiles": self.target_smiles,
            "config": self.config.to_dict(),
            "requested_hint_count": requested_hint_count,
            "calls": self.calls,
            "candidates_seen": self.candidates_seen,
            "candidates_kept": self.candidates_kept,
            "candidates_rejected": self.candidates_rejected,
            "hint_comparisons": self.hint_comparisons,
            "exact_set_matches": self.exact_set_matches,
            "exact_component_matches": self.exact_component_matches,
            "similarity_matches": self.similarity_matches,
            "terminal_blacklist_matches": self.terminal_blacklist_matches,
            "terminal_stock_exclusions_requested": self.terminal_stock_exclusions_requested,
            "terminal_stock_exclusions_removed": self.terminal_stock_exclusions_removed,
            "score_adjusted": self.score_adjusted,
            "reranked_calls": self.reranked_calls,
            "rejected_by_reason": dict(sorted(self.rejected_by_reason.items())),
            "matched_hint_smiles": sorted(self.matched_hint_smiles),
            "requested_reaction_prior_count": requested_reaction_prior_count,
            "preferred_reaction_classes": list(self.config.preferred_reaction_classes),
            "preferred_retrons": list(self.config.preferred_retrons),
            "reaction_prior_candidates_with_metadata": self.reaction_prior_candidates_with_metadata,
            "reaction_class_comparisons": self.reaction_class_comparisons,
            "retron_comparisons": self.retron_comparisons,
            "reaction_class_matches": self.reaction_class_matches,
            "retron_matches": self.retron_matches,
            "reaction_prior_score_adjusted": self.reaction_prior_score_adjusted,
            "matched_reaction_classes": sorted(self.matched_reaction_classes),
            "matched_retrons": sorted(self.matched_retrons),
            "reaction_prior_metadata_observed": bool(
                self.reaction_prior_candidates_with_metadata
            ),
            "reaction_prior_applied": bool(self.reaction_prior_score_adjusted),
            "rejected_candidate_audit_count": self.candidates_rejected,
            "rejected_candidate_audits": list(self.rejected_candidate_audits),
            "rejected_candidate_audits_truncated": max(
                0, self.candidates_rejected - len(self.rejected_candidate_audits)
            ),
            "hint_comparison_executed": bool(requested_hint_count and self.hint_comparisons),
            "ranking_signal_applied": bool(self.score_adjusted),
            "hard_filter_executed": bool(self.candidates_seen),
            "raw_reaction_injection": False,
        }


class ChemEnzyGuidedOneStepWrapper:
    """Filter and rerank native proposals without injecting new reactions."""

    def __init__(self, one_step: Any, *, config: ChemEnzyGuidanceConfig, state: ChemEnzyGuidanceState) -> None:
        self.one_step = one_step
        self.config = config
        self.state = state
        self.one_step_models = dict(getattr(one_step, "one_step_models", {}) or {})
        hints = set(config.preferred_smiles) | set(config.anchor_smiles)
        self._hint_fingerprints = {}
        for smiles in hints:
            fingerprint = _fingerprint(smiles)
            if fingerprint is not None:
                self._hint_fingerprints[smiles] = fingerprint

    def run(self, target: str, *args: Any, **kwargs: Any) -> dict[str, Any]:
        result = self.one_step.run(target, *args, **kwargs)
        self.state.calls += 1
        if not isinstance(result, dict):
            return result
        reactants = _as_list(result.get("reactants"))
        candidate_count = len(reactants)
        if not candidate_count:
            return result

        self.state.candidates_seen += candidate_count
        target_key = _canonical_smiles(target)
        rows: list[dict[str, Any]] = []
        for index, raw_reactants in enumerate(reactants):
            components = _canonical_components(raw_reactants)
            admission = _host_admission_audit(target_key, components, self.config)
            reasons = [str(item) for item in admission.get("reasons") or []]
            if reasons:
                self.state.candidates_rejected += 1
                self.state.rejected_by_reason.update(reasons)
                provenance = _candidate_provenance(result, index)
                rejection = retrosynthetic_admission_record(
                    admission,
                    stage="pre_moltree_one_step",
                    source=provenance["source"],
                    model=provenance["model"],
                    template=provenance["template"],
                    candidate_index=index,
                )
                if len(self.state.rejected_candidate_audits) < 100:
                    self.state.rejected_candidate_audits.append(rejection)
                continue
            row = self._scored_row(
                result,
                index=index,
                components=components,
                admission=admission,
            )
            rows.append(row)

        self.state.candidates_kept += len(rows)
        original_order = [row["index"] for row in rows]
        if self.config.enabled:
            rows.sort(key=lambda row: (row["guided_cost"], row["index"]))
            if original_order != [row["index"] for row in rows]:
                self.state.reranked_calls += 1
        return _project_result(result, rows, candidate_count=candidate_count)

    def _scored_row(
        self,
        result: dict[str, Any],
        *,
        index: int,
        components: tuple[str, ...],
        admission: dict[str, Any],
    ) -> dict[str, Any]:
        hints = set(self.config.preferred_smiles) | set(self.config.anchor_smiles)
        self.state.hint_comparisons += len(hints)
        component_set = set(components)
        exact_set = any(Counter(components) == Counter(candidate_set) for candidate_set in self.config.preferred_precursor_sets)
        exact_components = sorted(component_set & hints)
        blacklist_matches = sorted(component_set & set(self.config.terminal_blacklist))
        similarity, similarity_hint = _best_similarity(components, self._hint_fingerprints)

        reward = 0.0
        if exact_set:
            reward += self.config.exact_set_cost_reward
            self.state.exact_set_matches += 1
        if exact_components:
            reward += self.config.exact_component_cost_reward
            self.state.exact_component_matches += 1
            self.state.matched_hint_smiles.update(exact_components)
        if similarity >= self.config.similarity_threshold and similarity_hint:
            scaled = (similarity - self.config.similarity_threshold) / max(1e-9, 1.0 - self.config.similarity_threshold)
            reward += self.config.similarity_cost_reward * max(0.0, min(1.0, scaled))
            self.state.similarity_matches += 1
            self.state.matched_hint_smiles.add(similarity_hint)
        penalty = self.config.terminal_blacklist_cost_penalty if blacklist_matches else 0.0
        if blacklist_matches:
            self.state.terminal_blacklist_matches += 1

        prior = _candidate_reaction_prior_matches(
            result,
            index=index,
            preferred_reaction_classes=self.config.preferred_reaction_classes,
            preferred_retrons=self.config.preferred_retrons,
        )
        if prior["metadata_observed"]:
            self.state.reaction_prior_candidates_with_metadata += 1
            self.state.reaction_class_comparisons += len(
                self.config.preferred_reaction_classes
            )
            self.state.retron_comparisons += len(self.config.preferred_retrons)
        reaction_class_matches = list(prior["reaction_class_matches"])
        retron_matches = list(prior["retron_matches"])
        prior_reward = 0.0
        if reaction_class_matches:
            prior_reward += self.config.reaction_class_cost_reward
            self.state.reaction_class_matches += 1
            self.state.matched_reaction_classes.update(reaction_class_matches)
        if retron_matches:
            prior_reward += self.config.retron_cost_reward
            self.state.retron_matches += 1
            self.state.matched_retrons.update(retron_matches)
        # This prior is intentionally bounded and can only reorder reactions
        # already returned by ChemEnzy.  It never creates a precursor edge.
        prior_reward = min(0.5, prior_reward)

        scores = _as_list(result.get("scores"))
        costs = _as_list(result.get("costs"))
        score = _safe_score(scores[index] if index < len(scores) else None)
        base_cost = _safe_cost(costs[index] if index < len(costs) else None, score=score)
        guided_cost = max(1e-6, base_cost - reward - prior_reward + penalty)
        guided_score = max(1e-9, min(1.0, math.exp(-guided_cost)))
        if abs(guided_cost - base_cost) > 1e-9:
            self.state.score_adjusted += 1
        if prior_reward > 0.0:
            self.state.reaction_prior_score_adjusted += 1
        return {
            "index": index,
            "guided_cost": guided_cost,
            "guided_score": guided_score,
            "annotation": {
                "schema_version": "chem_enzy_guidance.candidate.v1",
                "policy_id": self.config.policy_id,
                "candidate_index": index,
                "reactants": list(components),
                "base_cost": base_cost,
                "guided_cost": guided_cost,
                "cost_adjustment": guided_cost - base_cost,
                "exact_precursor_set_match": exact_set,
                "exact_hint_components": exact_components,
                "best_hint_similarity": round(similarity, 6),
                "best_similarity_hint": similarity_hint,
                "terminal_blacklist_matches": blacklist_matches,
                "host_admission_edge_digest": admission.get("edge_digest"),
                "host_admission_accepted": admission.get("accepted") is True,
                "reaction_prior": {
                    "metadata_observed": prior["metadata_observed"],
                    "reaction_class_matches": reaction_class_matches,
                    "retron_matches": retron_matches,
                    "cost_reward": prior_reward,
                    "applied": prior_reward > 0.0,
                    "native_candidates_only": True,
                    "raw_reaction_injection": False,
                },
                "raw_reaction_injection": False,
            },
        }


def guided_policy_config_from_flags(search_flags: dict[str, Any] | None) -> ChemEnzyGuidanceConfig:
    return ChemEnzyGuidanceConfig.from_flags(search_flags)


def reset_guided_policy_state(planner: Any, target_smiles: str) -> None:
    state = getattr(planner, "_autoplanner_guided_policy_state", None)
    if isinstance(state, ChemEnzyGuidanceState):
        state.reset_for_target(str(target_smiles or ""))


def guided_policy_stats(planner: Any) -> dict[str, Any] | None:
    state = getattr(planner, "_autoplanner_guided_policy_state", None)
    if isinstance(state, ChemEnzyGuidanceState):
        return state.to_dict()
    return None


def install_canonical_ancestor_cycle_filter(mol_tree_module: Any) -> None:
    """Install the host admission gate immediately before MolTree insertion.

    The one-step wrapper normally filters the same candidates earlier.  This
    second gate is deliberately independent: cached trees, alternate vendor
    adapters, or a wrapper regression cannot insert a structurally rejected
    edge into search state.
    """
    tree_class = getattr(mol_tree_module, "MolTree", None)
    if tree_class is None:
        return
    original_prepare = getattr(
        tree_class, "_autoplanner_original_prepare_expansion", None
    )
    if original_prepare is None:
        original_prepare = getattr(tree_class, "prepare_expansion", None)
        if callable(original_prepare):
            tree_class._autoplanner_original_prepare_expansion = original_prepare
    if callable(original_prepare):

        def prepare_with_exact_multisets(
            self: Any, *args: Any, **kwargs: Any
        ) -> Any:
            prepared = original_prepare(self, *args, **kwargs)
            if not isinstance(prepared, tuple) or len(prepared) != 4:
                return prepared
            result = args[1] if len(args) > 1 else kwargs.get("result")
            if not isinstance(result, dict):
                return prepared
            raw_candidates = _as_list(result.get("reactants"))
            exact_reactants = [
                _raw_precursor_components(candidate) for candidate in raw_candidates
            ]
            current_reactants, costs, templates, annotations = prepared
            if (
                len(exact_reactants) != len(current_reactants)
                or any(not components for components in exact_reactants)
            ):
                return prepared
            # The vendor implementation historically used ``set`` here,
            # silently changing C.C into C.  Preserve the exact multiset so
            # both the insertion audit and the resulting tree describe the
            # same reaction candidate emitted by the model.
            return exact_reactants, costs, templates, annotations

        tree_class.prepare_expansion = prepare_with_exact_multisets
    original = getattr(tree_class, "_autoplanner_original_add_reaction_and_mol_nodes", None)
    if original is None:
        original = tree_class._add_reaction_and_mol_nodes
        tree_class._autoplanner_original_add_reaction_and_mol_nodes = original

    def guarded(self: Any, cost: Any, mols: Any, parent: Any, template: Any, ancestors: Any, cascade_annotation: Any = None):
        product = str(getattr(parent, "mol", "") or "")
        admission = audit_retrosynthetic_candidate(
            product,
            list(mols or []),
            forbidden_return_smiles=list(ancestors or []),
        )
        if admission.get("accepted") is not True:
            provenance = _template_provenance(template, cascade_annotation)
            rejection = retrosynthetic_admission_record(
                admission,
                stage="pre_moltree_insert",
                source=provenance["source"],
                model=provenance["model"],
                template=template,
            )
            trace = getattr(self, "cascade_expansion_trace", None)
            if isinstance(trace, list):
                trace.append(
                    {
                        "event": "cascade_expansion_hard_filtered",
                        "stage": "pre_moltree_insert",
                        "parent_mol": product,
                        "reactants": list(mols or []),
                        "reasons": list(admission.get("reasons") or []),
                        "edge_digest": admission.get("edge_digest"),
                        "source": provenance["source"],
                        "model": provenance["model"],
                        "admission_record": rejection,
                        "host_audit_authority": True,
                        "raw_reaction_injection": False,
                    }
                )
            return None
        return original(
            self,
            cost,
            mols,
            parent,
            template,
            ancestors,
            cascade_annotation=cascade_annotation,
        )

    tree_class._add_reaction_and_mol_nodes = guarded


def exclude_guided_terminal_blacklist(
    starting_mols: Any,
    *,
    state: ChemEnzyGuidanceState,
) -> Any:
    """Remove blacklisted advisory terminals from a loaded trusted stock set."""
    blacklist = set(state.config.terminal_blacklist)
    state.terminal_stock_exclusions_requested = len(blacklist)
    if not blacklist or not isinstance(starting_mols, (set, list, tuple)):
        return starting_mols
    kept = [
        smiles
        for smiles in starting_mols
        if _canonical_smiles(smiles) not in blacklist
    ]
    state.terminal_stock_exclusions_removed = len(starting_mols) - len(kept)
    if isinstance(starting_mols, set):
        return set(kept)
    if isinstance(starting_mols, tuple):
        return tuple(kept)
    return kept


def _project_result(result: dict[str, Any], rows: list[dict[str, Any]], *, candidate_count: int) -> dict[str, Any]:
    out: dict[str, Any] = {}
    indices = [row["index"] for row in rows]
    for key, value in result.items():
        sequence = _as_list(value) if isinstance(value, (list, tuple)) or hasattr(value, "tolist") else None
        if sequence is not None and len(sequence) == candidate_count:
            out[key] = [sequence[index] for index in indices]
        else:
            out[key] = value
    out["scores"] = [row["guided_score"] for row in rows]
    out["costs"] = [row["guided_cost"] for row in rows]
    out["chem_enzy_guidance"] = [row["annotation"] for row in rows]
    return out


def _host_admission_audit(
    product_smiles: str,
    reactants: tuple[str, ...],
    config: ChemEnzyGuidanceConfig,
) -> dict[str, Any]:
    return audit_retrosynthetic_candidate(
        product_smiles,
        reactants,
        policy=RetrosyntheticAdmissionPolicy(
            hard_filter_element_inventory=config.hard_filter_element_inventory,
            max_tolerated_missing_heavy_atoms=(
                config.max_tolerated_missing_heavy_atoms
            ),
            hard_filter_large_atom_jump=config.hard_filter_large_atom_jump,
            large_atom_jump_threshold=config.large_atom_jump_threshold,
            hard_filter_self_loop=config.hard_filter_self_loop,
        ),
    )


def _candidate_provenance(result: dict[str, Any], index: int) -> dict[str, Any]:
    template = _candidate_field(result, "template", index)
    model = next(
        (
            str(value)
            for key in ("model_full_name", "model_name", "source_model")
            if (value := _candidate_field(result, key, index)) not in (None, "")
        ),
        "",
    )
    source = next(
        (
            str(value)
            for key in ("source", "reaction_source")
            if (value := _candidate_field(result, key, index)) not in (None, "")
        ),
        "",
    )
    if isinstance(template, dict):
        model = model or str(
            template.get("model_full_name") or template.get("model_name") or ""
        )
        source = source or str(
            template.get("source_model") or template.get("source") or ""
        )
    return {"source": source, "model": model, "template": template}


def _template_provenance(template: Any, annotation: Any) -> dict[str, str]:
    template_row = template if isinstance(template, dict) else {}
    annotation_row = annotation if isinstance(annotation, dict) else {}
    return {
        "source": str(
            annotation_row.get("source")
            or annotation_row.get("source_model")
            or template_row.get("source")
            or template_row.get("source_model")
            or ""
        ),
        "model": str(
            annotation_row.get("model_full_name")
            or annotation_row.get("model_name")
            or template_row.get("model_full_name")
            or template_row.get("model_name")
            or ""
        ),
    }


def _candidate_reaction_prior_matches(
    result: dict[str, Any],
    *,
    index: int,
    preferred_reaction_classes: tuple[str, ...],
    preferred_retrons: tuple[str, ...],
) -> dict[str, Any]:
    values = [
        _candidate_field(result, key, index)
        for key in (
            "template",
            "model_full_name",
            "model_name",
            "source_model",
            "source",
            "reaction_class",
            "reaction_type",
            "reaction_family",
            "retron",
            "product_retron",
            "derived_from_retron",
        )
    ]
    texts: list[str] = []
    for value in values:
        texts.extend(_prior_metadata_texts(value))
    normalized_texts = tuple(
        dict.fromkeys(
            normalized
            for text in texts
            if (normalized := _normalize_prior_text(text))
        )
    )
    return {
        "metadata_observed": bool(normalized_texts),
        "reaction_class_matches": _matching_prior_hints(
            preferred_reaction_classes, normalized_texts
        ),
        "retron_matches": _matching_prior_hints(preferred_retrons, normalized_texts),
    }


def _candidate_field(result: dict[str, Any], key: str, index: int) -> Any:
    value = result.get(key)
    if isinstance(value, (list, tuple)) or hasattr(value, "tolist"):
        rows = _as_list(value)
        return rows[index] if index < len(rows) else None
    return value


def _prior_metadata_texts(value: Any, *, depth: int = 0) -> list[str]:
    if value is None or depth > 2:
        return []
    if isinstance(value, dict):
        texts: list[str] = []
        for key in (
            "template_id",
            "name",
            "model_full_name",
            "model_name",
            "source_model",
            "source",
            "reaction_class",
            "reaction_type",
            "reaction_family",
            "retron_type",
            "retron",
            "product_retron",
            "derived_from_retron",
        ):
            if key in value:
                texts.extend(_prior_metadata_texts(value.get(key), depth=depth + 1))
        return texts
    if isinstance(value, (list, tuple, set)):
        return [
            text
            for item in list(value)[:32]
            for text in _prior_metadata_texts(item, depth=depth + 1)
        ]
    text = str(value or "").strip()
    return [text[:512]] if text else []


def _matching_prior_hints(
    hints: tuple[str, ...], metadata_texts: tuple[str, ...]
) -> list[str]:
    matches: list[str] = []
    for hint in hints:
        normalized = _normalize_prior_text(hint)
        if not normalized:
            continue
        if any(
            normalized == text
            or (len(normalized) >= 4 and normalized in text)
            or (len(text) >= 4 and text in normalized)
            for text in metadata_texts
        ):
            matches.append(hint)
    return matches


def _normalize_prior_text(value: Any) -> str:
    text = str(value or "").strip().lower()
    for token in ("_", "-", "/", ":"):
        text = text.replace(token, " ")
    return " ".join(text.split())


def _best_similarity(
    components: tuple[str, ...],
    hint_fingerprints: dict[str, Any],
) -> tuple[float, str]:
    best = 0.0
    best_hint = ""
    for component in components:
        fp = _fingerprint(component)
        if fp is None:
            continue
        for hint, hint_fp in hint_fingerprints.items():
            score = float(DataStructs.TanimotoSimilarity(fp, hint_fp))
            if score > best:
                best = score
                best_hint = hint
    return best, best_hint


def _fingerprint(smiles: str) -> Any | None:
    mol = Chem.MolFromSmiles(str(smiles or ""))
    if mol is None:
        return None
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
    return generator.GetFingerprint(mol)


def _canonical_smiles(value: Any) -> str:
    mol = Chem.MolFromSmiles(str(value or "").strip())
    if mol is None:
        return ""
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(mol, isomericSmiles=True)


def _canonical_components(value: Any) -> tuple[str, ...]:
    if isinstance(value, (list, tuple)):
        raw = [
            fragment
            for item in value
            for fragment in str(item or "").split(".")
        ]
    else:
        raw = str(value or "").split(".")
    canonical = [_canonical_smiles(item) for item in raw if str(item or "").strip()]
    if not canonical or any(not item for item in canonical):
        return ()
    # Preserve repeated components.  Callers that only need identity
    # membership (self-loop and individual hint matching) explicitly build a
    # set, while exact precursor-set matching and element inventory use this
    # tuple as a multiset via ``Counter``.
    return tuple(sorted(canonical))


def _raw_precursor_components(value: Any) -> list[str]:
    values = value if isinstance(value, (list, tuple)) else [value]
    return [
        fragment.strip()
        for item in values
        for fragment in str(item or "").split(".")
        if fragment.strip()
    ]


def _canonical_smiles_tuple(values: Any) -> tuple[str, ...]:
    out: set[str] = set()
    for value in values if isinstance(values, (list, tuple, set)) else [values]:
        components = _canonical_components(value)
        out.update(components)
    return tuple(sorted(out))


def _canonical_precursor_sets(values: Any) -> tuple[tuple[str, ...], ...]:
    rows: set[tuple[str, ...]] = set()
    for value in values if isinstance(values, (list, tuple, set)) else [values]:
        components = _canonical_components(value)
        if components:
            rows.add(components)
    return tuple(sorted(rows))


def _bounded_hint_tuple(values: Any) -> tuple[str, ...]:
    raw_values = values if isinstance(values, (list, tuple, set)) else [values]
    hints = {
        str(value or "").strip()[:128]
        for value in list(raw_values)[:64]
        if str(value or "").strip()
    }
    return tuple(sorted(hints)[:32])


def _element_counts(smiles: str) -> Counter[str]:
    mol = Chem.MolFromSmiles(smiles or "")
    if mol is None:
        return Counter()
    return Counter(atom.GetSymbol() for atom in mol.GetAtoms() if atom.GetAtomicNum() != 1)


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if hasattr(value, "tolist"):
        converted = value.tolist()
        return converted if isinstance(converted, list) else [converted]
    return [value]


def _safe_score(value: Any) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        score = 1e-3
    if not math.isfinite(score):
        score = 1e-3
    return max(1e-9, min(1.0, score))


def _safe_cost(value: Any, *, score: float) -> float:
    try:
        cost = float(value)
    except (TypeError, ValueError):
        cost = -math.log(max(score, 1e-9))
    if not math.isfinite(cost) or cost < 0.0:
        cost = -math.log(max(score, 1e-9))
    return cost


def _float(value: Any, default: float, *, lo: float, hi: float | None = None) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        out = float(default)
    out = max(lo, out)
    return min(out, hi) if hi is not None else out


def _int(value: Any, default: int, *, lo: int) -> int:
    try:
        out = int(value)
    except (TypeError, ValueError):
        out = int(default)
    return max(lo, out)
