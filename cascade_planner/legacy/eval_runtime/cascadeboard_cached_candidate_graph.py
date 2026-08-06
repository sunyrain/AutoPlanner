"""Strict candidate-graph adapter for persisted proposal caches."""
from __future__ import annotations

from cascade_planner.cascadeboard.candidate_cache import canon_smiles
from cascade_planner.legacy.eval_runtime.cascadeboard_candidate_graph import (
    CandidateHypergraph,
    CandidateReaction,
)


class StrictCachedGraph(CandidateHypergraph):
    """Candidate graph backed only by cache entries; no mock fallback."""

    def __init__(self, cache: dict[str, list[dict]], **kwargs):
        super().__init__(**kwargs)
        self._cache = cache
        self.cache_misses: set[str] = set()

    def _get_candidates(self, product, depth, compiled):
        product_key = canon_smiles(product) or product
        rows = self._cache.get(product_key) or self._cache.get(product) or []
        if not rows:
            self.cache_misses.add(product)
            return []
        candidates = []
        for row in rows:
            candidates.append(
                CandidateReaction(
                    product=row.get("product") or product,
                    main_reactant=row.get("main_reactant", ""),
                    aux_reactants=row.get("aux_reactants", []),
                    reaction_smiles=row.get("reaction_smiles") or row.get("rxn_smiles", ""),
                    reaction_type=row.get("reaction_type") or row.get("type", ""),
                    ec=row.get("ec"),
                    enzyme_uid=row.get("enzyme_uid"),
                    score=float(row.get("score", 0.0)),
                    source=row.get("source", "cache"),
                    metadata={
                        key: row[key]
                        for key in (
                            "rank", "T", "pH", "solvent", "e_enzyme",
                            "dual_tower_score", "enzyme_source",
                        )
                        if key in row
                    },
                )
            )
        return sorted(candidates, key=lambda candidate: -candidate.score)


__all__ = ["StrictCachedGraph"]
