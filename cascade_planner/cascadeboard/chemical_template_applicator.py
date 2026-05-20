"""USPTO chemical template rescue for recall-oriented candidate generation."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, TYPE_CHECKING

from rdkit import Chem

from cascade_planner.cascadeboard.retrorules_applicator import RetroRulesApplicator

if TYPE_CHECKING:
    from cascade_planner.cascadeboard.chemical_template_pair_ranker import ChemicalTemplatePairRanker
    from cascade_planner.cascadeboard.chemical_template_preselector import ChemicalTemplatePreselector


DEFAULT_USPTO_TEMPLATE_PATHS = (
    Path("data_external/retrorules/templates_uspto.csv.gz"),
)


class ChemicalTemplateApplicator(RetroRulesApplicator):
    """Apply USPTO retrosynthesis templates as a chemical fallback source."""

    def __init__(
        self,
        *args,
        preselector: ChemicalTemplatePreselector | None = None,
        pair_ranker: ChemicalTemplatePairRanker | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.preselector = preselector
        self.pair_ranker = pair_ranker

    def predict(
        self,
        product_smiles: str,
        top_k: int = 10,
        *,
        ec_token: str = "",
        skel_type: str = "",
    ) -> list[dict[str, Any]]:
        rows = super().predict(product_smiles, top_k=top_k, ec_token=ec_token, skel_type=skel_type)
        for row in rows:
            row["source"] = "uspto_template"
            row.setdefault("type", skel_type or "chemical_template")
            row.setdefault("reaction_type", row.get("type"))
            evidence = dict(row.get("evidence") or {})
            evidence["template_family"] = "uspto"
            row["evidence"] = evidence
        return rows

    def _rank_templates(self, templates, product_mol: Chem.Mol, *, ec_token: str = "", skel_type: str = ""):
        ranked = super()._rank_templates(templates, product_mol, ec_token=ec_token, skel_type=skel_type)
        product_smiles = Chem.MolToSmiles(product_mol)
        if self.pair_ranker and self.pair_ranker.available:
            pair_scores = self.pair_ranker.score_templates(product_smiles, ranked)
            if pair_scores:
                base_rank = {id(row): idx for idx, row in enumerate(ranked)}
                weight = _env_float("AUTOPLANNER_CHEM_TEMPLATE_PAIR_RANKER_WEIGHT", 25.0)
                return sorted(
                    ranked,
                    key=lambda row: -float(base_rank.get(id(row), 0)) + weight * pair_scores.get(row.template_id, 0.0),
                    reverse=True,
                )
        if not self.preselector or not self.preselector.available:
            return ranked
        scores = self.preselector.score_template_ids(product_smiles, [row.template_id for row in ranked])
        if not scores:
            return ranked
        base_rank = {id(row): idx for idx, row in enumerate(ranked)}
        return sorted(
            ranked,
            key=lambda row: (
                scores.get(row.template_id, 0.0),
                1.0 if row.template_id in scores else 0.0,
                -float(base_rank.get(id(row), 0)),
            ),
            reverse=True,
        )

    @classmethod
    def from_env(cls) -> "ChemicalTemplateApplicator":
        from cascade_planner.cascadeboard.chemical_template_pair_ranker import (
            ChemicalTemplatePairRanker,
            pair_ranker_enabled,
        )
        from cascade_planner.cascadeboard.chemical_template_preselector import (
            ChemicalTemplatePreselector,
            preselector_enabled,
        )

        raw_paths = os.environ.get("AUTOPLANNER_CHEM_TEMPLATES_PATHS", "")
        paths = [Path(p) for p in raw_paths.split(os.pathsep) if p] if raw_paths else DEFAULT_USPTO_TEMPLATE_PATHS
        pair_ranker = ChemicalTemplatePairRanker.from_env() if pair_ranker_enabled() else None
        preselector = (
            ChemicalTemplatePreselector.from_env()
            if not (pair_ranker and pair_ranker.available) and preselector_enabled()
            else None
        )
        return cls(
            paths,
            max_templates=_env_int("AUTOPLANNER_CHEM_TEMPLATES_MAX_TEMPLATES", 20000),
            max_per_ec1=_env_int("AUTOPLANNER_CHEM_TEMPLATES_MAX_PER_EC1", 20000),
            max_templates_per_query=_env_int("AUTOPLANNER_CHEM_TEMPLATES_MAX_PER_QUERY", 500),
            max_outcomes_per_template=_env_int("AUTOPLANNER_CHEM_TEMPLATES_OUTCOMES_PER_TEMPLATE", 1),
            generalize=_env_int("AUTOPLANNER_CHEM_TEMPLATES_GENERALIZE", 0),
            preselector=preselector,
            pair_ranker=pair_ranker,
        )


def chemical_templates_enabled() -> bool:
    raw = os.environ.get("AUTOPLANNER_ENABLE_CHEM_TEMPLATES")
    if raw is None:
        return any(path.exists() for path in DEFAULT_USPTO_TEMPLATE_PATHS)
    return str(raw).lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default
