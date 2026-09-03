"""Bind Director worker records to the isolated AiZ ReactionJSON policy.

This adapter deliberately consumes the existing worker artifact and the
existing ``RouteJSONCompiler`` path.  It does not introduce a second parser or
permit model-declared precursor SMILES to become structural authority.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

from rdkit import Chem

from cascade_planner.agent.codex_worker import WorkerRunRecord
from cascade_planner.application.routejson_compiler import RouteJSONCompiler
from cascade_planner.interfaces.aizynthfinder_reactionjson_expansion import (
    ReactionJsonExpansionCandidate,
    ReactionJsonExpansionRequest,
)
from cascade_planner.orchestration.sequential_strategy_director import (
    _reactionjson_candidates_from_record,
    _step_row,
)


WorkerRecordProvider = Callable[[ReactionJsonExpansionRequest], WorkerRunRecord]


@dataclass(frozen=True, slots=True)
class WorkerRecordAdapterDiagnostic:
    call_index: int
    product_smiles: str
    compiled_candidates: int
    rejected_candidates: tuple[Mapping[str, Any], ...]


class WorkerRecordReactionJsonCandidateProvider:
    """Compile one Director response into AiZ-ready reaction candidates."""

    def __init__(
        self,
        record_provider: WorkerRecordProvider,
        *,
        max_candidates: int = 1,
        compiler: RouteJSONCompiler | None = None,
    ) -> None:
        if not callable(record_provider):
            raise TypeError("worker record_provider must be callable")
        if max_candidates < 1:
            raise ValueError("worker record candidate limit must be positive")
        self.record_provider = record_provider
        self.max_candidates = int(max_candidates)
        self.compiler = compiler or RouteJSONCompiler()
        self.records: list[WorkerRunRecord] = []
        self.diagnostics: list[WorkerRecordAdapterDiagnostic] = []

    def __call__(
        self, request: ReactionJsonExpansionRequest
    ) -> list[ReactionJsonExpansionCandidate]:
        record = self.record_provider(request)
        if not isinstance(record, WorkerRunRecord):
            raise TypeError("record_provider did not return WorkerRunRecord")
        self.records.append(record)

        selected, selected_mapped = _selected_product_from_record(record, request)
        compiled, rejected = _reactionjson_candidates_from_record(
            record,
            expected_product=selected,
            mapped_product_smiles=selected_mapped,
            require_reaction_operations=True,
            compiler=self.compiler,
            max_candidates=self.max_candidates,
        )
        self.diagnostics.append(
            WorkerRecordAdapterDiagnostic(
                call_index=request.call_index,
                product_smiles=selected,
                compiled_candidates=len(compiled),
                rejected_candidates=tuple(dict(row) for row in rejected),
            )
        )

        result: list[ReactionJsonExpansionCandidate] = []
        for item in compiled:
            expansion = item.expansion
            step = _step_row(
                expansion,
                step_id=(
                    expansion.step_id
                    or (
                        f"aiz:strategy:{request.strategy_id}:call:{request.call_index}:"
                        f"candidate:{item.candidate_index + 1}"
                    )
                ),
                strategy_anchor=request.depth == 0,
            )
            result.append(
                ReactionJsonExpansionCandidate(
                    candidate_id=item.candidate_id,
                    product_smiles=selected,
                    mapped_product_smiles=selected_mapped,
                    precursor_smiles=tuple(expansion.precursor_smiles),
                    mapped_precursor_smiles=tuple(
                        expansion.mapped_precursor_smiles
                    ),
                    route_step=step,
                    prior=item.score,
                    candidate_key=item.candidate_key,
                )
            )
        return result


def _selected_product_from_record(
    record: WorkerRunRecord,
    request: ReactionJsonExpansionRequest,
) -> tuple[str, str]:
    payload = dict(dict(record.output_artifact or {}).get("payload") or {})
    raw_products = {
        _canonical_smiles(row.get("product_smiles"))
        for row in payload.get("candidates") or ()
        if isinstance(row, Mapping) and _canonical_smiles(row.get("product_smiles"))
    }
    expandable = {
        _canonical_smiles(smiles): (_canonical_smiles(smiles), mapped)
        for smiles, mapped in zip(
            request.expandable_smiles,
            request.expandable_mapped_smiles,
        )
        if _canonical_smiles(smiles) and str(mapped or "")
    }
    if not raw_products and len(expandable) == 1:
        return next(iter(expandable.values()))
    matches = [expandable[product] for product in raw_products if product in expandable]
    if len(matches) != 1:
        raise ValueError(
            "worker record must bind every candidate to one expandable AiZ molecule"
        )
    return matches[0]


def _canonical_smiles(value: Any) -> str:
    molecule = Chem.MolFromSmiles(str(value or "").strip())
    if molecule is None:
        return ""
    for atom in molecule.GetAtoms():
        atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)
