"""Frozen checkpoint-backed cascade value adapter."""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from cascade_planner.cascade_search.state import CascadeProgramState, ConditionEnvelope
from cascade_planner.cascade_search.value import (
    CascadeValuePrediction,
    HeuristicCascadeValueModel,
)


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _mean(values: list[float | None]) -> float:
    clean = [float(value) for value in values if value is not None]
    return sum(clean) / len(clean) if clean else 0.0


def _span(values: list[float | None]) -> float:
    clean = [float(value) for value in values if value is not None]
    return max(clean) - min(clean) if clean else 0.0


def _norm(value: Any) -> str:
    return str(value or "").strip().lower()


class LearnedCascadeValueModel:
    """Torch checkpoint-backed value model for CascadeProgramState objects."""

    is_learned_value_model = True

    def __init__(self, checkpoint_path: str | Path, *, device: str = "cpu"):
        import numpy as np
        import torch
        import torch.nn as nn
        from rdkit import Chem, DataStructs, RDLogger
        from rdkit.Chem import AllChem

        RDLogger.DisableLog("rdApp.*")
        self._np = np
        self._torch = torch
        self._nn = nn
        self._Chem = Chem
        self._DataStructs = DataStructs
        self._AllChem = AllChem
        self.checkpoint_path = str(checkpoint_path)
        self.device = torch.device(device)
        checkpoint = torch.load(str(checkpoint_path), map_location=self.device)
        self.feature_schema = dict(checkpoint.get("feature_schema") or {})
        self.binary_labels = list(
            checkpoint.get("binary_labels")
            or self.feature_schema.get("binary_labels")
            or []
        )
        hidden = int(checkpoint.get("hidden") or 192)
        in_dim = int(self.feature_schema.get("feature_dim") or 0)
        if in_dim <= 0:
            raise ValueError(
                f"invalid learned cascade value checkpoint feature_dim: {checkpoint_path}"
            )
        self.model = _RuntimeCascadeValueNetwork(
            self._nn,
            in_dim,
            hidden=hidden,
            n_binary=len(self.binary_labels),
        ).to(self.device)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()

    def predict(self, state: CascadeProgramState) -> CascadeValuePrediction:
        x = self._state_features(state)
        with self._torch.no_grad():
            tensor = self._torch.tensor(
                x[None, :],
                dtype=self._torch.float32,
                device=self.device,
            )
            out = self.model(tensor)
            probs = (
                self._torch.sigmoid(out["binary_logits"])[0]
                .detach()
                .cpu()
                .numpy()
                .tolist()
            )
            depth = float(out["depth"][0].detach().cpu().item()) * 8.0
        values = {
            label: float(probs[idx]) for idx, label in enumerate(self.binary_labels)
        }
        heuristic = HeuristicCascadeValueModel().predict(state)
        p_stock = values.get("p_stock_closed", heuristic.p_stock_closed)
        p_condition = values.get(
            "p_condition_compatible", heuristic.p_condition_compatible
        )
        p_cofactor = values.get("p_cofactor_closed", heuristic.p_cofactor_closed)
        p_enzyme = values.get(
            "p_enzyme_evidence_valid", heuristic.p_enzyme_evidence_valid
        )
        p_gt = values.get("p_gt_like_cascade", heuristic.p_gt_like_cascade)
        p_stage_transition = values.get("p_stage_transition_needed")
        value = (
            0.22 * p_stock
            + 0.22 * p_condition
            + 0.18 * p_cofactor
            + 0.13 * p_enzyme
            + 0.25 * p_gt
        )
        if p_stage_transition is not None and state.stage_graph.n_stages <= 1:
            value -= 0.05 * p_stage_transition
        return CascadeValuePrediction(
            p_stock_closed=_clip01(p_stock),
            p_condition_compatible=_clip01(p_condition),
            p_cofactor_closed=_clip01(p_cofactor),
            p_enzyme_evidence_valid=_clip01(p_enzyme),
            p_gt_like_cascade=_clip01(p_gt),
            expected_remaining_depth=max(0.0, depth),
            value=_clip01(value),
            metadata={
                "model_family": "learned_cascade_value",
                "checkpoint_path": self.checkpoint_path,
                "raw_probabilities": values,
                "p_stage_transition_needed": p_stage_transition,
            },
        )

    def _state_features(self, state: CascadeProgramState) -> Any:
        schema = self.feature_schema
        n_bits = int(schema.get("n_bits") or 128)
        steps = list(state.step_annotations or [])
        target_fp = self._fp_many([state.target_smiles], n_bits=n_bits)
        open_leaves = list(state.open_molecule_leaves or state.open_leaves or [])
        start_fp = self._fp_many(open_leaves or [state.target_smiles], n_bits=n_bits)
        total_steps = max(1.0, float(len(steps) or 1))
        rxn_steps = float(sum(1 for step in steps if step.rxn_smiles))
        reactant_count = float(sum(len(step.reactant_smiles or []) for step in steps))
        product_count = float(
            len({step.product_smiles for step in steps if step.product_smiles})
        )
        catalyst_tokens = self._catalyst_classes(state)
        catalyst_count = max(1.0, float(len(catalyst_tokens)))
        conditions = [step.condition for step in steps if step.condition is not None]
        temp_values = [
            value
            for env in conditions
            for value in (env.temperature_c_min, env.temperature_c_max)
            if value is not None
        ]
        ph_values = [
            value
            for env in conditions
            for value in (env.ph_min, env.ph_max)
            if value is not None
        ]
        enzyme_count = float(sum(1 for step in steps if step.is_enzymatic))
        identifier_count = float(
            sum(1 for step in steps if step.ec_numbers or step.uniprot_ids)
        )
        cofactor_count = float(
            sum(1 for step in steps if step.all_cofactor_requirements())
        )
        condition_count = float(
            sum(self._condition_field_count(env) for env in conditions)
        )
        evidence_values = [
            step.evidence_confidence
            for step in steps
            if step.evidence_confidence is not None
        ]
        failures = {failure.category for failure in state.unresolved_failure_modes or []}
        unclosed = state.cofactor_ledger.unclosed_requirements()
        scalar = self._np.asarray(
            [
                total_steps / 8.0,
                rxn_steps / total_steps,
                catalyst_count / max(total_steps, 1.0) / 4.0,
                reactant_count / 16.0,
                product_count / 16.0,
                0.0,
                condition_count / 12.0,
                0.0,
                float(state.stock_closed),
                float(bool(conditions)),
                float(bool(state.unresolved_failure_modes)),
                0.0,
                _mean(evidence_values),
                enzyme_count / max(catalyst_count, 1.0),
                identifier_count / max(catalyst_count, 1.0),
                cofactor_count / max(catalyst_count, 1.0),
                _mean(temp_values) / 100.0,
                _span(temp_values) / 100.0,
                _mean(ph_values) / 14.0,
                _span(ph_values) / 14.0,
                0.0,
                float(bool(unclosed)),
                float("ConditionConflict" in failures),
                0.0,
                0.0,
            ],
            dtype=self._np.float32,
        )
        bags = [
            self._bag([self._cascade_type(state)], schema.get("cascade_type_vocab") or []),
            self._bag(
                self._transformations(state),
                schema.get("transformation_vocab") or [],
                denom=total_steps,
            ),
            self._bag(
                self._step_modes(state),
                schema.get("step_mode_vocab") or [],
                denom=total_steps,
            ),
            self._bag(
                catalyst_tokens,
                schema.get("catalyst_class_vocab") or [],
                denom=catalyst_count,
            ),
            self._bag(
                self._ec1_values(state),
                schema.get("ec1_vocab") or [],
                denom=catalyst_count,
            ),
            self._bag(
                self._solvents(state),
                schema.get("solvent_vocab") or [],
                denom=total_steps,
            ),
            self._bag(
                self._compatibility_tokens(state),
                schema.get("compatibility_vocab") or [],
            ),
        ]
        features = self._np.concatenate([target_fp, start_fp, scalar, *bags]).astype(
            self._np.float32
        )
        expected = int(schema.get("feature_dim") or len(features))
        if len(features) != expected:
            raise ValueError(
                "learned cascade value feature dimension mismatch: "
                f"got {len(features)}, expected {expected}"
            )
        return features

    def _fp_many(self, smiles_values: list[str], *, n_bits: int) -> Any:
        arr = self._np.zeros(n_bits, dtype=self._np.float32)
        for smiles in smiles_values:
            mol = self._mol_from_smiles(smiles)
            if mol is None:
                continue
            bv = self._AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=n_bits)
            tmp = self._np.zeros(n_bits, dtype=self._np.float32)
            self._DataStructs.ConvertToNumpyArray(bv, tmp)
            arr = self._np.maximum(arr, tmp)
        return arr

    def _mol_from_smiles(self, smiles: str | None) -> Any:
        text = str(smiles or "")
        chunks = sorted(
            [part.strip() for part in text.replace(";", ".").split(".") if part.strip()],
            key=len,
            reverse=True,
        )
        for chunk in chunks:
            mol = self._Chem.MolFromSmiles(chunk)
            if mol is not None:
                return mol
        return None

    def _bag(self, values: list[str], vocab: list[str], *, denom: float = 1.0) -> Any:
        counts = Counter(_norm(value) for value in values if _norm(value))
        return self._np.asarray(
            [counts.get(token, 0) / max(float(denom), 1.0) for token in vocab],
            dtype=self._np.float32,
        )

    def _cascade_type(self, state: CascadeProgramState) -> str:
        steps = list(state.step_annotations or [])
        if not steps:
            return ""
        enzymatic = [step.is_enzymatic for step in steps]
        if all(enzymatic):
            return "all_enzymatic"
        if any(enzymatic):
            return "chemoenzymatic"
        return "all_chemical"

    def _transformations(self, state: CascadeProgramState) -> list[str]:
        return [
            _norm(step.reaction_type)
            for step in state.step_annotations
            if _norm(step.reaction_type)
        ]

    def _step_modes(self, state: CascadeProgramState) -> list[str]:
        if state.stage_graph.n_stages > 1:
            return ["sequential_addition"] * max(1, len(state.step_annotations))
        return ["charged_at_t0"] * max(1, len(state.step_annotations))

    def _catalyst_classes(self, state: CascadeProgramState) -> list[str]:
        out = []
        for step in state.step_annotations:
            out.append("enzyme" if step.is_enzymatic else "unknown")
        return out or ["unknown"]

    def _ec1_values(self, state: CascadeProgramState) -> list[str]:
        out = []
        for step in state.step_annotations:
            for ec in step.ec_numbers:
                token = str(ec or "").split(".", 1)[0].strip().lower()
                if token:
                    out.append(token)
        return out

    def _solvents(self, state: CascadeProgramState) -> list[str]:
        out = []
        for step in state.step_annotations:
            if step.condition is not None:
                out.extend(
                    _norm(value)
                    for value in step.condition.solvents
                    if _norm(value)
                )
        return out

    def _compatibility_tokens(self, state: CascadeProgramState) -> list[str]:
        out = [failure.category for failure in state.unresolved_failure_modes or []]
        if state.stage_graph.n_stages > 1:
            out.append("sequential_addition")
        if not state.cofactor_ledger.unclosed_requirements():
            out.append("empirically_compatible")
        return [_norm(value) for value in out if _norm(value)]

    @staticmethod
    def _condition_field_count(env: ConditionEnvelope) -> int:
        return sum(
            int(bool(value))
            for value in (
                env.temperature_c_min,
                env.temperature_c_max,
                env.ph_min,
                env.ph_max,
                env.solvents,
                env.catalysts,
                env.buffer,
                env.cofactors,
            )
        )


class _RuntimeCascadeValueNetwork:
    def __init__(self, nn: Any, in_dim: int, *, hidden: int, n_binary: int):
        class Net(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                h2 = max(48, hidden // 2)
                self.backbone = nn.Sequential(
                    nn.Linear(in_dim, hidden),
                    nn.GELU(),
                    nn.Dropout(0.12),
                    nn.Linear(hidden, hidden),
                    nn.GELU(),
                    nn.Dropout(0.08),
                    nn.Linear(hidden, h2),
                    nn.GELU(),
                )
                self.binary_head = nn.Linear(h2, n_binary)
                self.depth_head = nn.Linear(h2, 1)

            def forward(self, x: Any) -> dict[str, Any]:
                h = self.backbone(x)
                return {
                    "binary_logits": self.binary_head(h),
                    "depth": self.depth_head(h).squeeze(-1),
                }

        self._model = Net()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._model, name)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._model(*args, **kwargs)


__all__ = ["LearnedCascadeValueModel"]
