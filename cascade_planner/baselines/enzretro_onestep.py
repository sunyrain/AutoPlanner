"""EnzRetro SSREdits one-step proposal provider for AutoPlanner.

This provider adapts the local EnzRetro reproduction package to AutoPlanner's
dict-based one-step proposal interface.  It is intentionally optional and lazy:
the heavy PyTorch checkpoint is loaded only when predict() is called.
"""
from __future__ import annotations

import argparse
import inspect
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from rdkit import Chem

from cascade_planner.baselines.proposal_gate import evaluate_step_candidate
from cascade_planner.cascadeboard.route_recovery import canonical_smiles


DEFAULT_ENZRETRO_PACKAGE_ROOT = Path(__file__).resolve().parents[3] / "enzretro_model_code_package"
ENZRETRO_PACKAGE_ROOT_ENV = "AUTOPLANNER_ENZRETRO_PACKAGE_ROOT"
ENZRETRO_MODEL_DIR_ENV = "AUTOPLANNER_ENZRETRO_MODEL_DIR"
ENZRETRO_CHECKPOINT_ENV = "AUTOPLANNER_ENZRETRO_CHECKPOINT"
ENZRETRO_VOCAB_FILE_ENV = "AUTOPLANNER_ENZRETRO_VOCAB_FILE"
ENZRETRO_DEVICE_ENV = "AUTOPLANNER_ENZRETRO_DEVICE"
ENZRETRO_BEAM_SIZE_ENV = "AUTOPLANNER_ENZRETRO_BEAM_SIZE"
ENZRETRO_RETURN_TOPK_ENV = "AUTOPLANNER_ENZRETRO_RETURN_TOPK"
ENZRETRO_DEDUPE_SUBSTRATE_ENV = "AUTOPLANNER_ENZRETRO_DEDUPE_SUBSTRATE"
ENZRETRO_LIPID_FILTER_ENV = "AUTOPLANNER_ENZRETRO_LIPID_FILTER"
ENZRETRO_REQUIRE_EXECUTE_ENV = "AUTOPLANNER_ENZRETRO_REQUIRE_EXECUTE"
ENZRETRO_CHEMISTRY_CONSTRAINTS_ENV = "AUTOPLANNER_ENZRETRO_CHEMISTRY_CONSTRAINTS"


@dataclass
class EnzRetroOneStepProposalProvider:
    """Expose EnzRetro product -> substrates + EC predictions as one-step rows."""

    package_root: Path | str = DEFAULT_ENZRETRO_PACKAGE_ROOT
    model_dir: Path | str | None = None
    checkpoint: str = "pytorch_model.pth"
    vocab_file: Path | str | None = None
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    beam_size: int = 1
    return_topk: int = 1
    decode_max_len: int = 180
    length_penalty: float = 0.7
    lipid_filter: bool = True
    require_execute: bool = True
    chemistry_constraints: bool = False
    dedupe_substrate: bool = False
    ec_level: int = 4
    _loaded: bool = field(default=False, init=False, repr=False)
    _model: Any | None = field(default=None, init=False, repr=False)
    _tokenizer: Any | None = field(default=None, init=False, repr=False)
    _device: torch.device | None = field(default=None, init=False, repr=False)
    _model_args: argparse.Namespace | None = field(default=None, init=False, repr=False)
    load_error: str = ""

    provider_name = "enzretro_ssredits"

    @classmethod
    def from_env(cls) -> "EnzRetroOneStepProposalProvider":
        package_root = Path(os.environ.get(ENZRETRO_PACKAGE_ROOT_ENV) or DEFAULT_ENZRETRO_PACKAGE_ROOT)
        return cls(
            package_root=package_root,
            model_dir=os.environ.get(ENZRETRO_MODEL_DIR_ENV) or None,
            checkpoint=os.environ.get(ENZRETRO_CHECKPOINT_ENV) or "pytorch_model.pth",
            vocab_file=os.environ.get(ENZRETRO_VOCAB_FILE_ENV) or None,
            device=os.environ.get(ENZRETRO_DEVICE_ENV) or ("cuda" if torch.cuda.is_available() else "cpu"),
            beam_size=_env_int(ENZRETRO_BEAM_SIZE_ENV, 1),
            return_topk=_env_int(ENZRETRO_RETURN_TOPK_ENV, 1),
            dedupe_substrate=_env_bool(ENZRETRO_DEDUPE_SUBSTRATE_ENV, True),
            lipid_filter=_env_bool(ENZRETRO_LIPID_FILTER_ENV, True),
            require_execute=_env_bool(ENZRETRO_REQUIRE_EXECUTE_ENV, True),
            chemistry_constraints=_env_bool(ENZRETRO_CHEMISTRY_CONSTRAINTS_ENV, False),
        )

    @property
    def available(self) -> bool:
        package_root = Path(self.package_root)
        model_dir = self._model_dir()
        vocab_file = self._vocab_file()
        return (
            package_root.exists()
            and (package_root / "scripts").exists()
            and (package_root / "enzretro").exists()
            and model_dir.exists()
            and (model_dir / self.checkpoint).exists()
            and vocab_file.exists()
        )

    def predict(self, product_smiles: str, top_k: int = 10, **_: Any) -> list[dict[str, Any]]:
        if not product_smiles:
            return []
        try:
            self._ensure_loaded()
            candidates = self._predict_candidates(product_smiles, top_k=max(1, int(top_k or 1)))
        except Exception as exc:
            self.load_error = f"{type(exc).__name__}:{exc}"
            return []
        rows: list[dict[str, Any]] = []
        for rank, candidate in enumerate(candidates, start=1):
            substrate = str(candidate.get("predicted_substrates") or "")
            reactants = [part for part in substrate.split(".") if part]
            if not reactants:
                continue
            main = _select_main_reactant(reactants, product_smiles)
            main_key = canonical_smiles(main) or main
            aux = [smi for smi in reactants if (canonical_smiles(smi) or smi) != main_key]
            rxn_smiles = ".".join(reactants) + f">>{product_smiles}"
            ec = str(candidate.get("predicted_ec") or "")
            source = self.provider_name
            proposal_gate = evaluate_step_candidate(
                product_smiles=product_smiles,
                reactant_smiles=reactants,
                rxn_smiles=rxn_smiles,
                source_model=source,
            )
            rows.append(
                {
                    "main_reactant": main,
                    "aux_reactants": aux,
                    "reactant_smiles": reactants,
                    "rxn_smiles": rxn_smiles,
                    "reaction_smiles": rxn_smiles,
                    "source": source,
                    "source_model": source,
                    "score": _safe_float(candidate.get("score"), 1.0 / rank),
                    "rank": rank,
                    "candidate_count": len(candidates),
                    "type": "enzymatic_ssredits",
                    "proposal_type": "enzretro_ssredits",
                    "model_full_name": "enzretro.backward_ecs4",
                    "ec": ec,
                    "enzyme_ec_annotations": [{"ec_number": ec, "source": source}] if ec else [],
                    "teacher_one_step": True,
                    "teacher_source": source,
                    "proposal_gate": proposal_gate,
                    "execute_ok": bool(candidate.get("execute_ok")),
                    "predicted_ssredits": candidate.get("predicted_ssredits"),
                    "logprob": candidate.get("logprob"),
                    "lipid_penalty": candidate.get("lipid_penalty"),
                    "raw_backend_metadata": {
                        "enzretro": {
                            "package_root": str(self.package_root),
                            "model_dir": str(self._model_dir()),
                            "checkpoint": self.checkpoint,
                            "beam_size": self.beam_size,
                            "return_topk": self.return_topk,
                            "chemistry_constraints": self.chemistry_constraints,
                        }
                    },
                }
            )
            if len(rows) >= max(1, int(top_k or 1)):
                break
        return rows

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        package_root = Path(self.package_root).resolve()
        scripts_root = package_root / "scripts"
        enzretro_root = package_root / "enzretro"
        sys.path.insert(0, str(scripts_root))
        sys.path.insert(0, str(enzretro_root))
        from evaluate_enzretro_retro_pipeline import load_model_args
        from train_enzretro_ssredits import AtomTokenizer, build_model

        args = argparse.Namespace(
            device=self.device,
            max_length=256,
            d_model=256,
            layers=6,
            heads=8,
            qkv_dim=32,
            ffn_dim=2048,
            dropout=0.1,
            relative_buckets=32,
            relative_max_distance=128,
        )
        model_dir = self._model_dir()
        load_model_args(model_dir, args)
        tokenizer = AtomTokenizer(self._vocab_file())
        device = torch.device(args.device)
        model = build_model(args, tokenizer.vocab_size, tokenizer.pad_id, tokenizer.bos_id, tokenizer.eos_id).to(device)
        state = torch.load(model_dir / self.checkpoint, map_location=device)
        state_dict = state["model_state"] if isinstance(state, dict) and "model_state" in state else state
        model.load_state_dict(state_dict)
        model.eval()
        self._model = model
        self._tokenizer = tokenizer
        self._device = device
        self._model_args = args
        self._loaded = True

    @torch.no_grad()
    def _predict_candidates(self, product_smiles: str, *, top_k: int) -> list[dict[str, Any]]:
        from evaluate_enzretro_beam_rerank import (
            beam_decode_single,
            dedupe_by_substrate,
            enrich_candidate,
            score_candidate,
        )
        from evaluate_enzretro_retro_pipeline import execute_edits, extract_ec
        from evaluate_enzretro_ssredits import greedy_decode

        assert self._model is not None
        assert self._tokenizer is not None
        assert self._device is not None
        assert self._model_args is not None

        tokenizer = self._tokenizer
        max_length = int(getattr(self._model_args, "max_length", 256))
        source = f"[EC][Backward]{product_smiles}[EC]{self.ec_level}"
        ids, mask = tokenizer.encode(source, max_length)
        input_ids = torch.tensor([ids], dtype=torch.long, device=self._device)
        input_mask = torch.tensor([mask], dtype=torch.long, device=self._device)
        if self.beam_size <= 1:
            pred_ids = greedy_decode(self._model, input_ids, input_mask, tokenizer, max_length)[0].cpu()
            ssredits = tokenizer.decode(pred_ids)
            try:
                substrates = execute_edits(product_smiles, ssredits)
                execute_ok = bool(substrates)
                error = ""
            except Exception as exc:
                substrates = ""
                execute_ok = False
                error = str(exc)
            return [
                {
                    "predicted_ec": extract_ec(ssredits),
                    "predicted_substrates": substrates,
                    "predicted_ssredits": ssredits,
                    "execute_ok": execute_ok,
                    "score": 1.0,
                    "logprob": None,
                    "error": error,
                }
            ]

        raw_candidates = beam_decode_single(
            self._model,
            input_ids,
            input_mask,
            tokenizer,
            beam_size=max(1, int(self.beam_size)),
            return_topk=max(top_k, int(self.return_topk or top_k)),
            max_len=int(self.decode_max_len),
            length_penalty=float(self.length_penalty),
        )
        enriched = [enrich_candidate(candidate, tokenizer, product_smiles) for candidate in raw_candidates]
        score_signature = inspect.signature(score_candidate)
        supports_chemistry_constraints = "use_chemistry_constraints" in score_signature.parameters
        for candidate in enriched:
            score_kwargs: dict[str, Any] = {
                "require_execute": self.require_execute,
                "use_lipid_filter": self.lipid_filter,
            }
            if supports_chemistry_constraints:
                score_kwargs["use_chemistry_constraints"] = self.chemistry_constraints
            candidate.rerank_score = score_candidate(candidate, **score_kwargs)
        if self.dedupe_substrate:
            enriched = dedupe_by_substrate(enriched)
        enriched.sort(key=lambda candidate: candidate.rerank_score, reverse=True)
        return [
            {
                "predicted_ec": candidate.ec,
                "predicted_substrates": candidate.substrate,
                "predicted_ssredits": candidate.text,
                "execute_ok": candidate.execute_ok,
                "score": candidate.rerank_score,
                "logprob": candidate.logprob,
                "lipid_penalty": candidate.lipid_penalty,
                "chemistry_score": getattr(candidate, "chemistry_score", 0.0),
                "chemistry_notes": getattr(candidate, "chemistry_notes", None) or [],
                "error": "" if candidate.execute_ok else "candidate could not be executed",
            }
            for candidate in enriched[:top_k]
        ]

    def _model_dir(self) -> Path:
        if self.model_dir is not None:
            return Path(self.model_dir)
        return Path(self.package_root) / "model" / "ecreact" / "backward_ecs4" / "model"

    def _vocab_file(self) -> Path:
        if self.vocab_file is not None:
            return Path(self.vocab_file)
        return Path(self.package_root) / "enzretro" / "tokenizer" / "vocab.txt"


def _largest_smiles(items: list[str]) -> str:
    if not items:
        return ""
    return max(items, key=lambda item: (_heavy_atom_count(item), len(canonical_smiles(item) or item), canonical_smiles(item) or item))


COMMON_AUXILIARY_FRAGMENTS = {
    "",
    "O",
    "[H+]",
    "[H-]",
    "[OH-]",
    "[Na+]",
    "[K+]",
    "[Cl-]",
    "[Br-]",
    "Cl",
    "Br",
    "O=O",
    "OO",
    "N",
    "C(=O)=O",
    "O=C=O",
    "O=P(O)(O)O",
    "O=P([O-])([O-])O",
    "O=P(O)(O)OP(=O)(O)O",
}


def _select_main_reactant(items: list[str], product_smiles: str) -> str:
    if not items:
        return ""
    non_aux = [item for item in items if not _is_auxiliary_fragment(item)]
    candidates = non_aux or items
    product_atoms = _heavy_atom_count(product_smiles)

    def key(item: str) -> tuple[float, int, int, str]:
        atoms = _heavy_atom_count(item)
        # Prefer substrate-sized fragments over very large nucleotide/cofactor
        # fragments that survived the auxiliary filter.
        size_penalty = abs(atoms - product_atoms) / max(product_atoms, 1)
        if product_atoms and atoms > product_atoms * 1.8:
            size_penalty += 1.0
        can = canonical_smiles(item) or item
        return (-size_penalty, atoms, len(can), can)

    return max(candidates, key=key)


def _heavy_atom_count(smiles: str) -> int:
    mol = Chem.MolFromSmiles(str(smiles or ""))
    if mol is None:
        return 0
    return sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1)


def _is_auxiliary_fragment(smiles: str) -> bool:
    can = canonical_smiles(smiles) or smiles
    if can in COMMON_AUXILIARY_FRAGMENTS or smiles in COMMON_AUXILIARY_FRAGMENTS:
        return True
    mol = Chem.MolFromSmiles(str(can or ""))
    if mol is None:
        return False
    atom_counts: dict[int, int] = {}
    for atom in mol.GetAtoms():
        atom_counts[atom.GetAtomicNum()] = atom_counts.get(atom.GetAtomicNum(), 0) + 1
    heavy_atoms = sum(count for atomic_num, count in atom_counts.items() if atomic_num > 1)
    carbon_count = atom_counts.get(6, 0)
    phosphorus_count = atom_counts.get(15, 0)
    lower = can.lower()
    # Nucleotide/redox cofactors are frequently predicted as necessary
    # auxiliaries, but they should not become the route expansion target.
    if phosphorus_count >= 2 and ("ncnc" in lower or "ncn" in lower) and heavy_atoms >= 25:
        return True
    if "nc(=o)" in lower and phosphorus_count >= 1 and heavy_atoms >= 25:
        return True
    if "s+" in lower and phosphorus_count == 0 and heavy_atoms >= 25:
        return True
    if phosphorus_count >= 1 and carbon_count == 0 and heavy_atoms <= 12:
        return True
    return False


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name) or default)
    except (TypeError, ValueError):
        return int(default)


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}
