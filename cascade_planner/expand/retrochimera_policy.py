"""RetroChimera expansion strategy for AiZynthFinder MCTS.

Wraps RetroChimera (Microsoft, NeurIPS 2025) as an AiZynthFinder-compatible
expansion policy, enabling its use in multi-step retrosynthesis search.

Usage in aiz_mcts_bridge:
    from cascade_planner.expand.retrochimera_policy import register_retrochimera
    register_retrochimera(finder, model_dir="data_external/retrochimera_model")
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from rdkit import Chem

logger = logging.getLogger(__name__)

_RC_AVAILABLE = False
try:
    from retrochimera import RetroChimeraModel
    _RC_AVAILABLE = True
except ImportError:
    pass

from aizynthfinder.chem import RetroReaction, TreeMolecule
from aizynthfinder.context.config import Configuration
from aizynthfinder.context.policy.expansion_strategies import ExpansionStrategy
from syntheseus.interface.molecule import Molecule


class RetroChimeraExpansionStrategy(ExpansionStrategy):
    """AiZynthFinder expansion strategy backed by RetroChimera."""

    _required_kwargs = ["model_dir"]

    def __init__(self, key: str, config: Configuration, **kwargs) -> None:
        super().__init__(key, config, **kwargs)
        model_dir = kwargs["model_dir"]
        self._cutoff = int(kwargs.get("cutoff_number", 50))

        if not _RC_AVAILABLE:
            raise ImportError("retrochimera not installed: pip install 'retrochimera[graphium]'")

        logger.info("Loading RetroChimera from %s", model_dir)
        self._model = RetroChimeraModel(model_dir=model_dir)
        logger.info("RetroChimera loaded")

    # ------------------------------------------------------------------
    def get_actions(
        self,
        molecules: Sequence[TreeMolecule],
        cache_molecules: Optional[Sequence[TreeMolecule]] = None,
    ) -> Tuple[List[RetroReaction], List[float]]:
        possible_actions: list[RetroReaction] = []
        priors: list[float] = []

        for mol in molecules:
            smi = mol.smiles
            try:
                synth_mol = Molecule(smi)
                preds = self._model([synth_mol], num_results=self._cutoff)
            except Exception as exc:
                logger.debug("RetroChimera failed on %s: %s", smi[:40], exc)
                continue

            for rank, rxn in enumerate(preds[0]):
                reactant_smiles = [r.smiles for r in rxn.reactants]
                # Build a pseudo-SMARTS string (product>>reactants) for metadata
                rxn_smiles = f"{'.'.join(reactant_smiles)}>>{smi}"

                prob = max(1.0 / (rank + 1), 0.01)  # rank-based prior
                metadata = {
                    "policy_probability": round(prob, 4),
                    "policy_probability_rank": rank,
                    "policy_name": self.key,
                    "retrochimera_rank": rank,
                }
                priors.append(prob)

                # Create a SmilesBasedRetroReaction (no template SMARTS needed)
                possible_actions.append(
                    _SmilesRetroReaction(
                        mol,
                        reactant_smiles=reactant_smiles,
                        metadata=metadata,
                    )
                )

        return possible_actions, priors


class _SmilesRetroReaction(RetroReaction):
    """A RetroReaction defined by explicit reactant SMILES (no template)."""

    def __init__(self, mol: TreeMolecule, reactant_smiles: list[str],
                 metadata: dict | None = None) -> None:
        super().__init__(mol, index=0, metadata=metadata or {})
        self._reactant_smiles = reactant_smiles

    def _apply(self):
        """Return the pre-computed reactants as TreeMolecule tuples."""
        mols = []
        for smi in self._reactant_smiles:
            rd_mol = Chem.MolFromSmiles(smi)
            if rd_mol is None:
                continue
            try:
                mols.append(TreeMolecule(parent=self.mol, smiles=smi, sanitize=True))
            except Exception:
                pass
        if not mols:
            return ()
        return (tuple(mols),)

    def _make_smiles(self) -> str:
        return f"{'.'.join(self._reactant_smiles)}>>{self.mol.smiles}"


def register_retrochimera(
    finder,
    model_dir: str = "data_external/retrochimera_model",
    key: str = "retrochimera",
    cutoff: int = 50,
) -> None:
    """Register RetroChimera as an expansion policy in an AiZynthFinder instance."""
    strategy = RetroChimeraExpansionStrategy(
        key, finder.config, model_dir=model_dir, cutoff_number=cutoff,
    )
    finder.expansion_policy.load(strategy)
    logger.info("Registered RetroChimera expansion policy as '%s'", key)
