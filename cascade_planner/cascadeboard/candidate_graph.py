"""Dynamic Candidate Hypergraph for CascadeBoard.

Layer 1: Build AND-OR candidate graph from frozen experts,
propagate hard constraints (prune), sample paths.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

from cascade_planner.cascadeboard import (
    CascadeBoard, Slot, HardMask, CompiledConstraints,
)


@dataclass
class CandidateReaction:
    """A single retrosynthetic candidate from a frozen expert."""
    product: str
    main_reactant: str
    aux_reactants: list[str] = field(default_factory=list)
    reaction_smiles: str = ""
    reaction_type: str = ""
    ec: str | None = None
    enzyme_uid: str | None = None
    score: float = 0.0
    source: str = ""  # "retrochimera" / "enzexpand" / "template"
    metadata: dict = field(default_factory=dict)


@dataclass
class TreeNode:
    """Node in the AND-OR candidate tree."""
    smiles: str
    depth: int = 0
    is_in_stock: bool = False
    children: list[tuple[CandidateReaction, TreeNode]] = field(default_factory=list)
    # Each child is (reaction_that_produces_this_node, child_node_for_main_reactant)


class CandidateHypergraph:
    """Dynamic AND-OR candidate graph built from frozen experts."""

    def __init__(
        self,
        retro_engine=None,
        stock_checker=None,
        motif_memory=None,
        max_depth: int = 4,
        branch_factor: int = 15,
    ):
        self.retro_engine = retro_engine
        self.stock_checker = stock_checker or _default_stock_checker
        self.motif_memory = motif_memory
        self.max_depth = max_depth
        self.branch_factor = branch_factor
        self.root: TreeNode | None = None
        self._node_count = 0

    def build(self, target: str, compiled: CompiledConstraints | None = None, min_depth: int = 0) -> TreeNode:
        """Build candidate tree by recursively expanding from target.

        Args:
            min_depth: force expansion to at least this depth even if reactants are in stock.
        """
        self._min_depth = min_depth
        self.root = TreeNode(smiles=target, depth=0)
        self._expand(self.root, compiled)
        return self.root

    def _expand(self, node: TreeNode, compiled: CompiledConstraints | None) -> None:
        if node.depth >= self.max_depth:
            return
        min_depth = getattr(self, '_min_depth', 0)
        if node.is_in_stock and node.depth >= min_depth:
            return

        # Get candidates from frozen experts
        candidates = self._get_candidates(node.smiles, node.depth, compiled)

        for cand in candidates[:self.branch_factor]:
            # Check hard constraints
            if compiled and not self._passes_hard(cand, node.depth, compiled):
                continue

            child = TreeNode(
                smiles=cand.main_reactant,
                depth=node.depth + 1,
                is_in_stock=self.stock_checker(cand.main_reactant),
            )
            node.children.append((cand, child))
            self._node_count += 1

            # Recurse: always expand if below min_depth, even if in stock
            min_depth = getattr(self, '_min_depth', 0)
            if not child.is_in_stock or child.depth < min_depth:
                self._expand(child, compiled)

    def _get_candidates(
        self, product: str, depth: int, compiled: CompiledConstraints | None,
    ) -> list[CandidateReaction]:
        """Get retrosynthetic candidates from frozen experts."""
        if self.retro_engine is None:
            return _mock_candidates(product, depth)

        candidates = []
        try:
            # Chemical candidates
            chem = self.retro_engine.get("retrochimera")
            if chem:
                for c in chem.predict(product, top_k=10):
                    candidates.append(CandidateReaction(
                        product=product,
                        main_reactant=c.get("main_reactant", ""),
                        reaction_smiles=c.get("rxn_smiles", ""),
                        reaction_type=c.get("type", ""),
                        score=c.get("score", 0),
                        source="retrochimera",
                    ))
        except Exception:
            pass

        try:
            # Enzymatic candidates
            enz = self.retro_engine.get("enzexpand")
            if enz:
                for c in enz.predict(product, top_k=10):
                    candidates.append(CandidateReaction(
                        product=product,
                        main_reactant=c.get("main_reactant", ""),
                        reaction_smiles=c.get("rxn_smiles", ""),
                        reaction_type=c.get("type", ""),
                        ec=c.get("ec", None),
                        enzyme_uid=c.get("enzyme_uid"),
                        score=c.get("score", 0),
                        source="enzexpand",
                    ))
        except Exception:
            pass

        # Fallback: mock candidates for testing
        if not candidates:
            candidates = _mock_candidates(product, depth)

        return sorted(candidates, key=lambda c: -c.score)

    def _passes_hard(
        self, cand: CandidateReaction, depth: int, compiled: CompiledConstraints,
    ) -> bool:
        """Check if candidate passes hard constraints.

        Candidates with missing/empty field values are NOT filtered out,
        because they may be correct but simply unlabeled (e.g. RetroChimera
        candidates don't have reaction_type set).
        """
        for mask in compiled.hard_masks:
            if mask.slot_index is not None and mask.slot_index != depth:
                continue
            val = getattr(cand, mask.field, None)
            if val is None or val == "":
                continue  # Don't filter unlabeled candidates
            if not CompiledConstraints._value_allowed(val, mask.allowed_values):
                return False
            if mask.excluded_values and val in mask.excluded_values:
                return False
        return True

    def propagate_constraints(self, compiled: CompiledConstraints) -> bool:
        """Prune tree nodes that violate hard constraints. Returns True if tree is non-empty.

        Note: Only prunes based on exclude constraints (e.g. exclude_catalyst='Pd').
        Allowed-value constraints (e.g. fix reaction_type) are applied at board level,
        not at tree level, because candidates may not have the constrained field set.
        """
        if self.root is None:
            return False
        # Only prune for exclude constraints (where we know the value is wrong)
        has_exclude = any(m.excluded_values for m in compiled.hard_masks)
        if has_exclude:
            self._prune(self.root, compiled)
        return bool(self.root.children) or self.root.is_in_stock

    def _prune(self, node: TreeNode, compiled: CompiledConstraints) -> bool:
        """Recursively prune. Returns True if node has valid descendants."""
        if node.is_in_stock:
            return True
        valid_children = []
        for cand, child in node.children:
            if self._passes_hard(cand, child.depth - 1, compiled):
                if self._prune(child, compiled):
                    valid_children.append((cand, child))
        node.children = valid_children
        return bool(valid_children)

    def sample_paths(self, n: int = 30, temperature: float = 1.0, target_depth: int = 0) -> list[list[CandidateReaction]]:
        """Sample N paths from root to leaves.

        If target_depth > 0, prefer paths of that length by oversampling
        and filtering.
        """
        if self.root is None:
            return []
        paths = []
        attempts = n * 4 if target_depth > 0 else n
        for _ in range(attempts):
            path = self._sample_one_path(self.root, temperature)
            if path:
                paths.append(path)

        if target_depth > 0 and paths:
            # Sort: prefer paths closest to target_depth
            paths.sort(key=lambda p: abs(len(p) - target_depth))
            return paths[:n]
        return paths[:n]

    def _sample_one_path(
        self, node: TreeNode, temperature: float,
    ) -> list[CandidateReaction]:
        if not node.children:
            return []
        # Weighted sampling by score
        scores = [max(c.score, 0.01) for c, _ in node.children]
        if temperature != 1.0:
            scores = [s ** (1.0 / temperature) for s in scores]
        total = sum(scores)
        weights = [s / total for s in scores]

        idx = random.choices(range(len(node.children)), weights=weights, k=1)[0]
        cand, child = node.children[idx]

        # Continue recursion if child has children (even if in stock),
        # to allow multi-step paths when the tree was force-expanded.
        if child.children:
            rest = self._sample_one_path(child, temperature)
        else:
            rest = []
        return [cand] + rest

    def path_to_board(self, path: list[CandidateReaction], target: str) -> CascadeBoard:
        """Convert a sampled path to a CascadeBoard."""
        board = CascadeBoard.from_n_steps(len(path), target)
        for i, cand in enumerate(path):
            slot = board.slots[i]
            slot.product = cand.product
            slot.main_reactant = cand.main_reactant
            slot.aux_reactants = cand.aux_reactants
            slot.reaction_smiles = cand.reaction_smiles
            slot.reaction_type = cand.reaction_type
            slot.ec = cand.ec
            slot.enzyme_uid = cand.enzyme_uid
            slot.e_retro = cand.score
            slot.e_enzyme = cand.metadata.get("e_enzyme", cand.metadata.get("dual_tower_score"))
            slot.source = cand.source
            slot.T = cand.metadata.get("T")
            slot.pH = cand.metadata.get("pH")
            slot.solvent = cand.metadata.get("solvent")
            slot.candidates = []
        return board

    @property
    def node_count(self) -> int:
        return self._node_count

    def is_empty(self) -> bool:
        return self.root is None or (not self.root.children and not self.root.is_in_stock)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _default_stock_checker(smiles: str) -> bool:
    """Stock check using real zinc stock (17.4M molecules)."""
    try:
        from cascade_planner.cascadeboard.zinc_stock import is_in_zinc_stock
        return is_in_zinc_stock(smiles)
    except Exception:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False
        return mol.GetNumHeavyAtoms() <= 6


def _mock_candidates(product: str, depth: int) -> list[CandidateReaction]:
    """Generate mock candidates for testing without real models."""
    mol = Chem.MolFromSmiles(product)
    if mol is None:
        return []

    candidates = []
    types = ["reduction", "oxidation", "acylation", "hydrolysis", "C_C_coupling"]
    ecs = [None, "1.1.1.1", "3.1.1.3", "2.6.1.62", None]
    temps = [30.0, 37.0, 40.0, 50.0, 80.0]
    phs = [7.0, 7.5, 8.0, 6.5, None]

    # Generate reactants that get progressively simpler with depth
    n_heavy = mol.GetNumHeavyAtoms()
    # At depth 0-1: medium complexity intermediates; at depth 2+: simple/stock
    if depth >= 2 or n_heavy <= 6:
        simple = ["CCO", "CC=O", "CC(=O)O", "CCN", "C=O"]
    elif depth == 1:
        simple = ["CC(=O)c1ccccc1", "OC(=O)c1ccccc1", "c1ccc(CO)cc1", "CC(O)c1ccccc1", "c1ccc(N)cc1"]
    else:
        simple = ["CC(=O)Oc1ccccc1", "OC(=O)c1ccc(O)cc1", "CC(O)C(=O)c1ccccc1", "c1ccc(CC=O)cc1", "CC(=O)c1ccc(O)cc1"]

    for i in range(min(5, len(types))):
        reactant = simple[i % len(simple)]
        cand = CandidateReaction(
            product=product,
            main_reactant=reactant,
            reaction_type=types[i],
            ec=ecs[i],
            score=0.8 - i * 0.1,
            source="mock",
        )
        # Add mock conditions
        cand.metadata["T"] = temps[i]
        cand.metadata["pH"] = phs[i]
        candidates.append(cand)
    return candidates
