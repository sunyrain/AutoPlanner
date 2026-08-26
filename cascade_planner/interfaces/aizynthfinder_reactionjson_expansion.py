"""AiZynthFinder search adapters for host-replayed ReactionJSON actions.

The SynthEx-style boundary is intentionally narrow:

* AutoPlanner/Codex proposes an atom-edit program.
* AutoPlanner replays it and supplies the resulting mapped precursors.
* This module turns the accepted result into an AiZynthFinder ``RetroReaction``.
* AiZynthFinder remains the sole owner of OR candidates, MCTS/UCB selection,
  cycle pruning, stock termination, and back-propagation.

This module belongs to the isolated ``requirements_aizynth.txt`` runtime.  It
must not be imported by the normal AutoPlanner process unless AiZynthFinder is
installed there.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
import json
import math
from pathlib import Path
import sqlite3
from typing import Any, Callable, Mapping, Sequence

from aizynthfinder.chem import SmilesBasedRetroReaction, TreeMolecule
from aizynthfinder.context.policy.expansion_strategies import ExpansionStrategy
from aizynthfinder.context.stock.queries import StockQueryMixin
from aizynthfinder.search.mcts.node import MctsNode
from aizynthfinder.search.mcts.search import MctsSearchTree

from cascade_planner.application.strategy_contract import reaction_edit_signature


class ReactionJsonPolicyError(RuntimeError):
    """Raised when a host-policy response cannot be bound to the AiZ state."""


@dataclass(frozen=True, slots=True)
class ReactionJsonExpansionCandidate:
    """One already replayed ReactionJSON disconnection.

    ``mapped_precursor_smiles`` is the structural authority.  The policy never
    asks AiZynthFinder to redraw or reapply the atom edits.
    """

    candidate_id: str
    product_smiles: str
    mapped_product_smiles: str
    precursor_smiles: tuple[str, ...]
    mapped_precursor_smiles: tuple[str, ...]
    route_step: Mapping[str, Any]
    prior: float = 1.0
    candidate_key: str = ""

    def __post_init__(self) -> None:
        if not self.candidate_id:
            raise ValueError("reactionjson candidate_id is required")
        if not self.product_smiles:
            raise ValueError("reactionjson product_smiles is required")
        if not self.mapped_product_smiles:
            raise ValueError("reactionjson mapped_product_smiles is required")
        if not self.precursor_smiles:
            raise ValueError("reactionjson precursor_smiles is required")
        if len(self.precursor_smiles) != len(self.mapped_precursor_smiles):
            raise ValueError("reactionjson mapped precursor cardinality mismatch")
        if any(not value for value in self.mapped_precursor_smiles):
            raise ValueError("reactionjson mapped precursor is empty")


@dataclass(frozen=True, slots=True)
class ReactionJsonExpansionRequest:
    """Compact node-local request passed to the Codex/fixture provider."""

    strategy_id: str
    strategy_text: str
    call_index: int
    max_calls: int
    depth: int
    expandable_smiles: tuple[str, ...]
    expandable_mapped_smiles: tuple[str, ...]
    route_steps: tuple[Mapping[str, Any], ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)


ReactionJsonCandidateProvider = Callable[
    [ReactionJsonExpansionRequest],
    Sequence[ReactionJsonExpansionCandidate] | "ReactionJsonPolicyResponse",
]


@dataclass(frozen=True, slots=True)
class ReactionJsonPolicyResponse:
    """One policy response; only Host/MCTS owns terminal decisions."""

    candidates: tuple[ReactionJsonExpansionCandidate, ...] = ()


@dataclass(frozen=True, slots=True)
class ReactionJsonBranchResult:
    """Best target-rooted AiZ branch after one strategy's bounded search."""

    strategy_id: str
    solved: bool
    route_steps: tuple[Mapping[str, Any], ...]
    open_leaf_states: tuple[Mapping[str, str], ...]
    policy_calls: int
    mcts_iterations: int
    diagnostics: Mapping[str, Any]


class AiZynthFinderReactionJsonExpansionStrategy(ExpansionStrategy):
    """AiZ expansion policy backed by host-validated ReactionJSON candidates."""

    def __init__(
        self,
        key: str,
        config: Any,
        *,
        candidate_provider: ReactionJsonCandidateProvider,
        strategy_id: str,
        strategy_text: str,
        max_policy_calls: int = 25,
        max_candidates_per_call: int = 1,
        request_metadata: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(key, config)
        if not callable(candidate_provider):
            raise TypeError("reactionjson candidate_provider must be callable")
        if max_policy_calls < 1:
            raise ValueError("reactionjson max_policy_calls must be positive")
        if max_candidates_per_call < 1:
            raise ValueError("reactionjson max_candidates_per_call must be positive")
        self.candidate_provider = candidate_provider
        self.strategy_id = str(strategy_id or "")
        self.strategy_text = str(strategy_text or "")
        self.max_policy_calls = int(max_policy_calls)
        self.max_candidates_per_call = int(max_candidates_per_call)
        self.request_metadata = dict(request_metadata or {})
        self.policy_calls = 0
        self.accepted_actions = 0
        self.duplicate_actions = 0
        self.rejected_candidates: list[dict[str, str]] = []
        self._node: MctsNode | None = None

    @property
    def calls_exhausted(self) -> bool:
        return self.policy_calls >= self.max_policy_calls

    def set_node_context(self, node: MctsNode) -> None:
        """Bind the node selected by AiZ immediately before policy expansion."""

        self._node = node

    def get_actions(
        self,
        molecules: Sequence[TreeMolecule],
        cache_molecules: Sequence[TreeMolecule] | None = None,
    ) -> tuple[list[Any], list[float]]:
        del cache_molecules  # AiZ cache hints must never create paid calls.
        active = list(molecules or ())
        if not active or self.calls_exhausted:
            return [], []
        if self._node is None:
            raise ReactionJsonPolicyError("reactionjson AiZ node context is missing")

        self.policy_calls += 1
        request = self._make_request(active)
        try:
            raw_response = self.candidate_provider(request)
        except Exception as exc:  # fail at the policy boundary; AiZ must not retry blindly
            raise ReactionJsonPolicyError(
                f"reactionjson candidate provider failed on call {self.policy_calls}"
            ) from exc

        if isinstance(raw_response, ReactionJsonPolicyResponse):
            proposed = list(raw_response.candidates)
        else:
            proposed = list(raw_response or ())

        mol_by_inchikey = {mol.inchi_key: mol for mol in active}
        actions: list[Any] = []
        priors: list[float] = []
        seen: set[tuple[str, tuple[str, ...]]] = set()
        for candidate in proposed:
            if len(actions) >= self.max_candidates_per_call:
                break
            try:
                product = TreeMolecule(parent=None, smiles=candidate.product_smiles)
            except Exception:
                self._reject(candidate, "product_not_parseable")
                continue
            mol = mol_by_inchikey.get(product.inchi_key)
            if mol is None:
                self._reject(candidate, "product_not_in_expandable_state")
                continue
            mapped_precursors = tuple(
                str(value).strip() for value in candidate.mapped_precursor_smiles
            )
            signature = (mol.inchi_key, tuple(sorted(mapped_precursors)))
            if signature in seen:
                self.duplicate_actions += 1
                continue
            seen.add(signature)
            prior = _bounded_prior(candidate.prior, rank=len(actions))
            metadata = {
                "policy_name": self.key,
                "policy_probability": prior,
                "policy_probability_rank": len(actions),
                "candidate_id": candidate.candidate_id,
                "candidate_key": candidate.candidate_key,
                "strategy_id": self.strategy_id,
                "reactionjson_host_replayed": True,
                "autoplanner_route_step": dict(candidate.route_step),
                "unmapped_precursor_smiles": list(candidate.precursor_smiles),
                "mapped_precursor_smiles": list(mapped_precursors),
                "mapped_product_smiles": candidate.mapped_product_smiles,
            }
            actions.append(
                SmilesBasedRetroReaction(
                    mol,
                    metadata=metadata,
                    reactants_str=".".join(mapped_precursors),
                    mapped_prod_smiles=candidate.mapped_product_smiles,
                )
            )
            priors.append(prior)

        self.accepted_actions += len(actions)
        return actions, priors

    def reset_cache(self) -> None:
        """There is no prediction cache; preserve the scientific call ledger."""

    def diagnostics(self) -> dict[str, Any]:
        return {
            "engine": "AiZynthFinder.MctsSearchTree",
            "policy": self.key,
            "strategy_id": self.strategy_id,
            "policy_calls": self.policy_calls,
            "max_policy_calls": self.max_policy_calls,
            "max_candidates_per_call": self.max_candidates_per_call,
            "accepted_actions": self.accepted_actions,
            "duplicate_actions": self.duplicate_actions,
            "rejected_candidates": list(self.rejected_candidates),
            "calls_exhausted": self.calls_exhausted,
        }

    def _make_request(
        self, molecules: Sequence[TreeMolecule]
    ) -> ReactionJsonExpansionRequest:
        actions, _nodes = self._node.path_to()
        route_steps = tuple(
            dict(action.metadata.get("autoplanner_route_step") or {})
            for action in actions
            if isinstance(action.metadata.get("autoplanner_route_step"), Mapping)
        )
        return ReactionJsonExpansionRequest(
            strategy_id=self.strategy_id,
            strategy_text=self.strategy_text,
            call_index=self.policy_calls,
            max_calls=self.max_policy_calls,
            depth=len(actions),
            expandable_smiles=tuple(mol.smiles for mol in molecules),
            expandable_mapped_smiles=_host_mapped_active_molecules(
                actions=actions,
                molecules=molecules,
            ),
            route_steps=route_steps,
            metadata=dict(self.request_metadata),
        )

    def _reject(
        self, candidate: ReactionJsonExpansionCandidate, reason: str
    ) -> None:
        self.rejected_candidates.append(
            {"candidate_id": candidate.candidate_id, "reason": reason}
        )


def _host_mapped_active_molecules(
    *,
    actions: Sequence[Any],
    molecules: Sequence[TreeMolecule],
) -> tuple[str, ...]:
    """Recover the host atom-map namespace for the current AiZ state.

    AiZ remaps atoms introduced by a retrosynthetic action to its own local
    namespace.  Those ``TreeMolecule.mapped_smiles`` values are useful inside
    AiZ, but they are not the namespace in which the host-compiled
    ReactionJSON program was created.  Bind each instantiated AiZ reactant
    object back to the mapped precursor emitted by the host action that made
    it, then read the current expandable objects from that registry.
    """

    if not actions:
        return tuple(str(mol.mapped_smiles) for mol in molecules)

    host_map_by_object: dict[int, str] = {}
    for action_index, action in enumerate(actions):
        metadata = dict(getattr(action, "metadata", {}) or {})
        mapped_precursors = tuple(
            str(value).strip()
            for value in metadata.get("mapped_precursor_smiles") or ()
            if str(value).strip()
        )
        precursor_smiles = tuple(
            str(value).strip()
            for value in metadata.get("unmapped_precursor_smiles") or ()
            if str(value).strip()
        )
        outcomes = tuple(getattr(action, "reactants", ()) or ())
        if (
            len(outcomes) != 1
            or not mapped_precursors
            or len(precursor_smiles) != len(mapped_precursors)
            or len(outcomes[0]) != len(mapped_precursors)
        ):
            raise ReactionJsonPolicyError(
                "reactionjson host precursor map binding cardinality mismatch "
                f"at action {action_index}"
            )
        bindings = _bind_host_precursor_maps(
            tree_molecules=outcomes[0],
            precursor_smiles=precursor_smiles,
            mapped_precursor_smiles=mapped_precursors,
        )
        host_map_by_object.update(bindings)

    resolved: list[str] = []
    for mol in molecules:
        mapped = host_map_by_object.get(id(mol))
        if not mapped:
            raise ReactionJsonPolicyError(
                "reactionjson active molecule is outside the host map registry"
            )
        resolved.append(mapped)
    return tuple(resolved)


def _bind_host_precursor_maps(
    *,
    tree_molecules: Sequence[TreeMolecule],
    precursor_smiles: Sequence[str],
    mapped_precursor_smiles: Sequence[str],
) -> dict[int, str]:
    """Bind isomorphic siblings by molecular identity and stable occurrence."""

    host_occurrences: dict[str, list[str]] = {}
    for precursor, mapped in zip(
        precursor_smiles, mapped_precursor_smiles, strict=True
    ):
        key = _molecule_inchikey(precursor)
        host_occurrences.setdefault(key, []).append(mapped)

    consumed: dict[str, int] = {}
    bindings: dict[int, str] = {}
    for molecule in tree_molecules:
        key = str(molecule.inchi_key or "")
        occurrence = consumed.get(key, 0)
        candidates = host_occurrences.get(key, [])
        if occurrence >= len(candidates):
            raise ReactionJsonPolicyError(
                "reactionjson host precursor identity binding mismatch"
            )
        bindings[id(molecule)] = candidates[occurrence]
        consumed[key] = occurrence + 1

    if any(consumed.get(key, 0) != len(values) for key, values in host_occurrences.items()):
        raise ReactionJsonPolicyError(
            "reactionjson host precursor occurrence binding mismatch"
        )
    return bindings


@lru_cache(maxsize=16_384)
def _molecule_inchikey(smiles: str) -> str:
    try:
        return str(TreeMolecule(parent=None, smiles=smiles).inchi_key or "")
    except Exception as exc:
        raise ReactionJsonPolicyError(
            "reactionjson host precursor identity is invalid"
        ) from exc


class ReactionJsonMctsNode(MctsNode):
    """AiZ MCTS node that exposes path context to context-aware policies."""

    def expand(self) -> None:
        for key in self._config.expansion_policy.selection or ():
            policy = self._config.expansion_policy[key]
            setter = getattr(policy, "set_node_context", None)
            if callable(setter):
                setter(self)
        super().expand()


class ReactionJsonMctsSearchTree(MctsSearchTree):
    """Stock AiZ MCTS/UCB with ``ReactionJsonMctsNode`` as its node class."""

    def __init__(self, config: Any, root_smiles: str) -> None:
        super().__init__(config=config, root_smiles=None)
        self.last_backpropagated_leaf: MctsNode | None = None
        self.root = ReactionJsonMctsNode.create_root(
            smiles=root_smiles,
            tree=self,
            config=config,
        )

    def one_iteration(self) -> bool:
        """Run one MCTS iteration while making empty policy calls retryable.

        AiZynthFinder's stock node implementation marks a node terminal when a
        policy returns no actions.  That is correct for a deterministic neural
        policy, but it is too strong for a bounded LLM policy: an invalid or
        rejected ReactionJSON response consumes one policy call and should let
        UCB revisit the same open leaf with the next call.  Without this
        boundary, a branch can stop at 2--8 calls although its 25-call paper
        allowance is still unused.  We retain the normal selection,
        instantiation and back-propagation logic; only the empty-action
        terminal transition is made retryable until the policy budget is
        exhausted.
        """

        self.profiling["iterations"] += 1
        leaf = self.select_leaf()
        leaf.expand()

        def policy_exhausted() -> bool:
            for key in self.config.expansion_policy.selection or ():
                policy = self.config.expansion_policy[key]
                if bool(getattr(policy, "calls_exhausted", False)):
                    return True
            return False

        # ``MctsNode.expand`` reverses an empty expansion by clearing both
        # flags.  Re-open only unsolved states and only while another LLM call
        # is available; a genuinely terminal stock state remains terminal.
        if not getattr(leaf, "_children_actions", None):
            if (
                not leaf.state.is_terminal
                and not policy_exhausted()
            ):
                leaf.is_expandable = True
                leaf.is_expanded = False
            self.backpropagate(leaf)
            return bool(leaf.state.is_solved)

        while not leaf.is_terminal():
            child = leaf.promising_child()
            if child:
                leaf = child
                child.expand()
                if not getattr(child, "_children_actions", None):
                    if (
                        not child.state.is_terminal
                        and not policy_exhausted()
                    ):
                        child.is_expandable = True
                        child.is_expanded = False
                    leaf = child
                    break
            else:
                # No applicable instantiated child.  Give the same open node
                # another UCB/policy opportunity rather than silently closing
                # the branch before its scientific call ceiling.
                if (
                    not leaf.state.is_terminal
                    and not policy_exhausted()
                ):
                    leaf.is_expandable = True
                    leaf.is_expanded = False
                break

        self.backpropagate(leaf)
        self.last_backpropagated_leaf = leaf
        return bool(leaf.state.is_solved)


def run_reactionjson_branch(
    *,
    target_smiles: str,
    strategy_id: str,
    strategy_text: str,
    candidate_provider: ReactionJsonCandidateProvider,
    stock_query: StockQueryMixin,
    max_policy_calls: int = 25,
    max_candidates_per_call: int = 1,
    max_transforms: int = 25,
    exploration_constant: float = 1.4,
    max_mcts_iterations: int | None = None,
) -> ReactionJsonBranchResult:
    """Run one independent strategy in a stock AiZ MCTS/UCB tree.

    The scientific expansion budget is counted by the policy.  MCTS may take
    additional selection/back-propagation iterations that do not call the
    provider, so those two counters are reported separately.
    """

    from aizynthfinder.context.config import Configuration

    config = Configuration()
    config.search.max_transforms = max(1, int(max_transforms))
    config.search.algorithm_config["C"] = float(exploration_constant)
    config.search.algorithm_config["use_prior"] = True
    config.search.algorithm_config["prune_cycles_in_search"] = True
    config.scorers.create_default_scorers()
    # A benchmark target can itself occur in a supplier-derived inventory.
    # Treating that root hit as a solved zero-step route would measure
    # purchasability instead of retrosynthesis.  Keep the frozen stock intact
    # for every other molecule, but apply the standard leave-target-out rule
    # at the search boundary (also preventing a cyclic route from closing by
    # returning to the target).
    target_inchikey = TreeMolecule(parent=None, smiles=target_smiles).inchi_key
    effective_stock = TargetExcludedStockQuery(
        stock_query,
        excluded_inchikeys=(target_inchikey,),
    )
    config.stock.load(effective_stock, "paper_zinc_emolecules")
    config.stock.select("paper_zinc_emolecules")
    policy = AiZynthFinderReactionJsonExpansionStrategy(
        "codex_reactionjson",
        config,
        candidate_provider=candidate_provider,
        strategy_id=strategy_id,
        strategy_text=strategy_text,
        max_policy_calls=max_policy_calls,
        max_candidates_per_call=max_candidates_per_call,
    )
    config.expansion_policy.load(policy)
    config.expansion_policy.select("codex_reactionjson")
    tree = ReactionJsonMctsSearchTree(config, root_smiles=target_smiles)

    iteration_limit = max_mcts_iterations
    if iteration_limit is None:
        iteration_limit = max(25, int(max_policy_calls) * 5)
    solved = False
    iterations = 0
    stagnant_iterations = 0
    previous_calls = -1
    for _ in range(max(1, int(iteration_limit))):
        iterations += 1
        stock_solved = bool(tree.one_iteration())
        solved_leaf = tree.last_backpropagated_leaf
        solved = bool(
            stock_solved
            and solved_leaf is not None
            and _node_satisfies_strategy_execution_contract(
                solved_leaf,
                strategy_text=strategy_text,
            )
        )
        if solved:
            break
        if policy.policy_calls == previous_calls:
            stagnant_iterations += 1
        else:
            stagnant_iterations = 0
        previous_calls = policy.policy_calls
        if policy.calls_exhausted and stagnant_iterations >= 10:
            break

    selected = _select_branch_projection_node(
        tree,
        strategy_text=strategy_text,
    )
    actions, _nodes = selected.path_to()
    selected_strategy_contract_satisfied = (
        _node_satisfies_strategy_execution_contract(
            selected,
            strategy_text=strategy_text,
        )
    )
    route_steps = tuple(
        dict(action.metadata.get("autoplanner_route_step") or {})
        for action in actions
        if isinstance(action.metadata.get("autoplanner_route_step"), Mapping)
    )
    # Keep the path/projection boundary observable.  A prior Editor bug could
    # collapse a long AiZ path after the search had finished; without these
    # counters the report looked like a legitimate shallow search.  The
    # action count is the AiZ tree depth, while route_step_count is the number
    # of host-replayed ReactionJSON rows that can be handed to the Director.
    missing_route_step_indices = [
        index
        for index, action in enumerate(actions)
        if not isinstance(action.metadata.get("autoplanner_route_step"), Mapping)
    ]
    open_leaf_states = tuple(
        {
            "smiles": mol.smiles,
            "mapped_smiles": mol.mapped_smiles,
        }
        for mol in selected.state.expandable_mols
    )
    diagnostics = {
        **policy.diagnostics(),
        "mcts_iterations": iterations,
        "tree_profiling": dict(tree.profiling),
        "tree_nodes": len(tree.nodes()),
        "maximum_tree_depth": max(
            (len(node.path_to()[0]) for node in tree.nodes()),
            default=0,
        ),
        "selected_depth": len(actions),
        "path_action_count": len(actions),
        "path_route_step_count": len(route_steps),
        "path_route_step_metadata_missing_indices": missing_route_step_indices,
        "path_route_projection_complete": len(actions) == len(route_steps),
        "selected_open_leaves": len(open_leaf_states),
        "selected_solved": bool(selected.state.is_solved),
        "selected_stock_solved": bool(selected.state.is_solved),
        "selected_realized_strategic_milestones": (
            _realized_strategy_milestone_count(selected)
        ),
        "maximum_realized_strategic_milestones_in_tree": max(
            (_realized_strategy_milestone_count(node) for node in tree.nodes()),
            default=0,
        ),
        "selected_strategy_execution_contract_satisfied": (
            selected_strategy_contract_satisfied
        ),
        # AiZ stores solved state on the selected descendant; the root state's
        # molecule set remains the original unsolved target and is therefore
        # diagnostic only, not the branch-completion authority.
        "root_solved": bool(tree.root and tree.root.state.is_solved),
    }
    return ReactionJsonBranchResult(
        strategy_id=strategy_id,
        solved=bool(selected.state.is_solved and selected_strategy_contract_satisfied),
        route_steps=route_steps,
        open_leaf_states=open_leaf_states,
        policy_calls=policy.policy_calls,
        mcts_iterations=iterations,
        diagnostics=diagnostics,
    )


class FullInchiKeySqliteStockQuery(StockQueryMixin):
    """Exact, constant-memory stock lookup for the paper-matched SQLite index."""

    def __init__(self, path: str) -> None:
        self.path = Path(path).resolve()
        if not self.path.is_file():
            raise FileNotFoundError(f"paper stock index is missing: {self.path}")
        self._connection = sqlite3.connect(str(self.path))
        self._connection.execute("PRAGMA query_only = ON")
        columns = {
            str(row[1])
            for row in self._connection.execute("PRAGMA table_info(stock)")
        }
        if "full_inchikey" not in columns:
            raise ValueError("paper stock index lacks stock.full_inchikey")

    def __contains__(self, mol: Any) -> bool:
        return self._contains_inchikey(str(mol.inchi_key or ""))

    @lru_cache(maxsize=100_000)
    def _contains_inchikey(self, inchikey: str) -> bool:
        if not inchikey:
            return False
        row = self._connection.execute(
            "SELECT 1 FROM stock WHERE full_inchikey = ? LIMIT 1",
            (inchikey,),
        ).fetchone()
        return row is not None

    def __len__(self) -> int:
        metadata = self._connection.execute(
            "SELECT value FROM metadata WHERE key = 'current_member_count'"
        ).fetchone()
        if metadata is not None:
            return int(metadata[0])
        return int(self._connection.execute("SELECT COUNT(*) FROM stock").fetchone()[0])

    def availability_string(self, mol: Any) -> str:
        return "ZINC+eMolecules" if mol in self else ""

    def clear_cache(self) -> None:
        self._contains_inchikey.cache_clear()

    def close(self) -> None:
        self._connection.close()


class TargetExcludedStockQuery(StockQueryMixin):
    """Read-only stock view that removes only the benchmark target identity."""

    def __init__(
        self,
        source: StockQueryMixin,
        *,
        excluded_inchikeys: Sequence[str],
    ) -> None:
        self.source = source
        self.excluded_inchikeys = frozenset(
            str(value).strip() for value in excluded_inchikeys if str(value).strip()
        )

    def __contains__(self, mol: Any) -> bool:
        inchikey = str(getattr(mol, "inchi_key", "") or "")
        return bool(inchikey and inchikey not in self.excluded_inchikeys and mol in self.source)

    def __len__(self) -> int:
        # Report the backing catalog size.  The one-case exclusion is recorded
        # as evaluation policy and must not masquerade as a different catalog.
        return len(self.source)

    def availability_string(self, mol: Any) -> str:
        if mol not in self:
            return ""
        formatter = getattr(self.source, "availability_string", None)
        return str(formatter(mol) or "") if callable(formatter) else "available"

    def clear_cache(self) -> None:
        clearer = getattr(self.source, "clear_cache", None)
        if callable(clearer):
            clearer()


def _bounded_prior(value: Any, *, rank: int) -> float:
    try:
        prior = float(value)
    except (TypeError, ValueError):
        prior = 1.0 / float(rank + 1)
    if not math.isfinite(prior):
        prior = 1.0 / float(rank + 1)
    return min(1.0, max(1.0e-6, prior))


def _select_branch_projection_node(
    tree: ReactionJsonMctsSearchTree,
    *,
    strategy_text: str = "",
) -> MctsNode:
    nodes = list(tree.nodes())
    solved = [
        node
        for node in nodes
        if node.state.is_solved
        and _node_satisfies_strategy_execution_contract(
            node,
            strategy_text=strategy_text,
        )
    ]
    if solved:
        # Stock closure remains the first authority.  Within solved routes,
        # retain the path that actually executed the greatest number of
        # route-internal StrategyCards before preferring the shorter route.
        return max(
            solved,
            key=lambda node: (
                _realized_strategy_milestone_count(node),
                -len(node.path_to()[0]),
            ),
        )

    # The root is not a retrosynthetic route.  AiZ's state reward can still
    # rank it above every incomplete descendant (especially when a first
    # disconnection exposes several non-stock leaves).  Projecting that root
    # silently erased already host-replayed ReactionJSON work from the
    # Director output.  Once at least one accepted action exists, retain the
    # best non-root target-connected descendant; use the root only when the
    # policy produced no usable action at all.
    routed_nodes = [node for node in nodes if node.path_to()[0]]
    candidates = routed_nodes or nodes

    # A branch projection is also the frozen RouteJSON sent to the Critic.  A
    # generic AiZ reward may prefer a shallower chemical-only prefix even when
    # the branch's immutable StrategyCard requires an actual chemoenzymatic
    # sequence.  If the search has materialized at least one strategy-complete
    # descendant, rank within that cohort instead of silently truncating the
    # biological step from the reported route.  This changes only projection;
    # AiZ remains the owner of UCB selection and back-propagation during search.
    strategy_complete = [
        node
        for node in candidates
        if _node_satisfies_strategy_execution_contract(
            node,
            strategy_text=strategy_text,
        )
    ]
    if strategy_complete:
        candidates = strategy_complete

    def key(node: MctsNode) -> tuple[int, int, int, float, int]:
        try:
            reward = float(tree.compute_reward(node))
        except (TypeError, ValueError):
            reward = -math.inf
        # An unsolved projection is the Route Builder's best *partial route*,
        # not an AiZ value-function snapshot.  The generic reward strongly
        # prefers a shallow prefix when a deeper descendant exposes additional
        # non-stock leaves.  That is useful for UCB selection, but it is the
        # wrong serialization boundary: Critic/Editor must receive the most
        # complete connected route the tree actually built.  Keep the
        # strategy-milestone cohort first, then prefer the deepest path; use
        # reward only as a tie-breaker.  Solved routes remain handled by the
        # stock-authoritative branch above.
        depth = len(node.path_to()[0])
        return (
            _realized_strategy_milestone_count(node),
            _realized_strategy_anchor_pair_count(node),
            depth,
            reward,
            -len(node.state.expandable_mols),
        )

    return max(candidates, key=key)


def _realized_strategy_milestone_count(node: MctsNode) -> int:
    """Count complete route-internal StrategyCards on one connected path."""

    actions, _nodes = node.path_to()
    grouped: dict[int, list[Mapping[str, Any]]] = {}
    for action in actions:
        step = action.metadata.get("autoplanner_route_step")
        if not isinstance(step, Mapping):
            continue
        try:
            index = int(step.get("strategy_milestone_index") or 1)
        except (TypeError, ValueError):
            index = 1
        grouped.setdefault(max(1, index), []).append(step)
    complete = 0
    for steps in grouped.values():
        required: set[tuple[int, int]] = set()
        realized: set[tuple[int, int]] = set()
        for step in steps:
            card = step.get("strategy_card")
            if isinstance(card, Mapping):
                required.update(_strategy_map_pairs(card))
            realized.update(_step_changed_map_pairs(step))
        if required and required.issubset(realized):
            complete += 1
    return complete


def _realized_strategy_anchor_pair_count(node: MctsNode) -> int:
    actions, _nodes = node.path_to()
    required: set[tuple[int, int]] = set()
    realized: set[tuple[int, int]] = set()
    for action in actions:
        step = action.metadata.get("autoplanner_route_step")
        if isinstance(step, Mapping):
            card = step.get("strategy_card")
            if isinstance(card, Mapping):
                required.update(_strategy_map_pairs(card))
            realized.update(_step_changed_map_pairs(step))
    return len(required.intersection(realized))


def _node_satisfies_strategy_execution_contract(
    node: MctsNode,
    *,
    strategy_text: str,
) -> bool:
    required = _strategy_required_execution_domains(strategy_text)
    present = _node_route_execution_domains(node)
    if not required.issubset(present):
        return False
    required_pairs = _strategy_map_pairs_from_text(strategy_text)
    if required_pairs:
        actions, _nodes = node.path_to()
        realized_pairs: set[tuple[int, int]] = set()
        for action in actions:
            step = action.metadata.get("autoplanner_route_step")
            if isinstance(step, Mapping):
                realized_pairs.update(_step_changed_map_pairs(step))
        if not required_pairs.issubset(realized_pairs):
            return False
    return True


def _strategy_map_pairs_from_text(strategy_text: str) -> frozenset[tuple[int, int]]:
    try:
        strategy = json.loads(str(strategy_text or ""))
    except (TypeError, ValueError, json.JSONDecodeError):
        return frozenset()
    return _strategy_map_pairs(strategy if isinstance(strategy, Mapping) else {})


def _strategy_map_pairs(card: Mapping[str, Any]) -> frozenset[tuple[int, int]]:
    pairs: set[tuple[int, int]] = set()
    values = card.get("anchor_bond_signature") or card.get("key_bond_signature") or ()
    for value in values:
        parts = str(value or "").strip().split(":")
        if len(parts) != 3 or parts[0] != "map_pair":
            continue
        try:
            pairs.add(tuple(sorted((int(parts[1]), int(parts[2])))))
        except ValueError:
            continue
    return frozenset(pairs)


def _step_changed_map_pairs(step: Mapping[str, Any]) -> frozenset[tuple[int, int]]:
    signature = reaction_edit_signature(step.get("reaction_operations") or ())
    return frozenset(
        tuple(sorted((int(pair[0]), int(pair[1]))))
        for pair in signature.get("changed_map_pairs") or ()
        if isinstance(pair, (list, tuple)) and len(pair) == 2
    )


def _strategy_required_execution_domains(strategy_text: str) -> frozenset[str]:
    try:
        strategy = json.loads(str(strategy_text or ""))
    except (TypeError, ValueError, json.JSONDecodeError):
        return frozenset()
    if not isinstance(strategy, Mapping):
        return frozenset()
    execution_domain = str(strategy.get("execution_domain") or "").strip().lower()
    if execution_domain in {"hybrid", "chemoenzymatic", "chemo-enzymatic"}:
        return frozenset({"chemical", "biological"})
    return frozenset()


def _node_route_execution_domains(node: MctsNode) -> frozenset[str]:
    actions, _nodes = node.path_to()
    domains: set[str] = set()
    for action in actions:
        step = action.metadata.get("autoplanner_route_step")
        if not isinstance(step, Mapping):
            continue
        domain = str(step.get("execution_domain") or "").strip().lower()
        if domain == "chemical":
            domains.add("chemical")
        if domain in {
            "biological",
            "biocatalytic",
            "enzymatic",
            "enzyme",
            "whole_cell",
            "whole-cell",
        } or step.get("enzyme") or isinstance(step.get("biocatalytic_step"), Mapping):
            domains.add("biological")
    return frozenset(domains)
