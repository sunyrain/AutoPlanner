"""Interactive sidecar for one Codex-guided AiZynthFinder MCTS branch.

The main AutoPlanner process owns model calls and ReactionJSON replay.  This
isolated Python 3.11 process owns AiZynthFinder MCTS state.  They exchange one
JSON object per line over stdin/stdout so neither environment imports the
other one's binary dependency stack.
"""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import sys
import traceback
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from aizynthfinder.chem import Molecule  # noqa: E402
from aizynthfinder.context.stock.queries import StockQueryMixin  # noqa: E402

from cascade_planner.interfaces.aizynthfinder_reactionjson_expansion import (  # noqa: E402
    FullInchiKeySqliteStockQuery,
    ReactionJsonExpansionCandidate,
    ReactionJsonPolicyResponse,
    run_reactionjson_branch,
)


SCHEMA = "aizynthfinder_reactionjson_branch_sidecar.v2"


class _InlineStock(StockQueryMixin):
    """Small deterministic stock used only by sidecar integration tests."""

    def __init__(self, smiles: list[str]) -> None:
        self._keys = {Molecule(smiles=value).inchi_key for value in smiles}

    def __contains__(self, mol: Any) -> bool:
        return str(getattr(mol, "inchi_key", "")) in self._keys

    def __len__(self) -> int:
        return len(self._keys)


def _send(payload: dict[str, Any]) -> None:
    sys.stdout.write(
        json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n"
    )
    sys.stdout.flush()


def _receive() -> dict[str, Any]:
    line = sys.stdin.readline()
    if not line:
        raise EOFError("aizynthfinder sidecar input closed")
    value = json.loads(line)
    if not isinstance(value, dict):
        raise ValueError("aizynthfinder sidecar message is not an object")
    return value


def main() -> int:
    stock: Any | None = None
    try:
        launch = _receive()
        if launch.get("schema_version") != SCHEMA:
            raise ValueError("aizynthfinder sidecar launch schema mismatch")
        inline_stock = [
            str(value) for value in launch.get("inline_stock_smiles") or []
            if str(value)
        ]
        if inline_stock:
            stock = _InlineStock(inline_stock)
        else:
            stock = FullInchiKeySqliteStockQuery(
                str(launch.get("stock_index_path") or "")
            )

        def candidate_provider(request: Any) -> ReactionJsonPolicyResponse:
            _send(
                {
                    "schema_version": SCHEMA,
                    "type": "expansion_request",
                    "request": asdict(request),
                }
            )
            response = _receive()
            if response.get("type") != "expansion_response":
                raise ValueError("aizynthfinder sidecar response type mismatch")
            if response.get("error"):
                raise RuntimeError(str(response.get("error")))
            return ReactionJsonPolicyResponse(
                candidates=tuple(
                    ReactionJsonExpansionCandidate(
                        candidate_id=str(row.get("candidate_id") or ""),
                        product_smiles=str(row.get("product_smiles") or ""),
                        mapped_product_smiles=str(
                            row.get("mapped_product_smiles") or ""
                        ),
                        precursor_smiles=tuple(
                            str(value)
                            for value in row.get("precursor_smiles") or []
                        ),
                        mapped_precursor_smiles=tuple(
                            str(value)
                            for value in row.get("mapped_precursor_smiles") or []
                        ),
                        route_step=dict(row.get("route_step") or {}),
                        prior=float(row.get("prior") or 0.0),
                        candidate_key=str(row.get("candidate_key") or ""),
                    )
                    for row in response.get("candidates") or []
                    if isinstance(row, dict)
                ),
                model_call_consumed=bool(
                    response.get("model_call_consumed", True)
                ),
                stop_search=bool(response.get("stop_search", False)),
                stop_reason=str(response.get("stop_reason") or ""),
            )

        result = run_reactionjson_branch(
            target_smiles=str(launch.get("target_smiles") or ""),
            strategy_id=str(launch.get("strategy_id") or ""),
            strategy_text=str(launch.get("strategy_text") or ""),
            candidate_provider=candidate_provider,
            stock_query=stock,
            max_policy_calls=int(launch.get("max_policy_calls") or 25),
            max_candidates_per_call=int(
                launch.get("max_candidates_per_call") or 1
            ),
            max_transforms=int(launch.get("max_transforms") or 25),
            exploration_constant=float(
                launch.get("exploration_constant") or 1.4
            ),
            max_mcts_iterations=int(
                launch.get("max_mcts_iterations") or 0
            )
            or None,
        )
        _send(
            {
                "schema_version": SCHEMA,
                "type": "result",
                "result": asdict(result),
            }
        )
        return 0
    except Exception as exc:
        _send(
            {
                "schema_version": SCHEMA,
                "type": "error",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc()[-8_000:],
            }
        )
        return 1
    finally:
        closer = getattr(stock, "close", None)
        if callable(closer):
            closer()


if __name__ == "__main__":
    raise SystemExit(main())
