"""Build draft route evidence cards from controlled bridge-gate routes.

The output is a P5 draft artifact. It combines controlled route-tree selections
with real bridge verifier evidence. These cards are not final live-provider
routes; they are structured evidence records used to decide which bridge cases
should be promoted to live ChemEnzy/proposal-provider benchmarks.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cascade_planner.cascade_search.bridge_retriever_v0 import BridgeRetrieverV0, BridgeVerifierV0Scorer


DEFAULT_PACK_DIR = Path("data/bridge_pack_v0")
DEFAULT_MODEL_PATH = Path("results/shared/bridge_verifier_v0_20260527/bridge_verifier_v0_lgbm.joblib")
DEFAULT_INPUT = Path("results/shared/bridge_gate_ablation_v0_20260527/bridge_route_gate_rows.jsonl")
DEFAULT_OUTPUT_DIR = Path("results/shared/bridge_gate_ablation_v0_20260527")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def render_markdown(cards: list[dict[str, Any]]) -> str:
    lines = [
        "# Draft Bridge Route Evidence Cards v0",
        "",
        "These cards are generated from controlled route-tree bridge-gate runs plus real bridge verifier evidence.",
        "",
        "Scope: draft selection artifact, not final live-provider route evidence.",
        "",
    ]
    for card in cards:
        lines.extend(
            [
                f"## {card['route_card_id']}",
                "",
                f"- Target: `{card['target_smiles']}`",
                f"- Selected route sources: {', '.join(card['selected_sources'])}",
                f"- Bridge direction: `{card['bridge_direction']}`",
                f"- Bridge source: `{card['bridge_source']}`",
                f"- Tanimoto: {card['tanimoto']}",
                f"- Verifier score: {card['verifier_score']}",
                f"- EC sample: {', '.join(card['enzyme_ec_sample']) if card['enzyme_ec_sample'] else 'N/A'}",
                f"- Enzyme-side molecule: `{card['enzyme_smiles']}`",
                "",
                "Evidence:",
                "",
            ]
        )
        for item in card["evidence"]:
            lines.append(f"- {item}")
        lines.append("")
    return "\n".join(lines)


def build_cards(
    rows: list[dict[str, Any]],
    retriever: BridgeRetrieverV0,
    *,
    limit: int,
) -> list[dict[str, Any]]:
    selected = [
        row
        for row in rows
        if row.get("policy") == "bridge_gate_verifier"
        and int(row.get("label") or 0) == 1
        and bool(row.get("selected_enzyme_route"))
    ]
    selected.sort(key=lambda row: (str(row.get("target_smiles") or "")))
    cards: list[dict[str, Any]] = []
    for row in selected:
        hits = retriever.retrieve(
            str(row.get("target_smiles") or ""),
            top_k=3,
            require_verifier_pass=True,
        )
        if not hits:
            continue
        hit = hits[0]
        cards.append(
            {
                "route_card_id": f"route_bridge_v0_{len(cards) + 1:03d}",
                "target_smiles": row.get("target_smiles") or "",
                "chemical_inchikey": row.get("chemical_inchikey") or "",
                "selected_sources": list(row.get("selected_sources") or []),
                "route_policy": row.get("policy") or "",
                "bridge_direction": hit.bridge_direction,
                "bridge_source": hit.source,
                "enzyme_smiles": hit.enzyme_smiles,
                "enzyme_inchikey": hit.enzyme_inchikey,
                "tanimoto": round(float(hit.tanimoto), 4),
                "verifier_score": round(float(hit.verifier_score or 0.0), 6),
                "verifier_pass": bool(hit.verifier_pass),
                "enzyme_ec_sample": list(hit.enzyme_ec_sample[:8]),
                "controlled_route_metrics": {
                    "enzyme_calls": row.get("enzyme_calls"),
                    "generated_actions": row.get("generated_actions"),
                    "expansions": row.get("expansions"),
                    "elapsed_s": row.get("elapsed_s"),
                },
                "evidence": [
                    "route selected an enzyme step under verifier-gated source allocation",
                    "bridge candidate has verifier_pass=True",
                    "chemical frontier molecule has exact/similarity bridge evidence",
                    "EC evidence present" if hit.enzyme_ec_sample else "EC evidence missing",
                    "controlled route-tree benchmark; live-provider confirmation pending",
                ],
            }
        )
        if len(cards) >= limit:
            break
    return cards


def main() -> None:
    parser = argparse.ArgumentParser(description="Build bridge route evidence cards v0")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--pack-dir", type=Path, default=DEFAULT_PACK_DIR)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()

    scorer = BridgeVerifierV0Scorer(args.model_path)
    retriever = BridgeRetrieverV0(args.pack_dir, scorer=scorer)
    cards = build_cards(read_jsonl(args.input), retriever, limit=max(0, int(args.limit)))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "bridge_route_evidence_cards.json"
    md_path = args.output_dir / "bridge_route_evidence_cards.md"
    json_path.write_text(json.dumps(cards, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    md_path.write_text(render_markdown(cards), encoding="utf-8")
    print(json.dumps({"cards": len(cards), "json": str(json_path), "md": str(md_path)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
