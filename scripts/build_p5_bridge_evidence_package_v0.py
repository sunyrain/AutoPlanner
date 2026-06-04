"""Build the P5 bridge evidence package from audited live routes."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DEFAULT_AUDIT_ROWS = Path("results/shared/bridge_route_quality_audit_v0_20260528/bridge_live_route_quality_audit_rows.jsonl")
DEFAULT_OUTPUT_DIR = Path("results/shared/p5_bridge_evidence_package_v0_20260528")
PRODUCTION_POLICIES = ("bridge_gate_verifier_bonus2", "normal_bridge_gated", "bridge_gate_verifier")
POLICY_PRIORITY = {policy: idx for idx, policy in enumerate(PRODUCTION_POLICIES)}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def select_cards(rows: list[dict[str, Any]], *, limit: int) -> list[dict[str, Any]]:
    candidates = [
        row
        for row in rows
        if row.get("route_quality_level") in {"production_candidate_partial", "production_candidate_stock_closed"}
        and str(row.get("policy") or "") in PRODUCTION_POLICIES
        and int(row.get("label") or 0) == 1
    ]
    candidates.sort(
        key=lambda row: (
            POLICY_PRIORITY.get(str(row.get("policy") or ""), 99),
            0 if not row.get("hard_flags") else 1,
            len(row.get("risk_flags") or []),
            str(row.get("target_smiles") or ""),
            int(row.get("route_rank") or 999),
        )
    )
    selected: list[dict[str, Any]] = []
    seen_routes: set[tuple[str, str, int]] = set()
    for row in candidates:
        key = (str(row.get("target_smiles") or ""), str(row.get("policy") or ""), int(row.get("route_rank") or 0))
        if key in seen_routes:
            continue
        seen_routes.add(key)
        selected.append(_card_from_record(row, len(selected) + 1))
        if len(selected) >= limit:
            break
    return selected


def _card_from_record(row: dict[str, Any], idx: int) -> dict[str, Any]:
    bridge = (row.get("bridge_evidence") or [{}])[0] or {}
    return {
        "card_id": f"p5_bridge_live_route_{idx:03d}",
        "target_smiles": row.get("target_smiles") or "",
        "chemical_inchikey": row.get("chemical_inchikey") or "",
        "policy": row.get("policy") or "",
        "route_rank": row.get("route_rank"),
        "route_quality_level": row.get("route_quality_level"),
        "selected_sources": list(row.get("selected_sources") or []),
        "step_count": row.get("step_count"),
        "enzyme_step_count": row.get("enzyme_step_count"),
        "stock_closed": bool(row.get("stock_closed")),
        "route_solved": bool(row.get("route_solved")),
        "hard_flags": list(row.get("hard_flags") or []),
        "risk_flags": list(row.get("risk_flags") or []),
        "bridge": {
            "source": bridge.get("source") or "",
            "bridge_direction": bridge.get("bridge_direction") or "",
            "verifier_score": bridge.get("verifier_score"),
            "tanimoto": bridge.get("tanimoto"),
            "enzyme_ec_sample": list(bridge.get("enzyme_ec_sample") or []),
        },
        "steps": list(row.get("steps") or []),
        "audited_enzyme_steps": list(row.get("audited_enzyme_steps") or []),
        "evidence_statement": [
            "真实 live provider 生成并被 route-tree 选中的酶步",
            "目标分子存在 verifier-pass bridge evidence",
            "质量审计未发现自环、产物不匹配或小试剂伪反应硬错误",
            "当前仍为 partial route，尚未 stock-closed",
        ],
    }


def summarize(cards: list[dict[str, Any]]) -> dict[str, Any]:
    policy_counts = Counter(card["policy"] for card in cards)
    quality_counts = Counter(card["route_quality_level"] for card in cards)
    unique_targets = {card["chemical_inchikey"] or card["target_smiles"] for card in cards}
    risk_counts = Counter(flag for card in cards for flag in card.get("risk_flags") or [])
    return {
        "cards": len(cards),
        "unique_targets": len(unique_targets),
        "policy_counts": dict(sorted(policy_counts.items())),
        "quality_counts": dict(sorted(quality_counts.items())),
        "stock_closed_cards": sum(int(card["stock_closed"]) for card in cards),
        "route_solved_cards": sum(int(card["route_solved"]) for card in cards),
        "hard_flag_cards": sum(int(bool(card.get("hard_flags"))) for card in cards),
        "risk_flag_counts": dict(sorted(risk_counts.items())),
    }


def render_markdown(package: dict[str, Any]) -> str:
    lines = [
        "# P5 Bridge Evidence Package v0",
        "",
        "本包只收录 production-policy live routes，排除了 enzyme-only diagnostic 路线和 ungated 对照路线。",
        "",
        "重要边界：这些路线是 evidence-supported partial routes，不是 stock-closed solved synthesis plans。",
        "",
        "## Summary",
        "",
        f"- Evidence cards: {package['summary']['cards']}",
        f"- Unique targets: {package['summary']['unique_targets']}",
        f"- Stock-closed cards: {package['summary']['stock_closed_cards']}",
        f"- Route-solved cards: {package['summary']['route_solved_cards']}",
        f"- Hard-flag cards: {package['summary']['hard_flag_cards']}",
        f"- Policy counts: {json.dumps(package['summary']['policy_counts'], ensure_ascii=False)}",
        f"- Risk flags: {json.dumps(package['summary']['risk_flag_counts'], ensure_ascii=False)}",
        "",
        "## Evidence Cards",
        "",
    ]
    for card in package["cards"]:
        bridge = card["bridge"]
        lines.extend(
            [
                f"### {card['card_id']}",
                "",
                f"- Target: `{card['target_smiles']}`",
                f"- Policy: `{card['policy']}`; route rank: {card['route_rank']}",
                f"- Sources: {', '.join(card['selected_sources'])}",
                f"- Quality: `{card['route_quality_level']}`; stock_closed={card['stock_closed']}; route_solved={card['route_solved']}",
                f"- Bridge: `{bridge['source']}` / `{bridge['bridge_direction']}`; verifier_score={bridge['verifier_score']}; tanimoto={bridge['tanimoto']}",
                f"- EC sample: {', '.join(bridge['enzyme_ec_sample']) if bridge['enzyme_ec_sample'] else 'N/A'}",
                f"- Hard flags: {', '.join(card['hard_flags']) if card['hard_flags'] else 'none'}",
                f"- Risk flags: {', '.join(card['risk_flags']) if card['risk_flags'] else 'none'}",
                "",
                "Steps:",
                "",
            ]
        )
        for idx, step in enumerate(card["steps"], start=1):
            lines.append(
                f"{idx}. `{step.get('source', '')}` `{step.get('main_reactant', '')}` >> `{step.get('product', '')}`"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build P5 bridge evidence package")
    parser.add_argument("--audit-rows", type=Path, default=DEFAULT_AUDIT_ROWS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()

    cards = select_cards(read_jsonl(args.audit_rows), limit=max(1, int(args.limit)))
    package = {
        "schema_version": "p5_bridge_evidence_package_v0",
        "source_audit_rows": str(args.audit_rows),
        "selection_policy": {
            "allowed_policies": list(PRODUCTION_POLICIES),
            "excluded": ["enzyme_only_bridge_gated", "ungated_default_source_gate"],
            "requires_positive_label": True,
            "requires_quality_level": ["production_candidate_partial", "production_candidate_stock_closed"],
        },
        "summary": summarize(cards),
        "cards": cards,
        "limitations": [
            "0 cards are stock-closed in the current package.",
            "Most enzyme steps carry generic EC labels such as 1.x; richer EC/reaction-center evidence is still needed.",
            "This package supports P5 diagnostic reporting, not final medicinal chemistry route recommendation.",
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "p5_bridge_evidence_package.json"
    md_path = args.output_dir / "p5_bridge_evidence_package.md"
    json_path.write_text(json.dumps(package, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    md_path.write_text(render_markdown(package), encoding="utf-8")
    print(json.dumps({"json": str(json_path), "md": str(md_path), "summary": package["summary"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
