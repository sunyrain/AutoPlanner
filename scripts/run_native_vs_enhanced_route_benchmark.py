"""Route-level comparison: native ChemEnzy vs enhanced AutoPlanner route-tree.

The benchmark intentionally records comparability caveats.  ChemEnzy native and
route-tree use different search implementations, but they are run on the same
targets with matched depth/result budgets so we can measure whether the new
enzyme coverage layer changes route-level behavior.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cascade_planner.baselines.chem_enzy_adapter import (
    ChemEnzyBackendAdapter,
    DEFAULT_ONE_STEP_MODELS,
    DEFAULT_STOCKS,
)
from cascade_planner.baselines.proposal_gate import gate_web_route
from cascade_planner.baselines.route_contract import RouteCandidate, RouteSearchConfig
from cascade_planner.baselines.route_plausibility import audit_route_plausibility
from cascade_planner.cascade_search.bridge_retriever_v0 import BridgeRetrieverV0
from cascade_planner.cascade_search.enzyme_sp_verifier_v1 import EnzymeSPVerifierV1Scorer
from cascade_planner.cascadeboard.live_retro import build_live_retro_engine, retro_engine_cache_stats
from cascade_planner.cascadeboard.route_export import route_metrics, route_result_to_dict
from cascade_planner.cascadeboard.route_recovery import canonical_smiles
from cascade_planner.route_tree.search import NeuralGuidedAOSearch
from cascade_planner.route_tree.source_gate import BridgeAwareSourceGate, SourceGate
from scripts.run_bridge_live_policy_benchmark_v0 import load_negative_targets, load_positive_targets


DEFAULT_PACK_DIR = Path("data/bridge_pack_v0")
DEFAULT_PROBE_ROWS = Path("results/shared/bridge_gate_ablation_v0_20260527/live_enzyme_bridge_target_probe_rows.jsonl")
DEFAULT_OUTPUT_DIR = Path("results/shared/native_vs_enhanced_route_benchmark_20260528")
FINAL_CLEAN_FASTCLOSURE_PRESET = "final_clean_fastclosure_p16n16"
FINAL_CLEAN_FASTCLOSURE_MATERIAL_GATE_PRESET = "final_clean_fastclosure_material_gate_p16n16"
FINAL_CLEAN_FASTCLOSURE_MATERIAL_GATE_SEMISYNTHESIS_PRESET = (
    "final_clean_fastclosure_material_gate_semisynthesis_p16n16"
)
FINAL_CLEAN_FASTCLOSURE_MATERIAL_GATE_SEMISYNTHESIS_CHEMICAL_ANCHOR_PRESET = (
    "final_clean_fastclosure_material_gate_semisynthesis_chemical_anchor_p16n16"
)
ENZYME_SOURCES = {
    "enzyme_precedent",
    "v3_retrieval",
    "enzyformer",
    "enzexpand",
    "retrorules",
    "rhea",
    "rhea_template",
    "retrieval",
    "chem_enzy_onmt",
    "chem_enzy_bionav",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack-dir", type=Path, default=DEFAULT_PACK_DIR)
    parser.add_argument("--probe-rows", type=Path, default=DEFAULT_PROBE_ROWS)
    parser.add_argument(
        "--target-rows",
        type=Path,
        default=None,
        help=(
            "Optional JSONL file with explicit benchmark targets. Each row should contain "
            "target_smiles and may contain label/label_source. When provided, "
            "--positives/--negatives are ignored except for report metadata."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--enhancement-preset",
        choices=(
            "",
            FINAL_CLEAN_FASTCLOSURE_PRESET,
            FINAL_CLEAN_FASTCLOSURE_MATERIAL_GATE_PRESET,
            FINAL_CLEAN_FASTCLOSURE_MATERIAL_GATE_SEMISYNTHESIS_PRESET,
            FINAL_CLEAN_FASTCLOSURE_MATERIAL_GATE_SEMISYNTHESIS_CHEMICAL_ANCHOR_PRESET,
        ),
        default="",
        help=(
            "Apply an audited enhanced route-tree configuration. "
            f"{FINAL_CLEAN_FASTCLOSURE_PRESET} reproduces the clean p16n16 fastclosure preset; "
            f"{FINAL_CLEAN_FASTCLOSURE_MATERIAL_GATE_PRESET} adds the audited enzyme_precedent "
            "material-quality gate; "
            f"{FINAL_CLEAN_FASTCLOSURE_MATERIAL_GATE_SEMISYNTHESIS_PRESET} also enables the "
            "source-supported semisynthesis rescue/stock sidecar; "
            f"{FINAL_CLEAN_FASTCLOSURE_MATERIAL_GATE_SEMISYNTHESIS_CHEMICAL_ANCHOR_PRESET} "
            "also enables the curated source-supported chemical anchor sidecar."
        ),
    )
    parser.add_argument("--positives", type=int, default=2)
    parser.add_argument("--negatives", type=int, default=2)
    parser.add_argument("--max-targets", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260528)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=25)
    parser.add_argument("--expansion-topk", type=int, default=50)
    parser.add_argument("--branch-factor", type=int, default=8)
    parser.add_argument("--expansion-budget", type=int, default=40)
    parser.add_argument("--n-results", type=int, default=3)
    parser.add_argument(
        "--preset-n-results-override",
        type=int,
        default=0,
        help=(
            "Explicitly override n_results after applying --enhancement-preset. "
            "Default 0 preserves the audited preset value."
        ),
    )
    parser.add_argument(
        "--preset-expansion-budget-override",
        type=int,
        default=0,
        help=(
            "Explicitly override expansion_budget after applying --enhancement-preset. "
            "Default 0 preserves the audited preset value."
        ),
    )
    parser.add_argument(
        "--preset-max-depth-override",
        type=int,
        default=0,
        help=(
            "Explicitly override max_depth after applying --enhancement-preset. "
            "Default 0 preserves the audited preset value."
        ),
    )
    parser.add_argument(
        "--preset-route-tree-timeout-s-override",
        type=float,
        default=0.0,
        help=(
            "Explicitly override route_tree_timeout_s after applying --enhancement-preset. "
            "Default 0 preserves the audited preset value."
        ),
    )
    parser.add_argument("--gpu", type=int, default=-1)
    parser.add_argument("--native-timeout-s", type=float, default=90.0)
    parser.add_argument("--route-tree-timeout-s", type=float, default=60.0)
    parser.add_argument(
        "--native-one-step-models",
        default=",".join(DEFAULT_ONE_STEP_MODELS),
        help=(
            "Comma-separated ChemEnzy one-step models for the native baseline. "
            "Use graphfp_models.USPTO-full_remapped,onmt_models.bionav_native_one_step "
            "for a strict original-BioNav baseline."
        ),
    )
    parser.add_argument(
        "--bridge-enzyme-bonus",
        type=float,
        default=0.0,
        help="Selection-cost bonus for bridge-supported enzymatic actions in enhanced route-tree.",
    )
    parser.add_argument(
        "--enzyme-precedent-min-budget",
        type=int,
        default=0,
        help="Minimum proposal budget reserved for enzyme_precedent when the source is available.",
    )
    parser.add_argument(
        "--source-min-budgets",
        default="",
        help=(
            "Comma-separated source:budget floors for enhanced route-tree, e.g. "
            "chem_enzy_bionav:8,enzyme_precedent:4,v3_retrieval:4. "
            "--enzyme-precedent-min-budget is merged as a backwards-compatible alias."
        ),
    )
    parser.add_argument(
        "--enable-enzyme-continuation-source-gate",
        action="store_true",
        help=(
            "After an enhanced route has already selected an enzymatic step, "
            "allow downstream intermediates to keep controlled enzymatic source budgets "
            "even when the bridge retriever has no direct hit for that intermediate."
        ),
    )
    parser.add_argument(
        "--enable-selected-enzyme-evidence-enrichment",
        action="store_true",
        help=(
            "After route search, enrich selected enzymatic steps with enzyme_precedent "
            "support and transition signatures. This audits the returned route only and "
            "does not affect proposal ranking."
        ),
    )
    parser.add_argument("--selected-enzyme-evidence-topk", type=int, default=3)
    parser.add_argument("--selected-enzyme-evidence-min-similarity", type=float, default=0.35)
    parser.add_argument(
        "--enable-sp-v1-enzyme-result-selector",
        action="store_true",
        help=(
            "When a bridge-supported target has both a solved chemical route and a solved "
            "SP-v1 accepted enzyme route in the route-tree result pool, return the enzyme "
            "route first. This does not create enzyme routes or bypass SP-v1 filtering."
        ),
    )
    parser.add_argument("--sp-v1-enzyme-result-pool-min", type=int, default=5)
    parser.add_argument("--sp-v1-enzyme-selector-max-rank", type=int, default=5)
    parser.add_argument("--sp-v1-enzyme-selector-max-extra-cost", type=float, default=0.0)
    parser.add_argument(
        "--enable-sp-v1-enzyme-selector-cost-exception",
        action="store_true",
        help=(
            "Allow the SP-v1 enzyme selector to promote a bridge-supported accepted enzyme "
            "route over a non-enzyme top route in the same solved tier when the route cost "
            "is within --sp-v1-enzyme-selector-cost-exception-max-extra-cost."
        ),
    )
    parser.add_argument("--sp-v1-enzyme-selector-cost-exception-max-extra-cost", type=float, default=0.0)
    parser.add_argument(
        "--enable-enzyme-sp-material-gate",
        action="store_true",
        help=(
            "Reject SP-v1 accepted enzymatic actions whose enzyme_step_quality_v1 material "
            "audit is reject. Default off; use --enzyme-sp-material-gate-sources to limit "
            "the gate to selected proposal sources."
        ),
    )
    parser.add_argument(
        "--enzyme-sp-material-gate-sources",
        default="",
        help=(
            "Optional comma-separated source allowlist for --enable-enzyme-sp-material-gate, "
            "for example enzyme_precedent. Empty applies the gate to all enzymatic sources."
        ),
    )
    parser.add_argument(
        "--enhanced-stock",
        choices=("none", "zinc"),
        default="zinc",
        help="Stock checker used by enhanced route-tree. Native ChemEnzy uses the configured vendor stock.",
    )
    parser.add_argument(
        "--disable-common-stock",
        action="store_true",
        help="Do not add conservative common-commodity molecules such as ammonia and oxygen to enhanced stock.",
    )
    parser.add_argument(
        "--disable-vendor-stock",
        action="store_true",
        help="Do not add the ChemEnzy vendor Zinc_Fix-stock SQLite supplement to enhanced stock.",
    )
    parser.add_argument(
        "--vendor-stock-index",
        type=Path,
        default=Path("results/shared/chemenzy_vendor_stock/zinc_fix_stock_smiles.sqlite"),
        help="SQLite index built from ChemEnzy Zinc_Fix-stock for enhanced stock parity.",
    )
    parser.add_argument(
        "--enable-semisynthesis-stock",
        action="store_true",
        help="Treat curated, source-supported semisynthesis precursors as stock. Default off.",
    )
    parser.add_argument(
        "--enable-semisynthesis-rescue-source",
        action="store_true",
        help="Enable a lightweight semisynthesis_rescue route-tree proposal source. Default off.",
    )
    parser.add_argument("--semisynthesis-rescue-min-budget", type=int, default=2)
    parser.add_argument(
        "--enable-chemical-anchor-stock",
        action="store_true",
        help="Treat curated, source-supported chemical anchor precursors as stock. Default off.",
    )
    parser.add_argument(
        "--enable-chemical-anchor-rescue-source",
        action="store_true",
        help="Enable a lightweight chemical_anchor_rescue route-tree proposal source. Default off.",
    )
    parser.add_argument("--chemical-anchor-rescue-min-budget", type=int, default=2)
    parser.set_defaults(stock_aware_action_rerank=True)
    parser.add_argument(
        "--stock-aware-action-rerank",
        dest="stock_aware_action_rerank",
        action="store_true",
        help="Prefer enhanced route-tree actions whose reactants exactly hit stock. Enabled by default.",
    )
    parser.add_argument(
        "--disable-stock-aware-action-rerank",
        dest="stock_aware_action_rerank",
        action="store_false",
        help="Disable stock-aware action bonuses for ablations.",
    )
    parser.add_argument("--exact-stock-reactant-bonus", type=float, default=2.0)
    parser.add_argument("--full-stock-action-bonus", type=float, default=5.0)
    parser.add_argument(
        "--enzyme-sp-accepted-bonus",
        type=float,
        default=0.0,
        help="Selection-cost bonus for SP-v1 accepted enzymatic actions in enhanced route-tree.",
    )
    parser.add_argument(
        "--enzyme-sp-score-bonus",
        type=float,
        default=0.0,
        help="Additional SP-v1 bonus weight multiplied by score-threshold margin.",
    )
    parser.add_argument(
        "--chem-template-max-per-query",
        type=int,
        default=0,
        help="Set AUTOPLANNER_CHEM_TEMPLATES_MAX_PER_QUERY for enhanced route-tree; 0 preserves default.",
    )
    parser.add_argument(
        "--chem-template-max-templates",
        type=int,
        default=0,
        help="Set AUTOPLANNER_CHEM_TEMPLATES_MAX_TEMPLATES for enhanced route-tree; 0 preserves default.",
    )
    parser.add_argument(
        "--enable-enhanced-chemenzy-assembly",
        action="store_true",
        help="Enable assembled ChemEnzy one-step enhancements inside enhanced route-tree.",
    )
    parser.add_argument(
        "--enable-enhanced-chemical-fusion-source",
        action="store_true",
        help="Enable GraphFP+dual-tower fusion as a separate chemical route-tree source.",
    )
    parser.add_argument(
        "--enable-template-relevance-source",
        action="store_true",
        help="Enable ChemEnzy template_relevance .mar models as a separate chemical route-tree source.",
    )
    parser.add_argument(
        "--disable-retrochimera-source",
        action="store_true",
        help="Disable the legacy RetroChimera source inside enhanced route-tree.",
    )
    parser.add_argument(
        "--disable-chemtemplates-after-depth",
        type=int,
        default=-1,
        help=(
            "Disable the chemtemplates source after this route-tree depth. "
            "Use 0 to allow root calls only; -1 preserves default behavior."
        ),
    )
    parser.add_argument(
        "--enable-stock-closing-probe",
        action="store_true",
        help="Enable late-stage direct stock-closing probe for low-rank chemical closure candidates.",
    )
    parser.add_argument(
        "--stock-closing-probe-sources",
        default="chem_enzy_graphfp_fusion,template_relevance,chemtemplates",
        help="Comma-separated route-tree sources queried by the late stock-closing probe.",
    )
    parser.add_argument("--stock-closing-probe-topk", type=int, default=75)
    parser.add_argument("--stock-closing-probe-topk-cap", type=int, default=75)
    parser.add_argument("--stock-closing-probe-max-actions", type=int, default=4)
    parser.add_argument("--stock-closing-probe-remaining-depth", type=int, default=2)
    parser.add_argument(
        "--template-relevance-models",
        default="template_relevance.reaxys",
        help="Comma-separated template_relevance models for the independent chemical source.",
    )
    parser.add_argument("--template-relevance-topk", type=int, default=20)
    parser.add_argument("--template-relevance-min-budget", type=int, default=4)
    parser.add_argument("--template-relevance-gpu", type=int, default=-1)
    parser.add_argument(
        "--template-relevance-vendor-root",
        default="vendor/ChemEnzyRetroPlanner",
        help="ChemEnzy vendor root containing retro_planner and downloaded .mar files.",
    )
    parser.add_argument(
        "--enable-enhanced-bionav-source",
        action="store_true",
        help="Enable enhanced BioNav ONMT as a separate enzymatic route-tree source.",
    )
    parser.add_argument(
        "--enhanced-bionav-model",
        type=Path,
        default=Path(
            "results/shared/bionav_v2_formal_ec_context_20260529_valid32/checkpoints/archive/"
            "bionav_v2_ec_context_step_15000_benchmarked.pt"
        ),
        help="Product-marker BioNav checkpoint used by the assembled enhanced ChemEnzy one-step source.",
    )
    parser.add_argument("--enhanced-chemenzy-topk", type=int, default=50)
    parser.add_argument("--enhanced-chemical-fusion-topk", type=int, default=50)
    parser.add_argument("--enhanced-chemical-fusion-min-budget", type=int, default=4)
    parser.add_argument(
        "--enhanced-fusion-mode",
        choices=("graphfp_first", "rrf", "best_rank", "score_sum"),
        default="graphfp_first",
        help=(
            "GraphFP/dual-tower fusion mode used by enhanced chemical sources. "
            "graphfp_first preserves native GraphFP ordering and appends dual-tower coverage."
        ),
    )
    parser.add_argument("--enhanced-bionav-source-topk", type=int, default=10)
    parser.add_argument("--enhanced-chemenzy-min-budget", type=int, default=6)
    parser.add_argument("--enhanced-chemenzy-protected-topk", type=int, default=4)
    parser.add_argument("--enhanced-chemenzy-protected-front", type=int, default=1)
    parser.add_argument(
        "--enhanced-chemenzy-route-mode",
        choices=("fallback", "adaptive", "eager", "off"),
        default="fallback",
        help="Route-tree scheduling mode for the assembled ChemEnzy one-step source.",
    )
    parser.add_argument("--enhanced-dualtower-device", default=None)
    parser.add_argument(
        "--enable-native-chemical-rescue",
        action="store_true",
        help=(
            "If enhanced route-tree returns no solved route and no enzyme route, "
            "run ChemEnzy native as a chemical-only rescue and mark appended routes."
        ),
    )
    parser.add_argument(
        "--native-chemical-rescue-one-step-models",
        default="graphfp_models.USPTO-full_remapped,onmt_models.bionav_native_one_step",
        help="Comma-separated ChemEnzy one-step models used by native chemical rescue.",
    )
    parser.add_argument(
        "--native-chemical-rescue-require-proposal-gate",
        action="store_true",
        help=(
            "Accept native chemical rescue routes only when the conservative "
            "proposal material gate keeps every step."
        ),
    )
    parser.add_argument("--native-chemical-rescue-timeout-s", type=float, default=90.0)
    parser.add_argument("--skip-native", action="store_true")
    parser.add_argument("--skip-enhanced", action="store_true")
    parser.add_argument("--reuse-live-engine", action="store_true")
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=0,
        help=(
            "Write partial rows and summaries every N completed targets. "
            "0 disables checkpoint writes; final report output is unchanged."
        ),
    )
    parser.add_argument(
        "--resume-rows",
        type=Path,
        default=None,
        help=(
            "Optional checkpoint/final rows JSONL to preload. Targets that already have "
            "all requested run rows are skipped, and final summaries include preloaded rows."
        ),
    )
    parser.add_argument(
        "--shuffle-targets",
        action="store_true",
        help="Shuffle after preserving the requested positive/negative counts.",
    )
    return apply_enhancement_preset(parser.parse_args())


def apply_enhancement_preset(args: argparse.Namespace) -> argparse.Namespace:
    preset = str(getattr(args, "enhancement_preset", "") or "")
    if not preset:
        return args
    if preset not in {
        FINAL_CLEAN_FASTCLOSURE_PRESET,
        FINAL_CLEAN_FASTCLOSURE_MATERIAL_GATE_PRESET,
        FINAL_CLEAN_FASTCLOSURE_MATERIAL_GATE_SEMISYNTHESIS_PRESET,
        FINAL_CLEAN_FASTCLOSURE_MATERIAL_GATE_SEMISYNTHESIS_CHEMICAL_ANCHOR_PRESET,
    }:
        raise ValueError(f"Unknown enhancement preset: {preset}")
    args.max_depth = 6
    args.expansion_topk = 60
    args.branch_factor = 8
    args.expansion_budget = 90
    args.n_results = 1
    args.route_tree_timeout_s = 240.0
    args.bridge_enzyme_bonus = 2.0
    args.source_min_budgets = "chem_enzy_bionav:8,enzyme_precedent:4,v3_retrieval:4,enzexpand:4,enzyformer:3"
    args.enable_enzyme_continuation_source_gate = True
    args.enable_selected_enzyme_evidence_enrichment = True
    args.selected_enzyme_evidence_topk = 3
    args.selected_enzyme_evidence_min_similarity = 0.35
    args.enable_sp_v1_enzyme_result_selector = True
    args.sp_v1_enzyme_result_pool_min = 5
    args.sp_v1_enzyme_selector_max_rank = 5
    args.sp_v1_enzyme_selector_max_extra_cost = 0.0
    args.enable_sp_v1_enzyme_selector_cost_exception = True
    args.sp_v1_enzyme_selector_cost_exception_max_extra_cost = 2.0
    args.enzyme_sp_accepted_bonus = 3.0
    args.enzyme_sp_score_bonus = 1.0
    args.enable_enhanced_chemical_fusion_source = True
    args.enhanced_chemical_fusion_topk = 60
    args.enhanced_chemical_fusion_min_budget = 4
    args.enhanced_fusion_mode = "graphfp_first"
    args.enable_template_relevance_source = True
    args.template_relevance_models = "template_relevance.reaxys"
    args.template_relevance_topk = 20
    args.template_relevance_min_budget = 4
    args.disable_retrochimera_source = True
    args.enable_enhanced_bionav_source = True
    args.enhanced_bionav_source_topk = 10
    args.disable_chemtemplates_after_depth = 0
    args.enable_stock_closing_probe = True
    args.stock_closing_probe_sources = "chem_enzy_graphfp_fusion,template_relevance"
    args.stock_closing_probe_topk = 60
    args.stock_closing_probe_topk_cap = 60
    args.stock_closing_probe_max_actions = 4
    args.stock_closing_probe_remaining_depth = 2
    if preset in {
        FINAL_CLEAN_FASTCLOSURE_MATERIAL_GATE_PRESET,
        FINAL_CLEAN_FASTCLOSURE_MATERIAL_GATE_SEMISYNTHESIS_PRESET,
        FINAL_CLEAN_FASTCLOSURE_MATERIAL_GATE_SEMISYNTHESIS_CHEMICAL_ANCHOR_PRESET,
    }:
        args.enable_enzyme_sp_material_gate = True
        args.enzyme_sp_material_gate_sources = "enzyme_precedent"
    if preset == FINAL_CLEAN_FASTCLOSURE_MATERIAL_GATE_SEMISYNTHESIS_PRESET:
        args.enable_semisynthesis_stock = True
        args.enable_semisynthesis_rescue_source = True
        args.semisynthesis_rescue_min_budget = 2
    if preset == FINAL_CLEAN_FASTCLOSURE_MATERIAL_GATE_SEMISYNTHESIS_CHEMICAL_ANCHOR_PRESET:
        args.enable_semisynthesis_stock = True
        args.enable_semisynthesis_rescue_source = True
        args.semisynthesis_rescue_min_budget = 2
        args.enable_chemical_anchor_stock = True
        args.enable_chemical_anchor_rescue_source = True
        args.chemical_anchor_rescue_min_budget = 2
    _apply_preset_overrides(args)
    return args


def _apply_preset_overrides(args: argparse.Namespace) -> None:
    overrides = (
        ("preset_n_results_override", "n_results", int),
        ("preset_expansion_budget_override", "expansion_budget", int),
        ("preset_max_depth_override", "max_depth", int),
        ("preset_route_tree_timeout_s_override", "route_tree_timeout_s", float),
    )
    for source, target, caster in overrides:
        value = getattr(args, source, 0)
        if value is None:
            continue
        try:
            numeric = caster(value)
        except (TypeError, ValueError):
            continue
        if numeric > 0:
            setattr(args, target, numeric)


def main() -> None:
    args = parse_args()
    started = time.monotonic()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    targets = load_benchmark_targets(args)
    rows: list[dict[str, Any]] = load_resume_rows(args.resume_rows) if args.resume_rows else []
    resumed_target_keys = completed_resume_target_keys(rows, args=args)
    if rows:
        print(
            json.dumps(
                {
                    "resume_rows": str(args.resume_rows),
                    "preloaded_rows": len(rows),
                    "resumed_targets": len(resumed_target_keys),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    native_adapter = ChemEnzyBackendAdapter(gpu=int(args.gpu)) if not args.skip_native else None
    native_rescue_adapter = None
    if bool(args.enable_native_chemical_rescue):
        native_rescue_adapter = native_adapter or ChemEnzyBackendAdapter(gpu=int(args.gpu))
    retriever = BridgeRetrieverV0(args.pack_dir, scorer=None)
    enzyme_sp = EnzymeSPVerifierV1Scorer() if not args.skip_enhanced else None
    shared_live_engine = None
    if args.reuse_live_engine and not args.skip_enhanced:
        with temporary_env(**enhanced_env_values(args)):
            shared_live_engine = build_live_retro_engine()

    for idx, target in enumerate(targets, start=1):
        target_smiles = str(target.get("target_smiles") or "")
        if target_resume_key(target) in resumed_target_keys:
            print(f"[{idx}/{len(targets)}] {target_smiles} (resumed)", flush=True)
            continue
        print(f"[{idx}/{len(targets)}] {target_smiles}", flush=True)
        if native_adapter is not None:
            rows.append(run_native(target, adapter=native_adapter, args=args))
        if not args.skip_enhanced:
            rows.append(
                run_enhanced_route_tree(
                    target,
                    live_engine=shared_live_engine,
                    retriever=retriever,
                    enzyme_sp=enzyme_sp,
                    native_rescue_adapter=native_rescue_adapter,
                    args=args,
                )
            )
        checkpoint_every = max(0, int(getattr(args, "checkpoint_every", 0) or 0))
        if checkpoint_every and idx % checkpoint_every == 0:
            write_checkpoint_outputs(
                args,
                rows,
                processed_targets=idx,
                total_targets=len(targets),
                started=started,
            )

    report = {
        "schema_version": "native_vs_enhanced_route_benchmark.v0",
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "inputs": {
            "pack_dir": str(args.pack_dir),
            "probe_rows": str(args.probe_rows),
            "target_rows": str(args.target_rows) if args.target_rows else "",
            "enhancement_preset": str(getattr(args, "enhancement_preset", "") or ""),
            "targets": len(targets),
            "positives": sum(1 for row in targets if int(row.get("label") or 0) == 1),
            "negatives": sum(1 for row in targets if int(row.get("label") or 0) == 0),
            "max_depth": int(args.max_depth),
            "iterations": int(args.iterations),
            "expansion_topk": int(args.expansion_topk),
            "branch_factor": int(args.branch_factor),
            "expansion_budget": int(args.expansion_budget),
            "n_results": int(args.n_results),
            "native_timeout_s": float(args.native_timeout_s),
            "native_one_step_models": native_one_step_models(args),
            "route_tree_timeout_s": float(args.route_tree_timeout_s),
            "bridge_enzyme_bonus": float(args.bridge_enzyme_bonus),
            "enzyme_precedent_min_budget": int(args.enzyme_precedent_min_budget),
            "source_min_budgets": source_min_budget_env(args),
            "enzyme_continuation_source_gate": bool(args.enable_enzyme_continuation_source_gate),
            "selected_enzyme_evidence_enrichment": bool(args.enable_selected_enzyme_evidence_enrichment),
            "selected_enzyme_evidence_topk": int(args.selected_enzyme_evidence_topk),
            "selected_enzyme_evidence_min_similarity": float(args.selected_enzyme_evidence_min_similarity),
            "sp_v1_enzyme_result_selector": bool(args.enable_sp_v1_enzyme_result_selector),
            "sp_v1_enzyme_result_pool_min": int(args.sp_v1_enzyme_result_pool_min),
            "sp_v1_enzyme_selector_max_rank": int(args.sp_v1_enzyme_selector_max_rank),
            "sp_v1_enzyme_selector_max_extra_cost": float(args.sp_v1_enzyme_selector_max_extra_cost),
            "sp_v1_enzyme_selector_cost_exception": bool(
                getattr(args, "enable_sp_v1_enzyme_selector_cost_exception", False)
            ),
            "sp_v1_enzyme_selector_cost_exception_max_extra_cost": float(
                getattr(args, "sp_v1_enzyme_selector_cost_exception_max_extra_cost", 0.0) or 0.0
            ),
            "enzyme_sp_material_gate": bool(getattr(args, "enable_enzyme_sp_material_gate", False)),
            "enzyme_sp_material_gate_sources": str(getattr(args, "enzyme_sp_material_gate_sources", "") or ""),
            "enhanced_stock": str(args.enhanced_stock),
            "common_stock_enabled": not bool(args.disable_common_stock),
            "vendor_stock_enabled": not bool(args.disable_vendor_stock),
            "vendor_stock_index": str(args.vendor_stock_index),
            "vendor_stock_index_available": Path(args.vendor_stock_index).exists(),
            "semisynthesis_stock_enabled": bool(getattr(args, "enable_semisynthesis_stock", False)),
            "semisynthesis_rescue_source_enabled": bool(
                getattr(args, "enable_semisynthesis_rescue_source", False)
            ),
            "semisynthesis_rescue_min_budget": int(getattr(args, "semisynthesis_rescue_min_budget", 2) or 2),
            "chemical_anchor_stock_enabled": bool(getattr(args, "enable_chemical_anchor_stock", False)),
            "chemical_anchor_rescue_source_enabled": bool(
                getattr(args, "enable_chemical_anchor_rescue_source", False)
            ),
            "chemical_anchor_rescue_min_budget": int(
                getattr(args, "chemical_anchor_rescue_min_budget", 2) or 2
            ),
            "stock_aware_action_rerank": bool(args.stock_aware_action_rerank),
            "exact_stock_reactant_bonus": float(args.exact_stock_reactant_bonus),
            "full_stock_action_bonus": float(args.full_stock_action_bonus),
            "enzyme_sp_accepted_bonus": float(args.enzyme_sp_accepted_bonus),
            "enzyme_sp_score_bonus": float(args.enzyme_sp_score_bonus),
            "chem_template_max_per_query": int(args.chem_template_max_per_query),
            "chem_template_max_templates": int(args.chem_template_max_templates),
            "enable_enhanced_chemenzy_assembly": bool(args.enable_enhanced_chemenzy_assembly),
            "enable_enhanced_chemical_fusion_source": bool(args.enable_enhanced_chemical_fusion_source),
            "enable_template_relevance_source": bool(args.enable_template_relevance_source),
            "retrochimera_source_enabled": not bool(args.disable_retrochimera_source),
            "disable_chemtemplates_after_depth": int(args.disable_chemtemplates_after_depth),
            "stock_closing_probe_enabled": bool(args.enable_stock_closing_probe),
            "stock_closing_probe_sources": stock_closing_probe_sources(args),
            "stock_closing_probe_topk": int(args.stock_closing_probe_topk),
            "stock_closing_probe_topk_cap": int(args.stock_closing_probe_topk_cap),
            "stock_closing_probe_max_actions": int(args.stock_closing_probe_max_actions),
            "stock_closing_probe_remaining_depth": int(args.stock_closing_probe_remaining_depth),
            "template_relevance_models": template_relevance_models(args),
            "template_relevance_topk": int(args.template_relevance_topk),
            "template_relevance_min_budget": int(args.template_relevance_min_budget),
            "template_relevance_gpu": int(args.template_relevance_gpu),
            "template_relevance_vendor_root": str(args.template_relevance_vendor_root),
            "enable_enhanced_bionav_source": bool(args.enable_enhanced_bionav_source),
            "enhanced_bionav_model": str(args.enhanced_bionav_model),
            "enhanced_chemenzy_topk": int(args.enhanced_chemenzy_topk),
            "enhanced_chemical_fusion_topk": int(args.enhanced_chemical_fusion_topk),
            "enhanced_chemical_fusion_min_budget": int(args.enhanced_chemical_fusion_min_budget),
            "enhanced_fusion_mode": str(args.enhanced_fusion_mode),
            "brenda_condition_prior": True,
            "route_tree_condition_prediction": True,
            "route_tree_condition_model": "rcr",
            "route_tree_condition_prediction_chemical_only": True,
            "enhanced_bionav_source_topk": int(args.enhanced_bionav_source_topk),
            "enhanced_chemenzy_min_budget": int(args.enhanced_chemenzy_min_budget),
            "enhanced_chemenzy_protected_topk": int(args.enhanced_chemenzy_protected_topk),
            "enhanced_chemenzy_protected_front": int(args.enhanced_chemenzy_protected_front),
            "enhanced_chemenzy_route_mode": str(args.enhanced_chemenzy_route_mode),
            "enable_native_chemical_rescue": bool(args.enable_native_chemical_rescue),
            "native_chemical_rescue_one_step_models": native_chemical_rescue_models(args),
            "native_chemical_rescue_require_proposal_gate": bool(
                args.native_chemical_rescue_require_proposal_gate
            ),
            "native_chemical_rescue_timeout_s": float(args.native_chemical_rescue_timeout_s),
            "skip_native": bool(args.skip_native),
            "skip_enhanced": bool(args.skip_enhanced),
        },
        "summaries": summarize(rows),
        "conclusion": conclusion(summarize(rows)),
        "rows_jsonl": str(args.output_dir / "native_vs_enhanced_route_rows.jsonl"),
        "comparability_notes": [
            "ChemEnzy native uses its vendor MCTS/planner and configured stock.",
            "Enhanced route-tree uses AutoPlanner proposal sources, bridge gate, enzyme_precedent retrieval, and SP-v1 gate.",
            "Small-molecule terminal handling can differ unless a shared stock checker is wired into both paths.",
            "This benchmark is route-level evidence, not yet a publication-scale benchmark.",
        ],
        "live_engine_cache_stats": retro_engine_cache_stats(shared_live_engine or {}) if shared_live_engine else {},
    }
    rows_path = args.output_dir / "native_vs_enhanced_route_rows.jsonl"
    report_json = args.output_dir / "native_vs_enhanced_route_report.json"
    report_md = args.output_dir / "native_vs_enhanced_route_report.md"
    rows_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    report_md.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({"report": str(report_json), "rows": str(rows_path), "conclusion": report["conclusion"]}, ensure_ascii=False, indent=2))


def load_resume_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        raise FileNotFoundError(f"resume rows not found: {path}")
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def completed_resume_target_keys(rows: list[dict[str, Any]], *, args: argparse.Namespace) -> set[str]:
    expected_runs: set[str] = set()
    if not bool(getattr(args, "skip_native", False)):
        expected_runs.add("native_chemenzy")
    if not bool(getattr(args, "skip_enhanced", False)):
        expected_runs.add("enhanced_route_tree")
    if not expected_runs:
        return set()
    seen: dict[str, set[str]] = {}
    for row in rows:
        key = target_resume_key(row)
        run = str(row.get("run") or "")
        if key and run in expected_runs:
            seen.setdefault(key, set()).add(run)
    return {key for key, runs in seen.items() if expected_runs.issubset(runs)}


def target_resume_key(payload: dict[str, Any]) -> str:
    smiles = str(
        payload.get("target_canonical")
        or payload.get("target_smiles")
        or payload.get("smiles")
        or payload.get("target")
        or ""
    )
    return canonical_smiles(smiles) or smiles


def write_checkpoint_outputs(
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
    *,
    processed_targets: int,
    total_targets: int,
    started: float,
) -> None:
    rows_path = args.output_dir / "native_vs_enhanced_route_rows.checkpoint.jsonl"
    summary_path = args.output_dir / "native_vs_enhanced_route_checkpoint.json"
    rows_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    payload = {
        "schema_version": "native_vs_enhanced_route_benchmark.checkpoint.v0",
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "processed_targets": int(processed_targets),
        "total_targets": int(total_targets),
        "rows": len(rows),
        "rows_jsonl": str(rows_path),
        "summaries": summarize(rows),
    }
    summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(
        json.dumps(
            {
                "checkpoint": str(summary_path),
                "rows": str(rows_path),
                "processed_targets": int(processed_targets),
                "total_targets": int(total_targets),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


def run_native(target: dict[str, Any], *, adapter: ChemEnzyBackendAdapter, args: argparse.Namespace) -> dict[str, Any]:
    target_smiles = str(target.get("target_smiles") or "")
    config = RouteSearchConfig(
        target_smiles=target_smiles,
        stock_names=list(DEFAULT_STOCKS),
        max_iterations=max(1, int(args.iterations)),
        max_depth=max(1, int(args.max_depth)),
        expansion_topk=max(1, int(args.expansion_topk)),
        one_step_models=native_one_step_models(args),
        search_flags={
            "gpu": int(args.gpu),
            "keep_search": True,
            "use_filter": False,
            "use_depth_value_fn": False,
            "cascade_search_context": {
                "enabled": True,
                "benchmark": "native_vs_enhanced_route_benchmark.v0",
                "target_smiles": target_smiles,
            },
        },
    )
    started = time.monotonic()
    with temporary_env(
        AUTOPLANNER_CHEMENZY_NATIVE_TIMEOUT_S=str(max(1.0, float(args.native_timeout_s))),
    ):
        result = adapter.run_target(config)
    elapsed = time.monotonic() - started
    routes = [native_route_payload(route) for route in result.routes[: max(1, int(args.n_results))]]
    return {
        **target_base(target),
        "run": "native_chemenzy",
        "ok": not bool(result.failures),
        "failure_categories": [failure.category for failure in result.failures],
        "failure_messages": [failure.message for failure in result.failures[:3]],
        "route_count": len(routes),
        "solved_routes": sum(1 for route in routes if bool(route.get("route_solved"))),
        "progressive_routes": sum(1 for route in routes if bool(route.get("progressive_route"))),
        "enzyme_routes": sum(1 for route in routes if bool(route.get("has_enzyme_step"))),
        "sp_v1_accepted_enzyme_routes": sum(1 for route in routes if bool(route.get("has_sp_v1_accepted_enzyme_step"))),
        "enzyme_proposal_calls": 0,
        "enzyme_proposal_candidates": 0,
        "mean_steps": mean(route.get("n_steps") for route in routes),
        "elapsed_s": round(float(elapsed), 3),
        "backend_elapsed_s": (result.raw_backend_metadata or {}).get("elapsed_s"),
        "stats": dict(result.raw_backend_metadata or {}),
        "routes": routes,
    }


def load_benchmark_targets(args: argparse.Namespace) -> list[dict[str, Any]]:
    target_rows = getattr(args, "target_rows", None)
    if target_rows:
        targets = load_explicit_target_rows(Path(target_rows))
        if args.max_targets > 0:
            targets = targets[: int(args.max_targets)]
        return targets
    positives = load_positive_targets(args.probe_rows, count=max(0, int(args.positives)))
    negatives = load_negative_targets(args.pack_dir, count=max(0, int(args.negatives)), seed=int(args.seed))
    targets = [*positives, *negatives]
    if bool(args.shuffle_targets):
        import random

        random.Random(int(args.seed)).shuffle(targets)
    if args.max_targets > 0:
        targets = targets[: int(args.max_targets)]
    return targets


def load_explicit_target_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        smiles = str(payload.get("target_smiles") or payload.get("smiles") or payload.get("target") or "")
        if not smiles:
            continue
        key = canonical_smiles(smiles) or smiles
        if key in seen:
            continue
        seen.add(key)
        row = dict(payload)
        row["target_smiles"] = smiles
        row["label"] = int(row.get("label") or 0)
        row.setdefault("label_source", "explicit_target_rows")
        rows.append(row)
    return rows


def native_one_step_models(args: argparse.Namespace) -> list[str]:
    raw = str(getattr(args, "native_one_step_models", "") or "")
    models = [item.strip() for item in raw.split(",") if item.strip()]
    return models or list(DEFAULT_ONE_STEP_MODELS)


def native_chemical_rescue_models(args: argparse.Namespace) -> list[str]:
    raw = str(getattr(args, "native_chemical_rescue_one_step_models", "") or "")
    models = [item.strip() for item in raw.split(",") if item.strip()]
    return models or ["graphfp_models.USPTO-full_remapped"]


def template_relevance_models(args: argparse.Namespace) -> list[str]:
    raw = str(getattr(args, "template_relevance_models", "") or "")
    models = [item.strip() for item in raw.split(",") if item.strip()]
    return models or ["template_relevance.reaxys"]


def stock_closing_probe_sources(args: argparse.Namespace) -> list[str]:
    raw = str(getattr(args, "stock_closing_probe_sources", "") or "")
    sources = [item.strip() for item in raw.split(",") if item.strip()]
    return sources or ["chem_enzy_graphfp_fusion", "template_relevance", "chemtemplates"]


def parse_source_budget_spec(raw: str) -> dict[str, int]:
    budgets: dict[str, int] = {}
    for item in str(raw or "").split(","):
        item = item.strip()
        if not item:
            continue
        if ":" in item:
            source, value = item.split(":", 1)
        elif "=" in item:
            source, value = item.split("=", 1)
        else:
            continue
        source = source.strip()
        if not source:
            continue
        try:
            budget = int(value)
        except ValueError:
            continue
        if budget > 0:
            budgets[source] = max(int(budgets.get(source) or 0), budget)
    return budgets


def source_min_budget_env(args: argparse.Namespace) -> str:
    budgets = parse_source_budget_spec(str(getattr(args, "source_min_budgets", "") or ""))
    enzyme_precedent_budget = max(0, int(getattr(args, "enzyme_precedent_min_budget", 0) or 0))
    if enzyme_precedent_budget > 0:
        budgets["enzyme_precedent"] = max(int(budgets.get("enzyme_precedent") or 0), enzyme_precedent_budget)
    return ",".join(f"{source}:{budget}" for source, budget in budgets.items() if budget > 0)


def build_enhanced_stock_checker(args: argparse.Namespace):
    if str(getattr(args, "enhanced_stock", "zinc") or "zinc") == "none":
        return None
    stock_checker = None
    try:
        from cascade_planner.cascadeboard.zinc_stock import is_in_zinc_stock

        stock_checker = is_in_zinc_stock
    except Exception:
        stock_checker = None
    if bool(getattr(args, "disable_common_stock", False)):
        wrapped = stock_checker
    else:
        from cascade_planner.cascadeboard.common_stock import wrap_with_common_commodity_stock

        wrapped = wrap_with_common_commodity_stock(stock_checker)
    if bool(getattr(args, "disable_vendor_stock", False)):
        stock = wrapped
    else:
        from cascade_planner.cascadeboard.vendor_stock import wrap_with_vendor_stock

        sqlite_path = getattr(args, "vendor_stock_index", None) or Path(
            "results/shared/chemenzy_vendor_stock/zinc_fix_stock_smiles.sqlite"
        )
        stock = wrap_with_vendor_stock(wrapped, sqlite_path=sqlite_path)
    if bool(getattr(args, "enable_semisynthesis_stock", False)):
        from cascade_planner.cascadeboard.semisynthesis_stock import wrap_with_semisynthesis_stock

        stock = wrap_with_semisynthesis_stock(stock)
    if bool(getattr(args, "enable_chemical_anchor_stock", False)):
        from cascade_planner.cascadeboard.chemical_anchor_stock import wrap_with_chemical_anchor_stock

        stock = wrap_with_chemical_anchor_stock(stock)
    return stock


def run_enhanced_route_tree(
    target: dict[str, Any],
    *,
    live_engine: dict[str, Any] | None,
    retriever: BridgeRetrieverV0,
    enzyme_sp: EnzymeSPVerifierV1Scorer | None,
    native_rescue_adapter: ChemEnzyBackendAdapter | None,
    args: argparse.Namespace,
) -> dict[str, Any]:
    target_smiles = str(target.get("target_smiles") or "")
    started = time.monotonic()
    stock_checker = build_enhanced_stock_checker(args)
    with temporary_env(**enhanced_env_values(args)):
        live_engine = live_engine if live_engine is not None else build_live_retro_engine()
        planner = NeuralGuidedAOSearch(
            retro_engine=live_engine,
            stock_checker=stock_checker,
            max_depth=max(1, int(args.max_depth)),
            branch_factor=max(1, int(args.branch_factor)),
            expansion_budget=max(1, int(args.expansion_budget)),
            controller=None,
            enzyme_sp_verifier=enzyme_sp,
        )
        planner.proposals.source_gate = BridgeAwareSourceGate(SourceGate(), retriever=retriever, require_verifier_pass=True)
        results = planner.search(target_smiles, n_results=max(1, int(args.n_results)))
        stats = planner.stats.to_dict()
    routes = [route_tree_payload(result, stock_checker=stock_checker) for result in results]
    rescue_metadata: dict[str, Any] = {
        "enabled": bool(getattr(args, "enable_native_chemical_rescue", False)),
        "attempted": False,
        "accepted_routes": 0,
    }
    if _should_attempt_native_chemical_rescue(routes, args=args) and native_rescue_adapter is not None:
        rescue_routes, rescue_metadata = run_native_chemical_rescue(
            target,
            adapter=native_rescue_adapter,
            args=args,
        )
        if rescue_routes:
            routes = _dedupe_benchmark_routes(
                [*rescue_routes, *routes],
                limit=max(1, int(args.n_results)),
            )
    stats["native_chemical_rescue"] = rescue_metadata
    elapsed = time.monotonic() - started
    return {
        **target_base(target),
        "run": "enhanced_route_tree",
        "ok": bool(routes),
        "failure_categories": [] if routes else ["no_route_returned"],
        "failure_messages": [],
        "route_count": len(routes),
        "solved_routes": sum(1 for route in routes if bool(route.get("route_solved"))),
        "progressive_routes": sum(1 for route in routes if bool(route.get("progressive_route"))),
        "enzyme_routes": sum(1 for route in routes if bool(route.get("has_enzyme_step"))),
        "sp_v1_accepted_enzyme_routes": sum(1 for route in routes if bool(route.get("has_sp_v1_accepted_enzyme_step"))),
        "enzyme_proposal_calls": enzyme_proposal_calls(stats),
        "enzyme_proposal_candidates": enzyme_proposal_candidates(stats),
        "mean_steps": mean(route.get("n_steps") for route in routes),
        "elapsed_s": round(float(elapsed), 3),
        "backend_elapsed_s": stats.get("elapsed_s"),
        "stats": stats,
        "routes": routes,
    }


def _should_attempt_native_chemical_rescue(routes: list[dict[str, Any]], *, args: argparse.Namespace) -> bool:
    if not bool(getattr(args, "enable_native_chemical_rescue", False)):
        return False
    if any(bool(route.get("route_solved")) for route in routes):
        return False
    if any(bool(route.get("has_enzyme_step")) for route in routes):
        return False
    return True


def run_native_chemical_rescue(
    target: dict[str, Any],
    *,
    adapter: ChemEnzyBackendAdapter,
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    target_smiles = str(target.get("target_smiles") or "")
    started = time.monotonic()
    metadata: dict[str, Any] = {
        "enabled": True,
        "attempted": True,
        "accepted_routes": 0,
        "models": native_chemical_rescue_models(args),
        "proposal_gate_required": bool(getattr(args, "native_chemical_rescue_require_proposal_gate", False)),
        "proposal_gate_rejected_routes": 0,
        "proposal_gate_reason_counts": {},
    }
    config = RouteSearchConfig(
        target_smiles=target_smiles,
        stock_names=list(DEFAULT_STOCKS),
        max_iterations=max(1, int(args.iterations)),
        max_depth=max(1, int(args.max_depth)),
        expansion_topk=max(1, int(args.expansion_topk)),
        one_step_models=native_chemical_rescue_models(args),
        search_flags={
            "gpu": int(args.gpu),
            "keep_search": True,
            "use_filter": False,
            "use_depth_value_fn": False,
            "cascade_search_context": {
                "enabled": True,
                "benchmark": "native_vs_enhanced_route_benchmark.native_chemical_rescue.v0",
                "target_smiles": target_smiles,
            },
        },
    )
    with temporary_env(
        AUTOPLANNER_CHEMENZY_NATIVE_TIMEOUT_S=str(
            max(1.0, float(getattr(args, "native_chemical_rescue_timeout_s", 90.0)))
        ),
    ):
        result = adapter.run_target(config)
    rescue_routes: list[dict[str, Any]] = []
    gate_reason_counts: list[str] = []
    gate_rejected_routes = 0
    for route in result.routes[: max(1, int(args.n_results))]:
        payload = _mark_native_chemical_rescue_route(native_route_payload(route))
        gate_report = gate_web_route(payload)
        payload["native_chemical_rescue_proposal_gate"] = _compact_proposal_gate_report(gate_report)
        if bool(gate_report.get("hard_reject")):
            gate_rejected_routes += 1
            for reason, count in (gate_report.get("reason_counts") or {}).items():
                gate_reason_counts.extend([str(reason)] * int(count or 0))
            if bool(getattr(args, "native_chemical_rescue_require_proposal_gate", False)):
                continue
        if bool(payload.get("has_enzyme_step")):
            continue
        rescue_routes.append(payload)
    metadata.update(
        {
            "elapsed_s": round(time.monotonic() - started, 3),
            "backend_elapsed_s": (result.raw_backend_metadata or {}).get("elapsed_s"),
            "accepted_routes": len(rescue_routes),
            "solved_routes": sum(1 for route in rescue_routes if bool(route.get("route_solved"))),
            "proposal_gate_rejected_routes": int(gate_rejected_routes),
            "proposal_gate_reason_counts": dict_count(gate_reason_counts),
            "failure_categories": [failure.category for failure in result.failures],
            "failure_messages": [failure.message for failure in result.failures[:3]],
        }
    )
    return rescue_routes, metadata


def _mark_native_chemical_rescue_route(route: dict[str, Any]) -> dict[str, Any]:
    out = dict(route)
    out["native_chemical_rescue"] = True
    out["route_tree_search_status"] = "native_chemical_rescue"
    steps = []
    for step in out.get("steps") or []:
        row = dict(step)
        source = str(row.get("source") or "ChemEnzyRetroPlanner")
        row["source"] = f"native_chemical_rescue:{source}"
        row["native_chemical_rescue"] = True
        steps.append(row)
    out["steps"] = steps
    out["source_counts"] = dict_count(step.get("source") for step in steps)
    return out


def _compact_proposal_gate_report(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": report.get("schema_version"),
        "decision": report.get("decision"),
        "hard_reject": bool(report.get("hard_reject")),
        "mode": report.get("mode"),
        "step_count": int(report.get("step_count") or 0),
        "rejected_step_count": int(report.get("rejected_step_count") or 0),
        "route_hard_reasons": list(report.get("route_hard_reasons") or []),
        "reason_counts": dict(report.get("reason_counts") or {}),
        "frontier": report.get("frontier") or None,
    }


def _dedupe_benchmark_routes(routes: list[dict[str, Any]], *, limit: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[tuple[str, ...]] = set()
    for route in routes:
        signature = tuple(
            str(step.get("reaction_smiles") or "")
            for step in route.get("steps") or []
            if step.get("reaction_smiles")
        )
        if signature and signature in seen:
            continue
        if signature:
            seen.add(signature)
        out.append(route)
        if len(out) >= max(1, int(limit or 1)):
            break
    return out


def enhanced_env_values(args: argparse.Namespace) -> dict[str, str]:
    values = {
        "AUTOPLANNER_ROUTE_TREE_ENZYME_PRECEDENT_RETRIEVAL": "1",
        "AUTOPLANNER_ROUTE_TREE_V3_RETRIEVAL_ALL": "1",
        "AUTOPLANNER_ROUTE_TREE_BRIDGE_EC_CONTEXT_PROPOSALS": "1",
        "AUTOPLANNER_ROUTE_TREE_BRIDGE_ENZYME_BONUS": str(max(0.0, float(args.bridge_enzyme_bonus))),
        "AUTOPLANNER_ROUTE_TREE_SOURCE_MIN_BUDGETS": source_min_budget_env(args),
        "AUTOPLANNER_ROUTE_TREE_HARD_TIMEOUT_S": str(max(1.0, float(args.route_tree_timeout_s))),
        "AUTOPLANNER_ROUTE_TREE_SOFT_TIMEOUT_S": str(max(1.0, float(args.route_tree_timeout_s) * 0.75)),
        "AUTOPLANNER_ENZYME_SP_VERIFIER_V1_REJECT_BELOW_THRESHOLD": "1",
        "AUTOPLANNER_ENZYME_SP_VERIFIER_V1_SCOPE": "enzyme_or_bridge_supported_enzyme",
        "AUTOPLANNER_ROUTE_TREE_ENZYME_SP_ACCEPTED_BONUS": str(max(0.0, float(args.enzyme_sp_accepted_bonus))),
        "AUTOPLANNER_ROUTE_TREE_ENZYME_SP_SCORE_BONUS": str(max(0.0, float(args.enzyme_sp_score_bonus))),
        "AUTOPLANNER_ROUTE_TREE_EXACT_STOCK_REACTANT_BONUS": (
            str(max(0.0, float(args.exact_stock_reactant_bonus))) if args.stock_aware_action_rerank else "0"
        ),
        "AUTOPLANNER_ROUTE_TREE_FULL_STOCK_ACTION_BONUS": (
            str(max(0.0, float(args.full_stock_action_bonus))) if args.stock_aware_action_rerank else "0"
        ),
        "AUTOPLANNER_ROUTE_TREE_BRENDA_CONDITION_PRIOR": "1",
        "AUTOPLANNER_ROUTE_TREE_CONDITION_PREDICTION": "1",
        "AUTOPLANNER_ROUTE_TREE_CONDITION_MODEL": "rcr",
        "AUTOPLANNER_ROUTE_TREE_CONDITION_PREDICTION_CHEMICAL_ONLY": "1",
        "AUTOPLANNER_ROUTE_TREE_CONDITION_VENDOR_ROOT": "vendor/ChemEnzyRetroPlanner",
    }
    if bool(getattr(args, "enable_enzyme_continuation_source_gate", False)):
        values["AUTOPLANNER_BRIDGE_GATE_ALLOW_ENZYME_CONTINUATION"] = "1"
    if bool(getattr(args, "enable_selected_enzyme_evidence_enrichment", False)):
        values["AUTOPLANNER_ROUTE_TREE_SELECTED_ENZYME_EVIDENCE_ENRICHMENT"] = "1"
        values["AUTOPLANNER_ROUTE_TREE_SELECTED_ENZYME_EVIDENCE_TOPK"] = str(
            max(1, int(getattr(args, "selected_enzyme_evidence_topk", 3) or 3))
        )
        values["AUTOPLANNER_ROUTE_TREE_SELECTED_ENZYME_EVIDENCE_MIN_SIMILARITY"] = str(
            max(0.0, float(getattr(args, "selected_enzyme_evidence_min_similarity", 0.35) or 0.0))
        )
    if bool(getattr(args, "enable_sp_v1_enzyme_result_selector", False)):
        values["AUTOPLANNER_ROUTE_TREE_SP_V1_ENZYME_RESULT_SELECTOR"] = "1"
        values["AUTOPLANNER_ROUTE_TREE_ENZYME_RESULT_POOL_MIN"] = str(
            max(1, int(getattr(args, "sp_v1_enzyme_result_pool_min", 5) or 5))
        )
        values["AUTOPLANNER_ROUTE_TREE_SP_V1_ENZYME_SELECTOR_MAX_RANK"] = str(
            max(1, int(getattr(args, "sp_v1_enzyme_selector_max_rank", 5) or 5))
        )
        values["AUTOPLANNER_ROUTE_TREE_SP_V1_ENZYME_SELECTOR_MAX_EXTRA_COST"] = str(
            max(0.0, float(getattr(args, "sp_v1_enzyme_selector_max_extra_cost", 0.0) or 0.0))
        )
        if bool(getattr(args, "enable_sp_v1_enzyme_selector_cost_exception", False)):
            values["AUTOPLANNER_ROUTE_TREE_SP_V1_ENZYME_SELECTOR_ALLOW_COST_EXCEPTION"] = "1"
            values["AUTOPLANNER_ROUTE_TREE_SP_V1_ENZYME_SELECTOR_COST_EXCEPTION_MAX_EXTRA_COST"] = str(
                max(
                    0.0,
                    float(getattr(args, "sp_v1_enzyme_selector_cost_exception_max_extra_cost", 0.0) or 0.0),
                )
            )
    if bool(getattr(args, "enable_enzyme_sp_material_gate", False)):
        values["AUTOPLANNER_ENZYME_SP_MATERIAL_GATE"] = "1"
        sources = str(getattr(args, "enzyme_sp_material_gate_sources", "") or "").strip()
        if sources:
            values["AUTOPLANNER_ENZYME_SP_MATERIAL_GATE_SOURCES"] = sources
    if bool(getattr(args, "disable_retrochimera_source", False)):
        values["AUTOPLANNER_DISABLE_RETROCHIMERA"] = "1"
    raw_disable_chemtemplates_depth = getattr(args, "disable_chemtemplates_after_depth", -1)
    disable_chemtemplates_depth = -1 if raw_disable_chemtemplates_depth is None else int(raw_disable_chemtemplates_depth)
    if disable_chemtemplates_depth >= 0:
        values["AUTOPLANNER_ROUTE_TREE_DISABLE_SOURCES_AFTER_DEPTH"] = _merge_disable_sources_after_depth(
            values.get("AUTOPLANNER_ROUTE_TREE_DISABLE_SOURCES_AFTER_DEPTH", ""),
            {"chemtemplates": disable_chemtemplates_depth},
        )
    if bool(getattr(args, "enable_stock_closing_probe", False)):
        values.update(stock_closing_probe_env(args))
    if bool(getattr(args, "enable_semisynthesis_rescue_source", False)):
        values["AUTOPLANNER_ENABLE_SEMISYNTHESIS_RESCUE_PROPOSALS"] = "1"
        values["AUTOPLANNER_SEMISYNTHESIS_RESCUE_MIN_BUDGET"] = str(
            max(1, int(getattr(args, "semisynthesis_rescue_min_budget", 2) or 2))
        )
    if bool(getattr(args, "enable_chemical_anchor_rescue_source", False)):
        values["AUTOPLANNER_ENABLE_CHEMICAL_ANCHOR_RESCUE_PROPOSALS"] = "1"
        values["AUTOPLANNER_CHEMICAL_ANCHOR_RESCUE_MIN_BUDGET"] = str(
            max(1, int(getattr(args, "chemical_anchor_rescue_min_budget", 2) or 2))
        )
    if int(args.chem_template_max_per_query or 0) > 0:
        values["AUTOPLANNER_CHEM_TEMPLATES_MAX_PER_QUERY"] = str(max(1, int(args.chem_template_max_per_query)))
    if int(args.chem_template_max_templates or 0) > 0:
        values["AUTOPLANNER_CHEM_TEMPLATES_MAX_TEMPLATES"] = str(max(1, int(args.chem_template_max_templates)))
    if bool(getattr(args, "enable_enhanced_chemenzy_assembly", False)):
        values.update(enhanced_chemenzy_assembly_env(args))
    if bool(getattr(args, "enable_enhanced_chemical_fusion_source", False)):
        values.update(enhanced_chemical_fusion_source_env(args))
    if bool(getattr(args, "enable_template_relevance_source", False)):
        values.update(template_relevance_source_env(args))
    if bool(getattr(args, "enable_enhanced_bionav_source", False)):
        values.update(enhanced_bionav_source_env(args))
    caps = enhanced_request_caps(args)
    if caps:
        values["AUTOPLANNER_ROUTE_TREE_SOURCE_REQUEST_CAPS"] = caps
    return values


def _merge_disable_sources_after_depth(raw: str, updates: dict[str, int]) -> str:
    merged: dict[str, int] = {}
    for item in str(raw or "").split(","):
        item = item.strip()
        if not item:
            continue
        if ":" in item:
            key, value = item.split(":", 1)
        elif "=" in item:
            key, value = item.split("=", 1)
        else:
            continue
        key = key.strip()
        if not key:
            continue
        try:
            merged[key] = int(value)
        except ValueError:
            continue
    for key, value in updates.items():
        if int(value) >= 0:
            merged[str(key)] = int(value)
    return ",".join(f"{key}:{value}" for key, value in merged.items())


def stock_closing_probe_env(args: argparse.Namespace) -> dict[str, str]:
    return {
        "AUTOPLANNER_ROUTE_TREE_STOCK_CLOSING_PROBE": "1",
        "AUTOPLANNER_ROUTE_TREE_STOCK_CLOSING_PROBE_SOURCES": ",".join(stock_closing_probe_sources(args)),
        "AUTOPLANNER_ROUTE_TREE_STOCK_CLOSING_PROBE_TOPK": str(
            max(1, int(getattr(args, "stock_closing_probe_topk", 75) or 75))
        ),
        "AUTOPLANNER_ROUTE_TREE_STOCK_CLOSING_PROBE_TOPK_CAP": str(
            max(1, int(getattr(args, "stock_closing_probe_topk_cap", 75) or 75))
        ),
        "AUTOPLANNER_ROUTE_TREE_STOCK_CLOSING_PROBE_MAX_ACTIONS": str(
            max(1, int(getattr(args, "stock_closing_probe_max_actions", 4) or 4))
        ),
        "AUTOPLANNER_ROUTE_TREE_STOCK_CLOSING_PROBE_REMAINING_DEPTH": str(
            max(0, int(getattr(args, "stock_closing_probe_remaining_depth", 2) or 0))
        ),
    }


def enhanced_chemenzy_assembly_env(args: argparse.Namespace) -> dict[str, str]:
    model = Path(args.enhanced_bionav_model).expanduser()
    if not model.is_absolute():
        model = (ROOT / model).resolve()
    values = {
        "AUTOPLANNER_ENABLE_CHEMENZY_ONESTEP_PROPOSALS": "1",
        "AUTOPLANNER_CHEMENZY_ONESTEP_MODELS": "graphfp_models.USPTO-full_remapped,onmt_models.bionav_one_step",
        "AUTOPLANNER_CHEMENZY_ONESTEP_TOPK": str(max(1, int(args.enhanced_chemenzy_topk))),
        "AUTOPLANNER_CHEMENZY_ONESTEP_MIN_BUDGET": str(max(1, int(args.enhanced_chemenzy_min_budget))),
        "AUTOPLANNER_CHEMENZY_ONESTEP_ROUTE_MODE": str(args.enhanced_chemenzy_route_mode),
        "AUTOPLANNER_ROUTE_TREE_SOURCE_REQUEST_CAPS": (
            f"chem_enzy_onestep:{max(1, int(args.enhanced_chemenzy_topk))}"
        ),
        "AUTOPLANNER_CHEMENZY_ONMT_MODEL_PATH": str(model),
        "AUTOPLANNER_CHEMENZY_ONMT_TOKENIZER": "pretokenized",
        "AUTOPLANNER_CHEMENZY_ONMT_SOURCE_PREFIX": "<product>",
        "AUTOPLANNER_CHEMENZY_ONMT_PRETOKENIZE_MODE": "char",
        "AUTOPLANNER_ENABLE_GRAPHFP_DUALTOWER_FUSION": "1",
        "AUTOPLANNER_GRAPHFP_FUSION_INTERNAL_TOPK": "50",
        "AUTOPLANNER_DUALTOWER_TEMPLATE_TOPK": "100",
        "AUTOPLANNER_GRAPHFP_DUALTOWER_FUSION_MODE": str(
            getattr(args, "enhanced_fusion_mode", "graphfp_first") or "graphfp_first"
        ),
        "AUTOPLANNER_GRAPHFP_DUALTOWER_FUSION_PROTECTED_TOPK": str(
            max(0, int(args.enhanced_chemenzy_protected_topk))
        ),
        "AUTOPLANNER_GRAPHFP_DUALTOWER_FUSION_PROTECTED_FRONT": str(
            max(0, int(args.enhanced_chemenzy_protected_front))
        ),
    }
    if str(getattr(args, "enhanced_dualtower_device", "") or "").strip():
        values["AUTOPLANNER_DUALTOWER_TEMPLATE_DEVICE"] = str(args.enhanced_dualtower_device).strip()
    return values


def enhanced_chemical_fusion_source_env(args: argparse.Namespace) -> dict[str, str]:
    topk = max(1, int(getattr(args, "enhanced_chemical_fusion_topk", 50) or 50))
    min_budget = max(1, int(getattr(args, "enhanced_chemical_fusion_min_budget", 4) or 4))
    values = {
        "AUTOPLANNER_ENABLE_CHEMENZY_GRAPHFP_FUSION_PROPOSALS": "1",
        "AUTOPLANNER_CHEMENZY_GRAPHFP_FUSION_MODELS": "graphfp_models.USPTO-full_remapped",
        "AUTOPLANNER_CHEMENZY_GRAPHFP_FUSION_TOPK": str(topk),
        "AUTOPLANNER_CHEMENZY_GRAPHFP_FUSION_MIN_BUDGET": str(min_budget),
        "AUTOPLANNER_ENABLE_GRAPHFP_DUALTOWER_FUSION": "1",
        "AUTOPLANNER_GRAPHFP_FUSION_INTERNAL_TOPK": str(topk),
        "AUTOPLANNER_DUALTOWER_TEMPLATE_TOPK": "100",
        "AUTOPLANNER_GRAPHFP_DUALTOWER_FUSION_MODE": str(
            getattr(args, "enhanced_fusion_mode", "graphfp_first") or "graphfp_first"
        ),
        "AUTOPLANNER_ROUTE_TREE_SOURCE_REQUEST_CAPS": f"chem_enzy_graphfp_fusion:{topk}",
    }
    if str(getattr(args, "enhanced_dualtower_device", "") or "").strip():
        values["AUTOPLANNER_DUALTOWER_TEMPLATE_DEVICE"] = str(args.enhanced_dualtower_device).strip()
    return values


def template_relevance_source_env(args: argparse.Namespace) -> dict[str, str]:
    topk = max(1, int(getattr(args, "template_relevance_topk", 20) or 20))
    min_budget = max(1, int(getattr(args, "template_relevance_min_budget", 4) or 4))
    raw_gpu = getattr(args, "template_relevance_gpu", -1)
    gpu = -1 if raw_gpu is None or str(raw_gpu).strip() == "" else int(raw_gpu)
    vendor_root = str(getattr(args, "template_relevance_vendor_root", "") or "vendor/ChemEnzyRetroPlanner")
    return {
        "AUTOPLANNER_ENABLE_TEMPLATE_RELEVANCE_PROPOSALS": "1",
        "AUTOPLANNER_TEMPLATE_RELEVANCE_MODELS": ",".join(template_relevance_models(args)),
        "AUTOPLANNER_TEMPLATE_RELEVANCE_TOPK": str(topk),
        "AUTOPLANNER_TEMPLATE_RELEVANCE_MIN_BUDGET": str(min_budget),
        "AUTOPLANNER_TEMPLATE_RELEVANCE_GPU": str(gpu),
        "AUTOPLANNER_TEMPLATE_RELEVANCE_VENDOR_ROOT": vendor_root,
        "AUTOPLANNER_ROUTE_TREE_SOURCE_REQUEST_CAPS": f"template_relevance:{topk}",
    }


def enhanced_bionav_source_env(args: argparse.Namespace) -> dict[str, str]:
    model = Path(args.enhanced_bionav_model).expanduser()
    if not model.is_absolute():
        model = (ROOT / model).resolve()
    topk = max(1, int(getattr(args, "enhanced_bionav_source_topk", 10) or 10))
    caps = [f"chem_enzy_bionav:{topk}"]
    if bool(getattr(args, "enable_enhanced_chemenzy_assembly", False)):
        caps.append(f"chem_enzy_onestep:{max(1, int(args.enhanced_chemenzy_topk))}")
    return {
        "AUTOPLANNER_ENABLE_CHEMENZY_BIONAV_PROPOSALS": "1",
        "AUTOPLANNER_CHEMENZY_BIONAV_MODELS": "onmt_models.bionav_one_step",
        "AUTOPLANNER_CHEMENZY_BIONAV_TOPK": str(topk),
        "AUTOPLANNER_CHEMENZY_ONMT_MODEL_PATH": str(model),
        "AUTOPLANNER_CHEMENZY_ONMT_TOKENIZER": "pretokenized",
        "AUTOPLANNER_CHEMENZY_ONMT_SOURCE_PREFIX": "<product>",
        "AUTOPLANNER_CHEMENZY_ONMT_PRETOKENIZE_MODE": "char",
        "AUTOPLANNER_ROUTE_TREE_SOURCE_REQUEST_CAPS": ",".join(caps),
    }


def enhanced_request_caps(args: argparse.Namespace) -> str:
    caps: list[str] = []
    if bool(getattr(args, "enable_enhanced_chemical_fusion_source", False)):
        topk = max(1, int(getattr(args, "enhanced_chemical_fusion_topk", 50) or 50))
        caps.append(f"chem_enzy_graphfp_fusion:{topk}")
    if bool(getattr(args, "enable_template_relevance_source", False)):
        topk = max(1, int(getattr(args, "template_relevance_topk", 20) or 20))
        caps.append(f"template_relevance:{topk}")
    if bool(getattr(args, "enable_enhanced_bionav_source", False)):
        topk = max(1, int(getattr(args, "enhanced_bionav_source_topk", 10) or 10))
        caps.append(f"chem_enzy_bionav:{topk}")
    if bool(getattr(args, "enable_enhanced_chemenzy_assembly", False)):
        topk = max(1, int(getattr(args, "enhanced_chemenzy_topk", 50) or 50))
        caps.append(f"chem_enzy_onestep:{topk}")
    if bool(getattr(args, "enable_semisynthesis_rescue_source", False)):
        caps.append("semisynthesis_rescue:8")
    if bool(getattr(args, "enable_chemical_anchor_rescue_source", False)):
        caps.append("chemical_anchor_rescue:4")
    return ",".join(dict.fromkeys(caps))


def native_route_payload(route: RouteCandidate) -> dict[str, Any]:
    sources = [str(step.source_model or "") for step in route.steps]
    has_enzyme = bool(route.enzymatic_step_present or any("enzyme" in source.lower() for source in sources))
    plausibility = audit_route_plausibility(route)
    return {
        "route_rank": int(route.route_rank or 0),
        "score": route.score,
        "n_steps": len(route.steps),
        "route_solved": bool(route.solved),
        "route_plausibility": plausibility,
        "route_plausibility_passed": bool(plausibility.get("passed")),
        "progressive_route": bool(route.steps),
        "has_enzyme_step": has_enzyme,
        "source_counts": dict_count(sources),
        "steps": [
            {
                "product": step.product_smiles,
                "reactants": list(step.reactant_smiles or []),
                "reaction_smiles": step.rxn_smiles,
                "source": step.source_model,
                "score": step.score,
                "has_enzyme_annotation": bool(step.has_enzyme_annotation),
            }
            for step in route.steps
        ],
    }


def route_tree_payload(result: Any, *, stock_checker: Any | None = None) -> dict[str, Any]:
    payload = route_result_to_dict(result, stock_checker=stock_checker)
    metrics = payload.get("metrics") or {}
    steps = payload.get("steps") or []
    explanation = payload.get("explanation") or {}
    uncertainty = explanation.get("uncertainty_table") or {}
    source_counts = metrics.get("candidate_source_counts") or dict_count(step.get("source") for step in steps)
    enzyme_steps = [
        step
        for step in steps
        if step.get("ec") or str(step.get("source") or "").lower() in ENZYME_SOURCES
    ]
    sp_v1_accepted_steps = [
        step
        for step in enzyme_steps
        if bool((step.get("enzyme_sp_verifier_v1") or {}).get("accepted"))
    ]
    return {
        "score": payload.get("score"),
        "n_steps": payload.get("n_steps"),
        "route_solved": bool(metrics.get("route_solved")),
        "progressive_route": bool(metrics.get("progressive_route")),
        "route_tree_search_status": uncertainty.get("route_tree_search_status"),
        "has_enzyme_step": bool(enzyme_steps),
        "has_sp_v1_accepted_enzyme_step": bool(sp_v1_accepted_steps),
        "source_counts": source_counts,
        "metrics": metrics,
        "explanation": explanation,
        "steps": steps,
    }


def target_base(target: dict[str, Any]) -> dict[str, Any]:
    smiles = str(target.get("target_smiles") or "")
    return {
        "target_smiles": smiles,
        "target_canonical": canonical_smiles(smiles) or smiles,
        "label": int(target.get("label") or 0),
        "label_source": str(target.get("label_source") or ""),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for run in sorted({row["run"] for row in rows}):
        subset = [row for row in rows if row["run"] == run]
        positives = [row for row in subset if int(row.get("label") or 0) == 1]
        negatives = [row for row in subset if int(row.get("label") or 0) == 0]
        out[run] = {
            "targets": len(subset),
            "ok_targets": sum(1 for row in subset if row.get("ok")),
            "route_count": sum(int(row.get("route_count") or 0) for row in subset),
            "targets_with_routes": sum(1 for row in subset if int(row.get("route_count") or 0) > 0),
            "targets_with_solved_route": sum(1 for row in subset if int(row.get("solved_routes") or 0) > 0),
            "targets_with_progressive_route": sum(1 for row in subset if int(row.get("progressive_routes") or 0) > 0),
            "targets_with_enzyme_route": sum(1 for row in subset if int(row.get("enzyme_routes") or 0) > 0),
            "targets_with_sp_v1_accepted_enzyme_route": sum(
                1 for row in subset if int(row.get("sp_v1_accepted_enzyme_routes") or 0) > 0
            ),
            "targets_with_plausibility_audited_route": sum(
                1 for row in subset if any(_route_has_plausibility_audit(route) for route in row.get("routes") or [])
            ),
            "targets_with_plausibility_passed_route": sum(
                1 for row in subset if any(_route_plausibility_passed(route) is True for route in row.get("routes") or [])
            ),
            "targets_with_plausibility_passed_solved_route": sum(
                1
                for row in subset
                if any(
                    bool(route.get("route_solved")) and _route_plausibility_passed(route) is True
                    for route in row.get("routes") or []
                )
            ),
            "partial_route_count": sum(
                max(0, int(row.get("route_count") or 0) - int(row.get("solved_routes") or 0))
                for row in subset
            ),
            "timeout_frontier_routes": sum(
                1
                for row in subset
                for route in row.get("routes") or []
                if str(route.get("route_tree_search_status") or "") == "timeout_frontier"
            ),
            "positive_enzyme_target_recall": ratio(
                sum(1 for row in positives if int(row.get("enzyme_routes") or 0) > 0),
                len(positives),
            ),
            "positive_enzyme_proposal_recall": ratio(
                sum(1 for row in positives if int(row.get("enzyme_proposal_candidates") or 0) > 0),
                len(positives),
            ),
            "negative_enzyme_target_rate": ratio(
                sum(1 for row in negatives if int(row.get("enzyme_routes") or 0) > 0),
                len(negatives),
            ),
            "negative_enzyme_proposal_rate": ratio(
                sum(1 for row in negatives if int(row.get("enzyme_proposal_candidates") or 0) > 0),
                len(negatives),
            ),
            "enzyme_proposal_calls": sum(int(row.get("enzyme_proposal_calls") or 0) for row in subset),
            "enzyme_proposal_candidates": sum(int(row.get("enzyme_proposal_candidates") or 0) for row in subset),
            "mean_routes": mean(row.get("route_count") for row in subset),
            "mean_steps": mean(row.get("mean_steps") for row in subset if row.get("mean_steps") is not None),
            "mean_elapsed_s": mean(row.get("elapsed_s") for row in subset),
            "failure_categories": dict_count(cat for row in subset for cat in row.get("failure_categories") or []),
            "enzyme_sp_rejections": sum(
                int(((row.get("stats") or {}).get("enzyme_sp_verifier_rejections") or 0))
                for row in subset
            ),
            "native_chemical_rescue_attempts": sum(
                1 for row in subset if bool(((row.get("stats") or {}).get("native_chemical_rescue") or {}).get("attempted"))
            ),
            "native_chemical_rescue_solved_targets": sum(
                1
                for row in subset
                if int(((row.get("stats") or {}).get("native_chemical_rescue") or {}).get("solved_routes") or 0) > 0
            ),
            "native_chemical_rescue_routes": sum(
                int(((row.get("stats") or {}).get("native_chemical_rescue") or {}).get("accepted_routes") or 0)
                for row in subset
            ),
            "proposal_source_calls": merge_source_calls(subset),
        }
    return out


def _route_has_plausibility_audit(route: dict[str, Any]) -> bool:
    if "route_plausibility_passed" in route:
        return True
    return isinstance(route.get("route_plausibility"), dict)


def _route_plausibility_passed(route: dict[str, Any]) -> bool | None:
    if "route_plausibility_passed" in route:
        return bool(route.get("route_plausibility_passed"))
    audit = route.get("route_plausibility")
    if isinstance(audit, dict) and "passed" in audit:
        return bool(audit.get("passed"))
    return None


def conclusion(summary: dict[str, Any]) -> str:
    native = summary.get("native_chemenzy") or {}
    enhanced = summary.get("enhanced_route_tree") or {}
    return (
        "Route-level smoke compares ChemEnzy native search with enhanced route-tree. "
        f"Native returned routes for {native.get('targets_with_routes', 0)}/{native.get('targets', 0)} targets; "
        f"enhanced returned routes for {enhanced.get('targets_with_routes', 0)}/{enhanced.get('targets', 0)} targets. "
        f"Enhanced solved targets={enhanced.get('targets_with_solved_route', 0)}, "
        f"progressive targets={enhanced.get('targets_with_progressive_route', 0)}, "
        f"timeout-frontier partial routes={enhanced.get('timeout_frontier_routes', 0)}. "
        f"Enhanced selected enzyme routes for {enhanced.get('targets_with_enzyme_route', 0)} targets "
        f"and SP-v1 accepted enzyme routes for {enhanced.get('targets_with_sp_v1_accepted_enzyme_route', 0)} targets "
        f"and generated {enhanced.get('enzyme_proposal_candidates', 0)} enzyme proposal candidates "
        f"with SP-v1 rejections={enhanced.get('enzyme_sp_rejections', 0)}. "
        "Use the detailed rows to inspect whether enzyme routes are useful; this is still a smoke benchmark, not final evidence."
    )


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Native ChemEnzy vs Enhanced Route-Tree Benchmark",
        "",
        "Route-level smoke benchmark on the same targets.",
        "",
        "| run | targets | targets w/routes | solved targets | progressive targets | enzyme route targets | SP-v1 accepted enzyme targets | routes | partial routes | timeout-frontier routes | enzyme proposal candidates | mean steps | mean elapsed s | SP-v1 rejects |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for run, row in (report.get("summaries") or {}).items():
        lines.append(
            "| {run} | {targets} | {targets_with_routes} | {targets_with_solved_route} | {targets_with_progressive_route} | "
            "{targets_with_enzyme_route} | {targets_with_sp_v1_accepted_enzyme_route} | "
            "{route_count} | {partial_route_count} | {timeout_frontier_routes} | "
            "{enzyme_proposal_candidates} | {mean_steps:.3f} | {mean_elapsed_s:.3f} | {enzyme_sp_rejections} |".format(
                run=run,
                **row,
            )
        )
    lines.extend(["", "## Conclusion", "", report["conclusion"], "", "## Comparability Notes", ""])
    for note in report.get("comparability_notes") or []:
        lines.append(f"- {note}")
    lines.append("")
    return "\n".join(lines)


def merge_source_calls(rows: list[dict[str, Any]]) -> dict[str, int]:
    out: dict[str, int] = {}
    for row in rows:
        stats = row.get("stats") or {}
        source_stats = stats.get("proposal_source_stats") or {}
        for source, payload in source_stats.items():
            out[source] = out.get(source, 0) + int((payload or {}).get("calls") or 0)
    return out


def enzyme_proposal_calls(stats: dict[str, Any]) -> int:
    source_stats = stats.get("proposal_source_stats") or {}
    return sum(
        int((source_stats.get(source) or {}).get("calls") or 0)
        for source in ENZYME_SOURCES
    )


def enzyme_proposal_candidates(stats: dict[str, Any]) -> int:
    source_stats = stats.get("proposal_source_stats") or {}
    return sum(
        int((source_stats.get(source) or {}).get("final_returned") or 0)
        for source in ENZYME_SOURCES
    )


def dict_count(values: Any) -> dict[str, int]:
    out: dict[str, int] = {}
    for value in values:
        key = str(value or "unknown")
        out[key] = out.get(key, 0) + 1
    return out


def mean(values: Any) -> float:
    vals = []
    for value in values:
        try:
            vals.append(float(value))
        except (TypeError, ValueError):
            continue
    return round(sum(vals) / len(vals), 4) if vals else 0.0


def ratio(num: int, den: int) -> float:
    return round(float(num) / float(den), 4) if den else 0.0


@contextmanager
def temporary_env(**values: str):
    old = {key: os.environ.get(key) for key in values}
    try:
        for key, value in values.items():
            os.environ[key] = str(value)
        yield
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


if __name__ == "__main__":
    main()
