"""Run and report a real bufotalin budget-exhaustion agentic blackboard case."""
from __future__ import annotations

import argparse
import io
import json
import sys
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import fitz
from PIL import Image, ImageDraw, ImageFont, JpegImagePlugin


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cascade_planner.harness.agentic_blackboard_controller import run_agentic_blackboard_controller
from cascade_planner.harness.tools import HarnessBudget


OUT_DIR = ROOT / "docs" / "agentic_blackboard" / "report_20260609"
DEFAULT_RUN_DIR = ROOT / "results" / "shared" / "bufotalin_agentic_blackboard_autonomous_budget_exhaustion_20260609"
PRIOR_SUCCESS_RUN_DIR = ROOT / "results" / "shared" / "bufotalin_fullflow_fresh_visual_existing_pdf_20260608_065053"
DEFAULT_PDF_PATH = ROOT / "1-s2.0-S0040402025001668-main.pdf"
DEFAULT_KEY_PATH = ROOT / "key.txt"
DEFAULT_BASE_URL = "https://api.wellau.com/v1"
DEFAULT_MODEL = "gpt-5.5"

BUFOTALIN_SMILES = (
    "CC(=O)O[C@H]1C[C@@]2([C@@H]3CC[C@@H]4C[C@H](CC[C@@]4"
    "([C@H]3CC[C@@]2([C@H]1C5=COC(=O)C=C5)C)C)O)O"
)
BUFOTALIN_FAMILY = "bufotalin, bufadienolide, steroid, C17 2-pyrone"
SOURCE_REF = "doi:10.1016/j.tet.2025.134610"
FONT_REGULAR = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
FONT_BOLD = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc")

PAGE_W, PAGE_H = 1240, 1754
MARGIN = 72
TEXT = (24, 35, 51)
MUTED = (82, 95, 115)
LINE = (205, 215, 227)
BG = (246, 248, 251)
PANEL = (255, 255, 255)
ACCENT = (14, 116, 144)
ACCENT_2 = (79, 70, 229)
GOOD = (22, 101, 52)
WARN = (146, 64, 14)
BAD = (153, 27, 27)

ACTION_ZH = {
    "generate_disconnection_hypotheses": "生成目标侧断键假设",
    "search_literature": "检索目标近端文献",
    "extract_pdf_literature_structures": "渲染/索引本地 PDF 结构证据",
    "extract_visual_literature_chain": "视觉抽取文献结构链",
    "compile_exact_literature_rows": "编译 exact 文献行",
    "rank_analogical_hypotheses": "排序类比假设",
    "run_guided_chemenzy": "运行 guided ChemEnzy",
    "expand_child_target": "扩展上游子目标",
    "stitch_parent_route": "拼接父路线证明",
    "build_failure_critic_report": "构建失败批判报告",
    "stop_unresolved": "停止并保持未解决",
}

REASON_ZH = {
    "no_deterministic_parent_route_proof": "缺少确定性的父路线证明",
    "target_equivalence_not_proven": "目标等价性未证明",
    "parent_route_verifier_not_accepted": "父路线 verifier 未接受",
    "stock_audit_not_passed": "库存审计未通过",
    "child_target_route_not_connected_to_parent_bridge": "子目标路线未连到父路线桥接点",
    "exact_literature_segment_not_connected_to_parent_route": "exact 文献片段未连到父路线",
    "route_expansion_child_targets_missing": "没有可扩展的子目标",
    "guided_policy_missing": "guided 搜索策略缺失",
    "visual_input_images_missing": "视觉输入图片缺失",
    "no_failure_evidence": "没有新的失败证据",
    "no_route_expansion_subgoal_verified_solved": "没有子目标被 verifier 证明闭合",
    "large_atom_jump": "出现无法解释的大重原子跳跃",
    "advanced_same_scaffold_terminal": "高级同骨架终端被拒绝",
    "literature_template_plugin_not_invoked": "文献模板插件未被后端调用",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(OUT_DIR))
    parser.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR))
    parser.add_argument("--pdf-path", default=str(DEFAULT_PDF_PATH))
    parser.add_argument("--skip-run", action="store_true", help="Only rebuild report from an existing run directory.")
    parser.add_argument("--max-rounds", type=int, default=5)
    parser.add_argument("--timeout-s", type=float, default=1200.0)
    parser.add_argument("--visual-timeout-s", type=float, default=420.0)
    parser.add_argument("--chemenzy-timeout-s", type=float, default=300.0)
    parser.add_argument("--chem-enzy-iterations", type=int, default=20)
    parser.add_argument("--chem-enzy-expansion-topk", type=int, default=50)
    parser.add_argument("--key-path", default=str(DEFAULT_KEY_PATH))
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--test-summary",
        default="python -m pytest tests/test_agentic_blackboard_controller.py -q: not run",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    run_dir = Path(args.run_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_run:
        run_budget_exhaustion_case(args=args, run_dir=run_dir)

    data = build_report_data(
        run_dir=run_dir,
        pdf_path=Path(args.pdf_path).resolve(),
        test_summary=str(args.test_summary),
        run_config={
            "max_rounds": int(args.max_rounds),
            "timeout_s": float(args.timeout_s),
            "visual_timeout_s": float(args.visual_timeout_s),
            "chemenzy_timeout_s": float(args.chemenzy_timeout_s),
            "chem_enzy_iterations": int(args.chem_enzy_iterations),
            "chem_enzy_expansion_topk": int(args.chem_enzy_expansion_topk),
            "source_ref": SOURCE_REF,
            "model": str(args.model),
        },
    )

    stem = "bufotalin_autonomous_budget_exhaustion_agentic_blackboard_zh_20260609"
    json_path = output_dir / f"{stem}.json"
    md_path = output_dir / f"{stem}.md"
    pdf_path = output_dir / f"{stem}.pdf"
    audit_path = output_dir / f"{stem}_audit.json"

    json_path.write_text(json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(data), encoding="utf-8")
    render_pdf(data, pdf_path)
    audit = build_audit(data, json_path=json_path, md_path=md_path, pdf_path=pdf_path)
    audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2, ensure_ascii=False, sort_keys=True))


def run_budget_exhaustion_case(*, args: argparse.Namespace, run_dir: Path) -> dict[str, Any]:
    budget = HarnessBudget(timeout_s=float(args.timeout_s))
    budget.guided_chemenzy_timeout_s = float(args.chemenzy_timeout_s)
    budget.max_route_expansion_subgoal_runs = 1
    return run_agentic_blackboard_controller(
        target_name="bufotalin",
        target_smiles=BUFOTALIN_SMILES,
        family_hint=BUFOTALIN_FAMILY,
        output_dir=run_dir,
        literature_pdf_path=Path(args.pdf_path).resolve(),
        literature_pdf_source_ref=SOURCE_REF,
        timeout_s=float(args.timeout_s),
        key_path=args.key_path,
        base_url=args.base_url,
        model=args.model,
        max_rounds=int(args.max_rounds),
        exhaust_round_budget=True,
        budget=budget,
    )


def build_report_data(*, run_dir: Path, pdf_path: Path, test_summary: str, run_config: dict[str, Any]) -> dict[str, Any]:
    blackboard = read_json(run_dir / "agent_blackboard.json")
    final = read_json(run_dir / "final_verdict.json")
    artifact_bundle = read_json(run_dir / "artifact_bundle.json")
    target_input = read_json(run_dir / "target_input.json")
    preflight = read_json(run_dir / "preflight.json")
    action_batches = [read_json(path) for path in sorted(run_dir.glob("action_batch_round_*.json"))]
    tool_calls = read_jsonl(run_dir / "tool_calls.jsonl")
    artifacts = dict(artifact_bundle.get("artifacts") or {})
    budget_state = dict(blackboard.get("budget_state") or {})
    termination = termination_summary(blackboard=blackboard, final=final)
    return {
        "schema_version": "bufotalin_budget_exhaustion_agentic_blackboard_report_data.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "case_run_dir": str(run_dir),
        "local_pdf_path": str(pdf_path),
        "local_pdf_exists": pdf_path.is_file(),
        "local_pdf_size_bytes": pdf_path.stat().st_size if pdf_path.exists() else 0,
        "target_input": target_input,
        "preflight": preflight,
        "target": blackboard.get("target_profile") or {},
        "run_config": run_config,
        "test_summary": test_summary,
        "action_batches": action_batches,
        "action_history": blackboard.get("action_history") or [],
        "decision_points": summarize_decisions(action_batches, blackboard),
        "tool_calls": tool_calls,
        "tool_summary": summarize_tools(tool_calls),
        "literature_evidence": blackboard.get("literature_evidence") or {},
        "route_failures": blackboard.get("route_failures") or [],
        "plugin_runtime_diagnostics": blackboard.get("plugin_runtime_diagnostics") or [],
        "bridge_tasks": blackboard.get("bridge_tasks") or [],
        "terminal_blacklist": blackboard.get("terminal_blacklist") or [],
        "analogical_hypothesis_ranking": blackboard.get("analogical_hypothesis_ranking") or {},
        "current_belief": blackboard.get("current_belief") or {},
        "parent_route_proof": blackboard.get("parent_route_proof") or {},
        "budget_state": budget_state,
        "termination": termination,
        "final_verdict": final,
        "planner_semantics": {
            "action_selection": "default_agent_action_planner.v1",
            "round_policy": "blackboard_state_driven",
            "exhaust_round_budget": True,
            "scripted_round_index_planner": False,
            "script_injected_action_payloads": False,
        },
        "prior_success_comparison": load_prior_success_comparison(PRIOR_SUCCESS_RUN_DIR),
        "artifact_refs": blackboard.get("artifact_refs") or {},
        "artifact_bundle_summary": {
            "artifact_keys": sorted(artifacts),
            "validation_count": len(artifact_bundle.get("validations") or []),
            "safety_flags": artifact_bundle.get("safety_flags") or [],
        },
        "real_artifact_summaries": {
            "pdf_evidence": summarize_pdf_evidence(artifacts, blackboard),
            "visual_chain": summarize_artifact(artifacts.get("visual_literature_chain_extraction")),
            "compiled_exact": summarize_artifact(artifacts.get("source_detail_chain_route")),
            "guided_chemenzy": summarize_artifact(artifacts.get("guided_chemenzy")),
            "route_expansion": summarize_artifact(artifacts.get("route_expansion_subgoal_search")),
            "stitched_route": summarize_artifact(artifacts.get("stitched_semisynthesis_route")),
        },
    }


def load_prior_success_comparison(run_dir: Path) -> dict[str, Any]:
    clean_final = read_json_optional(run_dir / "final_verdict_strict_visual_clean.json")
    clean_summary = read_json_optional(run_dir / "bufotalin_strict_visual_continuation_clean_summary.json")
    source_audit = read_json_optional(run_dir / "source_detail_chain_route_strict_visual" / "source_detail_route_chain_audit.json")
    visual = read_json_optional(run_dir / "visual_literature_chain_extraction_strict_visual" / "visual_literature_chain_extraction_result.json")
    stitched = read_json_optional(run_dir / "stitched_semisynthesis_route_strict_visual" / "stitched_semisynthesis_route.json")
    route_expansion = read_json_optional(run_dir / "route_expansion_subgoal_search_result.json")
    labels = []
    for row in source_audit.get("chain") or []:
        if isinstance(row, dict):
            labels.append(str(row.get("product_label") or row.get("label") or row.get("step_id") or ""))
    return {
        "schema_version": "bufotalin_prior_success_comparison.v1",
        "run_dir": str(run_dir),
        "exists": run_dir.is_dir(),
        "final_verdict": {
            "verdict": clean_final.get("verdict"),
            "route_status": clean_final.get("route_status"),
            "solved": bool(clean_final.get("solved")),
            "reasons": [str(item) for item in clean_final.get("reasons") or []],
        },
        "strict_visual_chain": {
            "accepted": bool(visual.get("accepted")),
            "status": str(visual.get("status") or ""),
            "candidate_step_count": int(visual.get("candidate_step_count") or 0),
            "terminal": dict(visual.get("strict_visual_terminal") or (clean_summary.get("strict_visual_terminal") or {})),
        },
        "source_detail_chain": {
            "accepted": bool(source_audit.get("accepted")),
            "step_count": int(source_audit.get("step_count") or ((source_audit.get("summary") or {}).get("chain_step_count") or 0)),
            "one_step_row_count": int((source_audit.get("summary") or {}).get("one_step_row_count") or 0),
            "terminal_reached": bool(source_audit.get("terminal_reached")),
            "terminal_name": str(source_audit.get("terminal_name") or "strict visual terminal 11"),
            "labels": [item for item in labels if item],
        },
        "child_route": {
            "accepted": bool(route_expansion.get("accepted")),
            "solved": bool(route_expansion.get("solved")),
            "status": str(route_expansion.get("status") or ""),
        },
        "stitched_route": {
            "accepted": bool(stitched.get("accepted")),
            "solved": bool(stitched.get("solved")),
            "route_status": str(stitched.get("route_status") or ""),
            "stock_audit_passed": bool(stitched.get("stock_audit_passed")),
            "combined_route": dict(stitched.get("combined_route") or (clean_summary.get("stitched_route") or {}).get("combined_route") or {}),
            "warnings": [str(item) for item in stitched.get("warnings") or []],
        },
    }


def summarize_decisions(action_batches: list[dict[str, Any]], blackboard: dict[str, Any]) -> list[dict[str, Any]]:
    history = [dict(row) for row in blackboard.get("action_history") or [] if isinstance(row, dict)]
    by_id = {str(row.get("action_id") or ""): row for row in history}
    rows: list[dict[str, Any]] = []
    for batch in action_batches:
        for raw in batch.get("actions") or []:
            if not isinstance(raw, dict):
                continue
            action_id = str(raw.get("action_id") or "")
            hist = by_id.get(action_id, {})
            rows.append(
                {
                    "round": batch.get("round_index"),
                    "action_id": action_id,
                    "action_type": str(raw.get("action_type") or ""),
                    "action_zh": ACTION_ZH.get(str(raw.get("action_type") or ""), str(raw.get("action_type") or "")),
                    "rationale": str(raw.get("rationale") or ""),
                    "expected_artifact": str(raw.get("expected_artifact") or ""),
                    "success_condition": str(raw.get("success_condition") or ""),
                    "status": str(hist.get("status") or "not_recorded"),
                    "useful_artifact": bool(hist.get("useful_artifact")),
                    "reasons": [str(item) for item in hist.get("reasons") or []],
                    "artifact_ref": str(hist.get("artifact_ref") or ""),
                }
            )
    return rows


def summarize_tools(tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in tool_calls:
        if not isinstance(row, dict):
            continue
        output = dict(row.get("output") or {})
        result = output.get("result") if isinstance(output.get("result"), dict) else output
        out.append(
            {
                "tool_name": str(row.get("tool_name") or ""),
                "status": str(row.get("status") or ""),
                "elapsed_s": row.get("elapsed_s"),
                "accepted": bool(output.get("accepted", result.get("accepted", row.get("status") == "accepted"))),
                "reasons": [str(item) for item in row.get("reasons") or output.get("reasons") or result.get("reasons") or []],
                "schema_version": str(result.get("schema_version") or output.get("schema_version") or ""),
            }
        )
    return out


def termination_summary(*, blackboard: dict[str, Any], final: dict[str, Any]) -> dict[str, Any]:
    budget = dict(blackboard.get("budget_state") or {})
    rounds_completed = int(budget.get("rounds_completed") or 0)
    max_rounds = int(budget.get("max_rounds") or 0)
    stop_selected = any(
        str(row.get("action_type") or "") == "stop_unresolved"
        for row in blackboard.get("action_history") or []
        if isinstance(row, dict)
    )
    proof = dict(blackboard.get("parent_route_proof") or {})
    proof_accepted = bool(proof.get("accepted") and proof.get("solved"))
    if proof_accepted:
        reason = "parent_route_proof_accepted"
    elif stop_selected:
        reason = "stop_unresolved_selected"
    elif max_rounds and rounds_completed >= max_rounds:
        reason = "max_round_budget_exhausted"
    else:
        reason = "controller_stopped_before_round_budget"
    return {
        "schema_version": "bufotalin_budget_exhaustion_termination.v1",
        "reason": reason,
        "rounds_completed": rounds_completed,
        "max_rounds": max_rounds,
        "budget_exhausted": reason == "max_round_budget_exhausted",
        "final_verdict": final.get("verdict"),
        "solved": bool(final.get("solved")),
    }


def summarize_pdf_evidence(artifacts: dict[str, Any], blackboard: dict[str, Any]) -> dict[str, Any]:
    evidence = artifacts.get("literature_pdf_structure_evidence")
    if not isinstance(evidence, dict):
        rows = ((blackboard.get("literature_evidence") or {}).get("pdf_structure_evidence") or [])
        evidence = dict(rows[0]) if rows and isinstance(rows[0], dict) else {}
    return summarize_artifact(evidence)


def summarize_artifact(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or not value:
        return {"present": False}
    summary = dict(value.get("summary") or {})
    out = {
        "present": True,
        "schema_version": str(value.get("schema_version") or ""),
        "accepted": bool(value.get("accepted", False)),
        "status": str(value.get("status") or value.get("route_status") or ""),
        "solved": bool(value.get("solved", False)),
        "reasons": [str(item) for item in value.get("reasons") or []],
        "summary": summary,
    }
    for key in (
        "candidate_step_count",
        "exact_row_count",
        "subgoal_count",
        "accepted_subgoal_count",
        "route_status",
        "source_pdf_path",
    ):
        if key in value:
            out[key] = value.get(key)
    if value.get("result") and isinstance(value.get("result"), dict):
        result = dict(value.get("result") or {})
        out["result_status"] = str(result.get("status") or result.get("search_status") or "")
        out["result_ok"] = bool(result.get("ok") or result.get("accepted", False))
        if result.get("exit_code") is not None:
            out["exit_code"] = result.get("exit_code")
    return out


def render_markdown(data: dict[str, Any]) -> str:
    final = data["final_verdict"]
    term = data["termination"]
    prior = data.get("prior_success_comparison") or {}
    lines = [
        "# Bufotalin Agentic Blackboard 自主预算耗尽中文报告",
        "",
        f"- 生成时间：{data['generated_at_utc']}",
        f"- run 目录：`{data['case_run_dir']}`",
        f"- 本地 PDF：`{data['local_pdf_path']}`，exists={data['local_pdf_exists']}，size={data['local_pdf_size_bytes']} bytes",
        f"- 最终结论：`{final.get('verdict')}` / route_status `{final.get('route_status')}` / solved `{final.get('solved')}`",
        f"- 停止原因：`{term.get('reason')}`，rounds `{term.get('rounds_completed')}/{term.get('max_rounds')}`",
        "- action 选择：默认 `agent_action_planner.v1` 依据 blackboard 状态自主选择；未传入按轮次写死的 action planner。",
        f"- 测试摘要：{data['test_summary']}",
        "",
        "## 为什么这一轮没有直接停止",
        "",
        "本次使用 controller 的 `exhaust_round_budget=True`，action batch 仍由默认 blackboard planner 动态选择。该模式只改变停止策略：当普通策略会因连续无新 artifact 而停下时，planner 必须先尝试 blackboard 中仍未耗尽、未 stale 的替代方向；只有父路线 proof 接受或轮次预算耗尽才收口。",
        "",
        "## Action 执行轨迹",
        "",
    ]
    for row in data["decision_points"]:
        reasons = ", ".join(translate_reason(item) for item in row.get("reasons") or []) or "无"
        lines.extend(
            [
                f"### 第 {row['round']} 轮：{row['action_zh']} (`{row['action_type']}`)",
                f"- rationale：{row['rationale']}",
                f"- expected_artifact：{row['expected_artifact']}",
                f"- success_condition：{row['success_condition']}",
                f"- status：`{row['status']}`，useful_artifact=`{row['useful_artifact']}`，reasons：{reasons}",
                "",
            ]
        )
    lines.extend(
        [
            "## 为什么这次失败，而之前 bufotalin 成功",
            "",
            f"- 之前成功 run：`{prior.get('run_dir', '')}`。",
            f"- 之前 clean verdict：`{(prior.get('final_verdict') or {}).get('verdict')}` / solved `{(prior.get('final_verdict') or {}).get('solved')}`。",
            f"- 之前 strict visual chain：accepted `{(prior.get('strict_visual_chain') or {}).get('accepted')}`，steps `{(prior.get('strict_visual_chain') or {}).get('candidate_step_count')}`，terminal `{((prior.get('strict_visual_chain') or {}).get('terminal') or {}).get('name', '')}`。",
            f"- 之前 source-detail exact chain：accepted `{(prior.get('source_detail_chain') or {}).get('accepted')}`，one_step_rows `{(prior.get('source_detail_chain') or {}).get('one_step_row_count')}`，terminal_reached `{(prior.get('source_detail_chain') or {}).get('terminal_reached')}`。",
            f"- 之前 stitched route：accepted `{(prior.get('stitched_route') or {}).get('accepted')}`，stock_audit `{(prior.get('stitched_route') or {}).get('stock_audit_passed')}`。",
            "",
            "反思：之前的 solved 不是 ChemEnzy 对 bufotalin 原生闭合，而是 `stock -> compound 11 -> bufotalin` 的拼接半合成证明：strict visual/source-detail 文献链把 bufotalin 连到 terminal 11，子目标搜索再把 terminal 11 从库存闭合，最后 stitch proof 通过。本轮自主 blackboard 运行没有重新生成 accepted exact literature chain，`compile_exact_literature_rows` 因缺少可用 source-detail rows 得到 0 行；后续 guided ChemEnzy/child expansion 即使产生候选，也缺少和父目标相连的 exact 文献段，因此 parent proof 不能通过。",
            "",
            "## Blackboard 和最终门禁",
            "",
            f"- bridge_tasks：{len(data['bridge_tasks'])}",
            f"- exact_rows：{len((data['literature_evidence'] or {}).get('exact_rows') or [])}",
            f"- selected_analogies：{len((data['analogical_hypothesis_ranking'] or {}).get('selected_hypotheses') or [])}",
            f"- parent_route_proof：`{(data['parent_route_proof'] or {}).get('route_status', 'missing')}`",
            "",
            "结论：没有 deterministic parent proof 时，final verdict 只能保持 unresolved/partial；类比、子目标或后端 solved flag 都不能升级为父目标 solved。",
        ]
    )
    return "\n".join(lines) + "\n"


def render_pdf(data: dict[str, Any], pdf_path: Path) -> None:
    pages = [
        page_cover(data),
        page_stop_reason(data),
        page_decision_timeline(data),
        page_literature_pdf(data),
        page_tool_results(data),
        page_prior_success_comparison(data),
        page_blackboard_and_gate(data),
        page_artifacts_and_tests(data),
    ]
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    doc = fitz.open()
    for image in pages:
        page = doc.new_page(width=PAGE_W, height=PAGE_H)
        stream = io.BytesIO()
        image.save(stream, format="JPEG", quality=88, optimize=True)
        page.insert_image(fitz.Rect(0, 0, PAGE_W, PAGE_H), stream=stream.getvalue())
    doc.save(pdf_path)
    doc.close()


def page_cover(data: dict[str, Any]) -> Image.Image:
    img, draw = new_page()
    final = data["final_verdict"]
    term = data["termination"]
    target = data["target"]
    y = 92
    y = draw_text(draw, "Bufotalin 自主预算耗尽版 Agentic Blackboard 报告", MARGIN, y, 43, bold=True, color=TEXT, max_width=PAGE_W - 2 * MARGIN)
    y = draw_text(draw, "真实运行：action 由默认 blackboard planner 选择；本地 PDF 只在 scout 选中后进入流程", MARGIN, y + 8, 24, bold=True, color=ACCENT, max_width=PAGE_W - 2 * MARGIN)
    y += 30
    metrics = [
        ("目标", str(target.get("target_name") or "bufotalin")),
        ("SMILES 重原子 / 环数", f"{target.get('heavy_atoms')} / {target.get('rings')}"),
        ("最终结论", f"{final.get('verdict')} ({final.get('route_status')})"),
        ("停止原因", f"{term.get('reason')}，rounds {term.get('rounds_completed')}/{term.get('max_rounds')}"),
        ("本地 PDF", f"{data['local_pdf_path']} ({data['local_pdf_size_bytes']} bytes)"),
    ]
    for label, value in metrics:
        y = metric(draw, label, value, MARGIN, y)
    draw_flow_strip(draw, MARGIN, 1040, PAGE_W - 2 * MARGIN)
    y = 1280
    draw_text(draw, "关键边界", MARGIN, y, 29, bold=True, color=TEXT, max_width=PAGE_W - 2 * MARGIN)
    draw_text(
        draw,
        "本报告展示的是持续探索到轮次预算耗尽，不是提前 stop。即使过程中出现文献候选、类比排序、guided ChemEnzy 或子目标尝试，最终 solved 仍只由 stitched_parent_route_proof.v1 决定。",
        MARGIN,
        y + 52,
        23,
        color=MUTED,
        max_width=PAGE_W - 2 * MARGIN,
    )
    return img


def page_stop_reason(data: dict[str, Any]) -> Image.Image:
    img, draw = new_page()
    y = page_title(draw, "为什么这一轮不是直接停止", "默认 blackboard planner 自主选 action；预算耗尽模式只改变停止策略")
    bullets = [
        f"实际停止原因：{data['termination']['reason']}。",
        f"轮次预算：{data['termination']['rounds_completed']}/{data['termination']['max_rounds']}，budget_exhausted={data['termination']['budget_exhausted']}。",
        "action batch 每轮仍经过 schema、预算和 raw reaction 注入校验；planner 不能直接宣称 solved。",
        "本地 PDF 没有预先塞给视觉工具；只有 `search_literature` 选择它以后，下一轮才执行 PDF 渲染。",
        "当普通策略会早停时，exhaust 模式先改走未耗尽的替代 action；没有 parent proof 时 final verdict 仍保持 non-solved。",
    ]
    for idx, item in enumerate(bullets, start=1):
        y = panel(draw, MARGIN, y, PAGE_W - 2 * MARGIN, 124, f"{idx}. {item}", color=ACCENT if idx <= 2 else TEXT)
        y += 18
    return img


def page_decision_timeline(data: dict[str, Any]) -> Image.Image:
    img, draw = new_page()
    y = page_title(draw, "Action 决策轨迹", "每个 action 都有 rationale、expected_artifact、success_condition 和执行记录")
    for row in data["decision_points"]:
        title = f"第 {row['round']} 轮 - {row['action_zh']} [{row['status']}, useful={row['useful_artifact']}]"
        y = panel(draw, MARGIN, y, PAGE_W - 2 * MARGIN, 128, title, color=ACCENT_2 if row["useful_artifact"] else WARN)
        inner_y = y - 74
        draw_text(draw, f"理由：{row['rationale']}", MARGIN + 26, inner_y, 17, color=MUTED, max_width=PAGE_W - 2 * MARGIN - 52)
        y += 12
        if y > PAGE_H - 170:
            break
    return img


def page_literature_pdf(data: dict[str, Any]) -> Image.Image:
    img, draw = new_page()
    y = page_title(draw, "文献与 PDF 证据链", "agent 请求文献后，用户提供的本地 PDF 才进入抽取流程")
    evidence = data["literature_evidence"] or {}
    source_lines = [
        f"{row.get('source_ref')}: local_pdf={bool(row.get('local_pdf'))}, recommendations={', '.join(row.get('extraction_task_recommendations') or [])}"
        for row in evidence.get("source_candidates") or []
        if isinstance(row, dict)
    ]
    y = section(draw, "Scout source candidates", source_lines, y, max_items=4)
    pdf_summary = data["real_artifact_summaries"]["pdf_evidence"]
    y = section(
        draw,
        "PDF render/index summary",
        [
            f"present={pdf_summary.get('present')}, accepted={pdf_summary.get('accepted')}, source={pdf_summary.get('source_pdf_path', '')}",
            f"counts={pdf_summary.get('summary')}",
            f"reasons={', '.join(translate_reason(r) for r in pdf_summary.get('reasons') or []) or '无'}",
        ],
        y,
        max_items=4,
    )
    visual = data["real_artifact_summaries"]["visual_chain"]
    section(
        draw,
        "Visual chain extraction",
        [
            f"present={visual.get('present')}, accepted={visual.get('accepted')}, status={visual.get('status')}",
            f"candidate_step_count={visual.get('candidate_step_count', 'n/a')}",
            f"reasons={', '.join(translate_reason(r) for r in visual.get('reasons') or []) or '无'}",
        ],
        y,
        max_items=4,
    )
    return img


def page_tool_results(data: dict[str, Any]) -> Image.Image:
    img, draw = new_page()
    y = page_title(draw, "真实工具执行摘要", "这里列出的都是 controller 记录的 tool_calls，不是报告脚本伪造结果")
    tool_lines = []
    for row in data["tool_summary"]:
        reasons = ", ".join(translate_reason(item) for item in row.get("reasons") or []) or "无"
        tool_lines.append(f"{row['tool_name']}: status={row['status']}, accepted={row['accepted']}, elapsed={row.get('elapsed_s')}s, reasons={reasons}")
    y = section(draw, "Tool calls", tool_lines, y, max_items=9)
    summaries = data["real_artifact_summaries"]
    compact = [
        f"compiled_exact: present={summaries['compiled_exact'].get('present')}, accepted={summaries['compiled_exact'].get('accepted')}, reasons={summaries['compiled_exact'].get('reasons')}",
        f"guided_chemenzy: present={summaries['guided_chemenzy'].get('present')}, accepted={summaries['guided_chemenzy'].get('accepted')}, route_status={summaries['guided_chemenzy'].get('route_status')}",
        f"route_expansion: present={summaries['route_expansion'].get('present')}, accepted={summaries['route_expansion'].get('accepted')}, subgoals={summaries['route_expansion'].get('subgoal_count')}",
        f"stitched_route: present={summaries['stitched_route'].get('present')}, accepted={summaries['stitched_route'].get('accepted')}, reasons={summaries['stitched_route'].get('reasons')}",
    ]
    section(draw, "Artifact result snapshot", compact, y, max_items=6)
    return img


def page_prior_success_comparison(data: dict[str, Any]) -> Image.Image:
    img, draw = new_page()
    y = page_title(draw, "与之前 bufotalin 成功版对比", "之前的 solved 来自 strict visual chain + compound 11 子目标 + stitch proof")
    prior = data.get("prior_success_comparison") or {}
    current_exact = len((data.get("literature_evidence") or {}).get("exact_rows") or [])
    current_proof = data.get("parent_route_proof") or {}
    prior_final = prior.get("final_verdict") or {}
    prior_visual = prior.get("strict_visual_chain") or {}
    prior_source = prior.get("source_detail_chain") or {}
    prior_stitch = prior.get("stitched_route") or {}
    rows = [
        f"旧成功 clean verdict: verdict={prior_final.get('verdict')}, solved={prior_final.get('solved')}",
        f"旧成功 strict visual chain: accepted={prior_visual.get('accepted')}, steps={prior_visual.get('candidate_step_count')}, terminal={((prior_visual.get('terminal') or {}).get('name') or '')}",
        f"旧成功 source-detail exact: accepted={prior_source.get('accepted')}, one_step_rows={prior_source.get('one_step_row_count')}, terminal_reached={prior_source.get('terminal_reached')}",
        f"旧成功 stitch: accepted={prior_stitch.get('accepted')}, stock_audit={prior_stitch.get('stock_audit_passed')}, status={prior_stitch.get('route_status')}",
        f"本轮自主 run: final={data['final_verdict'].get('verdict')}, exact_rows={current_exact}, proof_status={current_proof.get('route_status', 'missing')}",
    ]
    y = section(draw, "关键事实", rows, y, max_items=6)
    reflection = [
        "旧成功不是 ChemEnzy 原生直接闭合 bufotalin，而是 stock -> compound 11 -> bufotalin 的拼接半合成证明。",
        "本轮自主 run 没有重新得到 accepted 的 15 步 strict visual/source-detail chain；compile exact rows 因此没有可用行。",
        "没有 exact 文献段连到父路线时，guided ChemEnzy 或 child target 的候选只能作为探索反馈，不能作为 parent solved proof。",
    ]
    section(draw, "失败反思", reflection, y, max_items=4)
    return img


def page_blackboard_and_gate(data: dict[str, Any]) -> Image.Image:
    img, draw = new_page()
    y = page_title(draw, "Blackboard 与最终门禁", "失败、类比和 exact evidence 都只能进入状态；solved 需要 parent proof")
    y = section(draw, "Bridge tasks", [format_task(row) for row in data["bridge_tasks"]], y, max_items=6)
    selected = [
        f"{row.get('hypothesis_id')} score={row.get('score')} verify={', '.join(row.get('required_verification') or [])}"
        for row in (data["analogical_hypothesis_ranking"] or {}).get("selected_hypotheses") or []
        if isinstance(row, dict)
    ]
    y = section(draw, "Selected analogical hypotheses", selected, y, max_items=4)
    proof = data["parent_route_proof"] or {}
    proof_lines = [
        f"accepted={proof.get('accepted')}, solved={proof.get('solved')}, route_status={proof.get('route_status', 'missing')}",
        f"reasons={', '.join(translate_reason(item) for item in proof.get('reasons') or []) or '无'}",
        f"final={data['final_verdict'].get('verdict')}, solved={data['final_verdict'].get('solved')}",
    ]
    section(draw, "Parent proof gate", proof_lines, y, max_items=4)
    return img


def page_artifacts_and_tests(data: dict[str, Any]) -> Image.Image:
    img, draw = new_page()
    y = page_title(draw, "审计文件与验证", "报告输出、run artifacts 和 focused tests")
    y = panel(draw, MARGIN, y, PAGE_W - 2 * MARGIN, 150, f"测试：{data['test_summary']}", color=GOOD)
    y += 28
    refs = [f"{key}: {value}" for key, value in sorted((data.get("artifact_refs") or {}).items())[:10]]
    y = section(draw, "Blackboard artifact refs", refs, y, max_items=8)
    keys = data["artifact_bundle_summary"]["artifact_keys"]
    section(draw, "Artifact bundle keys", [", ".join(keys[:16])], y, max_items=2)
    return img


def build_audit(data: dict[str, Any], *, json_path: Path, md_path: Path, pdf_path: Path) -> dict[str, Any]:
    page_count = 0
    if pdf_path.exists():
        with fitz.open(pdf_path) as doc:
            page_count = len(doc)
    return {
        "schema_version": "bufotalin_budget_exhaustion_agentic_blackboard_report_audit.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "accepted": pdf_path.exists() and pdf_path.stat().st_size > 20_000 and page_count >= 8,
        "json_path": str(json_path),
        "markdown_path": str(md_path),
        "pdf_path": str(pdf_path),
        "case_run_dir": data["case_run_dir"],
        "page_count": page_count,
        "pdf_size_bytes": pdf_path.stat().st_size if pdf_path.exists() else 0,
        "termination_reason": data["termination"].get("reason"),
        "budget_exhausted": bool(data["termination"].get("budget_exhausted")),
        "final_verdict": data["final_verdict"].get("verdict"),
        "solved": bool(data["final_verdict"].get("solved")),
        "action_count": len(data["decision_points"]),
        "tool_call_count": len(data["tool_summary"]),
        "test_summary": data["test_summary"],
    }


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_json_optional(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return read_json(path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))
    return rows


def translate_reason(reason: Any) -> str:
    text = str(reason or "")
    return REASON_ZH.get(text, text)


def format_task(row: dict[str, Any]) -> str:
    return f"{row.get('task_type')}: {row.get('target_handle')} -> {row.get('required_bridge')}"


def new_page() -> tuple[Image.Image, ImageDraw.ImageDraw]:
    img = Image.new("RGB", (PAGE_W, PAGE_H), BG)
    draw = ImageDraw.Draw(img)
    draw.rectangle([36, 36, PAGE_W - 36, PAGE_H - 36], fill=PANEL, outline=LINE, width=2)
    return img, draw


def page_title(draw: ImageDraw.ImageDraw, title: str, subtitle: str) -> int:
    y = 78
    y = draw_text(draw, title, MARGIN, y, 39, bold=True, color=TEXT, max_width=PAGE_W - 2 * MARGIN)
    y = draw_text(draw, subtitle, MARGIN, y + 4, 22, color=MUTED, max_width=PAGE_W - 2 * MARGIN)
    draw.line([MARGIN, y + 24, PAGE_W - MARGIN, y + 24], fill=LINE, width=2)
    return y + 56


def metric(draw: ImageDraw.ImageDraw, label: str, value: str, x: int, y: int) -> int:
    draw.rounded_rectangle([x, y, PAGE_W - MARGIN, y + 108], radius=12, fill=(241, 245, 249), outline=LINE, width=1)
    draw_text(draw, label, x + 24, y + 17, 18, bold=True, color=ACCENT, max_width=230)
    draw_text(draw, value, x + 260, y + 17, 20, color=TEXT, max_width=PAGE_W - MARGIN - x - 284)
    return y + 126


def panel(draw: ImageDraw.ImageDraw, x: int, y: int, w: int, h: int, text: str, *, color: tuple[int, int, int]) -> int:
    draw.rounded_rectangle([x, y, x + w, y + h], radius=12, fill=(248, 250, 252), outline=LINE, width=1)
    draw.rectangle([x, y, x + 10, y + h], fill=color)
    draw_text(draw, text, x + 28, y + 20, 20, bold=True, color=TEXT, max_width=w - 56)
    return y + h


def section(draw: ImageDraw.ImageDraw, title: str, items: list[str], y: int, *, max_items: int) -> int:
    draw_text(draw, title, MARGIN, y, 25, bold=True, color=TEXT, max_width=PAGE_W - 2 * MARGIN)
    y += 42
    if not items:
        items = ["未记录"]
    for item in items[:max_items]:
        draw.ellipse([MARGIN + 4, y + 9, MARGIN + 16, y + 21], fill=ACCENT)
        y = draw_text(draw, str(item), MARGIN + 30, y, 18, color=MUTED, max_width=PAGE_W - 2 * MARGIN - 30) + 6
        if y > PAGE_H - 120:
            break
    return y + 24


def draw_flow_strip(draw: ImageDraw.ImageDraw, x: int, y: int, width: int) -> None:
    labels = ["Blackboard", "Planner", "Validator", "Executor", "Critic", "Parent Proof"]
    gap = 16
    box_w = int((width - gap * (len(labels) - 1)) / len(labels))
    for idx, label in enumerate(labels):
        bx = x + idx * (box_w + gap)
        color = ACCENT if idx in {0, 1} else ACCENT_2 if idx in {2, 3} else WARN if idx == 4 else GOOD
        draw.rounded_rectangle([bx, y, bx + box_w, y + 88], radius=12, fill=(239, 246, 255), outline=color, width=3)
        draw_text(draw, label, bx + 10, y + 29, 17, bold=True, color=color, max_width=box_w - 20, align="center")
        if idx < len(labels) - 1:
            ax = bx + box_w + 4
            draw.line([ax, y + 44, ax + gap - 8, y + 44], fill=MUTED, width=3)
            draw.polygon([(ax + gap - 8, y + 38), (ax + gap - 8, y + 50), (ax + gap, y + 44)], fill=MUTED)


def draw_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    x: int,
    y: int,
    size: int,
    *,
    bold: bool = False,
    color: tuple[int, int, int] = TEXT,
    max_width: int,
    align: str = "left",
) -> int:
    font = get_font(size, bold=bold)
    line_h = int(size * 1.38)
    for paragraph in str(text or "").split("\n"):
        lines = wrap_text(draw, paragraph, font, max_width)
        if not lines:
            y += line_h
            continue
        for line in lines:
            tx = x
            if align == "center":
                bbox = draw.textbbox((0, 0), line, font=font)
                tx = x + max(0, int((max_width - (bbox[2] - bbox[0])) / 2))
            draw.text((tx, y), line, font=font, fill=color)
            y += line_h
    return y


def wrap_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont, max_width: int) -> list[str]:
    if not text:
        return []
    chunks = textwrap.wrap(text, width=96, break_long_words=False, replace_whitespace=False) or [text]
    lines: list[str] = []
    for chunk in chunks:
        current = ""
        for token in chunk.split(" "):
            candidate = token if not current else f"{current} {token}"
            if text_width(draw, candidate, font) <= max_width:
                current = candidate
                continue
            if current:
                lines.append(current)
            if text_width(draw, token, font) <= max_width:
                current = token
                continue
            piece = ""
            current = ""
            for ch in token:
                if text_width(draw, piece + ch, font) <= max_width:
                    piece += ch
                else:
                    if piece:
                        lines.append(piece)
                    piece = ch
            current = piece
        if current:
            lines.append(current)
    return lines


def text_width(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont) -> int:
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0]


def get_font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    path = FONT_BOLD if bold and FONT_BOLD.exists() else FONT_REGULAR
    if not path.exists():
        path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    return ImageFont.truetype(str(path), size)


if __name__ == "__main__":
    # Pillow may warn when large PDF-rendered screenshots are embedded in text pages.
    JpegImagePlugin.MAXBLOCK = 1024 * 1024
    main()
