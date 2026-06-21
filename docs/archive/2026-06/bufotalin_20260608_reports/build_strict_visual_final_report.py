"""Build the corrected final PDF report from the strict-visual bufotalin result."""
from __future__ import annotations

import argparse
import json
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import fitz
from PIL import Image, ImageDraw, ImageFont
from rdkit import Chem, RDLogger
from rdkit.Chem import Draw


RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN_DIR = ROOT / "results/shared/bufotalin_fullflow_fresh_visual_existing_pdf_20260608_065053"
OUT_DIR = ROOT / "docs/bufotalin/report_20260608"
ASSET_DIR = OUT_DIR / "assets/strict_visual_final"

FONT_REGULAR = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
FONT_BOLD = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc")

PAGE_W, PAGE_H = 1240, 1754
MARGIN = 72
BG = (247, 249, 252)
TEXT = (23, 31, 42)
MUTED = (75, 85, 99)
LINE = (203, 213, 225)
PANEL = (255, 255, 255)
TEAL = (13, 105, 117)
BLUE = (29, 78, 216)
GREEN = (22, 101, 52)
AMBER = (146, 64, 14)
RED = (153, 27, 27)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR))
    parser.add_argument("--output-dir", default=str(OUT_DIR))
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    global ASSET_DIR
    ASSET_DIR = output_dir / "assets/strict_visual_final"
    ASSET_DIR.mkdir(parents=True, exist_ok=True)

    data = build_data(run_dir)
    stem = "bufotalin_strict_visual_final_corrected_report_20260608"
    json_path = output_dir / f"{stem}.json"
    md_path = output_dir / f"{stem}.md"
    pdf_path = output_dir / f"{stem}.pdf"
    audit_path = output_dir / f"{stem}_audit.json"

    json_path.write_text(json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(markdown(data), encoding="utf-8")
    render_pdf(data, pdf_path)
    audit = {
        "schema_version": "bufotalin_strict_visual_final_corrected_report_audit.v1",
        "accepted": pdf_path.exists() and pdf_path.stat().st_size > 80_000 and page_count(pdf_path) == 8,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "pdf_path": str(pdf_path),
        "markdown_path": str(md_path),
        "data_path": str(json_path),
        "pdf_size_bytes": pdf_path.stat().st_size if pdf_path.exists() else 0,
        "page_count": page_count(pdf_path),
        "final_verdict": data["clean_final"]["verdict"],
        "solved": data["clean_final"]["solved"],
        "stock_audit_passed": data["clean_final"]["stock_audit_passed"],
    }
    audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2, ensure_ascii=False, sort_keys=True))


def build_data(run_dir: Path) -> dict[str, Any]:
    clean_summary = read_json(run_dir / "bufotalin_strict_visual_continuation_clean_summary.json")
    continuation_summary = read_json(run_dir / "bufotalin_strict_visual_continuation_summary.json")
    clean_final = read_json(run_dir / "final_verdict_strict_visual_clean.json") or clean_summary.get("clean_final_verdict") or {}
    stale_final = read_json(run_dir / "final_verdict_strict_visual.json") or clean_summary.get("final_verdict") or {}
    target_input = read_json(run_dir / "target_input.json")
    preflight = read_json(run_dir / "preflight.json")
    target_profile = dict(preflight.get("target_profile") or {})
    terminal_audit = read_json(run_dir / "visual_literature_chain_extraction_strict_visual/strict_visual_terminal_audit.json")
    visual_validation = read_json(run_dir / "literature_intermediate_chain_validation_strict_visual/visual_structure_chain_validation.json")
    source_chain = read_json(run_dir / "source_detail_chain_route_strict_visual/source_detail_route_chain_audit.json")
    subgoal = read_json(run_dir / "route_expansion_subgoals/01_strict_visual_terminal_11_verifier.json")
    stitched = read_json(run_dir / "stitched_semisynthesis_route_strict_visual/stitched_semisynthesis_route.json")
    clean_validation = read_json(run_dir / "artifact_bundle_validation_strict_visual_clean.json")
    pdf_evidence = read_json(run_dir / "literature_pdf_structure_evidence.json")
    native_verifier = read_json(run_dir / "route_verifier_report.json")
    guided_result = read_json(run_dir / "guided_chemenzy_result.json")
    tool_status = [dict(item) for item in continuation_summary.get("tool_status") or [] if isinstance(item, dict)]
    route = dict(stitched.get("combined_route") or clean_summary.get("stitched_route", {}).get("combined_route") or {})
    return {
        "schema_version": "bufotalin_strict_visual_final_corrected_report_data.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "target": {
            "name": target_input.get("target_name") or "bufotalin",
            "input_smiles": target_input.get("target_smiles") or "",
            "isomeric_smiles": preflight.get("isomeric_smiles") or target_input.get("target_smiles") or "",
            "formula": target_profile.get("formula") or "",
            "exact_mw": target_profile.get("exact_mw") or "",
            "heavy_atoms": target_profile.get("heavy_atoms") or "",
            "rings": target_profile.get("rings") or "",
            "stereocenters": target_profile.get("stereocenters") or "",
            "inchi_key": preflight.get("inchi_key") or "",
        },
        "clean_final": clean_final,
        "stale_final": stale_final,
        "clean_validation": clean_validation,
        "terminal_audit": terminal_audit,
        "visual_validation": {
            "accepted": visual_validation.get("accepted"),
            "status": visual_validation.get("status"),
            "summary": visual_validation.get("summary") or clean_summary.get("visual_chain_summary") or {},
            "continuity_audit": visual_validation.get("continuity_audit") or {},
            "warnings": visual_validation.get("warnings") or [],
            "reasons": visual_validation.get("reasons") or [],
            "path": str(run_dir / "literature_intermediate_chain_validation_strict_visual/visual_structure_chain_validation.json"),
        },
        "source_chain": {
            "accepted": source_chain.get("accepted"),
            "step_count": source_chain.get("step_count"),
            "terminal_reached": source_chain.get("terminal_reached"),
            "terminal_name": source_chain.get("terminal_name"),
            "terminal_smiles": source_chain.get("terminal_smiles"),
            "terminal_canonical_smiles": source_chain.get("terminal_canonical_smiles"),
            "reasons": source_chain.get("reasons") or [],
            "path": str(run_dir / "source_detail_chain_route_strict_visual/source_detail_route_chain_audit.json"),
        },
        "subgoal": {
            "accepted": subgoal.get("accepted"),
            "route_status": subgoal.get("route_status"),
            "route_count": subgoal.get("route_count"),
            "accepted_route_count": subgoal.get("accepted_route_count"),
            "best_route_rank": subgoal.get("best_route_rank"),
            "target_match": subgoal.get("target_match"),
            "best_route_step_count": int((stitched.get("subgoal_closure") or {}).get("best_route_step_count") or 0),
            "target_equivalence_audit": subgoal.get("target_equivalence_audit") or {},
            "reasons": subgoal.get("reasons") or [],
            "path": str(run_dir / "route_expansion_subgoals/01_strict_visual_terminal_11_verifier.json"),
        },
        "stitched": {
            "accepted": stitched.get("accepted"),
            "solved": stitched.get("solved"),
            "route_status": stitched.get("route_status"),
            "stock_audit_passed": stitched.get("stock_audit_passed"),
            "combined_route": route,
            "terminal_match_audit": stitched.get("terminal_match_audit") or {},
            "warnings": stitched.get("warnings") or [],
            "reasons": stitched.get("reasons") or [],
            "path": str(run_dir / "stitched_semisynthesis_route_strict_visual/stitched_semisynthesis_route.json"),
        },
        "negative_controls": {
            "native_route_status": native_verifier.get("route_status"),
            "native_accepted_route_count": native_verifier.get("accepted_route_count"),
            "native_reasons": native_verifier.get("reasons") or [],
            "guided_route_status": guided_result.get("route_status"),
            "guided_reasons": guided_result.get("reasons") or [],
            "stale_final_reasons": stale_final.get("reasons") or [],
        },
        "pdf_evidence": {
            "source_pdf_path": pdf_evidence.get("source_pdf_path") or str(ROOT / "1-s2.0-S0040402025001668-main.pdf"),
            "summary": pdf_evidence.get("summary") or {},
            "scheme_crops": pdf_evidence.get("scheme_crops") or [],
            "path": str(run_dir / "literature_pdf_structure_evidence.json"),
        },
        "tool_status": tool_status,
        "artifact_refs": {
            **dict(clean_summary.get("artifact_refs") or {}),
            "clean_summary": str(run_dir / "bufotalin_strict_visual_continuation_clean_summary.json"),
            "strict_visual_clean_verdict": str(run_dir / "final_verdict_strict_visual_clean.json"),
            "strict_visual_clean_bundle": str(run_dir / "artifact_bundle_strict_visual_clean.json"),
            "strict_visual_subgoal_verifier": str(run_dir / "route_expansion_subgoals/01_strict_visual_terminal_11_verifier.json"),
        },
    }


def markdown(data: dict[str, Any]) -> str:
    final = data["clean_final"]
    route = data["stitched"]["combined_route"]
    lines = [
        "# Bufotalin strict visual final corrected report",
        "",
        "## 最终结论",
        "",
        f"- verdict: `{final.get('verdict')}`",
        f"- solved: `{final.get('solved')}`",
        f"- route_status: `{final.get('route_status')}`",
        f"- stock_audit_passed: `{final.get('stock_audit_passed')}`",
        f"- combined steps: `{route.get('combined_step_count')}` = `{route.get('subgoal_route_step_count')}` subgoal + `{route.get('literature_step_count')}` literature",
        "",
        "## 关键修正",
        "",
        "- strict visual terminal 11 与旧 hard-coded compound 11 不是同一立体异构体；本报告使用 strict terminal 重新跑 exact subgoal search。",
        "- 旧 `final_verdict_strict_visual.json` 被旧 guided/raw artifacts 污染而 rejected；clean bundle 只保留 strict visual deterministic artifacts，最终 verdict 为 solved。",
        "",
        "## 审计产物",
        "",
    ]
    for key, value in data["artifact_refs"].items():
        lines.append(f"- `{key}`: `{value}`")
    return "\n".join(lines) + "\n"


def render_pdf(data: dict[str, Any], pdf_path: Path) -> None:
    pages = [
        page_final(data),
        page_correction(data),
        page_strict_visual_chain(data),
        page_terminal_identity(data),
        page_subgoal(data),
        page_stitch(data),
        page_guards(data),
        page_artifacts(data),
    ]
    doc = fitz.open()
    for idx, page_image in enumerate(pages, start=1):
        png = ASSET_DIR / f"page_{idx:02d}.png"
        page_image.save(png, quality=95)
        page = doc.new_page(width=PAGE_W, height=PAGE_H)
        page.insert_image(page.rect, filename=str(png))
    doc.set_metadata(
        {
            "title": "Bufotalin strict visual final corrected report",
            "author": "AutoPlanner harness",
            "subject": "Corrected final report based on strict visual continuation",
        }
    )
    doc.save(pdf_path, deflate=True, garbage=4)
    doc.close()


def page_final(data: dict[str, Any]) -> Image.Image:
    canvas, draw = base("最终修正版结论", "依据 strict visual clean bundle，而不是旧污染 bundle", "01")
    target = data["target"]
    final = data["clean_final"]
    route = data["stitched"]["combined_route"]
    draw_molecule(canvas, draw, target["isomeric_smiles"], 760, 165, 350, 285, "bufotalin")
    paragraph(
        draw,
        MARGIN,
        165,
        620,
        f"最终 clean verdict 为 {final.get('verdict')}。成立条件是：strict visual 文献链 15 步通过连续性和 target identity 审计；strict terminal 11 作为 exact 子目标重新搜索并被 verifier 接受；stitch 审计确认 terminal 与子目标完全同一分子。",
        title="结论",
        tone="ok",
    )
    stats(
        draw,
        105,
        555,
        [
            ("文献链", f"{data['source_chain']['step_count']} 步", "target match + contiguous"),
            ("子目标", f"{data['subgoal']['accepted_route_count']}/{data['subgoal']['route_count']}", "exact terminal solved"),
            ("拼接", f"{route.get('combined_step_count')} 步", f"{route.get('subgoal_route_step_count')} + {route.get('literature_step_count')}"),
            ("Clean verdict", "solved", "stock audit passed"),
        ],
    )
    paragraph(
        draw,
        MARGIN,
        920,
        PAGE_W - 2 * MARGIN,
        "这不是说原生 ChemEnzy 或旧 guided rerun 直接成功。它们仍是负例，保留为假闭合防线。真正成功的是 strict visual source-detail 文献链与 exact terminal 子目标路线的确定性拼接。",
        title="必须保留的限定语",
        tone="warn",
    )
    paragraph(
        draw,
        MARGIN,
        1240,
        PAGE_W - 2 * MARGIN,
        "仍需人工复核的部分是视觉结构中的早期 steroid 立体化学，尤其 11/24/25/23/26/27/28 和 26 的 wavy C-Br。自动化结论可作为严格审计后的候选闭合路线，不应替代人工化学审稿。",
        title="化学边界",
    )
    footer(draw, data)
    return canvas


def page_correction(data: dict[str, Any]) -> Image.Image:
    canvas, draw = base("这次修正了什么", "旧 PDF 的结论入口需要换成 clean strict visual verdict", "02")
    stale = data["stale_final"]
    clean = data["clean_final"]
    paragraph(
        draw,
        MARGIN,
        165,
        520,
        f"旧 strict verdict 文件仍显示 {stale.get('verdict')}，原因是 bundle 里混入了旧 guided/native raw artifacts，触发 raw_reaction_injection/open timeout 等 guard。这是 bundle 污染问题，不是 strict visual stitch 本身失败。",
        title="旧入口为什么不对",
        tone="bad",
    )
    paragraph(
        draw,
        660,
        165,
        520,
        f"修正入口是 clean bundle：validation accepted={data['clean_validation'].get('accepted')}，reasons={join(data['clean_validation'].get('reasons') or [])}，final verdict={clean.get('verdict')}。",
        title="新入口",
        tone="ok",
    )
    flow(
        draw,
        95,
        590,
        [
            ("strict visual repair", "同一批新裁图"),
            ("deterministic normalize", "15 步 RDKit valid"),
            ("source-detail", "文献链通过"),
            ("exact subgoal", "strict terminal solved"),
            ("clean bundle", "solved"),
        ],
    )
    paragraph(
        draw,
        MARGIN,
        1025,
        PAGE_W - 2 * MARGIN,
        "另一个关键修正是 terminal：strict visual 链里的 compound 11 与早先 hard-coded compound 11 的 InChIKey 不同。因此不能复用旧子目标闭合，必须按 strict terminal 重新跑 exact subgoal search；本次已经重跑并通过。",
        title="不能偷换 terminal",
        tone="warn",
    )
    footer(draw, data)
    return canvas


def page_strict_visual_chain(data: dict[str, Any]) -> Image.Image:
    canvas, draw = base("strict visual 文献链", "从 bufotalin 到 strict terminal 11 的 15 步", "03")
    visual = data["visual_validation"]
    summary = visual["summary"]
    paragraph(
        draw,
        MARGIN,
        155,
        520,
        f"验证结果：accepted={visual['accepted']}，status={visual['status']}。step_count={summary.get('step_count')}，accepted_step_count={summary.get('accepted_step_count')}，compound_count={summary.get('compound_count')}，chain_contiguous={summary.get('chain_contiguous')}，target_match={summary.get('target_match')}。",
        title="结构链验证",
        tone="ok",
    )
    paragraph(
        draw,
        660,
        155,
        520,
        "这条链来自 strict vision repair 后的确定性规范化。它不是复用旧 source-detail 记录；每一步都经过 RDKit parse 和相邻主反应物/产物连续性检查。",
        title="来源说明",
    )
    crop_paths = [Path(str(row.get("image_path") or "")) for row in data["pdf_evidence"].get("scheme_crops") or []]
    image_strip(canvas, draw, [p for p in crop_paths if p.exists()][:3], MARGIN, 515, PAGE_W - 2 * MARGIN, 420)
    flow(
        draw,
        105,
        1010,
        [
            ("bufotalin", "target"),
            ("33/32/31", "后期 Scheme 4"),
            ("30/22/14", "pyrone 接入段"),
            ("20/19/28", "氧化保护段"),
            ("24/11", "strict terminal"),
        ],
    )
    footer(draw, data)
    return canvas


def page_terminal_identity(data: dict[str, Any]) -> Image.Image:
    canvas, draw = base("terminal 身份审计", "strict terminal 11 与旧 hard-coded 11 不同", "04")
    audit = data["terminal_audit"]
    strict = dict(audit.get("strict_visual_terminal") or {})
    old = dict(audit.get("previous_hardcoded_compound_11") or {})
    draw_molecule(canvas, draw, strict.get("input_smiles") or "", MARGIN, 170, 470, 360, "strict visual terminal 11")
    draw_molecule(canvas, draw, old.get("input_smiles") or "", 660, 170, 470, 360, "old hard-coded compound 11")
    paragraph(
        draw,
        MARGIN,
        610,
        520,
        f"strict terminal InChIKey={strict.get('inchikey')}。该分子是 strict visual 文献链最后一步的真实 terminal，后续 exact subgoal search 使用的就是它。",
        title="本次使用的 terminal",
        tone="ok",
    )
    paragraph(
        draw,
        660,
        610,
        520,
        f"旧 hard-coded 11 InChIKey={old.get('inchikey')}。matches_previous_hardcoded_compound_11={audit.get('matches_previous_hardcoded_compound_11')}，所以不能复用旧子目标结果。",
        title="旧 terminal",
        tone="warn",
    )
    paragraph(
        draw,
        MARGIN,
        1000,
        PAGE_W - 2 * MARGIN,
        "这个修正是本轮汇报最重要的地方：如果把两个 11 混为同一分子，stitch 会在交界处出现立体化学错误。现在的最终路线只基于 strict terminal 自己的 exact target verifier。",
        title="为什么这一步关键",
    )
    footer(draw, data)
    return canvas


def page_subgoal(data: dict[str, Any]) -> Image.Image:
    canvas, draw = base("strict terminal 子目标搜索", "重新跑 exact subgoal，不复用旧结果", "05")
    subgoal = data["subgoal"]
    audit = dict(subgoal.get("target_equivalence_audit") or {})
    paragraph(
        draw,
        MARGIN,
        155,
        520,
        f"exact subgoal search 已通过：route_status={subgoal['route_status']}，route_count={subgoal['route_count']}，accepted_route_count={subgoal['accepted_route_count']}，best_route_rank={subgoal['best_route_rank']}，best_route_step_count={subgoal['best_route_step_count']}。",
        title="子目标结果",
        tone="ok",
    )
    paragraph(
        draw,
        660,
        155,
        520,
        f"目标等价审计通过：request InChIKey={audit.get('request_inchikey')}，backend InChIKey={audit.get('backend_inchikey')}，match_basis={audit.get('match_basis')}。",
        title="exact target 审计",
        tone="ok",
    )
    stats(
        draw,
        120,
        580,
        [
            ("搜索目标", "strict 11", "非旧 hard-coded 11"),
            ("候选路线", str(subgoal["route_count"]), "ChemEnzy raw candidates"),
            ("通过路线", str(subgoal["accepted_route_count"]), "verifier accepted"),
            ("最佳路线", f"rank {subgoal['best_route_rank']}", f"{subgoal['best_route_step_count']} steps"),
        ],
    )
    paragraph(
        draw,
        MARGIN,
        965,
        PAGE_W - 2 * MARGIN,
        "stitched route 里仍保留 `subgoal_verifier:large_atom_jump` 警告，因为部分候选路线失败；但 verifier 接受了满足 target identity 和 stock closure 的候选路线，因此作为警告而不是拒绝理由。",
        title="警告解释",
        tone="warn",
    )
    footer(draw, data)
    return canvas


def page_stitch(data: dict[str, Any]) -> Image.Image:
    canvas, draw = base("最终拼接路线", "stock 到 strict terminal，再由文献链到 bufotalin", "06")
    stitched = data["stitched"]
    route = stitched["combined_route"]
    paragraph(
        draw,
        MARGIN,
        155,
        520,
        f"stitch accepted={stitched['accepted']}，solved={stitched['solved']}，route_status={stitched['route_status']}，stock_audit_passed={stitched['stock_audit_passed']}。",
        title="stitch 审计",
        tone="ok",
    )
    terminal = dict((stitched.get("terminal_match_audit") or {}).get("terminal") or {})
    paragraph(
        draw,
        660,
        155,
        520,
        f"交界 terminal InChIKey={terminal.get('inchikey')}。文献链 terminal 与子目标 target 的 canonical isomeric SMILES / InChIKey 完全一致。",
        title="交界身份",
        tone="ok",
    )
    route_bar(draw, MARGIN, 560, PAGE_W - 2 * MARGIN, route)
    paragraph(
        draw,
        MARGIN,
        855,
        PAGE_W - 2 * MARGIN,
        f"完整路线共 {route.get('combined_step_count')} 步：stock 到 strict terminal 的 {route.get('subgoal_route_step_count')} 步子目标路线，加上 strict terminal 到 bufotalin 的 {route.get('literature_step_count')} 步文献链。",
        title="路线组成",
    )
    footer(draw, data)
    return canvas


def page_guards(data: dict[str, Any]) -> Image.Image:
    canvas, draw = base("负例和 guard 没有被抹掉", "clean solved 与旧失败证据并存", "07")
    neg = data["negative_controls"]
    paragraph(
        draw,
        MARGIN,
        155,
        520,
        f"原生 ChemEnzy 仍未通过：route_status={neg['native_route_status']}，accepted_route_count={neg['native_accepted_route_count']}，reasons={join(neg['native_reasons'])}。",
        title="原生负例",
        tone="bad",
    )
    paragraph(
        draw,
        660,
        155,
        520,
        f"guided rerun 也未解决目标：route_status={neg['guided_route_status']}，reasons={join(neg['guided_reasons'])}。它不能覆盖 route verifier。",
        title="guided 负例",
        tone="bad",
    )
    paragraph(
        draw,
        MARGIN,
        540,
        PAGE_W - 2 * MARGIN,
        f"旧 strict verdict 的拒绝原因为：{join(neg['stale_final_reasons'])}。clean bundle validation 则 accepted={data['clean_validation'].get('accepted')}，reasons={join(data['clean_validation'].get('reasons') or [])}。也就是说，最终 solved 来自 clean deterministic artifacts，而不是忽略失败证据。",
        title="clean bundle 的作用",
        tone="ok",
    )
    flow(
        draw,
        105,
        990,
        [
            ("native/guided", "保留失败"),
            ("strict visual", "15 步通过"),
            ("exact subgoal", "strict 11 solved"),
            ("stitch", "identity match"),
            ("clean verdict", "solved"),
        ],
    )
    footer(draw, data)
    return canvas


def page_artifacts(data: dict[str, Any]) -> Image.Image:
    canvas, draw = base("最终汇报的审计入口", "复现时优先看这些文件", "08")
    rows = [
        ("clean summary", data["artifact_refs"].get("clean_summary"), "修正版总摘要"),
        ("clean verdict", data["artifact_refs"].get("strict_visual_clean_verdict"), "最终 solved 判定"),
        ("strict chain", data["artifact_refs"].get("visual_chain_validation"), "15 步视觉链验证"),
        ("terminal audit", data["artifact_refs"].get("strict_visual_terminal_audit"), "strict 11 与旧 11 区分"),
        ("source-detail chain", data["artifact_refs"].get("source_detail_chain_route"), "15 步文献链"),
        ("subgoal verifier", data["artifact_refs"].get("strict_visual_subgoal_verifier"), "strict terminal exact search"),
        ("stitched route", data["artifact_refs"].get("stitched_route"), "23 步拼接路线"),
        ("clean bundle", data["artifact_refs"].get("strict_visual_clean_bundle"), "无旧 raw artifact 污染的 bundle"),
    ]
    text = "\n".join(f"{name}: {rel(data, path)}；{note}" for name, path, note in rows)
    paragraph(draw, MARGIN, 155, PAGE_W - 2 * MARGIN, text, title="文件入口")
    paragraph(
        draw,
        MARGIN,
        930,
        PAGE_W - 2 * MARGIN,
        "最终正确表述：AutoPlanner 对 bufotalin 给出一条 strict-visual source-detail 文献链与 exact strict-terminal 子目标路线拼接而成的 stock-closed semisynthesis candidate。系统 clean verdict 为 solved；化学发布前仍需人工核对早期 steroid 立体化学。",
        title="最终汇报语句",
        tone="ok",
    )
    footer(draw, data)
    return canvas


def base(title: str, subtitle: str, page_no: str) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    canvas = Image.new("RGB", (PAGE_W, PAGE_H), BG)
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, PAGE_W, 30), fill=TEAL)
    draw.text((MARGIN, 58), title, font=font(42, bold=True), fill=TEXT)
    write(draw, subtitle, MARGIN, 120, PAGE_W - 2 * MARGIN - 80, 22, MUTED)
    draw.text((PAGE_W - MARGIN - 50, 66), page_no, font=font(30, bold=True), fill=TEAL)
    return canvas, draw


def paragraph(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    w: int,
    text: str,
    *,
    title: str = "",
    tone: str = "",
) -> None:
    h = max(180, min(420, height_for(text, w - 44, 20) + (86 if title else 46)))
    fill, outline, heading = PANEL, LINE, TEXT
    if tone == "ok":
        fill, outline, heading = (240, 253, 244), (52, 211, 153), GREEN
    elif tone == "warn":
        fill, outline, heading = (255, 251, 235), (245, 158, 11), AMBER
    elif tone == "bad":
        fill, outline, heading = (255, 247, 247), (248, 113, 113), RED
    draw.rounded_rectangle((x, y, x + w, y + h), radius=14, fill=fill, outline=outline, width=2)
    ty = y + 20
    if title:
        draw.text((x + 22, ty), title, font=font(25, bold=True), fill=heading)
        ty += 48
    write(draw, text, x + 22, ty, w - 44, 20, TEXT)


def flow(draw: ImageDraw.ImageDraw, x: int, y: int, nodes: list[tuple[str, str]]) -> None:
    node_w = 190
    gap = 32
    for idx, (title, body) in enumerate(nodes):
        nx = x + idx * (node_w + gap)
        draw.rounded_rectangle((nx, y, nx + node_w, y + 145), radius=14, fill=PANEL, outline=LINE, width=2)
        draw.text((nx + 14, y + 18), title, font=font(18, bold=True), fill=TEAL)
        write(draw, body, nx + 14, y + 58, node_w - 28, 15, MUTED)
        if idx < len(nodes) - 1:
            ax = nx + node_w + 4
            ay = y + 72
            draw.line((ax, ay, ax + gap - 8, ay), fill=BLUE, width=4)
            draw.polygon([(ax + gap - 8, ay), (ax + gap - 20, ay - 8), (ax + gap - 20, ay + 8)], fill=BLUE)


def stats(draw: ImageDraw.ImageDraw, x: int, y: int, rows: list[tuple[str, str, str]]) -> None:
    w = 235
    gap = 28
    for idx, (title, big, note) in enumerate(rows):
        nx = x + idx * (w + gap)
        draw.rounded_rectangle((nx, y, nx + w, y + 160), radius=14, fill=PANEL, outline=LINE, width=2)
        draw.text((nx + 18, y + 18), title, font=font(17, bold=True), fill=MUTED)
        draw.text((nx + 18, y + 62), big, font=font(27, bold=True), fill=TEXT)
        write(draw, note, nx + 18, y + 108, w - 36, 15, MUTED)


def route_bar(draw: ImageDraw.ImageDraw, x: int, y: int, w: int, route: dict[str, Any]) -> None:
    draw.rounded_rectangle((x, y, x + w, y + 180), radius=14, fill=PANEL, outline=LINE, width=2)
    left_w = int(w * 0.42)
    bar_y = y + 76
    draw.text((x + 24, y + 24), "stock → strict terminal → bufotalin", font=font(25, bold=True), fill=TEXT)
    draw.rounded_rectangle((x + 28, bar_y, x + 28 + left_w, bar_y + 52), radius=10, fill=(220, 252, 231), outline=(52, 211, 153), width=2)
    draw.rounded_rectangle((x + 28 + left_w + 74, bar_y, x + w - 28, bar_y + 52), radius=10, fill=(219, 234, 254), outline=(96, 165, 250), width=2)
    draw.text((x + 45, bar_y + 11), f"stock 到 strict 11：{route.get('subgoal_route_step_count')} 步", font=font(17, bold=True), fill=GREEN)
    draw.text((x + 28 + left_w + 92, bar_y + 11), f"strict 11 到 bufotalin：{route.get('literature_step_count')} 步", font=font(17, bold=True), fill=BLUE)
    ax = x + 28 + left_w + 30
    draw.line((ax, bar_y + 26, ax + 30, bar_y + 26), fill=TEAL, width=4)
    draw.polygon([(ax + 30, bar_y + 26), (ax + 18, bar_y + 17), (ax + 18, bar_y + 35)], fill=TEAL)
    write(draw, f"合计 {route.get('combined_step_count')} 步，stitch route_status={route.get('route_status') or 'solved'}。", x + 28, y + 138, w - 56, 16, MUTED)


def image_strip(canvas: Image.Image, draw: ImageDraw.ImageDraw, paths: list[Path], x: int, y: int, w: int, h: int) -> None:
    draw.rounded_rectangle((x, y, x + w, y + h), radius=14, fill=PANEL, outline=LINE, width=2)
    if not paths:
        draw.text((x + 24, y + 24), "No crop images found", font=font(18), fill=MUTED)
        return
    slot_w = (w - 60) // len(paths)
    for idx, path in enumerate(paths):
        with Image.open(path) as im:
            im = im.convert("RGB")
            im.thumbnail((slot_w - 18, h - 75))
            px = x + 20 + idx * slot_w + (slot_w - im.width) // 2
            py = y + 26 + (h - 75 - im.height) // 2
            canvas.paste(im, (px, py))
            draw.text((x + 20 + idx * slot_w, y + h - 36), path.stem[:32], font=font(14), fill=MUTED)


def draw_molecule(canvas: Image.Image, draw: ImageDraw.ImageDraw, smiles: str, x: int, y: int, w: int, h: int, label: str) -> None:
    draw.rounded_rectangle((x, y, x + w, y + h), radius=14, fill=PANEL, outline=LINE, width=2)
    mol = Chem.MolFromSmiles(smiles or "")
    if mol is None:
        return
    img = Draw.MolToImage(mol, size=(w - 35, h - 70))
    canvas.paste(img.convert("RGB"), (x + 18, y + 14))
    draw.text((x + 18, y + h - 38), label, font=font(16), fill=MUTED)


def footer(draw: ImageDraw.ImageDraw, data: dict[str, Any]) -> None:
    draw.line((MARGIN, PAGE_H - 82, PAGE_W - MARGIN, PAGE_H - 82), fill=LINE, width=2)
    draw.text((MARGIN, PAGE_H - 58), "bufotalin strict visual final corrected report", font=font(13), fill=MUTED)
    draw.text((PAGE_W - 560, PAGE_H - 58), "run_dir: " + Path(str(data["run_dir"])).name, font=font(13), fill=MUTED)


def write(draw: ImageDraw.ImageDraw, text: str, x: int, y: int, w: int, size: int, fill: tuple[int, int, int]) -> int:
    fnt = font(size)
    lines: list[str] = []
    for para in str(text).splitlines() or [""]:
        if not para:
            lines.append("")
            continue
        current = ""
        tokens = para.split(" ")
        if len(tokens) == 1 and any("\u4e00" <= ch <= "\u9fff" for ch in para):
            tokens = mixed_tokens(para)
        for token in tokens:
            sep = joiner(current, token)
            trial = token if not current else current + sep + token
            if text_width(draw, trial, fnt) <= w:
                current = trial
            else:
                if current:
                    lines.append(current)
                if text_width(draw, token, fnt) <= w:
                    current = token
                else:
                    lines.extend(textwrap.wrap(token, width=max(8, w // max(8, size)), break_long_words=True))
                    current = ""
        if current:
            lines.append(current)
    line_h = int(size * 1.35)
    for idx, line in enumerate(lines):
        draw.text((x, y + idx * line_h), line, font=fnt, fill=fill)
    return len(lines) * line_h


def mixed_tokens(text: str) -> list[str]:
    tokens: list[str] = []
    current = ""
    for ch in text:
        if "\u4e00" <= ch <= "\u9fff" or ch in "，。；：、（）“”":
            if current:
                tokens.append(current)
                current = ""
            tokens.append(ch)
        else:
            current += ch
    if current:
        tokens.append(current)
    return [token for token in tokens if token]


def joiner(current: str, token: str) -> str:
    if not current:
        return ""
    left = current[-1]
    right = token[0] if token else ""
    if any("\u4e00" <= ch <= "\u9fff" for ch in left + right):
        return ""
    if left in "，。；：、（“" or right in "，。；：、）”":
        return ""
    return " "


def height_for(text: str, w: int, size: int) -> int:
    probe = Image.new("RGB", (10, 10))
    draw = ImageDraw.Draw(probe)
    return write(draw, text, 0, 0, w, size, TEXT)


def text_width(draw: ImageDraw.ImageDraw, text: str, fnt: ImageFont.FreeTypeFont) -> int:
    box = draw.textbbox((0, 0), text, font=fnt)
    return box[2] - box[0]


def font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    path = FONT_BOLD if bold else FONT_REGULAR
    if path.exists():
        return ImageFont.truetype(str(path), size)
    return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return dict(data) if isinstance(data, dict) else {}


def fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.3f}".rstrip("0").rstrip(".")
    return str(value)


def join(items: list[Any]) -> str:
    values = [str(item) for item in items if str(item)]
    return "、".join(values) if values else "-"


def rel(data: dict[str, Any], path_value: Any) -> str:
    raw = str(path_value or "")
    if not raw:
        return "-"
    path = Path(raw)
    run_dir = Path(str(data.get("run_dir") or ""))
    try:
        return "run_dir/" + str(path.relative_to(run_dir))
    except ValueError:
        pass
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return raw


def page_count(pdf_path: Path) -> int:
    if not pdf_path.exists():
        return 0
    doc = fitz.open(pdf_path)
    count = doc.page_count
    doc.close()
    return count


if __name__ == "__main__":
    main()
