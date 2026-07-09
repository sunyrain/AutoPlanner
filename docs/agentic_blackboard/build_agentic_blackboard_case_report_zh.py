"""Build a Chinese PDF report for the agentic blackboard case run."""
from __future__ import annotations

import argparse
import io
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import fitz
from PIL import Image, ImageDraw, ImageFont, JpegImagePlugin


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from build_agentic_blackboard_case_report import (  # noqa: E402
    ACCENT,
    ACCENT_2,
    BAD,
    BG,
    FONT_BOLD,
    FONT_REGULAR,
    GOOD,
    LINE,
    MARGIN,
    MUTED,
    PAGE_H,
    PAGE_W,
    PANEL,
    TEXT,
    WARN,
    build_report_data,
    get_font,
    new_page,
    page_title,
    section_list,
    text_width,
)


OUT_DIR = ROOT / "docs" / "agentic_blackboard" / "report_20260609"

ACTION_ZH = {
    "generate_disconnection_hypotheses": "生成目标侧断键假设",
    "build_failure_critic_report": "构建失败批判报告",
    "search_literature": "检索目标近端文献",
    "compile_exact_literature_rows": "编译精确文献行",
    "rank_analogical_hypotheses": "排序类比假设",
    "stop_unresolved": "停止并保持未解决",
    "run_guided_chemenzy": "运行 guided ChemEnzy",
    "expand_child_target": "扩展子目标",
    "stitch_parent_route": "拼接父路线证明",
}

REASON_ZH = {
    "large_atom_jump": "出现无法解释的大重原子跳跃",
    "advanced_same_scaffold_terminal": "把高级同骨架中间体误当作库存终端",
    "literature_template_plugin_not_invoked": "文献模板插件未被后端调用",
    "plugin_product_hits=0": "插件产物命中为零，文献行暂未连到目标",
    "fake_closed_rejected": "伪闭合被拒绝",
    "no_deterministic_parent_route_proof": "缺少确定性的父路线证明",
}

TASK_ZH = {
    "target_proximal_bridge_required": "需要目标近端桥接",
    "upstream_terminal_synthesis": "需要高级终端的上游合成",
    "bridge_to_literature_product_required": "需要桥接到文献产物",
    "target_side_bridge_before_source_replay": "精确复播前需要目标侧桥接",
    "target_proximal_bridge": "目标近端桥接任务",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(OUT_DIR))
    parser.add_argument(
        "--test-summary",
        default="pytest -q: 279 passed, 2 skipped, 7 warnings in 76.17s",
        help="Completed test gate summary to embed in the report.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    run_dir = output_dir / "case_run_zh"
    data = build_report_data(run_dir=run_dir, test_summary=str(args.test_summary))
    data["language"] = "zh-CN"
    data["report_title"] = "Agentic Blackboard 中文案例报告"

    stem = "agentic_blackboard_mla_case_report_zh_20260609"
    json_path = output_dir / f"{stem}.json"
    md_path = output_dir / f"{stem}.md"
    pdf_path = output_dir / f"{stem}.pdf"
    audit_path = output_dir / f"{stem}_audit.json"

    json_path.write_text(json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown_zh(data), encoding="utf-8")
    render_pdf_zh(data, pdf_path)
    audit = build_audit_zh(data, json_path=json_path, md_path=md_path, pdf_path=pdf_path)
    audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2, ensure_ascii=False, sort_keys=True))


def render_markdown_zh(data: dict[str, Any]) -> str:
    final = data["final_verdict"]
    target = data["target"]
    lines = [
        "# Agentic Blackboard 中文案例报告",
        "",
        f"- 生成时间：{data['generated_at_utc']}",
        f"- 案例运行目录：`{data['case_run_dir']}`",
        f"- 目标：{target.get('target_name')}（重原子 {target.get('heavy_atoms')}，环数 {target.get('rings')}）",
        f"- 测试门禁：{data['test_summary']}",
        f"- 最终结论：`{final.get('verdict')}` / route_status `{final.get('route_status')}`",
        "",
        "## 一、架构结论",
        "",
        "本案例展示的是 Policy-driven DAG + Blackboard。系统不再固定执行一条线性 fullflow，而是在每一轮读取 blackboard 状态，由 action planner 选择 typed actions；随后 validator、executor、critic 和 parent proof gate 逐层约束输出。",
        "",
        "## 二、Agent 决策过程",
        "",
    ]
    for row in data["decision_points"]:
        action_type = str(row["action_type"])
        lines.extend(
            [
                f"### 第 {row['round']} 轮：`{action_type}`（{ACTION_ZH.get(action_type, action_type)}）",
                f"- 决策理由：{translate_decision_text(str(row['rationale']))}",
                f"- 期望产物：{row['expected_artifact']}",
                f"- 成功条件：{translate_decision_text(str(row['success_condition']))}",
                f"- 执行结果：status `{row['status']}`，useful_artifact `{row['useful_artifact']}`",
                "",
            ]
        )
    lines.extend(
        [
            "## 三、Blackboard 更新",
            "",
            f"- route_failures：{', '.join(translate_reason(row.get('reason', '')) for row in data['route_failures'])}",
            f"- bridge_tasks：{len(data['bridge_tasks'])} 个",
            f"- terminal_blacklist：{len(data['terminal_blacklist'])} 个",
            f"- exact_rows：{len((data['literature_evidence'] or {}).get('exact_rows') or [])} 个",
            f"- selected_analogies：{len((data['analogical_hypothesis_ranking'] or {}).get('selected_hypotheses') or [])} 个",
            "",
            "## 四、最终门禁",
            "",
            "本案例没有 stitched_parent_route_proof.v1，因此即使已有文献行和类比排序，也不能宣称 solved。最终状态保持非 solved，这是新架构的关键安全边界。",
        ]
    )
    return "\n".join(lines) + "\n"


def render_pdf_zh(data: dict[str, Any], pdf_path: Path) -> None:
    pages = [
        page_cover_zh(data),
        page_architecture_zh(data),
        page_decisions_zh(data),
        page_blackboard_zh(data),
        page_final_gate_zh(data),
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


def page_cover_zh(data: dict[str, Any]) -> Image.Image:
    img, draw = new_page()
    final = data["final_verdict"]
    target = data["target"]
    y = 92
    y = draw_text_zh(draw, "Agentic Blackboard 中文案例报告", MARGIN, y, 46, bold=True, color=TEXT, max_width=PAGE_W - 2 * MARGIN)
    y = draw_text_zh(draw, "重点：新架构下 agent 如何根据 blackboard 自主选择下一步", MARGIN, y + 8, 28, bold=True, color=ACCENT, max_width=PAGE_W - 2 * MARGIN)
    y += 34
    metrics = [
        ("目标", str(target.get("target_name") or "")),
        ("重原子 / 环数", f"{target.get('heavy_atoms')} / {target.get('rings')}"),
        ("最终结论", f"{final.get('verdict')} ({final.get('route_status')})"),
        ("测试门禁", data["test_summary"]),
    ]
    for label, value in metrics:
        y = draw_metric_zh(draw, label, value, MARGIN, y)
    draw_architecture_strip_zh(draw, MARGIN, 1000, PAGE_W - 2 * MARGIN)
    y = 1320
    draw_text_zh(draw, "核心结论", MARGIN, y, 28, bold=True, color=TEXT, max_width=PAGE_W - 2 * MARGIN)
    draw_text_zh(
        draw,
        "系统不再固定执行 failure -> scout -> extract -> rerun。每轮由 blackboard 汇总目标、失败、文献、类比和预算状态，planner 只选择 typed actions；最终 solved 必须由 deterministic parent proof 证明。",
        MARGIN,
        y + 48,
        24,
        color=MUTED,
        max_width=PAGE_W - 2 * MARGIN,
    )
    return img


def page_architecture_zh(data: dict[str, Any]) -> Image.Image:
    img, draw = new_page()
    y = page_title_zh(draw, "新架构说明", "Policy-driven DAG + Blackboard，外加确定性验证边界")
    guards = [
        "Planner 只能选择 action，不能直接输出 reaction SMILES 或 solved verdict。",
        "每个 action 都有 rationale、expected_artifact、success_condition，并经过 schema/budget 校验。",
        "Failure critic 会把 large_atom_jump、plugin_not_invoked、advanced terminal 转成 bridge tasks。",
        "精确文献行进入 exact evidence；类比假设只影响排序和搜索策略，不能作为 proof。",
        "最终 solved 需要 stitched_parent_route_proof.v1；child solved 不能自动升级 parent solved。",
    ]
    for idx, guard in enumerate(guards, start=1):
        y = draw_panel_zh(draw, MARGIN, y, PAGE_W - 2 * MARGIN, 122, f"{idx}. {guard}", color=ACCENT if idx <= 2 else TEXT)
        y += 18
    y += 18
    draw_text_zh(draw, "本次 blackboard 摘要", MARGIN, y, 26, bold=True, color=TEXT, max_width=PAGE_W - 2 * MARGIN)
    details = [
        f"route_failures={len(data['route_failures'])}",
        f"bridge_tasks={len(data['bridge_tasks'])}",
        f"terminal_blacklist={len(data['terminal_blacklist'])}",
        f"exact_rows={len((data['literature_evidence'] or {}).get('exact_rows') or [])}",
        f"selected_analogies={len((data['analogical_hypothesis_ranking'] or {}).get('selected_hypotheses') or [])}",
    ]
    draw_text_zh(draw, " | ".join(details), MARGIN, y + 48, 23, color=MUTED, max_width=PAGE_W - 2 * MARGIN)
    return img


def page_decisions_zh(data: dict[str, Any]) -> Image.Image:
    img, draw = new_page()
    y = page_title_zh(draw, "Agent 决策过程", "每一轮都从 blackboard 状态出发，选择一组 typed actions")
    for row in data["decision_points"]:
        action_type = str(row["action_type"])
        h = 188
        title = f"第 {row['round']} 轮 - {ACTION_ZH.get(action_type, action_type)} [{action_type}]"
        y = draw_panel_zh(draw, MARGIN, y, PAGE_W - 2 * MARGIN, h, title, color=ACCENT_2 if row["useful_artifact"] else WARN)
        inner_y = y - h + 52
        draw_text_zh(draw, f"决策理由：{translate_decision_text(str(row['rationale']))}", MARGIN + 24, inner_y, 19, color=TEXT, max_width=PAGE_W - 2 * MARGIN - 48)
        draw_text_zh(draw, f"期望产物：{row['expected_artifact']}", MARGIN + 24, inner_y + 56, 19, color=MUTED, max_width=PAGE_W - 2 * MARGIN - 48)
        draw_text_zh(draw, f"成功条件：{translate_decision_text(str(row['success_condition']))}", MARGIN + 24, inner_y + 98, 19, color=MUTED, max_width=PAGE_W - 2 * MARGIN - 48)
        y += 16
    return img


def page_blackboard_zh(data: dict[str, Any]) -> Image.Image:
    img, draw = new_page()
    y = page_title_zh(draw, "Blackboard 状态更新", "失败不是最终报告里的文字，而是下一轮行动的输入")
    y = section_list_zh(draw, "路线失败", [translate_reason(row.get("reason", "")) for row in data["route_failures"]], y)
    y = section_list_zh(draw, "桥接任务", [format_bridge_task(row) for row in data["bridge_tasks"]], y)
    y = section_list_zh(draw, "终端黑名单", [row.get("canonical_smiles", "") for row in data["terminal_blacklist"]], y)
    selected = [
        f"{row.get('hypothesis_id')}，score={row.get('score')}，必须验证：{', '.join(row.get('required_verification') or [])}"
        for row in (data["analogical_hypothesis_ranking"] or {}).get("selected_hypotheses") or []
    ]
    section_list_zh(draw, "被选中的 advisory hypotheses", selected, y)
    return img


def page_final_gate_zh(data: dict[str, Any]) -> Image.Image:
    img, draw = new_page()
    y = page_title_zh(draw, "测试、PDF 审计与最终门禁", "测试通过不等于 solved；solved 仍必须有 parent proof")
    y = draw_panel_zh(draw, MARGIN, y, PAGE_W - 2 * MARGIN, 150, f"全量测试门禁：{data['test_summary']}", color=GOOD)
    y += 28
    final = data["final_verdict"]
    text = (
        f"最终 verdict 为 {final.get('verdict')}，route_status 为 {final.get('route_status')}。"
        "这是预期行为：文献 exact rows 和 analogy ranking 只能支持探索方向，不能替代证明。"
        "若要宣称 solved，必须出现 stitched_parent_route_proof.v1，并同时满足 target equivalence、parent verifier accepted、stock audit、无 unexplained large atom jump、child-parent 连通和 exact-literature 连通。"
    )
    y = draw_panel_zh(draw, MARGIN, y, PAGE_W - 2 * MARGIN, 292, text, color=BAD if final.get("solved") else WARN)
    y += 32
    refs = data["artifact_refs"]
    ref_lines = [f"{key}: {shorten_path(value)}" for key, value in sorted(refs.items())[:8]]
    section_list_zh(draw, "关键 artifact refs", ref_lines, y)
    return img


def build_audit_zh(data: dict[str, Any], *, json_path: Path, md_path: Path, pdf_path: Path) -> dict[str, Any]:
    page_count = 0
    if pdf_path.exists():
        with fitz.open(pdf_path) as doc:
            page_count = len(doc)
    return {
        "schema_version": "agentic_blackboard_chinese_case_report_audit.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "accepted": pdf_path.exists() and pdf_path.stat().st_size > 20_000 and page_count >= 5,
        "language": "zh-CN",
        "pdf_path": str(pdf_path),
        "markdown_path": str(md_path),
        "json_path": str(json_path),
        "case_run_dir": data["case_run_dir"],
        "page_count": page_count,
        "pdf_size_bytes": pdf_path.stat().st_size if pdf_path.exists() else 0,
        "decision_count": len(data["decision_points"]),
        "final_verdict": data["final_verdict"].get("verdict"),
        "solved": bool(data["final_verdict"].get("solved")),
        "test_summary": data["test_summary"],
    }


def translate_decision_text(text: str) -> str:
    replacements = {
        "Initial MLA-like target needs target-side handles before any rerun.": "MLA-like 目标在任何重跑前都需要先识别目标侧功能手柄和断键区域。",
        "Prior route verifier shows large atom jump and advanced terminal; normalize that into bridge tasks.": "先前 route verifier 已发现大重原子跳跃和高级终端，需要归一化成 bridge tasks。",
        "Bridge tasks need target-proximal source candidates before exact replay.": "bridge tasks 需要目标近端文献候选，之后才适合 exact replay。",
        "Mock source-detail row is available; compile it as exact evidence, not as proof.": "已有 mock source-detail 行；它只能编译为 exact evidence，不能作为 solved proof。",
        "Rank advisory hypotheses after exact-row context is present.": "在 exact row 上下文存在后，对 advisory hypotheses 排序。",
        "No stitched parent proof exists; stop without solved claim.": "不存在 stitched parent proof，因此停止探索并避免 solved claim。",
        "Aryl ester, imide, cage, and amine advisory tasks appear.": "生成 aryl ester、imide、cage、amine 等 advisory tasks。",
        "Target bridge, terminal blacklist, and next-action bias are recorded.": "记录目标侧 bridge、terminal blacklist 和下一轮 action bias。",
        "Scout emits source candidates and extraction recommendations.": "scout 产出 source candidates 和 extraction recommendations。",
        "One exact row enters literature_evidence.exact_rows.": "一个 exact row 进入 literature_evidence.exact_rows。",
        "Selected hypotheses carry required verification and no_solved_claim.": "被选中的 hypotheses 保留 required verification 和 no_solved_claim。",
        "Final verdict stays unresolved or partial, never solved.": "最终 verdict 保持 unresolved/partial/rejected，不会变成 solved。",
    }
    return replacements.get(text, text)


def translate_reason(reason: str) -> str:
    text = str(reason or "")
    return f"{REASON_ZH.get(text, text)} [{text}]" if text else ""


def format_bridge_task(row: dict[str, Any]) -> str:
    task_type = str(row.get("task_type") or "")
    task = TASK_ZH.get(task_type, task_type)
    bridge = str(row.get("required_bridge") or row.get("target_handle") or "")
    return f"{task} [{task_type}]：{bridge}"


def shorten_path(value: Any) -> str:
    text = str(value or "")
    marker = "/docs/agentic_blackboard/"
    if marker in text:
        return "docs/agentic_blackboard/" + text.split(marker, 1)[1]
    return text


def page_title_zh(draw: ImageDraw.ImageDraw, title: str, subtitle: str) -> int:
    y = 78
    y = draw_text_zh(draw, title, MARGIN, y, 40, bold=True, color=TEXT, max_width=PAGE_W - 2 * MARGIN)
    y = draw_text_zh(draw, subtitle, MARGIN, y + 4, 23, color=MUTED, max_width=PAGE_W - 2 * MARGIN)
    draw.line([MARGIN, y + 24, PAGE_W - MARGIN, y + 24], fill=LINE, width=2)
    return y + 56


def draw_metric_zh(draw: ImageDraw.ImageDraw, label: str, value: str, x: int, y: int) -> int:
    draw.rounded_rectangle([x, y, PAGE_W - MARGIN, y + 112], radius=14, fill=(241, 245, 249), outline=LINE, width=1)
    draw_text_zh(draw, label, x + 24, y + 16, 18, bold=True, color=ACCENT, max_width=240)
    draw_text_zh(draw, value, x + 260, y + 16, 22, color=TEXT, max_width=PAGE_W - MARGIN - x - 284)
    return y + 132


def draw_panel_zh(draw: ImageDraw.ImageDraw, x: int, y: int, w: int, h: int, text: str, *, color: tuple[int, int, int]) -> int:
    draw.rounded_rectangle([x, y, x + w, y + h], radius=14, fill=(248, 250, 252), outline=LINE, width=1)
    draw.rectangle([x, y, x + 10, y + h], fill=color)
    draw_text_zh(draw, text, x + 28, y + 18, 21, bold=True, color=TEXT, max_width=w - 56)
    return y + h


def section_list_zh(draw: ImageDraw.ImageDraw, title: str, items: list[str], y: int) -> int:
    draw_text_zh(draw, title, MARGIN, y, 26, bold=True, color=TEXT, max_width=PAGE_W - 2 * MARGIN)
    y += 44
    if not items:
        items = ["无"]
    for item in items[:7]:
        draw.ellipse([MARGIN + 4, y + 10, MARGIN + 16, y + 22], fill=ACCENT)
        y = draw_text_zh(draw, str(item), MARGIN + 30, y, 20, color=MUTED, max_width=PAGE_W - 2 * MARGIN - 30) + 6
    return y + 24


def draw_architecture_strip_zh(draw: ImageDraw.ImageDraw, x: int, y: int, width: int) -> None:
    labels = ["Blackboard\n状态源", "Planner\n选动作", "Validator\n校验", "Executor\n执行", "Critic\n反馈", "Parent Proof\n最终证明"]
    gap = 18
    box_w = int((width - gap * (len(labels) - 1)) / len(labels))
    for idx, label in enumerate(labels):
        bx = x + idx * (box_w + gap)
        color = ACCENT if idx in {0, 1} else ACCENT_2 if idx in {2, 3} else WARN if idx == 4 else GOOD
        draw.rounded_rectangle([bx, y, bx + box_w, y + 104], radius=12, fill=(239, 246, 255), outline=color, width=3)
        draw_text_zh(draw, label, bx + 12, y + 22, 17, bold=True, color=color, max_width=box_w - 24, align="center")
        if idx < len(labels) - 1:
            ax = bx + box_w + 4
            draw.line([ax, y + 52, ax + gap - 8, y + 52], fill=MUTED, width=3)
            draw.polygon([(ax + gap - 8, y + 46), (ax + gap - 8, y + 58), (ax + gap, y + 52)], fill=MUTED)


def draw_text_zh(
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
    line_h = int(size * 1.42)
    for paragraph in str(text or "").split("\n"):
        lines = wrap_text_zh(draw, paragraph, font, max_width)
        if not lines:
            y += line_h
            continue
        for line in lines:
            tx = x
            if align == "center":
                tx = x + max(0, int((max_width - text_width(draw, line, font)) / 2))
            draw.text((tx, y), line, font=font, fill=color)
            y += line_h
    return y


def wrap_text_zh(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont, max_width: int) -> list[str]:
    if not text:
        return []
    lines: list[str] = []
    current = ""
    for token in split_zh_tokens(text):
        candidate = token if not current else current + token
        if text_width(draw, candidate, font) <= max_width:
            current = candidate
            continue
        if current:
            lines.append(current)
        if text_width(draw, token, font) <= max_width:
            current = token
        else:
            current = ""
            piece = ""
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


def split_zh_tokens(text: str) -> list[str]:
    tokens: list[str] = []
    current = ""
    for ch in text:
        if ch == " ":
            if current:
                tokens.append(current)
                current = ""
            tokens.append(" ")
            continue
        if "\u4e00" <= ch <= "\u9fff" or ch in "，。；：、（）":
            if current:
                tokens.append(current)
                current = ""
            tokens.append(ch)
            continue
        current += ch
    if current:
        tokens.append(current)
    return tokens


if __name__ == "__main__":
    main()
