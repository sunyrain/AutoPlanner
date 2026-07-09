"""Build a Chinese process-first PDF report for the bufotalin stitched full-flow run."""
from __future__ import annotations

import argparse
import json
import sys
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import fitz
from PIL import Image, ImageDraw, ImageFont
from rdkit import Chem, RDLogger
from rdkit.Chem import Draw


RDLogger.DisableLog("rdApp.*")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from build_stitched_fullflow_expert_report import build_report_data  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN_DIR = ROOT / "results" / "shared" / "bufotalin_stitched_fullflow_existing_pdf_20260608_0345"
OUT_DIR = ROOT / "docs" / "bufotalin" / "report_20260608"
ASSET_DIR = OUT_DIR / "assets" / "stitched_fullflow_chinese_process"

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
    ASSET_DIR.mkdir(parents=True, exist_ok=True)

    data = build_report_data(run_dir)
    stem = "bufotalin_stitched_fullflow_chinese_process_20260608"
    json_path = output_dir / f"{stem}.json"
    md_path = output_dir / f"{stem}.md"
    pdf_path = output_dir / f"{stem}.pdf"
    audit_path = output_dir / f"{stem}_audit.json"

    json_path.write_text(json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(markdown(data), encoding="utf-8")
    render_pdf(data, pdf_path)
    audit = {
        "schema_version": "bufotalin_stitched_fullflow_chinese_process_audit.v1",
        "accepted": pdf_path.exists() and pdf_path.stat().st_size > 70_000,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "pdf_path": str(pdf_path),
        "markdown_path": str(md_path),
        "data_path": str(json_path),
        "pdf_size_bytes": pdf_path.stat().st_size if pdf_path.exists() else 0,
        "page_count": page_count(pdf_path),
        "final_verdict": data["final"]["verdict"],
        "solved": data["final"]["solved"],
        "run_dir": str(run_dir),
    }
    audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2, ensure_ascii=False, sort_keys=True))


def markdown(data: dict[str, Any]) -> str:
    lines = [
        "# Bufotalin 中文流程汇报",
        "",
        "## 一句话结论",
        "",
        f"bufotalin 本轮最终结论是 `{data['final']['verdict']}`。它不是 ChemEnzy 原生路线直接成功，而是：文献 source-detail 链闭合到 compound 11，compound 11 子目标被 ChemEnzy/verifier 证明可由 stock 闭合，最后由 stitch 工具做身份审计后拼成完整半合成路线。",
        "",
        "## 关键数字",
        "",
        f"- 文献链：{data['source_detail_chain']['step_count']} 步，terminal reached={data['source_detail_chain']['terminal_reached']}",
        f"- 子目标：{data['subgoal']['accepted_route_count']} / {data['subgoal']['route_count']} 条路线被 verifier 接受",
        f"- 拼接路线：{data['stitch']['combined_route'].get('combined_step_count')} 步",
        f"- 原生 ChemEnzy：{data['native_chemenzy']['accepted_route_count']} 条 verifier accepted",
        f"- guided ChemEnzy 补跑：{data['guided_patched']['route_status']}",
        "",
        "## 审计入口",
        "",
    ]
    for key, value in data["artifact_refs"].items():
        lines.append(f"- {key}: `{value}`")
    return "\n".join(lines) + "\n"


def render_pdf(data: dict[str, Any], pdf_path: Path) -> None:
    pages = [
        page_conclusion(data),
        page_flow(data),
        page_gates(data),
        page_bufotalin_timeline(data),
        page_pdf_usage(data),
        page_failures(data),
        page_stitch(data),
        page_artifacts(data),
    ]
    doc = fitz.open()
    for idx, img in enumerate(pages, start=1):
        png = ASSET_DIR / f"page_{idx:02d}.png"
        img.save(png, quality=95)
        page = doc.new_page(width=PAGE_W, height=PAGE_H)
        page.insert_image(page.rect, filename=str(png))
    doc.set_metadata({
        "title": "Bufotalin 中文流程汇报",
        "author": "AutoPlanner harness",
        "subject": "高级天然产物 canonical 全流程与 bufotalin 示例",
    })
    doc.save(pdf_path, deflate=True, garbage=4)
    doc.close()


def page_conclusion(data: dict[str, Any]) -> Image.Image:
    canvas, draw = base("结论先讲清楚", "bufotalin 本轮为什么可以判定闭合", "01")
    draw_molecule(canvas, draw, data["target"]["isomeric_smiles"], 740, 165, 360, 290, "bufotalin")
    paragraph(
        draw,
        MARGIN,
        165,
        610,
        "这次的闭合不是说原生 ChemEnzy 一步到位，也不是说图片识别直接给了路线。真正成立的是一条被拆开审计、再拼回来的半合成路线：先用文献结构链从 bufotalin 逆推到 compound 11，再证明 compound 11 这个子目标可以由 stock 闭合，最后用拼接工具检查两个片段的交界分子完全一致。",
        title="一句话结论",
        tone="ok",
    )
    route = data["stitch"]["combined_route"]
    flow(
        draw,
        120,
        600,
        [
            ("输入 SMILES", "bufotalin"),
            ("文献链", f"{route.get('literature_step_count')} 步到 compound 11"),
            ("子目标搜索", f"{route.get('subgoal_route_step_count')} 步从 stock 到 compound 11"),
            ("身份审计", "同一 InChIKey / canonical SMILES"),
            ("最终判定", "闭合"),
        ],
    )
    stats(
        draw,
        130,
        985,
        [
            ("现场规划器", "通过", "运行语义合法"),
            ("文献链", f"{data['source_detail_chain']['step_count']} 步", "目标一致"),
            ("子目标", f"{data['subgoal']['accepted_route_count']}/{data['subgoal']['route_count']}", "通过验证"),
            ("拼接", f"{route.get('combined_step_count')} 步", "从 stock 到目标"),
        ],
    )
    paragraph(
        draw,
        MARGIN,
        1240,
        PAGE_W - 2 * MARGIN,
        "所以这里的结论是“stock → compound 11 → bufotalin”的拼接半合成路线闭合。原生和引导 ChemEnzy 的失败证据仍然保留在审计里，不会被最终结论隐藏。",
        title="边界",
    )
    footer(draw, data)
    return canvas


def page_flow(data: dict[str, Any]) -> Image.Image:
    canvas, draw = base("一个高级天然产物进来后，系统怎么走", "从 SMILES 到最终结论的中文流程", "02")
    steps = [
        ("1. 结构预检", "解析 SMILES，得到规范结构、分子式、手性中心、环数和天然产物风险标签。非法结构直接停止。"),
        ("2. 现场规划器", "让规划器给出工具计划，但先过格式门。字段、工具名、运行语义不合格就不执行工具。"),
        ("3. 原生 ChemEnzy", "先让逆合成引擎搜索路线。这里的闭合只是原始输出，不能直接代表系统最终闭合。"),
        ("4. 路线验证器", "逐条检查目标身份、stock 闭合、隐藏高级中间体、大原子跳变和同骨架假闭合。"),
        ("5. 文献/检索入口", "如果路线不闭合或疑似假闭合，就进入 frontier、文献、source access 和 PDF fallback。"),
        ("6. PDF 与文献结构链", "PDF 先变成页面、裁图、文本片段和证据清单；只有结构化产物/反应物 SMILES 通过 RDKit 和链连续性后才成为路线证据。"),
        ("7. 引导重跑 / 子目标", "文献链的末端或 frontier 可以发起子目标搜索。子目标闭合只说明这个子目标可解。"),
        ("8. 拼接审计", "检查文献链末端与子目标是否同一分子；通过后才把两个片段拼为完整路线。"),
        ("9. 最终结论", "确定性验证器汇总证据，输出闭合、未闭合或假闭合拒绝等结论。"),
    ]
    timeline(draw, MARGIN, 155, PAGE_W - 2 * MARGIN, steps)
    footer(draw, data)
    return canvas


def page_gates(data: dict[str, Any]) -> Image.Image:
    canvas, draw = base("三个最重要的硬门", "防止规划器或路线生成器过度宣称", "03")
    paragraph(
        draw,
        MARGIN,
        165,
        500,
        "规划器只能提出计划，不能越过格式门。现在运行语义必须落在合法枚举里；对象形式会被记录规范化审计，非法语义仍然拒绝。",
        title="硬门一：规划器格式",
        tone="ok",
    )
    paragraph(
        draw,
        660,
        165,
        500,
        "ChemEnzy 的原始闭合不是最终闭合。路线验证器会拒绝隐藏高级中间体、同骨架假闭合、大原子跳变和目标不一致的路线。",
        title="硬门二：路线验证器",
        tone="warn",
    )
    paragraph(
        draw,
        MARGIN,
        540,
        500,
        "子目标闭合不会自动升级为目标闭合。只有文献链末端的规范 SMILES 和 InChIKey 与子目标完全一致，且文献链、子目标验证都通过，拼接才成立。",
        title="硬门三：拼接身份审计",
        tone="ok",
    )
    paragraph(
        draw,
        660,
        540,
        500,
        f"本轮现场规划器已通过：运行语义={data['planner']['run_semantics']}，工具计划包含 PDF、文献结构链、子目标搜索和拼接审计。",
        title="bufotalin 对应结果",
    )
    flow(
        draw,
        110,
        970,
        [
            ("规划器", "计划合法"),
            ("ChemEnzy", "可产生候选"),
            ("验证器", "拒绝假闭合"),
            ("文献链", "结构化证据"),
            ("拼接", "闭合成立"),
        ],
    )
    footer(draw, data)
    return canvas


def page_bufotalin_timeline(data: dict[str, Any]) -> Image.Image:
    canvas, draw = base("bufotalin 本次真实运行", "按执行顺序看，不混淆失败与成功", "04")
    items = [
        ("现场规划器", "通过", f"格式门通过，运行语义={data['planner']['run_semantics']}"),
        ("原生 ChemEnzy", "失败", f"{fmt(data['native_chemenzy']['elapsed_s'])}s，{data['native_chemenzy']['route_count']} 条候选路线，通过验证={data['native_chemenzy']['accepted_route_count']}"),
        ("open research", "超时但留痕", f"{fmt(data['open_research']['elapsed_s'])}s，原因：{join(data['open_research']['reasons'])}；后续仍使用本地 PDF 继续"),
        ("PDF 重处理", "通过", f"重新渲染 {data['pdf_evidence']['summary'].get('rendered_page_count')} 页，生成 {data['pdf_evidence']['summary'].get('scheme_crop_count')} 个 crop"),
        ("文献结构链", "通过", f"{data['source_detail_chain']['step_count']} 步，terminal 到 compound 11"),
        ("引导重跑补跑", "失败", f"{fmt(data['guided_patched']['elapsed_s'])}s，验证器仍判 {data['guided_patched']['route_status']}"),
        ("compound 11 子目标", "通过", f"{data['subgoal']['accepted_route_count']} / {data['subgoal']['route_count']} 条路线通过，最佳名次={data['subgoal']['best_route_rank']}"),
        ("拼接 + 最终结论", "通过", f"拼接 {data['stitch']['combined_route'].get('combined_step_count')} 步，最终结论={data['final']['verdict']}"),
    ]
    status_timeline(draw, MARGIN, 155, PAGE_W - 2 * MARGIN, items)
    footer(draw, data)
    return canvas


def page_pdf_usage(data: dict[str, Any]) -> Image.Image:
    canvas, draw = base("PDF 是怎么被使用的", "已有 PDF 可以用，但图像和结构化链要真实重走", "05")
    pdf = data["pdf_evidence"]
    paragraph(
        draw,
        MARGIN,
        155,
        520,
        f"使用本地文件：{pdf['source_pdf_path']}。本轮没有直接把 PDF 当路线闭合证据，而是重新做页面渲染、裁图和证据清单；后续仍然要求结构化验证。",
        title="输入 PDF",
        tone="ok",
    )
    parsed = data["visual_audit"]["parsed_output"]
    paragraph(
        draw,
        660,
        155,
        520,
        "视觉审计很保守：它只清楚看到后期 Scheme 4 的 31→32→33→bufotalin 以及 SeO2 表格区域，没有单独声称完整路线闭合。",
        title="视觉审计结论",
        tone="warn",
    )
    crops = [Path(str(item.get("image_path") or "")) for item in pdf.get("scheme_crops") or []]
    image_strip(canvas, draw, [p for p in crops if p.exists()][:3], MARGIN, 520, PAGE_W - 2 * MARGIN, 430)
    flow(
        draw,
        105,
        1030,
        [
            ("PDF", "本地文件"),
            ("render/crop", "真实重做"),
            ("视觉审计", join(parsed.get("visible_sequence") or [])),
            ("结构化链", "15 步 RDKit 通过"),
            ("路线证据", "文献结构链"),
        ],
    )
    footer(draw, data)
    return canvas


def page_failures(data: dict[str, Any]) -> Image.Image:
    canvas, draw = base("为什么说原生逆合成失败了", "失败不是坏事，是防止假闭合", "06")
    paragraph(
        draw,
        MARGIN,
        160,
        520,
        f"原生 ChemEnzy 真实跑了 {fmt(data['native_chemenzy']['elapsed_s'])} 秒，产生 {data['native_chemenzy']['route_count']} 条候选路线。但路线验证器接受数是 {data['native_chemenzy']['accepted_route_count']}，原因包括：{join(data['native_chemenzy']['reasons'])}。",
        title="原生 ChemEnzy",
        tone="bad",
    )
    paragraph(
        draw,
        660,
        160,
        520,
        f"修复后的引导补跑也真实执行了 {fmt(data['guided_patched']['elapsed_s'])} 秒，但仍被路线验证器判为 {data['guided_patched']['route_status']}。这说明文献插件不能强行覆盖验证器。",
        title="引导补跑",
        tone="bad",
    )
    paragraph(
        draw,
        MARGIN,
        550,
        PAGE_W - 2 * MARGIN,
        "这和最终闭合不矛盾：失败的是“直接从目标跑出完整 stock route”的尝试；成功的是“文献链先到 compound 11，再单独证明 compound 11 可解，然后精确拼接”。系统把这两类证据分开审计，所以不会把假闭合当成功。",
        title="关键区别",
        tone="ok",
    )
    flow(
        draw,
        120,
        945,
        [
            ("直接逆合成", "verifier 拒绝"),
            ("文献链", "到 compound 11"),
            ("子目标", "compound 11 闭合"),
            ("stitch", "同一 terminal"),
            ("目标", "闭合"),
        ],
    )
    footer(draw, data)
    return canvas


def page_stitch(data: dict[str, Any]) -> Image.Image:
    canvas, draw = base("拼接为什么成立", "不是把两段随便拼起来，而是有身份审计", "07")
    terminal = data["stitch"]["terminal_match_audit"]
    target = data["stitch"]["target_identity_audit"]
    route = data["stitch"]["combined_route"]
    paragraph(
        draw,
        MARGIN,
        155,
        520,
        f"文献链末端和子目标的规范 SMILES 与 InChIKey 完全一致。交界分子 InChIKey={((terminal.get('terminal') or {}).get('inchikey'))}。",
        title="交界分子一致",
        tone="ok",
    )
    paragraph(
        draw,
        660,
        155,
        520,
        f"输入 bufotalin 与文献链目标也通过身份审计：目标一致={target.get('target_match')}。因此文献链确实服务于当前目标，不是相似物路线。",
        title="目标身份一致",
        tone="ok",
    )
    route_bar(draw, MARGIN, 520, PAGE_W - 2 * MARGIN, route)
    paragraph(
        draw,
        MARGIN,
        800,
        PAGE_W - 2 * MARGIN,
        f"最终拼接路线共 {route.get('combined_step_count')} 步：前半段是 stock 到 compound 11 的 {route.get('subgoal_route_step_count')} 步已验证子目标路线；后半段是 compound 11 到 bufotalin 的 {route.get('literature_step_count')} 步文献结构链。",
        title="完整路线的含义",
        tone="ok",
    )
    paragraph(
        draw,
        MARGIN,
        1120,
        PAGE_W - 2 * MARGIN,
        f"最终结论={data['final']['verdict']}，闭合={data['final']['solved']}，stock 审计通过={data['final']['stock_audit_passed']}。子目标验证器中仍有 large_atom_jump 警告，因为部分候选失败；但存在通过验证的路线并通过 stock closure，因此作为警告保留。",
        title="最终判定",
    )
    footer(draw, data)
    return canvas


def page_artifacts(data: dict[str, Any]) -> Image.Image:
    canvas, draw = base("最少需要看的审计产物", "只列复现和质疑时真正有用的文件", "08")
    useful = [
        ("总摘要", "bufotalin_stitched_fullflow_summary.json", "全流程关键数字和最终结论"),
        ("现场规划器", "codex_planner_run_record.json", "确认 live planner 和运行语义"),
        ("PDF 证据", "literature_pdf_structure_evidence.json", "确认本地 PDF、重渲染页和裁图"),
        ("文献链验证", "visual_structure_chain_validation.json", "确认 15 步链连续且目标一致"),
        ("文献结构链", "source_detail_route_chain_audit.json", "确认 terminal 到 compound 11"),
        ("子目标搜索", "route_expansion_subgoals/", "确认 compound 11 的验证结果"),
        ("拼接审计", "stitched_semisynthesis_route.json", "确认交界分子身份一致"),
        ("最终结论", "final_verdict.json", "系统最终闭合判定"),
        ("引导补跑", "guided_chemenzy_patched_tool_call.json", "确认引导重跑真实执行但被拒"),
        ("负例保留", "artifact_bundle_validation.json", "确认假闭合和超时证据没有被隐藏"),
    ]
    lines = [f"{name}：{filename}；{purpose}" for name, filename, purpose in useful]
    paragraph(draw, MARGIN, 155, PAGE_W - 2 * MARGIN, "\n".join(lines), title="审计入口")
    paragraph(
        draw,
        MARGIN,
        880,
        PAGE_W - 2 * MARGIN,
        "这版报告只讲流程和判断逻辑，不写实验步骤。完整绝对路径保存在配套 JSON 和审计 JSON 中；如果要质疑闭合，优先检查文献结构链、compound 11 验证器和拼接身份审计。",
        title="阅读顺序建议",
        tone="ok",
    )
    footer(draw, data)
    return canvas


def base(title: str, subtitle: str, page_no: str) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    canvas = Image.new("RGB", (PAGE_W, PAGE_H), BG)
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, PAGE_W, 30), fill=TEAL)
    draw.text((MARGIN, 58), title, font=font(43, bold=True), fill=TEXT)
    write(draw, subtitle, MARGIN, 120, PAGE_W - 2 * MARGIN - 80, 22, MUTED)
    draw.text((PAGE_W - MARGIN - 50, 66), page_no, font=font(30, bold=True), fill=TEAL)
    return canvas, draw


def paragraph(draw: ImageDraw.ImageDraw, x: int, y: int, w: int, text: str, *, title: str = "", tone: str = "") -> None:
    h = height_for(text, w, 21) + (70 if title else 36)
    h = max(180, min(360, h))
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
        draw.text((nx + 18, y + 62), big, font=font(28, bold=True), fill=TEXT)
        write(draw, note, nx + 18, y + 108, w - 36, 15, MUTED)


def timeline(draw: ImageDraw.ImageDraw, x: int, y: int, w: int, steps: list[tuple[str, str]]) -> None:
    row_h = 145
    for idx, (title, body) in enumerate(steps):
        yy = y + idx * row_h
        cx = x + 28
        draw.ellipse((cx - 16, yy + 10, cx + 16, yy + 42), fill=TEAL)
        draw.text((cx - 7, yy + 13), str(idx + 1), font=font(15, bold=True), fill=(255, 255, 255))
        if idx < len(steps) - 1:
            draw.line((cx, yy + 45, cx, yy + row_h - 5), fill=LINE, width=4)
        draw.text((x + 70, yy + 5), title, font=font(23, bold=True), fill=TEXT)
        write(draw, body, x + 70, yy + 44, w - 90, 18, MUTED)


def status_timeline(draw: ImageDraw.ImageDraw, x: int, y: int, w: int, rows: list[tuple[str, str, str]]) -> None:
    row_h = 154
    for idx, (title, status, body) in enumerate(rows):
        yy = y + idx * row_h
        color = GREEN if status == "通过" else RED if status == "失败" else AMBER
        draw.rounded_rectangle((x, yy, x + w, yy + 118), radius=12, fill=PANEL, outline=LINE, width=2)
        draw.rounded_rectangle((x + 18, yy + 24, x + 118, yy + 72), radius=10, fill=color)
        draw.text((x + 38, yy + 34), status, font=font(17, bold=True), fill=(255, 255, 255))
        draw.text((x + 145, yy + 22), title, font=font(23, bold=True), fill=TEXT)
        write(draw, body, x + 145, yy + 61, w - 165, 18, MUTED)


def image_strip(canvas: Image.Image, draw: ImageDraw.ImageDraw, paths: list[Path], x: int, y: int, w: int, h: int) -> None:
    draw.rounded_rectangle((x, y, x + w, y + h), radius=14, fill=PANEL, outline=LINE, width=2)
    slot_w = (w - 60) // max(1, len(paths))
    for idx, path in enumerate(paths):
        with Image.open(path) as im:
            im = im.convert("RGB")
            im.thumbnail((slot_w - 18, h - 75))
            px = x + 20 + idx * slot_w + (slot_w - im.width) // 2
            py = y + 26 + (h - 75 - im.height) // 2
            canvas.paste(im, (px, py))
            draw.text((x + 20 + idx * slot_w, y + h - 36), path.stem[:32], font=font(14), fill=MUTED)


def route_bar(draw: ImageDraw.ImageDraw, x: int, y: int, w: int, route: dict[str, Any]) -> None:
    draw.rounded_rectangle((x, y, x + w, y + 175), radius=14, fill=PANEL, outline=LINE, width=2)
    left_w = int(w * 0.43)
    bar_y = y + 72
    draw.text((x + 24, y + 22), "拼接路线", font=font(25, bold=True), fill=TEXT)
    draw.rounded_rectangle((x + 28, bar_y, x + 28 + left_w, bar_y + 48), radius=10, fill=(220, 252, 231), outline=(52, 211, 153), width=2)
    draw.rounded_rectangle((x + 28 + left_w + 72, bar_y, x + w - 28, bar_y + 48), radius=10, fill=(219, 234, 254), outline=(96, 165, 250), width=2)
    draw.text((x + 45, bar_y + 10), f"stock 到 compound 11：{route.get('subgoal_route_step_count')} 步", font=font(17, bold=True), fill=GREEN)
    draw.text((x + 28 + left_w + 92, bar_y + 10), f"compound 11 到 bufotalin：{route.get('literature_step_count')} 步", font=font(17, bold=True), fill=BLUE)
    ax = x + 28 + left_w + 30
    draw.line((ax, bar_y + 24, ax + 30, bar_y + 24), fill=TEAL, width=4)
    draw.polygon([(ax + 30, bar_y + 24), (ax + 18, bar_y + 15), (ax + 18, bar_y + 33)], fill=TEAL)
    write(draw, f"合计 {route.get('combined_step_count')} 步；方向是 stock → 文献 terminal → 目标天然产物。", x + 28, y + 135, w - 56, 16, MUTED)


def draw_molecule(canvas: Image.Image, draw: ImageDraw.ImageDraw, smiles: str, x: int, y: int, w: int, h: int, label: str) -> None:
    draw.rounded_rectangle((x, y, x + w, y + h), radius=14, fill=PANEL, outline=LINE, width=2)
    mol = Chem.MolFromSmiles(smiles or "")
    if mol is None:
        return
    im = Draw.MolToImage(mol, size=(w - 35, h - 70))
    canvas.paste(im.convert("RGB"), (x + 18, y + 14))
    draw.text((x + 18, y + h - 38), label, font=font(16), fill=MUTED)


def footer(draw: ImageDraw.ImageDraw, data: dict[str, Any]) -> None:
    draw.line((MARGIN, PAGE_H - 82, PAGE_W - MARGIN, PAGE_H - 82), fill=LINE, width=2)
    draw.text((MARGIN, PAGE_H - 58), "bufotalin stitched full-flow 中文流程汇报", font=font(13), fill=MUTED)
    draw.text((PAGE_W - 520, PAGE_H - 58), "运行目录：" + Path(str(data["run_dir"])).name, font=font(13), fill=MUTED)


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
            if width(draw, trial, fnt) <= w:
                current = trial
            else:
                if current:
                    lines.append(current)
                if width(draw, token, fnt) <= w:
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
    if any("\u4e00" <= ch <= "\u9fff" for ch in (left + right)):
        return ""
    if left in "，。；：、（“" or right in "，。；：、）”":
        return ""
    return " "


def height_for(text: str, w: int, size: int) -> int:
    probe = Image.new("RGB", (10, 10))
    draw = ImageDraw.Draw(probe)
    return write(draw, text, 0, 0, w, size, TEXT)


def width(draw: ImageDraw.ImageDraw, text: str, fnt: ImageFont.FreeTypeFont) -> int:
    box = draw.textbbox((0, 0), text, font=fnt)
    return box[2] - box[0]


def font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    path = FONT_BOLD if bold else FONT_REGULAR
    if path.exists():
        return ImageFont.truetype(str(path), size)
    return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)


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
        return ""
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
