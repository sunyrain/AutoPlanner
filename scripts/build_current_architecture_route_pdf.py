"""Build a display-ready PDF for the current runtime and GRIA migration route."""

from __future__ import annotations

import argparse
import html
import subprocess
from pathlib import Path

import markdown


ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs" / "architecture" / "CURRENT_ARCHITECTURE_STATUS.md"
ASSETS = ROOT / "docs" / "assets" / "current-architecture"
OUT = ROOT / "docs" / "deliverables"
HTML_OUT = OUT / "autoplanner-current-architecture-route.html"
PDF_OUT = OUT / "AutoPlanner_当前架构路线与GRIA迁移图_2026-07-16.pdf"

NAVY = "071824"
NAVY_2 = "0D2638"
INK = "173243"
MUTED = "6B7F8C"
PAPER = "F4F8FA"
CYAN = "22B8CF"
BLUE = "3977F6"
TEAL = "22A884"
GREEN = "43B86A"
AMBER = "E9A23B"
CORAL = "E9685B"
RED = "D94B55"
VIOLET = "7759D9"


def _esc(value: str) -> str:
    return html.escape(value, quote=True)


def _node(
    x: int,
    y: int,
    w: int,
    h: int,
    title: str,
    subtitle: str,
    color: str,
    *,
    badge: str = "",
) -> str:
    badge_svg = ""
    if badge:
        badge_svg = (
            f'<rect x="{x + w - 90}" y="{y + 14}" width="72" height="24" rx="12" '
            f'fill="#{color}" opacity=".16"/>'
            f'<text x="{x + w - 54}" y="{y + 31}" text-anchor="middle" '
            f'class="badge" fill="#{color}">{_esc(badge)}</text>'
        )
    return f"""
      <g class="node">
        <rect x="{x}" y="{y}" width="{w}" height="{h}" rx="18" fill="#FFFFFF" stroke="#{color}" stroke-width="3"/>
        <rect x="{x}" y="{y}" width="8" height="{h}" rx="4" fill="#{color}"/>
        {badge_svg}
        <text x="{x + 26}" y="{y + 38}" class="title" fill="#{INK}">{_esc(title)}</text>
        <text x="{x + 26}" y="{y + 68}" class="sub" fill="#{MUTED}">{_esc(subtitle)}</text>
      </g>
    """


def _arrow(x1: int, y1: int, x2: int, y2: int, color: str = CYAN) -> str:
    return (
        f'<path d="M{x1},{y1} L{x2},{y2}" fill="none" stroke="#{color}" '
        'stroke-width="4" stroke-linecap="round" marker-end="url(#arrow)"/>'
    )


def _svg_shell(width: int, height: int, title: str, body: str) -> str:
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<defs>
  <filter id="shadow" x="-10%" y="-20%" width="120%" height="150%"><feDropShadow dx="0" dy="6" stdDeviation="7" flood-color="#0B2637" flood-opacity=".12"/></filter>
  <marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto"><path d="M0,0 L0,6 L9,3 z" fill="#{CYAN}"/></marker>
  <style>
    text {{ font-family: "Microsoft YaHei", "Noto Sans CJK SC", Arial, sans-serif; }}
    .heading {{ font-size: 31px; font-weight: 700; letter-spacing: .4px; }}
    .kicker {{ font-size: 15px; font-weight: 700; letter-spacing: 2px; }}
    .title {{ font-size: 21px; font-weight: 700; }}
    .sub {{ font-size: 14px; }}
    .badge {{ font-size: 12px; font-weight: 700; }}
    .node {{ filter: url(#shadow); }}
    .label {{ font-size: 14px; font-weight: 700; }}
  </style>
</defs>
<rect width="100%" height="100%" rx="26" fill="#{PAPER}"/>
<text x="48" y="52" class="kicker" fill="#{CYAN}">AUTOPLANNER · SYSTEM ROUTE</text>
<text x="48" y="94" class="heading" fill="#{NAVY_2}">{_esc(title)}</text>
{body}
</svg>"""


def build_current_runtime_svg() -> str:
    body = [
        _node(
            50, 140, 260, 94, "Target intake", "SMILES · acceptance · budgets", BLUE, badge="INPUT"
        ),
        _node(360, 140, 260, 94, "RunKernel", "事件 · 恢复 · 任务 · 成本", BLUE, badge="AUTH"),
        _node(
            670, 140, 280, 94, "Global Director", "全局策略；proposal only", VIOLET, badge="PLAN"
        ),
        _node(
            1000,
            140,
            330,
            94,
            "Workers / Providers",
            "ChemEnzy · 文献 · 库存 · 实验交接",
            AMBER,
            badge="WORK",
        ),
        _node(
            1000,
            330,
            330,
            94,
            "Canonical admission",
            "身份 · 元素 · 映射 · 循环 · 权威",
            CORAL,
            badge="GATE",
        ),
        _node(640, 330, 310, 94, "Canonical Hypergraph", "分子 · 反应边 · 来源 · procedure", TEAL),
        _node(
            170, 535, 290, 94, "DeficitFrontier", "唯一待办 · 调度下一项工作", AMBER, badge="LOOP"
        ),
        _node(
            570,
            535,
            290,
            94,
            "Proof Portfolio",
            "最弱边/叶 · 多样性 · acceptance",
            GREEN,
            badge="DECIDE",
        ),
        _node(
            970,
            535,
            330,
            94,
            "Route Workbench",
            "同 revision 只读展示；不创造 proof",
            CYAN,
            badge="VIEW",
        ),
        _arrow(310, 187, 360, 187),
        _arrow(620, 187, 670, 187),
        _arrow(950, 187, 1000, 187),
        _arrow(1165, 234, 1165, 330),
        _arrow(1000, 377, 950, 377),
        _arrow(795, 424, 715, 535),
        _arrow(640, 377, 315, 535),
        _arrow(860, 582, 970, 582),
        '<path d="M170,582 C70,582 70,285 1000,285" fill="none" stroke="#E9A23B" stroke-width="3" stroke-dasharray="9 8" marker-end="url(#arrow)"/>',
        '<text x="88" y="455" class="label" fill="#B87920">未闭合 → 继续工作</text>',
        '<text x="1010" y="681" class="label" fill="#3977F6">只有 acceptance 可以宣布完成</text>',
    ]
    return _svg_shell(1380, 720, "当前真实运行路线 · Canonical V4", "".join(body))


def build_gria_svg() -> str:
    generators = [
        (60, "文献程序", "HTML / PDF / SI", BLUE),
        (320, "常规化学", "模型 / 规则 / ChemEnzy", CORAL),
        (580, "酶与 whole-cell", "单酶 / 级联 / 发酵", TEAL),
        (840, "机理一跳", "bond edit / critic", VIOLET),
        (1100, "路线重组", "共享 ChemicalState", AMBER),
    ]
    body = [
        _node(50, 130, 250, 92, "SMILES admission", "立体 · 盐型 · 互变体 · 身份", BLUE),
        _node(370, 130, 300, 92, "Target ChemicalState", "可区分的合成物质状态", BLUE),
        _arrow(300, 176, 370, 176),
        '<text x="50" y="282" class="kicker" fill="#6B7F8C">PARALLEL PROGRAM GENERATORS</text>',
    ]
    for x, title, subtitle, color in generators:
        body.append(_node(x, 310, 230, 82, title, subtitle, color))
        body.append(_arrow(520, 222, x + 115, 310))
    body.extend(
        [
            _node(
                400,
                460,
                600,
                88,
                "Unified HostAdmission",
                "identity → balance → atom map → centre → authority separation",
                CORAL,
                badge="GATE",
            ),
            *[_arrow(x + 115, 392, 700, 460) for x, *_ in generators],
            _node(
                70,
                625,
                320,
                94,
                "Transformation Program Graph",
                "program + operation + claim",
                TEAL,
            ),
            _node(
                470,
                625,
                300,
                94,
                "Pareto Optimizer",
                "证据 · 步数 · 风险 · 成本 · 选择性",
                GREEN,
                badge="RANK",
            ),
            _node(
                850,
                625,
                420,
                94,
                "Route Portfolio",
                "evidence / shortest / enzyme / novel / fallback",
                CYAN,
                badge="OUTPUT",
            ),
            _arrow(700, 548, 230, 625),
            _arrow(390, 672, 470, 672),
            _arrow(770, 672, 850, 672),
            _node(
                850,
                790,
                420,
                94,
                "Validation & Experiment Frontier",
                "request · RunKernel handoff · domain gate",
                AMBER,
            ),
            _arrow(1060, 719, 1060, 790),
            '<path d="M850,837 C520,920 100,900 230,719" fill="none" stroke="#22A884" stroke-width="4" stroke-dasharray="10 8" marker-end="url(#arrow)"/>',
            '<text x="330" y="915" class="label" fill="#168366">exact-boundary dirty hint · 只读子任务重算</text>',
        ]
    )
    return _svg_shell(1380, 940, "目标路线 · 一个 SMILES 进入 GRIA 之后", "".join(body))


def build_migration_svg() -> str:
    phases = [
        ("0", "冻结错误扩展", "已执行", GREEN),
        ("1", "Program 基座", "影子准入", AMBER),
        ("2", "Route 切换", "未实现", RED),
        ("3", "真实超步", "影子准入", AMBER),
        ("4", "结构化机理", "重拼 + 三态验证", AMBER),
        ("5", "Program 优化", "只读 Pareto", AMBER),
        ("6", "实验反馈", "Claim + 受限派发", AMBER),
    ]
    body = []
    for index, (phase, title, state, color) in enumerate(phases):
        x = 45 + index * 188
        body.append(
            f'<circle cx="{x + 70}" cy="205" r="33" fill="#{color}"/>'
            f'<text x="{x + 70}" y="216" text-anchor="middle" class="heading" fill="#FFFFFF">{phase}</text>'
            f'<rect x="{x}" y="260" width="150" height="118" rx="16" fill="#FFFFFF" stroke="#{color}" stroke-width="3"/>'
            f'<text x="{x + 75}" y="302" text-anchor="middle" class="title" fill="#{INK}">{_esc(title)}</text>'
            f'<text x="{x + 75}" y="342" text-anchor="middle" class="label" fill="#{color}">{_esc(state)}</text>'
        )
        if index < len(phases) - 1:
            body.append(_arrow(x + 104, 205, x + 174, 205))
    body.append(
        '<rect x="45" y="430" width="1280" height="72" rx="16" fill="#FFF5E5" stroke="#E9A23B"/>'
        '<text x="685" y="461" text-anchor="middle" class="title" fill="#8D5A0B">阶段 6 已接 host-trusted catalog、manual handoff 与 RunKernel 单账本恢复；真实设备仍未接入</text>'
        '<text x="685" y="487" text-anchor="middle" class="sub" fill="#8D5A0B">唯一 DeficitFrontier 不变；dispatch、result、Claim 与 Program admission 均不授予路线闭合</text>'
    )
    return _svg_shell(1380, 550, "迁移路线 · 从 V4 反应边到 GRIA 程序图", "".join(body))


def write_diagrams() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    diagrams = {
        "current-runtime-route.svg": build_current_runtime_svg(),
        "gria-smiles-route.svg": build_gria_svg(),
        "gria-migration-route.svg": build_migration_svg(),
    }
    for name, content in diagrams.items():
        normalized = "\n".join(line.rstrip() for line in content.splitlines()) + "\n"
        (ASSETS / name).write_text(normalized, encoding="utf-8")


def _status_badges(body: str) -> str:
    replacements = {
        "已实现": "status-done",
        "已实现但保留兼容等级": "status-partial",
        "已实现主链，真实案例仍不足": "status-partial",
        "过渡实现": "status-bridge",
        "过渡实现（Program 候选）": "status-bridge",
        "过渡实现（影子准入）": "status-bridge",
        "过渡实现（只读）": "status-bridge",
        "过渡实现（影子持久）": "status-bridge",
        "过渡实现（单反应节点）": "status-bridge",
        "已实现（只读）": "status-done",
        "未实现": "status-missing",
        "已执行；现有案例仅作回归": "status-done",
        "未开始生产实现": "status-missing",
        "有过渡原型，核心未实现": "status-bridge",
        "影子 store/oracle；7 类回归（3 canonical replay + 4 candidate）；3 类 current replay 路线双读通过": "status-bridge",
    }
    for label in sorted(replacements, key=len, reverse=True):
        body = body.replace(
            f"<td>{label}</td>",
            f'<td><span class="status {replacements[label]}">{label}</span></td>',
        )
    return body


def build_html() -> Path:
    write_diagrams()
    raw = DOC.read_text(encoding="utf-8")
    body_md = raw.split("\n", 1)[1] if raw.startswith("# ") else raw
    body = markdown.markdown(
        "[TOC]\n\n" + body_md,
        extensions=["tables", "fenced_code", "toc", "sane_lists"],
        extension_configs={"toc": {"title": "内容导航", "toc_depth": "2"}},
    )
    body = body.replace(
        'src="../assets/',
        f'src="{(ROOT / "docs" / "assets").as_uri()}/',
    )
    body = _status_badges(body)
    css = f"""
    @page {{ size: A4 landscape; margin: 13mm 15mm 13mm; }}
    @page:first {{ margin: 0; }}
    * {{ box-sizing: border-box; }}
    html {{ background: white; }}
    body {{ margin: 0; color: #{INK}; font-family: "Microsoft YaHei", "Noto Sans CJK SC", Arial, sans-serif;
            font-size: 10.6pt; line-height: 1.62; background: white; }}
    .cover {{ width: 297mm; height: 210mm; page-break-after: always; position: relative; overflow: hidden;
              color: white; background: linear-gradient(130deg, #{NAVY} 0%, #{NAVY_2} 56%, #104F66 100%); }}
    .cover::before {{ content: ""; position: absolute; width: 150mm; height: 150mm; border-radius: 50%;
                     right: -28mm; top: -45mm; border: 20mm solid rgba(34,184,207,.13); }}
    .cover::after {{ content: ""; position: absolute; width: 96mm; height: 96mm; border-radius: 50%;
                    right: 32mm; bottom: -53mm; border: 12mm solid rgba(67,184,106,.13); }}
    .cover-grid {{ position: absolute; inset: 0; opacity: .12;
                   background-image: linear-gradient(rgba(255,255,255,.12) 1px, transparent 1px),
                   linear-gradient(90deg, rgba(255,255,255,.12) 1px, transparent 1px);
                   background-size: 12mm 12mm; }}
    .cover-copy {{ position: absolute; z-index: 2; left: 24mm; top: 33mm; width: 205mm; }}
    .eyebrow {{ color: #{CYAN}; font-weight: 700; letter-spacing: 3px; font-size: 10pt; }}
    .cover h1 {{ margin: 8mm 0 5mm; font-size: 31pt; line-height: 1.2; letter-spacing: .5px; color: white; }}
    .cover .subtitle {{ width: 180mm; color: #d7e8ef; font-size: 14pt; line-height: 1.75; }}
    .truth {{ display: inline-block; margin-top: 10mm; padding: 3.2mm 5mm; border: 1px solid rgba(255,255,255,.3);
              border-left: 4px solid #{AMBER}; border-radius: 2mm; background: rgba(4,17,28,.45); font-size: 10.5pt; }}
    .meta {{ position: absolute; z-index: 2; left: 24mm; bottom: 17mm; color: #a9c4cf; font-size: 9pt; }}
    main {{ max-width: 267mm; margin: 0 auto; }}
    .toc {{ background: #eef6f8; border: 1px solid #d6e7ec; border-left: 5px solid #{CYAN};
            border-radius: 3mm; padding: 7mm 9mm; margin: 0; }}
    .toc > ul {{ columns: 2; column-gap: 18mm; padding-left: 6mm; }}
    .toc li {{ margin: 2mm 0; break-inside: avoid; }}
    .toc a {{ color: #14586F; text-decoration: none; font-weight: 600; }}
    h1, h2, h3 {{ color: #{NAVY_2}; line-height: 1.32; page-break-after: avoid; }}
    h2 {{ font-size: 20pt; margin: 0 0 6mm; padding: 0 0 3mm; border-bottom: 2px solid #d8e6eb;
          page-break-before: always; }}
    h2[id="4"] {{ page-break-before: auto; margin-top: 8mm; }}
    h2[id="7"] {{ page-break-before: auto; margin-top: 8mm; }}
    h2[id="8"] {{ page-break-before: auto; margin-top: 8mm; }}
    h2::before {{ content: "AUTOPLANNER  /  CURRENT ROUTE"; display: block; color: #{CYAN}; font-size: 8.5pt;
                  letter-spacing: 1.8px; margin-bottom: 2.5mm; }}
    h3 {{ font-size: 13.5pt; color: #155C73; margin: 7mm 0 3mm; }}
    p {{ margin: 2.5mm 0; orphans: 3; widows: 3; }}
    p > img {{ display: block; width: 100%; max-height: 148mm; object-fit: contain; margin: 4mm auto 5mm;
               border: 1px solid #d6e5ea; border-radius: 4mm; background: #{PAPER}; page-break-inside: avoid; }}
    p > img[src*="gria-migration-route.svg"] {{ height: 45mm; margin-bottom: 1mm; }}
    ul, ol {{ padding-left: 7mm; margin: 2mm 0 4mm; }}
    li {{ margin: 1.3mm 0; }}
    h2[id="4"] + p + pre {{ padding: 3mm 5mm; font-size: 8pt; line-height: 1.22; }}
    h2[id="4"] + p + pre + p + ol {{ font-size: 9pt; line-height: 1.3; }}
    h2[id="1"] + p + ul {{ columns: 2; column-gap: 9mm; column-rule: 1px solid #d8e6eb;
                            font-size: 8.7pt; line-height: 1.38; margin-top: 2mm; }}
    h2[id="1"] + p + ul li {{ margin: .8mm 0; break-inside: avoid-column; }}
    h2[id="7"] + p + ol {{ columns: 2; column-gap: 9mm; column-rule: 1px solid #d8e6eb;
                            font-size: 8.2pt; line-height: 1.2; margin-top: 1mm; }}
    h2[id="7"] + p + ol li {{ margin: .3mm 0; break-inside: avoid-column; }}
    h2[id="8"] ~ p, h2[id="8"] ~ ul {{ font-size: 9.4pt; line-height: 1.38; }}
    h2[id="8"] ~ ul {{ margin: 1mm 0 2mm; }}
    h2[id="8"] ~ ul li {{ margin: .6mm 0; }}
    strong {{ color: #0B4D64; }}
    code {{ font-family: Consolas, "Microsoft YaHei", monospace; color: #08657A; background: #ECF5F7; padding: .2mm .8mm; border-radius: 1mm; }}
    pre {{ padding: 5mm 6mm; background: #F3F8FA; border: 1px solid #d2e3e9; border-left: 5px solid #{CYAN};
           border-radius: 3mm; font-size: 9pt; line-height: 1.5; white-space: pre-wrap; page-break-inside: avoid; }}
    pre code {{ background: transparent; padding: 0; color: #26495A; }}
    table {{ width: 100%; border-collapse: collapse; margin: 3mm 0 3mm; font-size: 7.65pt; line-height: 1.34; }}
    thead {{ display: table-header-group; }}
    tr {{ page-break-inside: avoid; }}
    th {{ padding: 1.55mm 2mm; background: #{NAVY_2}; color: white; text-align: left; font-weight: 700; }}
    td {{ padding: 1.35mm 2mm; border: 1px solid #D4E2E8; vertical-align: top; }}
    tr:nth-child(even) td {{ background: #F5F9FA; }}
    h2 + table {{ margin-top: 2mm; font-size: 7.15pt; line-height: 1.22; }}
    h2 + table th {{ padding: 1.15mm 1.7mm; }}
    h2 + table td {{ padding: 1.05mm 1.7mm; }}
    h2 + table .status {{ padding: .75mm 1.7mm; font-size: 7.1pt; }}
    .status {{ display: inline-block; padding: 1mm 2.1mm; border-radius: 10mm; font-size: 7.7pt; font-weight: 700; white-space: nowrap; }}
    .status-done {{ color: #116D43; background: #DDF5E8; }}
    .status-partial {{ color: #985F09; background: #FFF0D1; }}
    .status-bridge {{ color: #8B5D0D; background: #FCE9C3; }}
    .status-missing {{ color: #A82F3A; background: #FBE0E3; }}
    blockquote {{ margin: 0 0 6mm; padding: 4mm 5mm; background: #FFF5E5; border-left: 5px solid #{AMBER}; color: #6B4D1C; }}
    a {{ color: #087A93; text-decoration: none; }}
    .report-note {{ margin: 2.5mm 0 0; padding: 1.8mm 4mm; color: #607580; background: #EEF5F7;
                    border-radius: 2mm; font-size: 7.8pt; line-height: 1.35; break-inside: avoid; }}
    """
    cover = f"""
    <section class="cover">
      <div class="cover-grid"></div>
      <div class="cover-copy">
        <div class="eyebrow">AUTOPLANNER · ARCHITECTURE STATUS</div>
        <h1>当前架构路线<br>与 GRIA 迁移图</h1>
        <div class="subtitle">从一个 SMILES 进入 Canonical V4 的真实路径，<br>到 Transformation Program 新架构的落地路线</div>
        <div class="truth">真实性口径：V4 主干可运行 · 创新层为过渡实现 · GRIA 核心尚未完整落地</div>
      </div>
      <div class="meta">状态基线 2026-07-16 · 代码、文档与运行语义交叉审计<br>AutoPlanner / Retrosynthesis Architecture Review</div>
    </section>
    """
    document = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
    <title>AutoPlanner 当前架构路线与 GRIA 迁移图</title><style>{css}</style></head>
    <body>{cover}<main>{body}<div class="report-note">本报告以仓库当前工作区为状态基线。设计文档中的实体名不代表生产实现；声明路线结构闭合由 route_closure 判定，反应、文献/条件、库存/采购与 process-ready 由各自 proof 和 acceptance 分轴判定。</div></main></body></html>"""
    OUT.mkdir(parents=True, exist_ok=True)
    HTML_OUT.write_text(document, encoding="utf-8")
    return HTML_OUT


def build_pdf(html_path: Path) -> Path:
    browsers = [
        Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
        Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
        Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
    ]
    browser = next((path for path in browsers if path.exists()), None)
    if browser is None:
        raise RuntimeError("Chrome or Edge is required for PDF rendering")
    command = [
        str(browser),
        "--headless",
        "--disable-gpu",
        "--no-sandbox",
        "--allow-file-access-from-files",
        "--no-pdf-header-footer",
        f"--print-to-pdf={PDF_OUT}",
        html_path.as_uri(),
    ]
    subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
    )
    if not PDF_OUT.exists() or PDF_OUT.stat().st_size < 100_000:
        raise RuntimeError("PDF rendering did not produce a valid output")
    return PDF_OUT


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-pdf", action="store_true")
    args = parser.parse_args()
    html_path = build_html()
    print(f"html={html_path}")
    if not args.skip_pdf:
        print(f"pdf={build_pdf(html_path)}")


if __name__ == "__main__":
    main()
