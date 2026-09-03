"""Generate deterministic SVG diagrams for each architecture-evolution stage."""

from __future__ import annotations

from html import escape
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "assets" / "architecture-evolution" / "stages"

NAVY = "#0b2438"
INK = "#17384d"
MUTED = "#617b8c"
PAPER = "#f4f8fa"
WHITE = "#ffffff"
CYAN = "#22a9c1"
TEAL = "#2bb89b"
AMBER = "#e7a83d"
CORAL = "#e87468"
BLUE = "#5a8dee"
PURPLE = "#7a78d9"
RED = "#d85d6b"
GREEN = "#35a873"


class SVG:
    def __init__(self, title: str, subtitle: str):
        self.parts = [
            f'''<svg xmlns="http://www.w3.org/2000/svg" width="1400" height="700" viewBox="0 0 1400 700" role="img" aria-label="{escape(title)}">
<defs>
  <marker id="arrow" markerWidth="10" markerHeight="8" refX="9" refY="4" orient="auto"><path d="M0,0 L10,4 L0,8 z" fill="#6b8798"/></marker>
  <filter id="shadow" x="-10%" y="-10%" width="120%" height="130%"><feDropShadow dx="0" dy="4" stdDeviation="6" flood-color="#17384d" flood-opacity=".12"/></filter>
</defs>
<rect width="1400" height="700" rx="28" fill="{PAPER}"/>
<text x="55" y="64" font-family="Microsoft YaHei, Noto Sans CJK SC, sans-serif" font-size="30" font-weight="700" fill="{NAVY}">{escape(title)}</text>
<text x="55" y="99" font-family="Microsoft YaHei, Noto Sans CJK SC, sans-serif" font-size="16" fill="{MUTED}">{escape(subtitle)}</text>'''
        ]

    def text(self, x, y, lines, size=20, color=INK, weight=400, anchor="middle", line_h=28):
        if isinstance(lines, str):
            lines = [lines]
        spans = []
        for i, line in enumerate(lines):
            dy = 0 if i == 0 else line_h
            spans.append(f'<tspan x="{x}" dy="{dy}">{escape(line)}</tspan>')
        self.parts.append(
            f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-family="Microsoft YaHei, Noto Sans CJK SC, sans-serif" font-size="{size}" font-weight="{weight}" fill="{color}">{"".join(spans)}</text>'
        )

    def box(self, x, y, w, h, title, body=(), accent=CYAN, fill=WHITE, dashed=False):
        dash = ' stroke-dasharray="8 7"' if dashed else ""
        self.parts.append(
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="18" fill="{fill}" stroke="{accent}" stroke-width="2"{dash} filter="url(#shadow)"/>'
        )
        self.parts.append(f'<rect x="{x}" y="{y}" width="9" height="{h}" rx="4" fill="{accent}"/>')
        self.text(x + w / 2 + 4, y + 38, title, 21, NAVY, 700)
        if body:
            self.text(x + w / 2 + 4, y + 72, list(body), 15, MUTED, 400, line_h=23)

    def pill(self, x, y, w, text, color=CYAN):
        self.parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="34" rx="17" fill="{color}"/>')
        self.text(x + w / 2, y + 23, text, 14, WHITE, 700)

    def arrow(self, x1, y1, x2, y2, dashed=False, color="#6b8798", width=3):
        dash = ' stroke-dasharray="9 7"' if dashed else ""
        self.parts.append(
            f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{width}"{dash} marker-end="url(#arrow)"/>'
        )

    def line(self, x1, y1, x2, y2, color="#6b8798", width=2, dashed=False):
        dash = ' stroke-dasharray="8 6"' if dashed else ""
        self.parts.append(
            f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{width}"{dash}/>'
        )

    def node(self, x, y, r=18, color=CYAN, label=None):
        self.parts.append(
            f'<circle cx="{x}" cy="{y}" r="{r}" fill="{color}" stroke="{WHITE}" stroke-width="4" filter="url(#shadow)"/>'
        )
        if label:
            self.text(x, y + r + 27, label, 13, MUTED, 600)

    def band(self, x, y, w, h, title, color=NAVY):
        self.parts.append(
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="16" fill="{color}"/>'
        )
        self.text(x + w / 2, y + h / 2 + 7, title, 18, WHITE, 700)

    def finish(self, filename: str):
        self.parts.append("</svg>")
        (OUT / filename).write_text("\n".join(self.parts), encoding="utf-8")


def stage1():
    s = SVG("阶段 I｜模型流水线 + 搜索器", "直觉：先把局部预测做准，再用搜索把局部能力串成路线")
    boxes = [
        (55, "数据与清洗", ("rxn_smiles", "EC / 条件"), BLUE),
        (320, "单步扩展", ("化学模型", "酶催化模板"), CORAL),
        (585, "条件与酶", ("T / pH", "enzyme rank"), AMBER),
        (850, "多步搜索", ("MCTS", "two-stage"), TEAL),
        (1115, "评估", ("Top-k / GT@5", "solve rate"), CYAN),
    ]
    for i, (x, t, b, c) in enumerate(boxes):
        s.box(x, 170, 220, 140, t, b, c)
        if i < len(boxes) - 1:
            s.arrow(x + 220, 240, x + 255, 240)
    s.band(75, 380, 370, 72, "局部指标是主要反馈", NAVY)
    s.band(515, 380, 370, 72, "搜索闭合近似路线完成", NAVY)
    s.band(955, 380, 370, 72, "组件各持一部分事实", NAVY)
    s.arrow(260, 380, 180, 316, dashed=True, color=BLUE)
    s.arrow(700, 380, 960, 316, dashed=True, color=TEAL)
    s.arrow(1140, 380, 1220, 316, dashed=True, color=CYAN)
    s.box(
        260,
        515,
        880,
        105,
        "迁移压力",
        ("单步更强 ≠ 多步更合理；条件、库存、来源和拓扑缺少统一事实模型",),
        RED,
        fill="#fff8f7",
    )
    s.finish("stage-1-model-pipeline.svg")


def stage2():
    s = SVG(
        "阶段 II｜CascadeBoard 全局路线补全",
        "直觉：让系统先看见整条路线，再填具体反应；冻结专家只提供候选或能量",
    )
    s.box(70, 160, 340, 120, "Layer 1 · Skeleton", ("全局反应类型 / EC / 条件骨架",), BLUE)
    s.box(
        530, 160, 340, 120, "Layer 2 · Molecular Fill", ("RetroChimera / EnzExpand 填分子",), AMBER
    )
    s.box(990, 160, 340, 120, "Layer 3 · Route Scoring", ("完整路线排序与兼容性",), TEAL)
    s.arrow(410, 220, 530, 220)
    s.arrow(870, 220, 990, 220)
    s.text(145, 360, "线性 Slot Chain", 22, NAVY, 700, anchor="start")
    for i, (label, c) in enumerate(
        [("目标", CORAL), ("Slot 1", BLUE), ("Slot 2", AMBER), ("Slot 3", TEAL), ("起始料", CYAN)]
    ):
        x = 175 + i * 215
        s.node(x, 440, 27, c, label)
        if i < 4:
            s.arrow(x + 30, 440, x + 180, 440, color="#8299a8")
    s.box(
        880,
        360,
        430,
        170,
        "隐含假设",
        ("主路线接近线性", "路线质量可由单一 scorer 排序", "候选空间覆盖了正确路线"),
        PURPLE,
        fill="#fbfaff",
    )
    s.box(
        250,
        560,
        900,
        90,
        "迁移压力",
        ("会聚 AND 语义、共享中间体、证据版本和开放研究无法自然装进 slot chain",),
        RED,
        fill="#fff8f7",
    )
    s.finish("stage-2-cascadeboard.svg")


def stage3():
    s = SVG(
        "阶段 III｜LLM 增强 Route Tree → Codex-entry Harness",
        "直觉：复杂目标的关键不是再跑一次搜索，而是先选择正确的工作流",
    )
    s.box(55, 165, 290, 120, "Target + Preflight", ("结构复杂度", "已知来源 / 约束"), BLUE)
    s.box(
        420,
        145,
        350,
        160,
        "Codex Controller",
        ("选择 ChemEnzy-first", "literature-first / hybrid"),
        CYAN,
    )
    s.box(845, 150, 230, 120, "ChemEnzy", ("局部候选", "无 solved 权威"), AMBER)
    s.box(1115, 150, 230, 120, "Literature", ("来源发现", "类比 / exact"), TEAL)
    s.arrow(345, 225, 420, 225)
    s.arrow(770, 205, 845, 205)
    s.arrow(770, 245, 1115, 245)
    s.box(
        420,
        385,
        350,
        145,
        "Deterministic Validators",
        ("结构 / 路线 / 库存 / 证据", "solved / partial / rejected"),
        GREEN,
    )
    s.arrow(960, 270, 735, 385)
    s.arrow(1230, 270, 760, 405)
    s.box(
        55,
        405,
        290,
        105,
        "旧直觉失效",
        ("候选多、planner 强", "不代表 parent route 闭合"),
        RED,
        fill="#fff8f7",
    )
    s.box(
        845,
        400,
        500,
        115,
        "迁移压力",
        ("固定 harness 难以支持多轮研究、失败转向与持续共享状态",),
        PURPLE,
        fill="#fbfaff",
    )
    s.pill(555, 565, 180, "编排权 ≠ 裁决权", CYAN)
    s.finish("stage-3-codex-harness.svg")


def stage4():
    s = SVG(
        "阶段 IV｜Agentic Blackboard",
        "直觉：让多个能力围绕一块持久共享状态协作，并把每个动作变成可审计事件",
    )
    s.box(
        500,
        210,
        400,
        210,
        "Agentic Blackboard",
        ("typed summaries", "artifact refs / deficits", "round budget / action history"),
        PURPLE,
        fill="#fbfaff",
    )
    satellites = [
        (70, 155, "Codex Action Planner", CYAN),
        (70, 420, "Failure Critic", CORAL),
        (1030, 155, "Literature / Vision", TEAL),
        (1030, 420, "ChemEnzy / Tools", AMBER),
    ]
    for x, y, t, c in satellites:
        s.box(x, y, 300, 105, t, (), c)
    s.arrow(370, 205, 500, 260)
    s.arrow(370, 470, 500, 375)
    s.arrow(1030, 205, 900, 260)
    s.arrow(1030, 470, 900, 375)
    s.arrow(700, 420, 700, 540, color=GREEN)
    s.box(520, 540, 360, 85, "Deterministic Parent Proof", ("唯一 final verdict 路径",), GREEN)
    s.box(85, 570, 350, 72, "贡献：开放研究可持续、可预算、可审计", (), BLUE, fill="#f5f9ff")
    s.box(965, 570, 350, 72, "缺陷：协作状态与化学真相可能漂移", (), RED, fill="#fff8f7")
    s.finish("stage-4-agentic-blackboard.svg")


def stage5():
    s = SVG(
        "阶段 V｜Evidence-first + Reaction Hypergraph V2",
        "直觉：允许广泛生成，但必须把提议、证据、证明和发布拆成不同权限平面",
    )
    planes = [
        (55, "Proposal", "断键 / 条件 / 替代", CORAL),
        (360, "Evidence", "来源 / 文档 / 页面", AMBER),
        (665, "Proof", "结构 / mapping / stock", TEAL),
        (970, "Publication", "一致 revision", CYAN),
    ]
    for i, (x, t, b, c) in enumerate(planes):
        s.box(x, 145, 260, 115, t, (b,), c)
        if i < 3:
            s.arrow(x + 260, 202, x + 300, 202)
    s.text(55, 340, "按 canonical product 组织的 AND/OR 超图", 21, NAVY, 700, anchor="start")
    coords = [
        (130, 440, CORAL, "P"),
        (330, 390, BLUE, "A"),
        (330, 500, AMBER, "B"),
        (570, 440, TEAL, "R1"),
        (790, 360, PURPLE, "C"),
        (790, 520, BLUE, "D"),
        (1030, 440, CYAN, "R2"),
        (1250, 440, GREEN, "T"),
    ]
    for x, y, color, label in coords:
        s.node(x, y, 22, color, label)
    for a, b in [
        ((153, 440), (307, 398)),
        ((153, 440), (307, 492)),
        ((353, 398), (547, 440)),
        ((353, 492), (547, 440)),
        ((593, 440), (768, 368)),
        ((593, 440), (768, 512)),
        ((813, 368), (1007, 440)),
        ((813, 512), (1007, 440)),
        ((1053, 440), (1227, 440)),
    ]:
        s.line(*a, *b, "#7893a3", 2)
    s.box(
        200,
        575,
        1000,
        78,
        "迁移压力",
        ("原则已经统一，但新超图、旧 Blackboard、RouteForest 与多个 queue 仍可能多写同一事实",),
        RED,
        fill="#fff8f7",
    )
    s.finish("stage-5-evidence-hypergraph.svg")


def stage6():
    s = SVG(
        "阶段 VI｜Canonical V4 单一权威链",
        "直觉：不是继续增加检查器，而是让每类事实只有一个写入权威",
    )
    s.box(
        475,
        135,
        450,
        100,
        "GlobalCampaignDirector",
        ("一次读取全局上下文；输出 proposal only",),
        CYAN,
    )
    s.arrow(700, 235, 700, 295, color=CYAN)
    blocks = [
        (55, "RunKernel", ("事件 / 恢复 / 预算",), BLUE),
        (385, "Canonical Hypergraph", ("分子 / 超边 / 来源 / 库存",), TEAL),
        (715, "DeficitFrontier", ("唯一待办与调度",), AMBER),
        (1045, "Proof Portfolio", ("最弱证明 / 多样性 / 验收",), CYAN),
    ]
    for x, t, b, c in blocks:
        s.box(x, 310, 300, 135, t, b, c)
    for x in (355, 685, 1015):
        s.arrow(x, 378, x + 30, 378)
    s.band(
        160,
        510,
        1080,
        70,
        "Host authority path：admission → materialize → validate → exact evidence → stock",
        NAVY,
    )
    s.arrow(700, 445, 700, 510, color=GREEN)
    s.box(
        240,
        615,
        920,
        55,
        "CLI / API / Web / 恢复 / 导出只读同一 canonical state",
        (),
        GREEN,
        fill="#f4fbf8",
    )
    s.finish("stage-6-canonical-v4.svg")


def stage7():
    s = SVG(
        "阶段 VII｜Target-only、Blind 与来源证据闭环",
        "直觉：让陌生目标进入同一权威链，同时用有界降级控制成本与证据强度",
    )
    chain = [
        (45, "Target-only", ("陌生 SMILES",), BLUE),
        (300, "Global plan", ("路线族 / frontier",), CYAN),
        (555, "Host validation", ("物化 / mapping",), TEAL),
        (810, "Exact evidence", ("HTML → PDF → OCR → vision",), AMBER),
        (1065, "Portfolio", ("proof / stock / deficits",), GREEN),
    ]
    for i, (x, t, b, c) in enumerate(chain):
        s.box(x, 145, 220, 120, t, b, c)
        if i < 4:
            s.arrow(x + 220, 205, x + 255, 205)
    s.text(120, 345, "来源不可逆降级链", 21, NAVY, 700, anchor="start")
    source = [
        ("官方 HTML/XML", GREEN),
        ("PDF 原生文本", TEAL),
        ("本地 OCR", AMBER),
        ("视觉 L0", CORAL),
    ]
    for i, (t, c) in enumerate(source):
        x = 120 + i * 290
        s.pill(x, 390, 210, t, c)
        if i < 3:
            s.arrow(x + 210, 407, x + 280, 407)
    s.box(
        95,
        500,
        390,
        110,
        "Self-evolution",
        ("exact row + accepted proof", "模板复用仍从 L0 重验"),
        PURPLE,
        fill="#fbfaff",
    )
    s.box(
        505,
        500,
        390,
        110,
        "Blind contract",
        ("冻结知识 / 库存 / 预算", "失败案例必须保留"),
        BLUE,
        fill="#f5f9ff",
    )
    s.box(
        915,
        500,
        390,
        110,
        "诚实输出",
        ("unresolved / budget_exhausted", "部分条件不伪装成完整"),
        CORAL,
        fill="#fff8f7",
    )
    s.finish("stage-7-target-evidence-loop.svg")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for builder in (stage1, stage2, stage3, stage4, stage5, stage6, stage7):
        builder()
    print(f"generated=7 output={OUT}")


if __name__ == "__main__":
    main()
