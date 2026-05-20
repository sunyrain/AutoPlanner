"""Build a static statin showcase page for docs/GitHub Pages."""
from __future__ import annotations

import argparse
import html
import json
import shutil
from pathlib import Path
from typing import Any

from rdkit import Chem, RDLogger
from rdkit.Chem.Draw import rdMolDraw2D


RDLogger.DisableLog("rdApp.*")

DEFAULT_PACKAGE = Path("results/shared/statin_panel_20260520/report_package_all9_static_showcase3")
DEFAULT_SITE = Path("docs/statins")
DEFAULT_OVERVIEW = Path("docs/index.html")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create static statin showcase HTML and assets.")
    parser.add_argument("--package-dir", default=str(DEFAULT_PACKAGE))
    parser.add_argument("--site-dir", default=str(DEFAULT_SITE))
    parser.add_argument("--overview", default=str(DEFAULT_OVERVIEW))
    parser.add_argument("--page-title", default="他汀类逆合成路线")
    parser.add_argument(
        "--page-lead",
        default=(
            "静态汇报版。每个目标最多展示 3 条路线；已排除 reject 和 needs_chemist_review，"
            "优先选择长路线和可解释的 late-stage / semisynthesis / fragment 路线。"
            "箭头条件为 RCR 模型预测，仅作为路线审阅辅助。"
        ),
    )
    parser.add_argument(
        "--run-label",
        default="depth 20 / iterations 200 / top-k 100",
        help="Short run label shown in the footer.",
    )
    parser.add_argument("--overview-title", default="他汀类逆合成路线")
    parser.add_argument(
        "--overview-description",
        default="九个他汀目标，每个最多三条路线，连续 SVG 路线图，箭头标注预测条件。",
    )
    args = parser.parse_args()

    package_dir = Path(args.package_dir)
    site_dir = Path(args.site_dir)
    overview_path = Path(args.overview)
    route_doc_dir = package_dir / "route_docs"
    figure_root = package_dir / "figures"
    if not route_doc_dir.exists():
        raise FileNotFoundError(route_doc_dir)
    if not figure_root.exists():
        raise FileNotFoundError(figure_root)

    site_dir.mkdir(parents=True, exist_ok=True)
    asset_dir = site_dir / "assets"
    if asset_dir.exists():
        shutil.rmtree(asset_dir)
    asset_dir.mkdir(parents=True, exist_ok=True)

    targets = []
    for doc_path in sorted(route_doc_dir.glob("*_routes.json")):
        doc = json.loads(doc_path.read_text(encoding="utf-8"))
        safe = doc_path.stem.replace("_top3_routes", "")
        target_asset_dir = asset_dir / safe
        target_asset_dir.mkdir(parents=True, exist_ok=True)
        figure_dir = figure_root / doc_path.stem
        if not figure_dir.exists():
            raise FileNotFoundError(figure_dir)
        target_svg = target_asset_dir / "target.svg"
        target_svg.write_text(_mol_svg(str(doc.get("target_smiles") or "")), encoding="utf-8")

        routes = []
        for route in [row for row in doc.get("routes") or [] if isinstance(row, dict)]:
            rank = int(route.get("rank") or route.get("display_rank") or len(routes) + 1)
            svg_name = f"scheme_route_{rank:02d}.svg"
            pdf_name = f"scheme_route_{rank:02d}.pdf"
            svg_src = figure_dir / svg_name
            pdf_src = figure_dir / pdf_name
            if svg_src.exists():
                shutil.copy2(svg_src, target_asset_dir / svg_name)
            if pdf_src.exists():
                shutil.copy2(pdf_src, target_asset_dir / pdf_name)
            audit = route.get("product_audit") or {}
            metrics = route.get("metrics") or {}
            routes.append(
                {
                    "rank": rank,
                    "steps": int(route.get("n_steps") or len(route.get("steps") or [])),
                    "class": audit.get("route_class") or "unknown",
                    "risk": audit.get("risk_order"),
                    "score": route.get("score"),
                    "original_rank": route.get("original_rank"),
                    "coverage": metrics.get("condition_coverage"),
                    "terminal_max": (audit.get("terminal_profile") or {}).get("effective_max_terminal_heavy_atoms")
                    or metrics.get("max_terminal_heavy_atoms"),
                    "tags": list(audit.get("tags") or [])[:5],
                    "issues": list(audit.get("issues") or [])[:5],
                    "svg": f"assets/{safe}/{svg_name}",
                    "pdf": f"assets/{safe}/{pdf_name}" if (target_asset_dir / pdf_name).exists() else "",
                }
            )
        targets.append(
            {
                "name": doc.get("target_name") or safe,
                "safe": safe,
                "smiles": doc.get("target_smiles"),
                "target_svg": f"assets/{safe}/target.svg",
                "source_route_count": doc.get("source_route_count"),
                "web_kept_route_count": doc.get("web_kept_route_count"),
                "showcase_route_count": doc.get("showcase_route_count"),
                "candidate_count": len(routes),
                "routes": routes,
                "selection_policy": doc.get("selection_policy") or {},
            }
        )

    summary = _summary(targets)
    (site_dir / "index.html").write_text(
        _render_statin_page(
            targets,
            summary,
            page_title=args.page_title,
            page_lead=args.page_lead,
            run_label=args.run_label,
        ),
        encoding="utf-8",
    )
    (site_dir / "summary.json").write_text(json.dumps({"summary": summary, "targets": targets}, indent=2, ensure_ascii=False), encoding="utf-8")
    overview_path.parent.mkdir(parents=True, exist_ok=True)
    overview_path.write_text(
        _render_overview(
            summary,
            overview_title=args.overview_title,
            overview_description=args.overview_description,
        ),
        encoding="utf-8",
    )
    (overview_path.parent / ".nojekyll").write_text("", encoding="utf-8")
    print(json.dumps({"site": str(site_dir / "index.html"), "overview": str(overview_path), "summary": summary}, indent=2, ensure_ascii=False))


def _summary(targets: list[dict[str, Any]]) -> dict[str, Any]:
    routes = [route for target in targets for route in target["routes"]]
    return {
        "target_count": len(targets),
        "route_count": len(routes),
        "long_route_count": sum(1 for route in routes if int(route.get("steps") or 0) >= 8),
        "condition_coverage_routes": sum(1 for route in routes if route.get("coverage") == 1.0),
        "max_steps": max((int(route.get("steps") or 0) for route in routes), default=0),
        "min_steps": min((int(route.get("steps") or 0) for route in routes), default=0),
    }


def _mol_svg(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return '<svg xmlns="http://www.w3.org/2000/svg" width="360" height="220"></svg>'
    drawer = rdMolDraw2D.MolDraw2DSVG(360, 220)
    opts = drawer.drawOptions()
    opts.clearBackground = False
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    return drawer.GetDrawingText()


def _render_statin_page(
    targets: list[dict[str, Any]],
    summary: dict[str, Any],
    *,
    page_title: str,
    page_lead: str,
    run_label: str,
) -> str:
    nav = "\n".join(
        f'<a href="#{_e(target["safe"])}">{_e(target["name"])} <span>{len(target["routes"])}</span></a>'
        for target in targets
    )
    sections = "\n".join(_target_section(target) for target in targets)
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{_e(page_title)} | AutoPlanner Cascade</title>
  <style>
    :root {{
      --bg: #f6f7f9; --panel: #fff; --text: #17202f; --muted: #647084;
      --line: #d9dee7; --brand: #0f766e; --ink: #0f172a; --soft: #eef7f5;
      --warn: #9a5b00; --shadow: 0 18px 48px rgba(15, 23, 42, .08);
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; overflow-x: hidden; background: var(--bg); color: var(--text); font: 14px/1.55 Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    .wrap {{ width: min(1480px, calc(100vw - 42px)); margin: 0 auto; }}
    header {{ background: #fff; border-bottom: 1px solid var(--line); }}
    .hero {{ padding: 28px 0 22px; display: grid; grid-template-columns: minmax(0, 1fr) auto; gap: 24px; align-items: end; }}
    h1 {{ margin: 0 0 10px; font-size: 32px; line-height: 1.12; letter-spacing: 0; color: var(--ink); }}
    .lead {{ color: var(--muted); max-width: 980px; }}
    .metrics {{ display: flex; gap: 10px; flex-wrap: wrap; justify-content: flex-end; }}
    .metric {{ min-width: 112px; border: 1px solid var(--line); padding: 10px 12px; background: #fbfcff; }}
    .metric b {{ display: block; font-size: 24px; color: var(--brand); line-height: 1.05; }}
    .metric span {{ color: var(--muted); font-size: 12px; }}
    nav {{ display: flex; gap: 8px; flex-wrap: wrap; padding: 14px 0 18px; }}
    nav a {{ color: var(--text); text-decoration: none; background: #fff; border: 1px solid var(--line); padding: 8px 11px; font-weight: 700; }}
    nav span {{ color: var(--brand); margin-left: 5px; }}
    main {{ padding: 22px 0 54px; }}
    .target {{ margin-bottom: 28px; max-width: 100%; min-width: 0; background: var(--panel); border: 1px solid var(--line); box-shadow: var(--shadow); }}
    .targetHead {{ display: grid; grid-template-columns: 310px minmax(0, 1fr); gap: 22px; padding: 18px; border-bottom: 1px solid var(--line); }}
    .mol {{ height: 210px; display: grid; place-items: center; border: 1px solid var(--line); background: #fff; overflow: hidden; }}
    .mol img {{ width: 100%; height: 100%; object-fit: contain; }}
    h2 {{ margin: 0 0 8px; font-size: 24px; letter-spacing: 0; }}
    .policy {{ color: var(--muted); max-width: 920px; }}
    .badges {{ display: flex; flex-wrap: wrap; gap: 8px; margin-top: 12px; }}
    .badge {{ border: 1px solid var(--line); background: #f8fafc; padding: 4px 8px; font-size: 12px; font-weight: 750; color: #344054; }}
    .badge.ok {{ color: #047857; background: #ecfdf5; border-color: #bbf7d0; }}
    .badge.warn {{ color: var(--warn); background: #fff7ed; border-color: #fed7aa; }}
    .routes {{ display: grid; gap: 18px; min-width: 0; padding: 18px; }}
    .route {{ min-width: 0; max-width: 100%; border: 1px solid var(--line); background: #fcfdff; }}
    .routeHead {{ padding: 12px 14px; display: flex; justify-content: space-between; gap: 12px; align-items: flex-start; border-bottom: 1px solid var(--line); }}
    .routeTitle {{ font-size: 17px; font-weight: 850; color: var(--ink); }}
    .routeMeta {{ display: flex; gap: 7px; flex-wrap: wrap; margin-top: 7px; }}
    .routeLinks a {{ color: var(--brand); font-weight: 800; text-decoration: none; margin-left: 12px; white-space: nowrap; }}
    .svgBox {{ min-width: 0; max-width: 100%; padding: 12px; background: #fff; overflow: hidden; }}
    .svgBox img {{ display: block; width: 100%; max-width: 100%; height: auto; }}
    .note {{ margin: 14px 0 0; padding: 10px 12px; background: var(--soft); border-left: 3px solid var(--brand); color: #31534f; }}
    footer {{ color: var(--muted); padding: 22px 0 40px; }}
    @media (max-width: 900px) {{
      .wrap {{ width: min(100vw - 24px, 1480px); }}
      .hero, .targetHead {{ display: block; }}
      .metrics {{ justify-content: flex-start; margin-top: 14px; }}
      .routeHead {{ display: block; }}
      .routeLinks {{ margin-top: 8px; }}
      .routeLinks a {{ margin-left: 0; margin-right: 12px; }}
    }}
  </style>
</head>
<body>
  <header>
    <div class="wrap hero">
      <div>
        <h1>{_e(page_title)}</h1>
        <div class="lead">{_e(page_lead)}</div>
      </div>
      <div class="metrics">
        <div class="metric"><b>{summary["target_count"]}</b><span>目标分子</span></div>
        <div class="metric"><b>{summary["route_count"]}</b><span>路线</span></div>
        <div class="metric"><b>{summary["long_route_count"]}</b><span>≥8 步路线</span></div>
        <div class="metric"><b>{summary["max_steps"]}</b><span>最长步数</span></div>
      </div>
    </div>
    <div class="wrap">{nav}</div>
  </header>
  <main class="wrap">
    {sections}
  </main>
  <footer class="wrap">Generated from AutoPlanner Cascade statin panel, {_e(run_label)}. Presentation filter excludes needs_chemist_review by default.</footer>
</body>
</html>
"""


def _target_section(target: dict[str, Any]) -> str:
    routes = target["routes"]
    route_cards = "\n".join(_route_card(target, route) for route in routes)
    short_note = ""
    if routes and max(int(route["steps"]) for route in routes) < 5:
        short_note = '<div class="note">严格筛选后该目标只剩短半合成候选；页面不使用 needs_chemist_review 路线补足数量。</div>'
    if len(routes) < 3:
        short_note += f'<div class="note">该目标严格筛选后只有 {len(routes)} 条可展示路线。</div>'
    return f"""
    <section class="target" id="{_e(target["safe"])}">
      <div class="targetHead">
        <div class="mol"><img src="{_e(target["target_svg"])}" alt="{_e(target["name"])}"></div>
        <div>
          <h2>{_e(target["name"])}</h2>
          <div class="policy">来源：web product-audit 后的路线池；隐藏 needs_chemist_review；不使用 terminal 原子数硬阈值，避免误伤用于引入基团的大型片段。</div>
          <div class="badges">
            <span class="badge ok">{len(routes)} 条展示路线</span>
            <span class="badge">原始路线 {target.get("source_route_count")}</span>
            <span class="badge">web 保留 {target.get("web_kept_route_count")}</span>
            <span class="badge">展示池 {target.get("showcase_route_count")}</span>
          </div>
          {short_note}
        </div>
      </div>
      <div class="routes">{route_cards}</div>
    </section>
    """


def _route_card(target: dict[str, Any], route: dict[str, Any]) -> str:
    tags = "".join(f'<span class="badge">{_e(tag)}</span>' for tag in route.get("tags") or [])
    issues = "".join(f'<span class="badge warn">{_e(issue)}</span>' for issue in route.get("issues") or [])
    coverage = route.get("coverage")
    coverage_text = "-" if coverage is None else f"{float(coverage) * 100:.0f}%"
    pdf = f'<a href="{_e(route["pdf"])}">PDF</a>' if route.get("pdf") else ""
    return f"""
      <article class="route">
        <div class="routeHead">
          <div>
            <div class="routeTitle">Route {route["rank"]} · {route["steps"]} steps · {_label(route["class"])}</div>
            <div class="routeMeta">
              <span class="badge ok">condition coverage {coverage_text}</span>
              <span class="badge">risk {route.get("risk")}</span>
              <span class="badge">original rank {route.get("original_rank")}</span>
              <span class="badge">terminal max heavy {route.get("terminal_max")}</span>
              {tags}{issues}
            </div>
          </div>
          <div class="routeLinks"><a href="{_e(route["svg"])}">SVG</a>{pdf}</div>
        </div>
        <div class="svgBox"><img src="{_e(route["svg"])}" alt="{_e(target["name"])} route {route["rank"]}"></div>
      </article>
    """


def _render_overview(
    summary: dict[str, Any],
    *,
    overview_title: str,
    overview_description: str,
) -> str:
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>AutoPlanner Cascade 展示总览</title>
  <style>
    body {{ margin: 0; background: #f6f7f9; color: #17202f; font: 15px/1.55 Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    .wrap {{ width: min(1120px, calc(100vw - 40px)); margin: 0 auto; padding: 42px 0; }}
    h1 {{ margin: 0 0 10px; font-size: 34px; letter-spacing: 0; }}
    .lead {{ color: #647084; max-width: 820px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 16px; margin-top: 24px; }}
    .card {{ display: block; padding: 18px; background: #fff; border: 1px solid #d9dee7; color: inherit; text-decoration: none; box-shadow: 0 18px 48px rgba(15,23,42,.08); }}
    .card h2 {{ margin: 0 0 8px; font-size: 22px; }}
    .card p {{ margin: 0; color: #647084; }}
    .metric {{ margin-top: 14px; color: #0f766e; font-weight: 850; }}
  </style>
</head>
<body>
  <main class="wrap">
    <h1>AutoPlanner Cascade 展示总览</h1>
    <div class="lead">静态汇报入口。当前重点展示他汀类分子的逆合成路线图，适合直接给专家浏览。</div>
    <div class="grid">
      <a class="card" href="statins/">
        <h2>{_e(overview_title)}</h2>
        <p>{_e(overview_description)}</p>
        <div class="metric">{summary["target_count"]} targets · {summary["route_count"]} routes · longest {summary["max_steps"]} steps</div>
      </a>
    </div>
  </main>
</body>
</html>
"""


def _label(route_class: str) -> str:
    return {
        "triage_semisynthesis": "semisynthesis",
        "triage_late_stage": "late-stage",
        "triage_fragment": "fragment assembly",
    }.get(str(route_class), str(route_class))


def _e(value: Any) -> str:
    return html.escape(str(value or ""), quote=True)


if __name__ == "__main__":
    main()
