import hashlib
import html
import json
from pathlib import Path
import re
import subprocess

import pytest

from cascade_planner.harness.v4_route_workbench import render_v4_route_workbench_html


SCRIPT = Path("cascade_planner/harness/route_forest_ui/script.js")
STYLES = Path("cascade_planner/harness/route_forest_ui/styles.css")


def test_camera_uses_one_world_transform_and_never_moves_svg_layer() -> None:
    script = SCRIPT.read_text(encoding="utf-8")

    assert "translate(${state.panX} ${state.panY}) scale(${state.zoom})" in script
    assert "world.setAttribute('transform', cameraTransform)" in script
    assert "applyPanTransform" not in script
    assert "translate3d" not in script
    assert "svg.style.transform =" not in script
    assert "viewport.setPointerCapture(event.pointerId)" in script
    assert script.index("viewport.setPointerCapture(event.pointerId)") > script.index(
        "Math.hypot(deltaX, deltaY)"
    )


def test_pointer_motion_is_latest_value_raf_batched_without_layout_render() -> None:
    script = SCRIPT.read_text(encoding="utf-8")
    move = script.split("window.addEventListener('pointermove'", 1)[1].split(
        "window.addEventListener('pointerup'", 1
    )[0]
    camera_frame = script.split("function commitPendingPanFrame", 1)[1].split(
        "function bindViewportEvents", 1
    )[0]

    assert "panSession.latestX = event.clientX" in move
    assert "panSession.latestY = event.clientY" in move
    assert "requestAnimationFrame(commitPendingPanFrame)" in move
    assert "applyViewportTransform({ updateMinimap: false })" in camera_frame
    assert "renderGraph(" not in move + camera_frame
    assert "buildGraphModel(" not in move + camera_frame


def test_large_graph_runtime_has_bounded_portfolio_lod_culling_and_probe() -> None:
    script = SCRIPT.read_text(encoding="utf-8")
    styles = STYLES.read_text(encoding="utf-8")

    assert "MAX_PORTFOLIO_ROUTES = 5" in script
    assert "portfolioDisplayLimit()" in script
    assert "autoplanner.route-forest-ui.v4" in script
    assert "CULLING_OBJECT_THRESHOLD" in script
    assert "updateViewportCulling" in script
    assert "is-canvas-culled" in styles
    assert "depictionCache" in script
    assert "graphModelCache" in script
    assert "data-zoom-band" not in script  # state is written through dataset, not markup.
    assert "viewport.dataset.zoomBand" in script
    assert "__AUTOPLANNER_ROUTE_PERF__" in script
    for metric in (
        "cameraFrames",
        "droppedFrames",
        "maximumFrameDelayMs",
        "maximumCameraFrameMs",
        "meanCameraFrameMs",
        "lastGraphUpdateMs",
        "renderedObjects",
        "culledObjects",
        "memoryBytes",
    ):
        assert metric in script


def test_camera_transform_is_the_only_composited_graph_layer() -> None:
    styles = STYLES.read_text(encoding="utf-8")
    graph_svg = styles.split(".graph-svg {", 1)[1].split("}", 1)[0]
    graph_world = styles.split(".graph-world {", 1)[1].split("}", 1)[0]

    assert "will-change" not in graph_svg
    assert "transform" not in graph_svg
    assert "will-change: transform" in graph_world


def test_long_current_routes_open_fully_fitted_instead_of_clipped() -> None:
    script = SCRIPT.read_text(encoding="utf-8")

    assert "function preferReadableFocus()" in script
    assert "return (lane.step_ids || []).length <= 4" in script
    assert "fitGraph({ readable: preferReadableFocus() })" in script
    assert "serpentine_long_route.v1" in script
    assert "maximumLayer >= 16" in script
    assert "Math.min(9, maximumLayer + 1)" in script
    assert "serpentineRowTurn" in script


def test_current_route_edges_use_fixed_ports_and_layer_channels() -> None:
    script = SCRIPT.read_text(encoding="utf-8")

    assert "state.mode === 'current' || state.edgeStyle !== 'trust'" in script
    assert "fixed-port-channels.v2" in script
    assert "data-edge-routing=" in script
    assert "const rowSeparated =" in script
    assert "packing === 'serpentine_long_route.v1' && rowSeparated" in script
    assert "`M ${x1} ${y1} V ${channelY} H ${x2} V ${y2}`" in script
    assert "`M ${x1} ${y1} H ${middle} V ${y2} H ${x2}`" in script


def test_route_arrows_and_responsive_camera_do_not_scale_with_edge_emphasis() -> None:
    script = SCRIPT.read_text(encoding="utf-8")

    assert 'markerUnits="userSpaceOnUse"' in script
    assert 'markerUnits="strokeWidth"' not in script
    assert "cameraMode: 'fit'" in script
    assert "function resizeGraphViewport()" in script
    assert "svg.setAttribute('viewBox'" in script
    assert "new ResizeObserver(handleViewportResize)" in script
    assert "fitGraph({ readable: preferReadableFocus(), remember: false })" in script
    assert "fitGraph({ readable: state.mode === 'current' })" not in script


def test_mixed_proof_route_labels_use_edge_distribution_not_weakest_only() -> None:
    script = SCRIPT.read_text(encoding="utf-8")

    assert "L1_source_reported: 'L1 文献报道'" in script
    assert "function routeProofMixLabel(lane)" in script
    assert "is-mixed-proof" in script
    assert "`${planner} 步 L0 规划`" in script
    assert "`${reported} 步 L1 文献`" in script
    assert "l1_source_reported_edges" in script
    assert "source_observation_records" in script
    assert "证据缺口与补证动作" in script
    assert "unexplained_element_gains" in script


def test_reaction_nodes_expose_condition_source_without_opening_inspector() -> None:
    script = SCRIPT.read_text(encoding="utf-8")
    styles = STYLES.read_text(encoding="utf-8")

    assert "function inlineConditionText(step)" in script
    assert "来源条件已绑定 · 字段待展开" in script
    assert "预测条件 · 非文献事实" in script
    assert "条件待取证" in script
    assert "normalizedConditionRows(step)" in script
    assert "source_observation_records" in script
    assert 'class="reaction-hit-target"' in script
    hit_target_index = script.index('class="reaction-hit-target"')
    assert hit_target_index < script.index('class="node-surface"', hit_target_index)
    assert "state.detailTab = 'step'" in script
    assert "if (state.layoutPreset === 'focus') state.layoutPreset = 'review'" in script
    assert 'class="reaction-condition-meta ${conditionClass}"' in script
    assert ".reaction-condition-meta.is-source-exact" in styles
    assert ".reaction-condition-meta.is-model-predicted" in styles
    assert ".reaction-condition-meta.is-missing" in styles
    assert ".graph-node--reaction .reaction-hit-target" in styles
    assert "fill: transparent !important" in styles


def test_reaction_inspector_groups_full_conditions_and_source_procedure_text() -> None:
    script = SCRIPT.read_text(encoding="utf-8")
    styles = STYLES.read_text(encoding="utf-8")

    assert "function conditionGroupHtml" in script
    assert "核心反应条件" in script
    assert "加料、后处理与纯化" in script
    assert "source-procedure-observation" in script
    assert "procedure_excerpt" in script
    assert "来源观察" in script
    assert ".condition-group" in styles
    assert ".source-procedure-observation" in styles
    assert "white-space: pre-wrap" in styles


def test_workbench_exposes_six_axis_proof_and_product_stage_filters() -> None:
    script = SCRIPT.read_text(encoding="utf-8")
    styles = STYLES.read_text(encoding="utf-8")
    template = Path("cascade_planner/harness/route_forest_ui/template.html").read_text(
        encoding="utf-8"
    )

    assert "function proofVectorHtml(value)" in script
    assert "科学 Proof vector" in script
    assert "路线 Proof vector" in script
    assert "来源过程" in script
    assert "失效事实" in script
    assert "路线已按当前权威降级" in script
    assert "哈希绑定过程" in script
    assert "condition_missing_required_groups" in script
    assert ".proof-vector-grid" in styles
    assert "grid-template-columns: repeat(2, minmax(0, 1fr))" in styles
    for stage in ("literature", "conditions", "procurement", "process"):
        assert f'data-stage-filter="{stage}"' in template


def test_headless_browser_drag_zoom_fit_selection_minimap_and_large_graph(
    tmp_path: Path,
) -> None:
    chrome = next(
        (
            path
            for path in (
                Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
                Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
            )
            if path.is_file()
        ),
        None,
    )
    if chrome is None:
        pytest.skip("Chromium browser unavailable for route UI interaction regression")

    workbench = _large_workbench(edge_count=70)
    page = tmp_path / "route-workbench.html"
    browser_profile = tmp_path / "chromium-profile"
    page.write_text(render_v4_route_workbench_html(workbench), encoding="utf-8")
    result = subprocess.run(
        [
            str(chrome),
            "--headless=new",
            "--disable-gpu",
            "--no-sandbox",
            "--no-first-run",
            "--disable-background-networking",
            "--disable-extensions",
            "--disable-sync",
            f"--user-data-dir={browser_profile}",
            "--run-all-compositor-stages-before-draw",
            "--virtual-time-budget=4000",
            "--dump-dom",
            f"{page.as_uri()}?route_ui_selftest=1",
        ],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=30,
    )
    match = re.search(
        r'<pre[^>]*id="routeUiSelfTest"[^>]*>(.*?)</pre>',
        result.stdout,
        re.DOTALL,
    )
    assert match, result.stderr[-2000:]
    report = json.loads(html.unescape(match.group(1)))
    assert report["status"] == "passed", report
    assert report["checks"] == {
        "drag": True,
        "singleWorldTransform": True,
        "zoomAnchor": True,
        "fit": True,
        "selection": True,
        "reactionHitTarget": True,
        "reactionInspector": True,
        "fullConditionGroups": True,
        "sourceProcedure": True,
        "minimap": True,
        "largeGraphCulling": True,
    }
    assert report["performance"]["renderedObjects"] > 120
    assert report["performance"]["cameraFrames"] >= 1


def _large_workbench(*, edge_count: int) -> dict:
    molecules = {
        f"m:{index}": {
            "molecule_id": f"m:{index}",
            "canonical_smiles": "C" * (index % 8 + 1),
            "role": "target" if index == 0 else "stock_leaf" if index == edge_count else "intermediate",
            "is_leaf": index == edge_count,
            "stock_closed": False,
            "stock_observation_id": "",
            "badges": [],
        }
        for index in range(edge_count + 1)
    }
    edges = {
        f"edge:{index}": {
            "edge_id": f"edge:{index}",
            "product_molecule_id": f"m:{index}",
            "precursor_molecule_ids": [f"m:{index + 1}"],
            "proof_level": 2,
            "proof_name": "L2_reaction_validated",
            "proof_color": "#3b82f6",
            "accepted": True,
            "origin_kinds": ["codex_global_director"],
            "source_kinds": [],
            "badges": ["reaction-validated"],
        }
        for index in range(edge_count)
    }
    route_id = "route:large"
    route = {
        "route_id": route_id,
        "route_family_id": "family:large",
        "strategy": "Large deterministic UI benchmark",
        "stage": "reaction_validated",
        "proof_level": 2,
        "proof_name": "L2_reaction_validated",
        "proof_color": "#3b82f6",
        "edge_ids": list(edges),
        "leaf_molecule_ids": [f"m:{edge_count}"],
        "root_edge_ids": ["edge:0"],
        "module_selections": {},
        "complete": False,
        "stock_closure_rate": 0.0,
        "independent_source_groups": [],
        "risk_score": 0.4,
        "convergence_score": 0.0,
        "deficit_count": 1,
        "badges": ["reaction-validated"],
    }
    payload = {
        "schema_version": "retrosynthesis_route_workbench.v1",
        "run_id": "large-ui-regression",
        "target": {"molecule_id": "m:0", "name": "large UI", "canonical_smiles": "C"},
        "revision": {
            "graph": 1,
            "evidence": 1,
            "graph_scientific_sha256": "fixture",
            "portfolio_sha256": "fixture",
        },
        "portfolio": {
            "route_ids": [route_id],
            "default_route_id": route_id,
            "route_count": 1,
            "accepted": False,
            "closeout": {"decision": "unresolved"},
            "metrics": {},
            "display_limit": 5,
        },
        "views": {
            "hypotheses": {"route_ids": [], "hypothesis_ids": [], "count": 0},
            "expanded": {"route_ids": [route_id], "count": 1},
            "reaction_validated": {"route_ids": [route_id], "count": 1},
            "stock_closed": {"route_ids": [], "count": 0},
        },
        "routes": {route_id: route},
        "molecules": molecules,
        "edges": edges,
        "hypotheses": {},
        "modules": {},
        "shared_intermediates": {},
        "layout": {"algorithm": "fixture", "nodes": [], "stable_ids": True},
        "inspectors": {
            "routes": {route_id: {}},
            "edges": {
                edge_id: {
                    "proof": {"reaction_validated": True},
                    "reaction_proofs": [],
                    "sources": [],
                    "exact_records": [],
                    "conflicts": [],
                    "provenance": [],
                    "rejection_reasons": [],
                }
                for edge_id in edges
            },
            "molecules": {molecule_id: {} for molecule_id in molecules},
            "rejections": [],
            "conflicts": {},
        },
        "proof_visuals": {},
        "semantics": {"read_model_only": True},
    }
    payload["content_sha256"] = hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    return payload
