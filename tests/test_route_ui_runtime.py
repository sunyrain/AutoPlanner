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
    assert script.index("viewport.setPointerCapture(event.pointerId)") < script.index(
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
