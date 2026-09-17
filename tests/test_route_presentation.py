"""Display grouping must preserve chemistry, count boundaries and replay access."""
from __future__ import annotations

from pathlib import Path
import subprocess

import pytest

from cascade_planner.web.route_display import STATIC_DIR


def test_reaction_details_keep_long_conditions_in_readable_sections() -> None:
    script = r"""
const assert = require('node:assert/strict');
const {reactionDetailHtml} = require(process.argv[1]);
const conditions = [
  'Use isolated acetoxy aldehyde free of upstream oxidant; add a low loading of potassium carbonate to its methanol solution at 0–10 °C and monitor acetate cleavage, avoiding prolonged basic exposure.',
  'Stereochemistry is inherited from the monoacetate precursor: cleavage at the acetyl group leaves the C3 bonds unchanged. Screen whether deacetylation outpaces aldehyde degradation or rearrangement while preserving the terminal alkyne and tertiary alcohol.',
  'At completion, promptly neutralize with dilute acetic acid under cooling, remove salts, and exchange out methanol under mild conditions; avoid strongly acidic workup and prolonged exposure.\nConfirm the exact product before advancing.'
];
const step = {conditions, catalyst:'K2CO3', precursor_smiles:['CC(=O)OCC=O'], product_smiles:'OCC=O',
  reaction_family:'Deacetylation', transformation_rationale:'Release the alcohol without altering the carbon skeleton.',
  critic_verdict:'uncertain', critic_reasons:['Exact substrate selectivity remains unverified.'],
  critic_condition_assessment:'Proposed conditions; not an experimentally established protocol.',
  critic_suggested_revision:'Check aldehyde stability.', builder_limitations:['No exact-substrate evidence.']};
const saved = JSON.stringify(step);
const html = reactionDetailHtml(step, {origin:'大模型 Builder',status:'host_materialized',
  relationLabel:'准备步骤',verdictLabel:'存在不确定性'});
assert.equal(JSON.stringify(step),saved);
assert.ok(html.includes('reactionDetailMeta'));
assert.ok(html.includes('reactionDetailColumns hasReview'));
assert.ok(html.includes('reactionConditions" open'));
assert.ok(html.includes('<ul class="reactionConditionList">'));
for (const condition of conditions) assert.ok(html.includes(`<li>${condition}</li>`));
assert.ok(!html.includes(conditions.join(' · ')));
assert.ok(!html.includes('<b>Use isolated'));
assert.ok(html.includes('催化体系</span>K2CO3'));
assert.ok(html.includes('Critic 依据'));
assert.ok(html.includes('条件判断'));
assert.ok(html.includes('建议修订'));
assert.ok(html.includes('Builder 限制'));
assert.ok(html.includes('结构与 SMILES'));
assert.ok(!html.includes('<details class="reactionTechnical" open'));
const short = reactionDetailHtml({conditions:['Water'], catalyst:'Water'});
assert.ok(!short.includes('hasReview'));
assert.ok(!short.includes('reactionDetailReview'));
assert.equal(short.match(/Water/g).length,1);
assert.ok(reactionDetailHtml({conditions:[]}).includes('未记录具体条件'));
assert.ok(reactionDetailHtml({step_origins:[{kind:'aizynthfinder_short_tail'}]}).includes('AiZ 未提供实验条件'));
const escaped = reactionDetailHtml({conditions:['<img src=x onerror=alert(1)>'], catalyst:'A&B',
  critic_reasons:['<script>unsafe</script>'], precursor_smiles:['<div>'], critic_verdict:'" onclick="unsafe'},
  {origin:'<source>',status:'"',relationLabel:'<role>'});
assert.ok(!escaped.includes('<img src=x'));
assert.ok(!escaped.includes('<script>'));
assert.ok(escaped.includes('&lt;img'));
assert.ok(escaped.includes('A&amp;B'));
assert.ok(escaped.includes('&lt;source&gt;'));
"""
    subprocess.run(["node", "-e", script, str(STATIC_DIR / "route_quality.js")], check=True, capture_output=True, text=True)


def test_auxiliary_reactant_is_shown_without_expanding_the_route_tree() -> None:
    script = r"""
const assert = require('node:assert/strict');
const view = require(process.argv[1]);
const bromide = 'COC(=O)C[C@@H](CBr)O[Si](C)(C)C(C)(C)C', thiol = 'Sc1nnnn1-c1ccccc1';
const step = {step_id:'substitution', product_smiles:'sulfide', precursor_smiles:[bromide],
  reaction_input_smiles:[bromide,thiol], auxiliary_reagent_smiles:[thiol]};
const saved = JSON.stringify(step);
assert.deepEqual(view.reactionInputs(step),[bromide,thiol]);
assert.deepEqual(view.additionalReactionInputs(step),[thiol]);
const html = view.reactionDetailHtml(step);
assert.ok(html.includes('本步反应物 · 2'));
assert.ok(html.includes('smiles='+encodeURIComponent(thiol)));
assert.ok(html.includes('data-molecule-smiles="'+thiol+'"'));
const graph = view.analyze([step], 'sulfide');
assert.equal(graph.groups[0].length,2);
assert.equal(graph.groups[0][0].smiles,bromide);
assert.equal(graph.groups[0][1].smiles,thiol);
assert.equal(graph.groups[0][1].co_reactant,true);
assert.deepEqual(graph.groups[0][1].child_step_indices,[]);
assert.equal(graph.recordCount,1);
assert.equal(graph.longest,1);
assert.equal(JSON.stringify(step),saved);
const offline = view.reactionDetailHtml(step,{molecule:smiles=>`<svg data-smiles="${smiles}"></svg>`});
assert.ok(offline.includes('<svg data-smiles="'+thiol+'"'));
assert.ok(!offline.includes('/api/v4/molecule.svg'));
assert.deepEqual(view.reactionInputs({precursor_smiles:[bromide,thiol]}),[bromide,thiol]);
assert.deepEqual(view.additionalReactionInputs({precursor_smiles:[bromide,thiol]}),[]);
const stock = {stock_hit_smiles:[thiol]};
assert.equal(view.stockPresentation(thiol,stock,'CO-REACTANT').label,'CO-REACTANT');
assert.equal(view.stockPresentation(thiol,stock,'OPEN LEAF').label,'STOCK LEAF');
assert.equal(view.stockPresentation(thiol,stock).className,'stockHit');
assert.ok(view.stockPresentation(thiol,stock).badge.includes('库存命中'));
assert.equal(view.stockPresentation(bromide,stock,'OPEN LEAF').badge,'');
assert.equal(view.stockPresentation(thiol,{}).className,'');
"""
    subprocess.run(["node", "-e", script, str(STATIC_DIR / "route_quality.js")], check=True, capture_output=True, text=True)


def test_live_and_offline_detail_handlers_use_shared_markup_and_valid_javascript() -> None:
    script = r"""
const assert = require('node:assert/strict'), fs = require('node:fs'), vm = require('node:vm');
for (const file of process.argv.slice(1)) {
  const html = fs.readFileSync(file,'utf8');
  assert.ok(html.includes('RoutePresentation.reactionDetailHtml(step,'));
  assert.ok(!html.includes('function reactionInsightMarkup'));
  assert.ok(!html.includes("conditions.join(' · ')"));
  assert.ok(html.includes("$('reactionDetailBody').scrollTop=0;"));
  assert.ok(html.includes("'SUMMARY'")); // Space on a disclosure must not start playback.
  for (const match of html.matchAll(/<script\b([^>]*)>([\s\S]*?)<\/script>/gi)) {
    if (/\btype=["']application\/json/.test(match[1])) continue;
    new vm.Script(match[2],{filename:file});
  }
}
"""
    subprocess.run(["node", "-e", script, *(str(STATIC_DIR / name) for name in
                    ("live_synthesis.html", "run_showcase.html"))], check=True, capture_output=True, text=True)


def test_sequence_grouping_uses_dependencies_without_changing_source() -> None:
    script = r"""
const assert = require('node:assert/strict');
const {analyze} = require(process.argv[1]);
const step = (id, product, precursors, extra = {}) => ({step_id:id, product_smiles:product, precursor_smiles:precursors, ...extra});
const rows = [step('root','T',['A','B']), step('a','A',['C']), step('b','B',['E']), step('c','C',['D'])];
const saved = JSON.stringify(rows), a = analyze(rows,'T');
assert.equal(a.recordCount,4); assert.equal(a.longest,3); assert.equal(a.unreachable,0);
assert.deepEqual(a.chain(0),[0]); assert.deepEqual(a.chain(1),[1,3]);
assert.equal(JSON.stringify(rows),saved);
assert.deepEqual(analyze(rows.map(s=>s.step_id==='c'?{...s,checkpoint_relation:'executes_checkpoint'}:s),'T').chain(1),[1]);
assert.deepEqual(analyze(rows.map(s=>s.step_id==='c'?{...s,critic_verdict:'reject'}:s),'T').chain(1),[1]);
const fork=[step('root','T',['A']),step('a1','A',['B']),step('a2','A',['C'])];
assert.equal(analyze(fork,'T').alternatives,true); assert.deepEqual(analyze(fork,'T').chain(0),[0]);
assert.equal(analyze([step('r1','T',['A']),step('r2','T',['B'])],'T').alternatives,true);
const disconnected=[step('root','T',['A']),step('orphan','X',['Y'])];
assert.equal(analyze(disconnected,'T').longest,1); assert.equal(analyze(disconnected,'T').unreachable,1);
assert.equal(analyze(disconnected,'missing').longest,null);
assert.equal(analyze([step('a','T',['A']),step('b','A',['T'])],'T').longest,null);
const shared=[step('root','T',['A','B']),step('a','A',['C']),step('b','B',['C']),step('c','C',['D'])];
assert.deepEqual(analyze(shared,'T').chain(1),[1]);
"""
    subprocess.run(["node", "-e", script, str(STATIC_DIR / "route_quality.js")], check=True, capture_output=True, text=True)


@pytest.mark.parametrize("kind", ["interaction", "graph", "route"])
def test_native_viewer_group_toggle_preserves_steps_conditions_and_counts(tmp_path: Path, kind: str) -> None:
    from test_v4_showcase_export import _saved_run
    from cascade_planner.web.v4_showcase_export import build_run_export_bundle, render_run_export_bundle_html
    from cascade_planner.web.v4_live_synthesis import _annotate_route_topology, render_molecule_svg

    playwright = pytest.importorskip("playwright.sync_api")
    chrome = Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe")
    if not chrome.is_file():
        pytest.skip("Chromium executable required")
    bundle = build_run_export_bundle(run_dir=_saved_run(tmp_path), branch_indices=(1,), export_kind=kind)
    structures = ["CCCCO", "CCCC=O", "CCCC(=O)O", "CCCC(=O)OC"]
    steps = [{"step_id":f"record-{i}", "product_smiles":structures[i], "precursor_smiles":[structures[i+1]],
              "reaction_family":f"Transformation {i}", "conditions":[f"Saved condition {i}"], "critic_verdict":"uncertain"} for i in range(3)]
    _annotate_route_topology(steps)
    bundle["metadata"]["target_smiles"] = structures[0]
    bundle["projection"]["target_smiles"] = structures[0]
    bundle["projection"]["branches"] = [{"branch_index":1,"steps":steps,"status":"complete"}]
    bundle["replay"] = {"frames":[{"kind":"final","branch_index":1,"branch_updates":bundle["projection"]["branches"]}], "frame_count":1}
    bundle["molecules"].update({smiles:render_molecule_svg(smiles)[0] for smiles in structures})
    path = tmp_path / (kind + ".html")
    path.write_text(render_run_export_bundle_html(bundle),encoding="utf-8")
    with playwright.sync_playwright() as runtime:
        browser = runtime.chromium.launch(executable_path=str(chrome),headless=True)
        try:
            page = browser.new_page(viewport={"width":1500,"height":1000})
            errors = []
            page.on("pageerror",lambda error:errors.append(str(error)))
            page.goto(path.as_uri())
            page.locator("[data-route-record-count]").wait_for()
            assert page.locator("[data-route-record-count]").inner_text() == "3"
            assert page.locator("[data-route-lls]").inner_text() == "3"
            if kind == "route":
                assert page.locator(".route-step").count() == 3
                assert page.locator("[data-paper-grouping]").count() == 0
                for i in range(3):
                    assert page.get_by_text(f"Saved condition {i}", exact=True).is_visible()
                assert not errors
                return
            group = page.locator("details.routeSequence")
            group.wait_for()
            assert group.count() == 1
            assert group.get_attribute("data-sequence-count") == "3"
            assert not group.evaluate("el=>el.open")
            assert page.locator("[data-route-record-count]").inner_text() == "3"
            assert page.locator("[data-route-lls]").inner_text() == "3"
            group.locator("summary").click()
            assert group.evaluate("el=>el.open")
            for i in range(3):
                assert group.get_by_text(f"Saved condition {i}",exact=True).is_visible()
            page.locator("[data-paper-grouping]").uncheck()
            assert page.locator("details.routeSequence").count() == 0
            assert page.locator("[data-route-record-count]").inner_text() == "3"
            assert page.locator("[data-route-lls]").inner_text() == "3"
            assert not errors
        finally:
            browser.close()
