"""Offline checks for the live site's display-language boundary (no browser)."""
import json
from pathlib import Path
import re
import subprocess

import pytest

from scripts.translate_route_exports import build_live_translation_bundle, text_id

STATIC = Path(__file__).resolve().parents[1] / 'cascade_planner/web/static'


def node(script, *args):
    subprocess.run(['node', '-e', script, *map(str, args)], check=True,
                   capture_output=True, text=True, encoding='utf-8')


def test_live_bundle_contains_only_reviewed_display_strings():
    entry = {'id': text_id('Reduce the ketone'), 'source': 'Reduce the ketone',
             'path': ['private', 'run'], 'field': 'query'}
    catalog = {'entries': [entry], 'runs': [{'run_dir': 'private/run'}]}
    translated = {'model': 'gpt-5.6-terra', 'translations': {entry['id']: '还原酮基'}}
    bundle = build_live_translation_bundle(catalog, translated)
    assert bundle['translations'] == {'Reduce the ketone': '还原酮基'}
    assert bundle['task_count'] == bundle['entry_count'] == 1
    assert 'private' not in json.dumps(bundle)
    assert 'conditions' in bundle['text_fields']
    assert 'step_id' not in bundle['text_fields']
    with pytest.raises(ValueError, match='Missing reviewed translation'):
        build_live_translation_bundle(catalog, {'translations': {}})


def test_language_boundary_is_cached_reversible_and_preserves_science():
    node(r"""
const assert=require('node:assert/strict');
const {create}=require(process.argv[1]);
const asset={locale:'zh-CN',text_fields:['query','conditions','reaction_family'],
 translations:{'Reduce the ketone':'还原酮基','Screen KRED at 20 °C.':'在 20 °C 下筛选 KRED。'}};
let saved='en';
const language=create({storage:{getItem:()=>saved,setItem:(_,v)=>{saved=v;}}});
assert.equal(language.locale,'en');
const raw={strategies:[{query:'Reduce the ketone'}], branches:[{branch_index:4,steps:[{
 step_id:'Reduce the ketone',reaction_family:'Reduce the ketone',conditions:['Screen KRED at 20 °C.','Untranslated new condition'],
 product_smiles:'CC(=O)O',precursor_smiles:['CC=O'],critic_verdict:'uncertain',
 reaction_operations:[{query:'Reduce the ketone'}],display_precursors:[{smiles:'CC=O',child_step_indices:[1]}]}]}],
 activities:[{query:'Reduce the ketone'}],condition_predictions:{conditions:['Screen KRED at 20 °C.']}};
const before=JSON.stringify(raw);
assert.strictEqual(language.view(raw),raw);
language.setLocale('zh-CN');assert.equal(saved,'zh-CN');
assert.strictEqual(language.view(raw),raw); // Loading/failure fallback.
language.setDictionary(asset);
const zh=language.view(raw),step=zh.branches[0].steps[0],old=raw.branches[0].steps[0];
assert.equal(zh.strategies[0].query,'还原酮基');
assert.equal(step.reaction_family,'还原酮基');
assert.deepEqual(step.conditions,['在 20 °C 下筛选 KRED。','Untranslated new condition']);
for(const key of ['step_id','product_smiles','precursor_smiles','critic_verdict','reaction_operations','display_precursors'])
 assert.strictEqual(step[key],old[key]);
assert.equal(zh.branches[0].branch_index,4);
assert.strictEqual(zh.activities,raw.activities);
assert.strictEqual(zh.condition_predictions,raw.condition_predictions);
assert.strictEqual(language.view(raw),zh);
assert.strictEqual(language.view(zh),zh);
assert.equal(JSON.stringify(raw),before);
language.setLocale('en');assert.strictEqual(language.view(raw),raw);
language.setLocale('zh-CN');assert.strictEqual(language.view(raw),zh);
const next=JSON.parse(before);next.strategies[0].query='New strategy';
assert.equal(language.view(next).strategies[0].query,'New strategy');
language.setDictionary({...asset,translations:{...asset.translations,'New strategy':'新策略'}});
assert.equal(language.view(next).strategies[0].query,'新策略');
const blocked=create({storage:{getItem(){throw Error('blocked');},setItem(){throw Error('blocked');}}});
blocked.setLocale('en');assert.equal(blocked.locale,'en');
""", STATIC / 'route_language.js')


def test_live_toggle_preserves_viewport_replay_and_open_details():
    # Execute just the real language handlers with synthetic view adapters.
    # No running browser or website DOM is inspected.
    node(r"""
const assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm');
const {create}=require(process.argv[1]);
const page=fs.readFileSync(process.argv[2],'utf8');
const start=page.indexOf('    function currentSnapshot()');
const end=page.indexOf('    function routeEvaluationFallback(',start);
const nodes={};const $=id=>nodes[id]??=( {dataset:{},scrollTop:0,hidden:false,open:false,
 setAttribute(k,v){this[k]=v;}} );
const raw={target_name:'Statin',strategies:[{query:'Reduce the ketone'}],branches:[]};
const language=create({locale:'en'});
language.setDictionary({locale:'zh-CN',text_fields:['query'],translations:{'Reduce the ketone':'还原酮基'}});
const appState={snapshot:raw,targetName:'Statin',zoom:1.7,panX:-230,panY:82,viewInitialized:true,
 selectedBranch:4,orientation:'horizontal',compactRoute:true,focusedLaneId:'lane-2',
 replay:{active:true,snapshots:[null,raw],index:1,playing:true,speed:2,timer:42}};
const before=JSON.stringify(appState);
$('reactionReplayJump').dataset.stepId='step-7';$('reactionDetailBody').scrollTop=117;
$('strategyDetail').open=true;$('strategyDetail').dataset.branch='4';$('strategyDetailBody').scrollTop=59;
let rendered,details=[],requests=0;
const context={$,appState,routeLanguage:language,routeTranslationState:'ready',
 renderLocalJobs(){},renderStrategies(value){rendered=value;},
 renderRoute(){$('reactionDetail').hidden=true;$('reactionReplayJump').dataset.stepId='';},
 showReactionDetail(id){details.push(id);$('reactionDetail').hidden=false;
   $('reactionReplayJump').dataset.stepId=id;$('reactionDetailBody').scrollTop=0;},
 showStrategyDetail(index){assert.equal(index,4);$('strategyDetailBody').scrollTop=0;},
 fetch(){requests++;throw Error('must not fetch');}};
vm.createContext(context);vm.runInContext(page.slice(start,end),context);
context.setRouteLanguage('zh-CN');
assert.equal(rendered[0].query,'还原酮基');assert.deepEqual(details,['step-7']);
assert.equal($('reactionDetailBody').scrollTop,117);assert.equal($('strategyDetailBody').scrollTop,59);
assert.equal($('routeLanguageZh')['aria-pressed'],'true');
assert.equal(JSON.stringify(appState),before);assert.equal(requests,0);
context.setRouteLanguage('en');assert.strictEqual(rendered,raw.strategies);
assert.equal(JSON.stringify(appState),before);assert.equal(requests,0);
appState.replay.active=false;
appState.snapshot={...raw,strategies:[{query:'New untranslated strategy'}]};
context.setRouteLanguage('zh-CN');assert.equal(rendered[0].query,'New untranslated strategy');
""", STATIC / 'route_language.js', STATIC / 'live_synthesis.html')


def test_live_page_script_syntax_and_translation_asset():
    page = (STATIC / 'live_synthesis.html').read_text(encoding='utf-8')
    for attrs, script in re.findall(r'<script\b([^>]*)>([\s\S]*?)</script>', page):
        if 'application/json' not in attrs and 'src=' not in attrs:
            subprocess.run(['node', '--check'], input=script, check=True,
                           capture_output=True, text=True, encoding='utf-8')
    assert 'id="routeLanguageZh"' in page and 'id="routeLanguageEn"' in page
    node(r"""
const assert=require('node:assert/strict'),fs=require('node:fs');
const {create}=require(process.argv[1]);
const asset=JSON.parse(fs.readFileSync(process.argv[2],'utf8'));
assert.equal(Object.keys(asset.translations).length,asset.entry_count);
const lang=create();lang.setDictionary(asset);
for(const [source,zh] of Object.entries(asset.translations))assert.equal(lang.text(source),zh);
lang.setLocale('en');for(const source of Object.keys(asset.translations))assert.equal(lang.text(source),source);
""", STATIC / 'route_language.js', STATIC / 'route_translations.zh-CN.json')
