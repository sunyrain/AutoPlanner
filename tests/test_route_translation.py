from copy import deepcopy
from threading import Event, Lock
import time

from scripts.translate_route_exports import display_entries, localized_copy, scientific_content, text_id, validate_translation
from scripts.route_translation_display import details_page
from scripts import translate_route_exports as translation


def test_translation_is_display_only_and_preserves_ids_structures_and_verdicts():
    original = {'metadata': {'target_name': 'Fluvastatin', 'run_id': 'Fluvastatin'},
                'projection': {'strategies': [{'query': 'Reduce the ketone', 'critical_assumption': 'Selectivity is unverified'}],
                    'branches': [{'status': 'complete', 'steps': [{'step_id': 'Reduce the ketone',
                        'product_smiles': 'COC(=O)C[C@H](O)CO', 'precursor_smiles': ['COC(=O)CC(=O)CO'],
                        'conditions': ['Screen KRED with NADPH at 20 °C.'], 'critic_verdict': 'uncertain',
                        'reaction_operations': [{'op': 'change_bond_order', 'map_a': 1, 'map_b': 2, 'delta': 1}]}]}]}}
    saved = deepcopy(original)
    entries = list(display_entries(original))
    assert {e['field'] for e in entries} == {'target_name', 'query', 'critical_assumption', 'conditions'}
    mapping = {text_id('Reduce the ketone'): '还原酮基', text_id('Screen KRED with NADPH at 20 °C.'): '在 20 °C 下使用 NADPH 筛选 KRED。'}
    translated = localized_copy(original, mapping)
    assert scientific_content(original) == scientific_content(translated)
    assert original == saved
    assert translated['projection']['strategies'][0]['query'] == '还原酮基'
    old_step = original['projection']['branches'][0]['steps'][0]
    step = translated['projection']['branches'][0]['steps'][0]
    assert step['conditions'] == ['在 20 °C 下使用 NADPH 筛选 KRED。']
    for field in ('step_id', 'product_smiles', 'precursor_smiles', 'critic_verdict', 'reaction_operations'):
        assert step[field] == old_step[field]


def test_translation_checks_detect_missing_rows_numbers_and_cofactors():
    source = 'Screen KRED with NADPH at 20 °C; activity is unverified.'
    entries = [{'id': text_id(source), 'source': source}]
    assert validate_translation(entries, {'translations': [{'id': text_id(source), 'zh': '在 20 °C 下使用 NADPH 筛选 KRED；活性尚未验证。'}]})[1] == []
    problems = validate_translation(entries, {'translations': [{'id': text_id(source), 'zh': '在 30 °C 下反应。'}]})[1]
    assert {p['kind'] for p in problems} == {'numeric_token_difference', 'protected_term_missing'}
    assert validate_translation(entries, {'translations': []})[1]
    # Exact identities, not entry order, bind translated text.
    assert validate_translation(entries, {'translations': [{'id': 'wrong', 'zh': '测试'}]})[1]


def test_bilingual_details_preserve_all_branches_refreshes_conditions_and_originals():
    original = {'projection': {
        'strategies': [{'index': 4, 'query': 'A query', 'critical_assumption': 'Unverified <activity>',
                        'strategy_refreshes': [{'milestone_index': 2, 'query': 'A later query'}]}],
        'branches': [{'branch_index': 4, 'steps': [{'step_id': 'step-1', 'reaction_family': 'Reduction',
            'conditions': ['First condition', 'Second condition'], 'critic_verdict': 'uncertain',
            'critic_reasons': ['Activity is unverified']}]}]}}
    mapping = {text_id('A query'): '策略设计', text_id('Unverified <activity>'): '活性尚未验证',
               text_id('A later query'): '后续调整', text_id('First condition'): '第一个条件',
               text_id('Second condition'): '第二个条件', text_id('Reduction'): '还原'}
    page = details_page(original, localized_copy(original, mapping), '测试')
    for value in ('strategy-4', 's4-r1', '后续调整', '第一个条件', '第二个条件', 'uncertain',
                  'First condition', 'Second condition', 'Unverified &lt;activity&gt;', '英文原文'):
        assert value in page
    assert '<activity>' not in page


def test_stereochemistry_temperature_sign_and_protecting_groups_are_checked():
    source = 'Use Cbz substrate (3R,5R) at −78 °C in THF.'
    entry = {'id': text_id(source), 'source': source}
    _, issues = validate_translation([entry], {'translations': [{'id': entry['id'], 'zh': 'Cbz底物(3S,5R)，78 °C，THF。'}]})
    assert {'id': entry['id'], 'kind': 'protected_term_missing', 'token': '3R'} in issues
    assert {'id': entry['id'], 'kind': 'negative_sign_difference'} in issues
    source = 'Review steps review-014 through review-011.'
    assert not validate_translation([{'id': text_id(source), 'source': source}],
        {'translations': [{'id': text_id(source), 'zh': '查看步骤 review-014 至 review-011。'}]})[1]


def test_bulk_resume_skips_completed_ids_and_uses_real_request_concurrency(tmp_path, monkeypatch):
    entries = [{'id': text_id(f'Reduce substrate {i}'), 'source': f'Reduce substrate {i}'} for i in range(8)]
    translation.write_json(tmp_path / 'catalog.json', {'entries': entries, 'runs': []})
    old = tmp_path / 'translations/gpt-5.6-terra-low-old'
    translation.write_json(old / 'input.json', {'entries': entries[:2]})
    translation.write_json(old / 'output.json', {'translations': [{'id': e['id'], 'zh': e['source']} for e in entries[:2]]})
    translation.write_json(old / 'result.json', {'exit_code': 0, 'tool_items': []})
    saved = (old / 'output.json').read_bytes()
    lock, enough_workers = Lock(), Event()
    state = {'active': 0, 'max_active': 0, 'seen': []}

    def fake_call(batch, destination, model, effort, **kwargs):
        with lock:
            state['active'] += 1
            state['max_active'] = max(state['active'], state['max_active'])
            state['seen'].extend(e['id'] for e in batch)
            if state['active'] >= 4:
                enough_workers.set()
        assert enough_workers.wait(2)
        time.sleep(.01)
        translation.write_json(destination / 'input.json', {'entries': batch})
        translation.write_json(destination / 'output.json', {'translations': [{'id': e['id'], 'zh': e['source']} for e in batch]})
        result = {'exit_code': 0, 'tool_items': []}
        translation.write_json(destination / 'result.json', result)
        with lock:
            state['active'] -= 1
        return result

    monkeypatch.setattr(translation, 'call_translator', fake_call)
    translation.translate(tmp_path, 'gpt-5.6-terra', 'low', workers=4, batch_chars=20)
    assert state['max_active'] == 4
    assert set(state['seen']) == {e['id'] for e in entries[2:]}
    assert len(state['seen']) == 6
    assert translation.missing_entries(tmp_path, 'gpt-5.6-terra', 'low') == []
    assert (old / 'output.json').read_bytes() == saved
