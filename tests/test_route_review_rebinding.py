import copy
import json

from cascade_planner.application.route_review_context import equivalent_whole_route_review
from cascade_planner.application.portfolio_selection import select_portfolio


def prompt(steps, **extra):
    return 'Policy\nPaperMatchedRouteCriticInput:\n' + json.dumps({
        'phase': 'final_route', 'steps': steps, **extra,
    })


def test_reordered_review_checks_whole_reaction_before_translating_slots():
    a = {'review_slot': 'review-001', 'mapped_product_smiles': '[CH3:1][OH:2]',
         'mapped_precursor_smiles': ['[CH4:1]', '[OH2:2]'],
         'reaction_operations': [{'op': 'break_bond', 'map_a': 1, 'map_b': 2}],
         'conditions': ['condition A']}
    b = {'review_slot': 'review-002', 'mapped_product_smiles': '[CH4:1]',
         'mapped_precursor_smiles': ['[CH3:1][Cl:3]'],
         'reaction_operations': [{'op': 'add_group', 'map_idx': 1, 'fragment_smiles': '*[Cl:3]'}],
         'conditions': ['condition B']}
    old = prompt([a, b], repair_requirements_to_reassess=['Check donor identity'])
    reordered = [{**b, 'review_slot': 'review-001'},
                 {**a, 'review_slot': 'review-002', 'mapped_precursor_smiles': list(reversed(a['mapped_precursor_smiles']))}]
    maps, slots = equivalent_whole_route_review(old, prompt(reordered))
    assert maps == {1: 1, 2: 2, 3: 3}
    assert slots == {'review-001': 'review-002', 'review-002': 'review-001'}
    for field, value in [('conditions', ['different formulation']),
                         ('reaction_operations', [{'op': 'remove_group', 'map_indices': [2]}]),
                         ('mapped_precursor_smiles', ['[CH4:1]', '[18OH2:2]'])]:
        changed = copy.deepcopy(reordered)
        changed[1][field] = value
        assert equivalent_whole_route_review(old, prompt(changed)) is None
    assert equivalent_whole_route_review(old, prompt(reordered,
        repair_requirements_to_reassess=['New incompatible consumer'])) is None
    assert equivalent_whole_route_review(old, prompt(reordered).replace('Policy', 'New policy')) is None


def test_portfolio_display_does_not_use_hand_weighted_chemical_scores():
    rows = [dict(route_id='a1', route_family_id='a', edge_ids=['x'], complete=False),
            dict(route_id='a2', route_family_id='a', edge_ids=['y'], complete=False),
            dict(route_id='b1', route_family_id='b', edge_ids=['z'], complete=False)]
    def choose(values):
        return [r['route_id'] for r in select_portfolio(values, minimum_count=2,
                maximum_count=2, require_distinct_edge_sets=True)]
    assert choose(rows) == ['a1', 'b1']
    altered = [{**r, 'risk_score': i * 100, 'strategic_value_score': 1000 - i,
                'length': 100 - i, 'pareto_optimal': i == 1,
                'evidence_maturity_score': i * 10} for i, r in enumerate(rows)]
    assert choose(altered) == choose(rows)
    assert choose([{**rows[0], 'complete': True}, *rows[1:]])[0] == 'a1'
