"""Prepare common blinded route reviews and explicit stage/stereo negative probes."""
from copy import deepcopy
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run import ROOT, read, write, d


def main():
    base = ROOT / 'results/discussion/reaction-granularity-20260907'
    output = base / 'acceptance'
    if output.exists():
        raise FileExistsError('Keep the frozen acceptance packet unchanged.')
    cases = []

    def add(name, target, steps, criterion):
        row = {'case': name, 'arm': 'anonymous', 'target': target, 'criterion': criterion}
        cases.append(row)
        write(output / (name + '--anonymous') / 'result.json', {'case': name, 'arm': 'anonymous',
            'status': 'fixed_chemical_probe', 'steps': steps, 'host_failures': [],
            'expected_boundary_reached': False, 'record': {}, 'criterion_not_shown_to_reviewer': criterion})

    for arm in ('strict', 'transformation'):
        result = read(base / ('aiz-canary--' + arm) / 'result.json')
        steps = result['branch']['steps']
        add('natural-' + arm, steps[0]['product_smiles'], steps,
            'Independent common-rule chemical assessment of the actual AiZ path; no supplied online judgments.')

    grignard = read(base / 'explicit-no-tools-v2/grignard--transformation/result.json')['steps'][0]
    add('ordered-addition', grignard['product_smiles'], [deepcopy(grignard)],
        'The stated separate dry activation/addition/workup should remain chemically coherent.')
    wet = deepcopy(grignard)
    wet['condition_predictions'] = [{'reagents': [
        'Prepare phenylmagnesium bromide from bromobenzene and magnesium directly in methanol containing water; '
        'add acetaldehyde while retaining this wet protic medium throughout the organometallic generation and addition. '
        'No solvent exchange or separate dry reagent preparation is performed.'
    ]}]
    add('incompatible-addition', wet['product_smiles'], [wet],
        'Reject: the organomagnesium nucleophile cannot survive the explicitly retained protic medium.')

    # Same carbonyl addition, but now demand a single enantiomer from an achiral
    # setup explicitly lacking resolution. Recompile the real net edit for the
    # new product; do not fabricate a provenance/stereo pass.
    chiral = deepcopy(grignard)
    target = d._canonical_smiles('C[C@H](O)c1ccccc1')
    materialized = d.RouteJSONCompiler().compile_step(mapped_product_smiles=d._mapped_smiles(target),
        operations=chiral['reaction_operations'], expected_product_smiles=target)
    chiral.update(product_smiles=target, mapped_product_smiles=materialized.mapped_product_smiles,
        precursor_smiles=list(materialized.precursor_smiles), mapped_precursor_smiles=list(materialized.mapped_precursor_smiles),
        reaction_operations=[dict(op) for op in materialized.reaction_operations],
        transformation_hypothesis='Deliver the specified single alcohol enantiomer by carbonyl addition.')
    chiral['condition_predictions'] = [{'reagents': [
        'Prepare phenylmagnesium bromide in dry ether, then add achiral acetaldehyde and quench with aqueous ammonium chloride. '
        'Use only achiral reagents and solvent; perform no chiral catalysis, auxiliary control, resolution or enantiomer separation. '
        'The isolated product is claimed to be exclusively the specified enantiomer.'
    ]}]
    add('unsupported-stereocontrol', target, [chiral],
        'Reject the exclusive enantiomer claim under explicitly achiral, nonresolving conditions; a product annotation is not selectivity.')

    product = d._canonical_smiles('C1=CCCCC1')
    mapped = '[CH:1]1=[CH:2][CH2:3][CH2:4][CH2:5][CH2:6]1'
    operations = [{'op': 'break_bond', 'map_a': 3, 'map_b': 4},
        {'op': 'break_bond', 'map_a': 5, 'map_b': 6},
        {'op': 'change_bond_order', 'map_a': 1, 'map_b': 2, 'delta': -1},
        {'op': 'change_bond_order', 'map_a': 2, 'map_b': 3, 'delta': 1},
        {'op': 'change_bond_order', 'map_a': 1, 'map_b': 6, 'delta': 1},
        {'op': 'change_bond_order', 'map_a': 4, 'map_b': 5, 'delta': 1}]
    m = d.RouteJSONCompiler().compile_step(mapped_product_smiles=mapped, operations=operations, expected_product_smiles=product)
    cycloaddition = {'step_id': 'coherent-cycloaddition', 'product_smiles': product,
        'mapped_product_smiles': m.mapped_product_smiles, 'precursor_smiles': list(m.precursor_smiles),
        'mapped_precursor_smiles': list(m.mapped_precursor_smiles), 'reaction_operations': operations,
        'transformation_hypothesis': 'Thermal Diels-Alder cycloaddition of butadiene and ethylene.',
        'condition_predictions': [{'reagents': ['Thermal high-pressure cycloaddition; substrate conversion and polymerization competition are unvalidated.']}]
    }
    add('coherent-multibond', product, [cycloaddition],
        'Multiple bond changes can be one coherent cycloaddition. Scope or polymerization risk may be uncertain; bond count is not grounds for splitting.')
    write(output / 'inputs.json', cases)
    print('Prepared six fixed acceptance packets; two contain explicit chemical contradictions.', flush=True)


if __name__ == '__main__':
    main()
