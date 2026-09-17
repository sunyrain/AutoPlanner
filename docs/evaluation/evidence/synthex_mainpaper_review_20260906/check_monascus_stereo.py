"""Reproduce the structural adjudication of the anonymous reviewer objection.

This is a case-specific diagnostic, not an edit to either planner or route.
It checks stereochemical consistency of the stated transformation, not feasibility.
"""
import json
from pathlib import Path
from rdkit import Chem

ROOT = Path(__file__).resolve().parents[4]
source = Path(__file__).with_name('npa026027_s3.json')
step = json.loads(source.read_text(encoding='utf-8'))['steps'][12]
lhs, rhs = step['rxn_smiles'].split('>>')
aryl, epoxide = map(Chem.MolFromSmiles, lhs.split('.'))
iodine = next(a.GetIdx() for a in aryl.GetAtoms() if a.GetAtomicNum() == 53)
aryl_carbon = aryl.GetAtomWithIdx(iodine).GetNeighbors()[0].GetIdx()
ring = next(r for r in epoxide.GetRingInfo().AtomRings() if len(r) == 3)
oxygen = next(i for i in ring if epoxide.GetAtomWithIdx(i).GetAtomicNum() == 8)
methylene = next(i for i in ring if epoxide.GetAtomWithIdx(i).GetAtomicNum() == 6
                 and epoxide.GetAtomWithIdx(i).GetTotalNumHs() == 2)
offset = aryl.GetNumAtoms()
edit = Chem.RWMol(Chem.CombineMols(aryl, epoxide))
edit.RemoveBond(offset + oxygen, offset + methylene)
edit.AddBond(aryl_carbon, offset + methylene, Chem.BondType.SINGLE)
edit.RemoveAtom(iodine)
product = edit.GetMol()
Chem.SanitizeMol(product)
Chem.AssignStereochemistry(product, cleanIt=True, force=True)
expected = Chem.MolToSmiles(product, isomericSmiles=True)
supplied_molecule = Chem.MolFromSmiles(rhs)
supplied = Chem.MolToSmiles(supplied_molecule, isomericSmiles=True)
result = {
    'source_route_id': 'npa026027_s3', 'source_step_idx': 12, 'review_slot': 'review-013',
    'method': 'Open the epoxide at CH2 and connect that CH2 to the aryl carbon formerly bonded to I; preserve the untouched stereocenter and its neighbor ordering.',
    'epoxide_cip': Chem.FindMolChiralCenters(epoxide, includeUnassigned=True),
    'expected_product': expected, 'supplied_product': supplied,
    'exact_isomeric_identity_match': expected == supplied,
    'expected_product_cip': Chem.FindMolChiralCenters(product, includeUnassigned=True),
    'supplied_product_cip': Chem.FindMolChiralCenters(supplied_molecule, includeUnassigned=True),
    'adjudication': ('The reviewer stereochemical-mismatch objection is refuted by explicit neighbor-preserving graph construction.'
                     if expected == supplied else 'Mismatch needs further analysis'),
    'limits': 'This checks the asserted stereochemical inconsistency, not the experimental yield or regioselectivity.',
}
output = ROOT / 'results/discussion/synthex-mainpaper-comparison-20260906/monascus-epoxide-stereo-adjudication.json'
output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(json.dumps(result, ensure_ascii=False, indent=2))
