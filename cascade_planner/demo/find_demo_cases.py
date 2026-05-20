"""Find a few representative cascade steps for the presentation."""
import json

from cascade_planner.paths import shared_dir

d = json.load(open('cascade_dataset_v2.normalized.json', encoding='utf-8'))
recs = d['records_kept']

cands = []
for r in recs:
    for c in r.get('cascades', []) or []:
        for s in c.get('steps', []) or []:
            cats = s.get('catalyst_components') or []
            enz = [cc for cc in cats if (cc.get('catalyst_class') or '').lower() == 'enzyme']
            if not enz:
                continue
            ec = enz[0].get('ec_number')
            rxn = s.get('rxn_smiles')
            cond = s.get('step_conditions') or {}
            T = cond.get('temperature_c')
            pH = cond.get('ph')
            sv = cond.get('solvent')
            if rxn and ec and T is not None and pH is not None and sv:
                # require RHS single product (no '.') for cleanness
                try:
                    lhs, rhs = rxn.split('>>')
                    if '.' in rhs:
                        continue
                except Exception:
                    continue
                cands.append({
                    'doi': r.get('doi'),
                    'step_id': s.get('step_id'),
                    'tsuper': s.get('transformation_superclass'),
                    'tname': s.get('transformation_name'),
                    'rxn': rxn,
                    'enz_name': enz[0].get('component_name'),
                    'ec': ec,
                    'org': enz[0].get('organism') or enz[0].get('uniprot_lookup_organism'),
                    'uniprot': enz[0].get('uniprot_id'),
                    'T': T, 'pH': pH, 'solvent': sv,
                })

print(f'Total cands: {len(cands)}')
# pick 5 diverse by EC1
seen = set()
picks = []
for c in cands:
    ec1 = (c['ec'] or '').split('.')[0]
    if ec1 in seen:
        continue
    seen.add(ec1)
    picks.append(c)
    if len(picks) >= 6:
        break

for i, c in enumerate(picks):
    print(f'\n=== DEMO {i+1}: EC{c["ec"]} ({c["tsuper"]}) ===')
    print(f'  doi      : {c["doi"]}')
    print(f'  step_id  : {c["step_id"]}')
    print(f'  rxn      : {c["rxn"]}')
    print(f'  enz      : {c["enz_name"]} (Uniprot={c["uniprot"]}, org={c["org"]})')
    print(f'  T={c["T"]}°C  pH={c["pH"]}  solvent={c["solvent"]}')

out_path = shared_dir() / 'demo_picks.json'
json.dump(picks, open(out_path, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
print(f'\nSaved -> {out_path}')
