"""Check the saved BCH ring-face assignment; this is not a transition-state calculation."""
from pathlib import Path
import json

import numpy as np
from rdkit import Chem, rdBase
from rdkit.Chem import AllChem

ROOT = Path(__file__).resolve().parents[4]
source = ROOT / "results/discussion/p0-route-reassessment-20260906/bch/branch-3/context.json"
context = json.loads(source.read_text(encoding="utf-8"))
step = next(s for s in context["steps"] if s["step_id"] == "codex:branch:3:node:6:candidate:1")
results = []
for invert in [False, True]:
    molecule = Chem.MolFromSmiles(step["mapped_precursor_smiles"][0])
    if invert:
        next(a for a in molecule.GetAtoms() if a.GetAtomMapNum() == 16).InvertChirality()
    molecule = Chem.AddHs(molecule)
    maps = {a.GetAtomMapNum(): a.GetIdx() for a in molecule.GetAtoms() if a.GetAtomMapNum()}
    parameters = AllChem.ETKDGv3()
    parameters.randomSeed = 20260906
    ids = list(AllChem.EmbedMultipleConfs(molecule, numConfs=8, params=parameters))
    rows = []
    for conformer in ids:
        convergence = AllChem.UFFOptimizeMolecule(molecule, confId=conformer, maxIters=500)
        xyz = np.array(molecule.GetConformer(conformer).GetPositions())
        ring = xyz[[maps[a] for a in [8, 15, 16, 17]]]
        normal = np.linalg.svd(ring - ring.mean(axis=0))[2][-1]
        tether_side = float(np.dot(xyz[maps[7]] - xyz[maps[8]], normal))
        bromide_side = float(np.dot(xyz[maps[46]] - xyz[maps[16]], normal))
        rows.append({"conformer": conformer, "uff_convergence_code": convergence,
                     "tether_signed_projection": tether_side,
                     "bromide_signed_projection": bromide_side,
                     "same_ring_face": tether_side * bromide_side > 0})
    results.append({"variant": "opposite_C16_control" if invert else "saved_precursor",
                    "mapped_smiles": Chem.MolToSmiles(Chem.RemoveHs(molecule)), "conformers": rows})
report = {"source": str(source), "step_id": step["step_id"], "rdkit_version": rdBase.rdkitVersion,
          "method": "ETKDGv3, seed 20260906, 8 conformers per variant, UFF; substituent projections onto best-fit cyclobutane plane",
          "scope": "Static ring-face assignment only; no barrier, reaction yield or kinetic feasibility calculation",
          "control_not_a_repaired_route": True, "results": results}
output = Path(__file__).with_name("bch_ring_face_check.json")
output.write_text(json.dumps(report, indent=2), encoding="utf-8")
for row in results:
    print(row["variant"], len(row["conformers"]), sum(c["same_ring_face"] for c in row["conformers"]))
