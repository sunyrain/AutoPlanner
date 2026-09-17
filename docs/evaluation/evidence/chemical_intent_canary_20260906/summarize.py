"""Read the isolated canaries and independently compare rejected stereo boundaries."""
from collections import Counter
import json
from pathlib import Path

from rdkit import Chem, rdBase

ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "results/discussion/chemical-intent-canary-20260906"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def identity(smiles, *, stereo):
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError("Invalid saved molecule")
    for atom in molecule.GetAtoms():
        atom.SetAtomMapNum(0)
    if not stereo:
        Chem.RemoveStereochemistry(molecule)
    return Chem.MolToSmiles(molecule, isomericSmiles=stereo)


def main():
    rows = []
    for name in ("bch3", "bch3-fixed-critic-budget64k"):
        root = BASE / name
        for journal in root.rglob("model-io.jsonl"):
            for line in journal.read_text(encoding="utf-8").splitlines():
                event = json.loads(line)
                if event.get("event") != "model_output":
                    continue
                diagnostics = event.get("usage_diagnostics") or {}
                rows.append({"canary": name, "task_id": event["task_id"],
                             "task_type": event["task_type"], "usage": event.get("usage") or {},
                             "tools": diagnostics.get("tool_counts") or {},
                             "responses": (diagnostics.get("requests") or {}).get("observed_completed_responses"),
                             "source": str(journal.relative_to(ROOT))})
    result = read(BASE / "bch3-fixed-critic-budget64k/result.json")
    branch = result["branch"]
    transaction = branch["path_repair_transactions"][0]
    boundaries = {row["step_id"]: row for row in transaction["reconnect_boundaries"]}
    checks = []
    for diagnostic in branch.get("materialization_diagnostics") or []:
        if diagnostic.get("reason") != "path_repair_reconnect_boundary_stereo_mismatch":
            continue
        expected = boundaries[diagnostic["boundary_step_id"]]["mapped_product_smiles"]
        actual = diagnostic["actual_mapped_precursor_smiles"]
        checks.append({"candidate_id": diagnostic["candidate_id"],
                       "same_connectivity_without_maps": identity(expected, stereo=False) == identity(actual, stereo=False),
                       "same_stereochemistry_without_maps": identity(expected, stereo=True) == identity(actual, stereo=True),
                       "expected_identity": identity(expected, stereo=True),
                       "actual_identity": identity(actual, stereo=True)})
    source = read(BASE / "bch3-fixed-critic-budget64k/manifest.json")["source_context"]
    original = read(Path(source))
    snapshot_unchanged = [
        (identity(row["product_smiles"], stereo=True),
         sorted(identity(s, stereo=True) for s in row["precursor_smiles"]))
        for row in original["steps"]
    ] == [
        (identity(row["product_smiles"], stereo=True),
         sorted(identity(s, stereo=True) for s in row["precursor_smiles"]))
        for row in branch["steps"]
    ]
    totals = Counter()
    for row in rows:
        totals.update(row["usage"])
    report = {
        "scope": "integration diagnostics, not a matched efficacy experiment",
        "rdkit_version": rdBase.rdkitVersion,
        "new_detection_assessment": read(BASE / "bch3/critic-result.json")["critique"]["overall_assessment"],
        "fixed_critic_repair_status": result["status"],
        "transaction_status": transaction["status"], "transaction_reason": transaction["reason"],
        "search_stop_reason": branch["path_repair_aizynthfinder_search"]["host_stop_reason"],
        "original_reaction_identities_retained": snapshot_unchanged,
        "stereo_boundary_checks": checks,
        "worker_calls": len(rows), "calls": rows, "usage_totals": dict(totals),
        "limitations": ["No successful rebuilt route or live re-Critic in this canary.",
                        "Map-free RDKit comparison checks identity, not reaction feasibility.",
                        "Historical fixed critique and new detection were not merged.",
                        "One earlier launch used an output budget below the Editor/Critic reserve; it made zero model calls."],
    }
    output = Path(__file__).with_name("results.json")
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("worker_calls", "usage_totals", "original_reaction_identities_retained",
                                            "stereo_boundary_checks")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
