"""End-to-end multi-step retrosynthesis driver.

Pipeline
--------
  1. Call AiZynthFinder MCTS in .venv_aizynth subprocess (USPTO + filter
     + ZINC stock). Returns up to N route trees with depths, policy
     priors, in_stock flags, template hashes.
  2. Walk each route tree, extract every reaction step as `reactants>>product`.
  3. In MAIN env: train condition heads on full snapshot, predict for each
     step: EC1 / transformation / catalyst / solvent / temperature, plus an
     enz-vs-chem gate.
  4. Pretty-print each route as an indented tree showing per-step
     conditions; dump structured JSON to results/multistep_routes.json.

Optional (Tier 2): for each non-stock leaf, run EnzExpand-EM single-step
to suggest an enzymatic alternative. Off by default to keep runtime bounded.

Usage
-----
  python -m cascade_planner.multistep.plan_route \\
      --product "CC(C)Cc1ccc(C(C)C(=O)O)cc1" \\
      --max-iter 100 --max-depth 5 --n-routes 5
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger
from sklearn.linear_model import LogisticRegression

from cascade_planner.conditions.predict_conditions import _topk_label
from cascade_planner.data.loader_v2 import load_v2
from cascade_planner.training.featurize_v2 import drfp_batch
from cascade_planner.paths import aizdata_dir

RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS = ROOT / "results"
AIZ_PY = ROOT / ".venv_aizynth" / "Scripts" / "python.exe"
AIZ_CONFIG = aizdata_dir() / "config.yml"
BRIDGE = ROOT / "cascade_planner" / "multistep" / "aiz_mcts_bridge.py"


# ----------------------- bridge call ----------------------- #

def call_mcts(product: str, max_iter: int, max_depth: int, n_routes: int,
              use_filter: bool = True, timeout: int = 600,
              config_path: Path | None = None,
              policies: list[str] | None = None,
              policy_weights: list[float] | None = None) -> dict:
    payload = {
        "product": product,
        "config": str(config_path or AIZ_CONFIG),
        "max_iter": max_iter,
        "max_depth": max_depth,
        "n_routes": n_routes,
        "use_filter": use_filter,
        "policies": policies or ["uspto"],
    }
    if policy_weights:
        payload["policy_weights"] = policy_weights
    print(f"[mcts] subprocess → AiZynth (max_iter={max_iter}, max_depth={max_depth})")
    t0 = time.time()
    proc = subprocess.run(
        [str(AIZ_PY), str(BRIDGE)],
        input=json.dumps(payload), capture_output=True, text=True,
        timeout=timeout,
    )
    print(f"  done in {time.time()-t0:.1f}s")
    if proc.returncode != 0:
        print(f"[mcts] FAILED rc={proc.returncode}")
        print(proc.stderr[-1500:])
        return {"target": product, "routes": [], "error": proc.stderr[-500:]}
    line = proc.stdout.strip().splitlines()[-1]
    return json.loads(line)


# ----------------------- condition heads ----------------------- #

def fit_clf(X, y):
    cls = sorted(set(y)); idx = {c: i for i, c in enumerate(cls)}
    yi = np.array([idx[v] for v in y])
    m = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=0)
    m.fit(X, yi)
    return m, cls


def predict_top1(m, cls, x):
    p = m.predict_proba(x.reshape(1, -1))[0]
    j = int(np.argmax(p))
    return cls[j], float(p[j])


def fit_T_by_ec1(T_vals, ec1_arr):
    means = {}
    for e in set(ec1_arr):
        mask = (ec1_arr == e) & (~np.isnan(T_vals))
        if mask.sum() > 0:
            means[e] = float(np.mean(T_vals[mask]))
    means["__global__"] = float(np.nanmean(T_vals))
    return means


def train_condition_heads(snapshot_path: str):
    print(f"[cond] loading {snapshot_path}")
    steps, _, _ = load_v2(snapshot_path)
    print(f"  steps={len(steps)} (training condition heads on full data)")
    rxns = [s.rxn_smiles for s in steps]
    X = drfp_batch(rxns)

    T_vals = np.array([s.temperature_c if s.temperature_c is not None else np.nan
                       for s in steps], dtype=float)
    ec1_arr = np.array([s.ec_number.split(".")[0] if s.ec_number else "" for s in steps])
    T_by_ec1 = fit_T_by_ec1(T_vals, ec1_arr)

    def mask(vals):
        return np.array([v is not None and v != "" for v in vals])

    m_ec = mask([s.ec_number for s in steps])
    ec1_y = np.array([s.ec_number.split(".")[0] for s in steps if s.ec_number])
    ec1_clf, ec1_cls = fit_clf(X[m_ec], ec1_y)

    m_tr = mask([s.transformation_superclass for s in steps])
    tr_y, _ = _topk_label([s.transformation_superclass for s in steps if s.transformation_superclass], 12)
    tr_clf, tr_cls = fit_clf(X[m_tr], tr_y)

    m_cat = mask([s.catalyst_class for s in steps])
    cat_y, _ = _topk_label([s.catalyst_class for s in steps if s.catalyst_class], 8)
    cat_clf, cat_cls = fit_clf(X[m_cat], cat_y)

    m_sv = mask([s.solvent_smiles for s in steps])
    sv_y, _ = _topk_label([s.solvent_smiles for s in steps if s.solvent_smiles], 12)
    sv_clf, sv_cls = fit_clf(X[m_sv], sv_y)

    is_enz = np.array(["enz" if s.ec_number else "chem" for s in steps])
    gate, gate_cls = fit_clf(X, is_enz)

    return dict(
        ec1=(ec1_clf, ec1_cls), tr=(tr_clf, tr_cls),
        cat=(cat_clf, cat_cls), sv=(sv_clf, sv_cls),
        gate=(gate, gate_cls), T_by_ec1=T_by_ec1,
    )


def annotate_step(rxn_smi: str, heads: dict) -> dict:
    """rxn_smi must be 'reactants>>product'."""
    try:
        x = drfp_batch([rxn_smi])
        if x.shape[0] == 0:
            return {}
        x0 = x[0]
    except Exception:
        return {}
    out = {}
    g, gc = heads["gate"]; out["gate"] = predict_top1(g, gc, x0)
    e, ec = heads["ec1"]; out["ec1"] = predict_top1(e, ec, x0)
    t, tc = heads["tr"]; out["transformation"] = predict_top1(t, tc, x0)
    c, cc = heads["cat"]; out["catalyst"] = predict_top1(c, cc, x0)
    s, sc = heads["sv"]; out["solvent"] = predict_top1(s, sc, x0)
    ec1_pred = out["ec1"][0]
    out["temperature_c"] = heads["T_by_ec1"].get(ec1_pred, heads["T_by_ec1"]["__global__"])
    return out


# ----------------------- tree walk + pretty print ----------------------- #

def walk_reactions(node, parent_product=None, out_list=None):
    """Yield dicts {step_idx, reactants_smi, product_smi, depth, prior, template_hash}."""
    if out_list is None:
        out_list = []
    if node is None:
        return out_list
    typ = node.get("type")
    if typ == "mol":
        prod = node["smiles"]
        for ch in node.get("children", []) or []:
            walk_reactions(ch, parent_product=prod, out_list=out_list)
    elif typ == "reaction":
        prod = parent_product
        meta = node.get("metadata", {}) or {}
        # product = parent mol, reactants = mol children
        reactant_smis = []
        for ch in node.get("children", []) or []:
            if ch.get("type") == "mol":
                reactant_smis.append(ch["smiles"])
        out_list.append({
            "product": prod,
            "reactants": reactant_smis,
            "reactants_dot": ".".join(reactant_smis),
            "rxn_smi": ".".join(reactant_smis) + ">>" + (prod or ""),
            "policy_prior": meta.get("policy_probability"),
            "policy_rank": meta.get("policy_probability_rank"),
            "template_hash": meta.get("template_hash"),
            "policy_name": meta.get("policy_name"),
        })
        for ch in node.get("children", []) or []:
            walk_reactions(ch, parent_product=parent_product, out_list=out_list)
    return out_list


def pretty_print_tree(node, heads=None, depth=0, lines=None):
    if lines is None:
        lines = []
    if node is None:
        return lines
    indent = "  " * depth
    typ = node.get("type")
    if typ == "mol":
        flag = "✓ STOCK" if node.get("in_stock") else "○"
        lines.append(f"{indent}{flag}  {node.get('smiles')}")
        for ch in node.get("children", []) or []:
            pretty_print_tree(ch, heads=heads, depth=depth + 1, lines=lines)
    elif typ == "reaction":
        meta = node.get("metadata", {}) or {}
        prior = meta.get("policy_probability")
        rank = meta.get("policy_probability_rank")
        pol = meta.get("policy_name", "?")
        head_str = f"{indent}  └─ rxn  policy={pol} rank={rank} prior={prior:.3g}" if prior is not None else f"{indent}  └─ rxn"
        # Build rxn smi from children + parent (parent is implicit via call site)
        if heads is not None:
            reactant_smis = [c["smiles"] for c in node.get("children", []) or [] if c.get("type") == "mol"]
            # parent product is one level up; we need to pass it in. Since we don't have it
            # here directly, use the reaction smiles field which aizynth gives in mapped form.
            # Fall back to skipping condition annotation if parent unknown.
            pass
        lines.append(head_str)
        for ch in node.get("children", []) or []:
            pretty_print_tree(ch, heads=heads, depth=depth + 1, lines=lines)
    return lines


# ----------------------- main ----------------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--product", required=True, help="Target SMILES")
    ap.add_argument("--data", default="cascade_dataset_v2.normalized.json")
    ap.add_argument("--max-iter", type=int, default=100)
    ap.add_argument("--max-depth", type=int, default=5)
    ap.add_argument("--n-routes", type=int, default=5)
    ap.add_argument("--no-filter", action="store_true",
                    help="Disable USPTO filter policy")
    ap.add_argument("--config", default=None,
                    help="Path to AiZynth config.yml (default: aizdata/config.yml)")
    ap.add_argument("--policies", nargs="+", default=["uspto"],
                    help="Expansion policy keys (e.g. uspto enzexpand)")
    ap.add_argument("--policy-weights", nargs="+", type=float, default=None,
                    help="Per-policy prior weight (must match --policies length; "
                         "auto-normalized to sum=1). e.g. 0.2 0.8 to bias enzexpand.")
    ap.add_argument("--out", default="results/multistep_routes.json")
    args = ap.parse_args()

    print("=" * 72)
    print(" MULTI-STEP RETROSYNTHESIS  (AiZynth MCTS + condition annotations)")
    print("=" * 72)
    print(f"target: {args.product}")

    # 1) MCTS multi-step search
    cfg = Path(args.config).resolve() if args.config else None
    res = call_mcts(args.product, args.max_iter, args.max_depth, args.n_routes,
                    use_filter=not args.no_filter,
                    config_path=cfg, policies=args.policies,
                    policy_weights=args.policy_weights)
    if not res.get("routes"):
        print("\n[error] no routes returned.")
        if "error" in res:
            print(res["error"])
        sys.exit(1)
    print(f"\n[mcts] {res['n_routes_total']} total routes, {res['n_routes_returned']} returned, "
          f"search_time={res['search_time_s']}s")

    # 2) Train condition heads
    heads = train_condition_heads(args.data)

    # 3) Walk + annotate
    out_routes = []
    for r_i, r in enumerate(res["routes"], 1):
        steps = walk_reactions(r["tree"])
        annot = []
        for st in steps:
            cond = annotate_step(st["rxn_smi"], heads)
            annot.append({**st, "conditions": cond})
        out_routes.append({
            "rank": r_i,
            "depth": r.get("depth"),
            "in_stock_frac": r.get("in_stock_frac"),
            "n_steps": len(steps),
            "steps": annot,
            "tree": r["tree"],
        })

    # 4) Pretty print
    print("\n" + "=" * 72)
    print(" TOP ROUTES")
    print("=" * 72)
    for r in out_routes:
        isf = r['in_stock_frac']
        isf_s = f"{isf:.2f}" if isinstance(isf, (int, float)) else "n/a"
        print(f"\n--- Route #{r['rank']}  steps={r['n_steps']}  in_stock_frac={isf_s}  ---")
        for j, st in enumerate(r["steps"], 1):
            cond = st.get("conditions", {})
            gate = cond.get("gate", ("?", 0))
            ec1 = cond.get("ec1", ("?", 0))
            tr = cond.get("transformation", ("?", 0))
            cat = cond.get("catalyst", ("?", 0))
            sv = cond.get("solvent", ("?", 0))
            T = cond.get("temperature_c")
            prior = st.get("policy_prior")
            print(f"  step{j}: {st['product']}")
            print(f"          ←  {' + '.join(st['reactants'])}")
            prior_s = f"{prior:.3g}" if prior is not None else "n/a"
            T_s = f"{T:.0f}°C" if T is not None else "n/a"
            print(f"          policy={st.get('policy_name')} rank={st.get('policy_rank')} prior={prior_s}")
            print(f"          gate={gate[0]}({gate[1]:.2f})  EC={ec1[0]}({ec1[1]:.2f})  "
                  f"trans={tr[0]}({tr[1]:.2f})  T≈{T_s}")
            print(f"          catalyst={cat[0]}({cat[1]:.2f})  solvent={sv[0]}({sv[1]:.2f})")

    # 5) Save JSON
    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "target": args.product,
            "search_time_s": res.get("search_time_s"),
            "n_routes_total": res.get("n_routes_total"),
            "routes": out_routes,
        }, f, indent=2, default=str)
    print(f"\n[save] {out_path}")


if __name__ == "__main__":
    main()
