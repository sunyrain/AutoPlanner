"""CascadeBoard++ CLI — minimal end-user interface.

Usage:
  python -m cascade_planner.cascadeboard.cli --target "CC(=O)O" --n-steps 2
  python -m cascade_planner.cascadeboard.cli --target "CC(=O)O" --constraints '{"one_pot": true, "max_delta_T": 15}'
  python -m cascade_planner.cascadeboard.cli --target "CC(=O)O" --objective green --checkpoint results/shared/cascadeboard_model_v13/best.pt
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
import time
from pathlib import Path

from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")


def _format_route(result, idx: int) -> str:
    """Format a single RouteResult for terminal display."""
    b = result.board
    lines = [f"Route {idx+1}  (score={result.score:.2f}, confidence={result.confidence:.2f})"]
    lines.append("─" * 60)

    for s in b.slots:
        ec_str = s.ec or ""
        type_str = s.reaction_type or ""
        src = s.source or "?"
        tag = f"[{src}]"
        if ec_str:
            tag += f" EC {ec_str}"
        if type_str:
            tag += f" {type_str}"

        prod = s.product or "?"
        react = s.main_reactant or "?"
        lines.append(f"  Step {s.index}: {react}")
        if s.aux_reactants:
            lines.append(f"         + {', '.join(s.aux_reactants)}")
        cond_parts = []
        if s.T is not None:
            cond_parts.append(f"{s.T:.0f}°C")
        if s.pH is not None:
            cond_parts.append(f"pH {s.pH:.1f}")
        if s.solvent:
            cond_parts.append(s.solvent)
        cond_str = ", ".join(cond_parts) if cond_parts else "conditions unknown"
        lines.append(f"       → {prod}  {tag}")
        lines.append(f"         ({cond_str})")

    # Diagnostics
    if result.bottleneck_reason:
        lines.append(f"  Bottleneck: {result.bottleneck_reason}")

    exp = result.explanation
    if exp and exp.constraints_satisfied:
        lines.append(f"  Constraints: {exp.constraints_satisfied}")
    if exp and exp.constraints_at_risk:
        lines.append(f"  At risk: {exp.constraints_at_risk}")

    # Quality
    q = result.quality_vector
    if q:
        q_parts = [f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}" for k, v in q.items()]
        lines.append(f"  Quality: {', '.join(q_parts)}")

    # Risk
    r = result.risk_vector
    if r:
        r_parts = [f"{k}={v:.2f}" for k, v in r.items()]
        lines.append(f"  Risk: {', '.join(r_parts)}")

    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(
        description="CascadeBoard++ — constraint-compiled cascade route planner",
    )
    ap.add_argument("--target", required=True, help="Target molecule SMILES")
    ap.add_argument("--n-steps", type=int, default=None, help="Number of steps (default: auto)")
    ap.add_argument("--constraints", default=None, help="JSON string of constraints")
    ap.add_argument("--objective", default="balanced", choices=["balanced", "industrial", "green", "novelty"])
    ap.add_argument("--checkpoint", default=None, help="Path to model checkpoint (.pt)")
    ap.add_argument("--n-results", type=int, default=5)
    ap.add_argument("--n-particles", type=int, default=32)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--json", action="store_true", help="Output as JSON instead of formatted text")
    ap.add_argument("--no-real-candidates", action="store_true", help="Use mock candidates (for testing)")
    ap.add_argument("--live", action="store_true", help="Use live RetroChimera + Skeleton MLP (for any molecule)")
    ap.add_argument("--domain", default="chemoenzymatic", choices=["all_enzymatic", "chemoenzymatic", "all_chemical"],
                     help="Route domain hint for skeleton generation")
    args = ap.parse_args()

    # Validate target
    mol = Chem.MolFromSmiles(args.target)
    if mol is None:
        print(f"Error: invalid SMILES '{args.target}'", file=sys.stderr)
        sys.exit(1)

    # Parse constraints
    constraints = None
    if args.constraints:
        try:
            constraints = json.loads(args.constraints)
        except json.JSONDecodeError as e:
            print(f"Error: invalid constraints JSON: {e}", file=sys.stderr)
            sys.exit(1)

    # Build planner
    from cascade_planner.cascadeboard.planner import CascadeBoardPlanner, load_cascadeboard_model
    from cascade_planner.cascadeboard.energy_api import EnergyAPI

    if args.live:
        # Unified pipeline: OA-ARM skeleton generation + live molecular fill.
        import logging; logging.disable(logging.CRITICAL)
        import warnings; warnings.filterwarnings("ignore")
        from cascade_planner.cascadeboard.skeleton_inpainter import plan_with_skeleton_inpainter
        from cascade_planner.cascadeboard.live_retro import build_live_retro_engine

        sink = io.StringIO() if args.json else None
        stream_ctx = contextlib.redirect_stdout(sink) if sink is not None else contextlib.nullcontext()
        with stream_ctx:
            retro = build_live_retro_engine()
            n_steps = args.n_steps or 3
            ckpt = args.checkpoint or "results/shared/skeleton_inpainter/best.pt"

            t0 = time.time()
            results = plan_with_skeleton_inpainter(
                target=args.target,
                n_steps=n_steps,
                domain=args.domain,
                objective=args.objective,
                constraints=constraints,
                fixed_slots=None,
                model_path=ckpt,
                retro_engine=retro,
                device=args.device,
                n_results=args.n_results,
                n_candidates_per_skeleton=2,
            )
            elapsed = time.time() - t0
    else:
        # Legacy pipeline: cache-based planner
        motif_path = Path("results/shared/cascade_motifs.json")
        motif_memory = json.loads(motif_path.read_text()) if motif_path.exists() else None
        model = load_cascadeboard_model(args.checkpoint, device=args.device) if args.checkpoint else None
        energy_api = EnergyAPI(motif_memory=motif_memory)
        planner = CascadeBoardPlanner(
            energy_api=energy_api, model=model, device=args.device,
            use_real_candidates=not args.no_real_candidates,
        )
        t0 = time.time()
        results = planner.plan(
            target=args.target, constraints=constraints, objective=args.objective,
            n_steps=args.n_steps, n_particles=args.n_particles, n_final=args.n_results,
        )
        elapsed = time.time() - t0

    if args.json:
        from cascade_planner.cascadeboard.route_export import route_results_payload
        out = route_results_payload(
            args.target,
            results,
            objective=args.objective,
            constraints=constraints,
            elapsed_s=elapsed,
        )
        print(json.dumps(out, indent=2, ensure_ascii=False))
    else:
        print(f"CascadeBoard++ — {len(results)} routes for {args.target} ({elapsed:.1f}s)")
        print(f"Objective: {args.objective}")
        if constraints:
            print(f"Constraints: {json.dumps(constraints)}")
        print()
        for i, r in enumerate(results):
            print(_format_route(r, i))
            print()


if __name__ == "__main__":
    main()
