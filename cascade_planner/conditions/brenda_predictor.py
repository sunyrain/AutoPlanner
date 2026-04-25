"""BRENDA-informed condition predictor.

Replaces the failing DRFP-based T/pH prediction (R² < 0) with enzyme-identity
lookup from BRENDA.  Temperature and pH are properties of the *enzyme*, not the
reaction graph — BRENDA has ~30-50K entries of enzyme-specific T_opt / pH_opt.

Fallback hierarchy (most to least specific):
  1. BRENDA exact (EC4, organism) match
  2. BRENDA EC4 median (across organisms)
  3. BRENDA EC3 prefix median
  4. BRENDA EC2 prefix median
  5. BRENDA EC1 class median
  6. AutoPlanner transformation_superclass median
  7. Global median (25 °C for T, 7.0 for pH)

Also provides ``TransformationRetriever`` for solvent / catalyst prediction
(frequency-based, where BRENDA doesn't help) and an evaluation harness.
"""
from __future__ import annotations

import argparse
import collections
import json
import logging
import statistics
import sys
from pathlib import Path
from typing import Any

import numpy as np

from cascade_planner.data.loader_v2 import StepRowV2, load_v2

log = logging.getLogger(__name__)

# Global fallback constants
_GLOBAL_T: float = 25.0
_GLOBAL_PH: float = 7.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ec_prefixes(ec: str) -> list[str]:
    """``"1.2.3.4"`` -> ``["1.2.3.4", "1.2.3", "1.2", "1"]``."""
    parts = ec.split(".")
    return [".".join(parts[:i]) for i in range(len(parts), 0, -1)]


def _source_label(prefix: str) -> str:
    """Map an EC prefix to a human-readable source tag."""
    n = prefix.count(".") + 1
    return f"brenda_ec{n}_median"


def _build_autoplanner_stats(
    steps: list[StepRowV2],
) -> dict[str, dict[str, float]]:
    """Build per-transformation_superclass median T/pH from AutoPlanner data."""
    t_by_trans: dict[str, list[float]] = {}
    ph_by_trans: dict[str, list[float]] = {}
    for s in steps:
        tr = s.transformation_superclass
        if not tr:
            continue
        if s.temperature_c is not None:
            t_by_trans.setdefault(tr, []).append(s.temperature_c)
        if s.ph is not None:
            ph_by_trans.setdefault(tr, []).append(s.ph)

    out: dict[str, dict[str, float]] = {}
    all_trans = set(t_by_trans) | set(ph_by_trans)
    for tr in all_trans:
        entry: dict[str, float] = {}
        if tr in t_by_trans:
            entry["T"] = statistics.median(t_by_trans[tr])
        if tr in ph_by_trans:
            entry["pH"] = statistics.median(ph_by_trans[tr])
        out[tr] = entry
    return out


# ---------------------------------------------------------------------------
# BRENDAConditionPredictor
# ---------------------------------------------------------------------------

class BRENDAConditionPredictor:
    """Predict T/pH using BRENDA enzyme-identity lookup with graceful fallback."""

    def __init__(
        self,
        brenda_lookup: Any | None = None,
        brenda_cache_path: str | Path | None = None,
        autoplanner_steps: list[StepRowV2] | None = None,
    ) -> None:
        # Try to load BRENDA lookup
        self._lookup = brenda_lookup
        if self._lookup is None and brenda_cache_path is not None:
            self._lookup = self._load_from_cache(brenda_cache_path)

        # AutoPlanner-internal fallback statistics
        self._ap_stats: dict[str, dict[str, float]] = {}
        if autoplanner_steps:
            self._ap_stats = _build_autoplanner_stats(autoplanner_steps)

    # -- cache loading -----------------------------------------------------

    @staticmethod
    def _load_from_cache(path: str | Path) -> Any | None:
        """Load a BRENDALookup from a JSON cache file."""
        p = Path(path)
        if not p.exists():
            log.warning("BRENDA cache not found: %s", p)
            return None
        try:
            from cascade_planner.data.brenda_conditions import (
                BRENDALookup,
                build_ec_lookup,
            )
            raw = json.loads(p.read_text(encoding="utf-8"))
            # Cache format: {"ec||organism": {"T_opt": float, "pH_opt": float}}
            brenda_data: dict[tuple[str, str], dict[str, float]] = {}
            for k, v in raw.items():
                parts = k.split("||", 1)
                if len(parts) == 2:
                    brenda_data[(parts[0], parts[1])] = v
            return build_ec_lookup(brenda_data)
        except Exception:
            log.warning("Failed to load BRENDA cache", exc_info=True)
            return None

    # -- core prediction ---------------------------------------------------

    def _predict(
        self,
        key: str,
        ec: str,
        organism: str | None = None,
        transformation: str | None = None,
    ) -> tuple[float | None, str]:
        """Shared logic for T and pH prediction.

        *key* is ``"T_opt"`` or ``"pH_opt"`` (BRENDA field names).
        """
        brenda_key = key  # "T_opt" or "pH_opt"
        ap_key = "T" if key == "T_opt" else "pH"
        global_default = _GLOBAL_T if key == "T_opt" else _GLOBAL_PH

        ec = (ec or "").strip()
        org = (organism or "").strip().lower()

        # 1. BRENDA exact (EC4, organism)
        if self._lookup is not None and ec and org:
            by_ec_org = getattr(self._lookup, "_by_ec_org", {})
            org_dict = by_ec_org.get(ec, {})
            entry = org_dict.get(org)
            if entry and brenda_key in entry:
                return entry[brenda_key], "brenda_ec4_organism"

        # 2-5. BRENDA prefix hierarchy
        if self._lookup is not None and ec:
            by_prefix = getattr(self._lookup, "_by_prefix", {})
            for prefix in _ec_prefixes(ec):
                pdata = by_prefix.get(prefix)
                if pdata and brenda_key in pdata:
                    return pdata[brenda_key], _source_label(prefix)

        # 6. AutoPlanner transformation_superclass median
        if transformation and transformation in self._ap_stats:
            val = self._ap_stats[transformation].get(ap_key)
            if val is not None:
                return val, "autoplanner_transform_median"

        # 7. Global median
        return global_default, "global_median"

    def predict_temperature(
        self,
        ec: str,
        organism: str | None = None,
        transformation: str | None = None,
    ) -> tuple[float | None, str]:
        """Return ``(predicted_T, source)``."""
        return self._predict("T_opt", ec, organism, transformation)

    def predict_ph(
        self,
        ec: str,
        organism: str | None = None,
        transformation: str | None = None,
    ) -> tuple[float | None, str]:
        """Return ``(predicted_pH, source)``."""
        return self._predict("pH_opt", ec, organism, transformation)

    def predict_conditions(
        self,
        ec: str,
        organism: str | None = None,
        transformation: str | None = None,
    ) -> dict:
        """Return full condition dict."""
        t_val, t_src = self.predict_temperature(ec, organism, transformation)
        ph_val, ph_src = self.predict_ph(ec, organism, transformation)
        return {
            "temperature_c": t_val,
            "temperature_source": t_src,
            "ph": ph_val,
            "ph_source": ph_src,
        }


# ---------------------------------------------------------------------------
# TransformationRetriever — solvent / catalyst frequency lookup
# ---------------------------------------------------------------------------

class TransformationRetriever:
    """Frequency-based solvent and catalyst prediction from AutoPlanner data."""

    def __init__(self, steps: list[StepRowV2]) -> None:
        # Index: (transformation, ec1) -> Counter of solvents / catalysts
        self._solv: dict[tuple[str | None, str | None], collections.Counter] = {}
        self._cat: dict[tuple[str | None, str | None], collections.Counter] = {}

        for s in steps:
            tr = s.transformation_superclass
            ec1 = s.ec_number.split(".")[0] if s.ec_number else None
            key = (tr, ec1)

            if s.solvent_smiles:
                self._solv.setdefault(key, collections.Counter())[s.solvent_smiles] += 1
            if s.catalyst_class:
                self._cat.setdefault(key, collections.Counter())[s.catalyst_class] += 1

    def _ranked(
        self,
        index: dict[tuple[str | None, str | None], collections.Counter],
        transformation: str | None,
        ec1: str | None,
    ) -> list[tuple[str, float]]:
        """Return frequency-ranked list of ``(label, fraction)``."""
        # Try exact (transformation, ec1), then (transformation, None),
        # then (None, ec1), then (None, None).
        for key in [
            (transformation, ec1),
            (transformation, None),
            (None, ec1),
            (None, None),
        ]:
            counter = index.get(key)
            if counter:
                total = sum(counter.values())
                return [(k, v / total) for k, v in counter.most_common()]
        return []

    def predict_solvent(
        self,
        transformation: str | None = None,
        ec1: str | None = None,
    ) -> list[tuple[str, float]]:
        """Return frequency-ranked solvents for matching steps."""
        return self._ranked(self._solv, transformation, ec1)

    def predict_catalyst(
        self,
        transformation: str | None = None,
        ec1: str | None = None,
    ) -> list[tuple[str, float]]:
        """Return frequency-ranked catalyst classes for matching steps."""
        return self._ranked(self._cat, transformation, ec1)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_brenda_predictor(
    steps: list[StepRowV2],
    predictor: BRENDAConditionPredictor,
) -> dict:
    """Evaluate predictor on steps with known T/pH.

    Returns dict with MAE, R², lift vs baselines, stratified by EC1 and
    transformation.
    """
    results: dict[str, Any] = {}

    for target, attr in [("temperature_c", "temperature_c"), ("ph", "ph")]:
        known = [(s, getattr(s, attr)) for s in steps if getattr(s, attr) is not None]
        if not known:
            results[target] = {"n": 0, "note": "no labelled data"}
            continue

        y_true = np.array([v for _, v in known])
        global_mean = float(np.mean(y_true))

        # Predictions
        preds = []
        sources: list[str] = []
        for s, _ in known:
            if target == "temperature_c":
                val, src = predictor.predict_temperature(
                    s.ec_number or "", None, s.transformation_superclass,
                )
            else:
                val, src = predictor.predict_ph(
                    s.ec_number or "", None, s.transformation_superclass,
                )
            preds.append(val if val is not None else global_mean)
            sources.append(src)
        y_pred = np.array(preds)

        # Baselines
        y_global_mean = np.full_like(y_true, global_mean)

        # mean_by_ec1
        ec1_vals: dict[str, list[float]] = {}
        for (s, _), v in zip(known, y_true):
            e1 = s.ec_number.split(".")[0] if s.ec_number else "_none_"
            ec1_vals.setdefault(e1, []).append(v)
        ec1_means = {k: float(np.mean(vs)) for k, vs in ec1_vals.items()}
        y_ec1_mean = np.array([
            ec1_means.get(
                s.ec_number.split(".")[0] if s.ec_number else "_none_",
                global_mean,
            )
            for s, _ in known
        ])

        def _metrics(y_p: np.ndarray) -> dict:
            mae = float(np.mean(np.abs(y_true - y_p)))
            ss_res = float(np.sum((y_true - y_p) ** 2))
            ss_tot = float(np.sum((y_true - global_mean) ** 2))
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
            return {"mae": round(mae, 3), "r2": round(r2, 4)}

        model_m = _metrics(y_pred)
        baseline_global = _metrics(y_global_mean)
        baseline_ec1 = _metrics(y_ec1_mean)

        # Source distribution
        src_counts = collections.Counter(sources)

        # Stratified by EC1
        ec1_strat: dict[str, dict] = {}
        for e1 in sorted(ec1_vals):
            mask = np.array([
                (s.ec_number.split(".")[0] if s.ec_number else "_none_") == e1
                for s, _ in known
            ])
            if mask.sum() < 5:
                continue
            sub_true = y_true[mask]
            sub_pred = y_pred[mask]
            sub_mean = float(np.mean(sub_true))
            ss_res = float(np.sum((sub_true - sub_pred) ** 2))
            ss_tot = float(np.sum((sub_true - sub_mean) ** 2))
            ec1_strat[f"EC{e1}"] = {
                "n": int(mask.sum()),
                "mae": round(float(np.mean(np.abs(sub_true - sub_pred))), 3),
                "r2": round(1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"), 4),
            }

        # Stratified by transformation
        trans_strat: dict[str, dict] = {}
        trans_vals: dict[str, list[int]] = {}
        for i, (s, _) in enumerate(known):
            tr = s.transformation_superclass or "_none_"
            trans_vals.setdefault(tr, []).append(i)
        for tr, idxs in sorted(trans_vals.items(), key=lambda x: -len(x[1])):
            if len(idxs) < 5:
                continue
            idx_arr = np.array(idxs)
            sub_true = y_true[idx_arr]
            sub_pred = y_pred[idx_arr]
            sub_mean = float(np.mean(sub_true))
            ss_res = float(np.sum((sub_true - sub_pred) ** 2))
            ss_tot = float(np.sum((sub_true - sub_mean) ** 2))
            trans_strat[tr] = {
                "n": len(idxs),
                "mae": round(float(np.mean(np.abs(sub_true - sub_pred))), 3),
                "r2": round(1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"), 4),
            }

        results[target] = {
            "n": len(known),
            "model": {"name": "brenda_predictor", **model_m},
            "baseline_global_mean": baseline_global,
            "baseline_mean_by_ec1": baseline_ec1,
            "lift_vs_global_mean_mae": round(baseline_global["mae"] - model_m["mae"], 3),
            "lift_vs_ec1_mean_mae": round(baseline_ec1["mae"] - model_m["mae"], 3),
            "source_distribution": dict(src_counts.most_common()),
            "by_ec1": ec1_strat,
            "by_transformation": trans_strat,
        }

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="BRENDA-informed condition predictor",
    )
    parser.add_argument("--data", default="cascade_dataset_v2.normalized.json",
                        help="Path to cascade dataset JSON")
    parser.add_argument("--brenda-cache", default=None,
                        help="Path to cached BRENDA lookup JSON")
    parser.add_argument("--eval", action="store_true",
                        help="Run evaluation on AutoPlanner data")
    parser.add_argument("--predict", default=None, metavar="EC",
                        help="Predict conditions for a single EC number")
    parser.add_argument("--organism", default=None,
                        help="Organism for single prediction")
    argv = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Load AutoPlanner data if needed
    steps: list[StepRowV2] = []
    if argv.eval or (argv.predict is None):
        data_path = Path(argv.data)
        if data_path.exists():
            steps, _, _ = load_v2(argv.data)
            log.info("Loaded %d steps from %s", len(steps), argv.data)
        else:
            log.warning("Data file not found: %s", argv.data)

    predictor = BRENDAConditionPredictor(
        brenda_cache_path=argv.brenda_cache,
        autoplanner_steps=steps,
    )

    if argv.predict:
        conds = predictor.predict_conditions(
            argv.predict, organism=argv.organism,
        )
        print(json.dumps(conds, indent=2))
        return

    if argv.eval:
        if not steps:
            print("No steps loaded — cannot evaluate.", file=sys.stderr)
            sys.exit(1)
        results = evaluate_brenda_predictor(steps, predictor)
        print(json.dumps(results, indent=2))
        return

    # Default: print summary
    print(f"Predictor ready. BRENDA lookup: {'loaded' if predictor._lookup else 'not available'}")
    print(f"AutoPlanner stats: {len(predictor._ap_stats)} transformations")
    if steps:
        n_ec = sum(1 for s in steps if s.ec_number)
        print(f"Steps: {len(steps)} total, {n_ec} with EC number")


if __name__ == "__main__":
    main()
