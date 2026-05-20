"""Build proposal recall supervision packs and audits.

Proposal packs are the recall-layer contract: candidates are labeled here, but
route choice remains in the route-tree controller.
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from cascade_planner.cascadeboard.route_recovery import (
    canonical_reaction,
    canonical_smiles,
    reaction_reactants,
)
from cascade_planner.vnext.features import read_jsonl


PROPOSAL_RECALL_SCHEMA_VERSION = "proposal_recall_pack.v1"
REQUIRED_FIELDS = (
    "product",
    "gt_reactants",
    "rxn_smiles",
    "reaction_type",
    "ec",
    "source",
    "conditions",
    "candidate_pool",
    "positive_candidate_indices",
    "hard_negative_type",
)

_CAN_SMILES_CACHE: dict[str, str] = {}
_CAN_RXN_CACHE: dict[str, str] = {}
_RXN_REACTANTS_CACHE: dict[str, set[str]] = {}
_WORKER_STEP_BY_ID: dict[str, dict[str, Any]] = {}
_WORKER_DERIVE_GT_FROM_POOL = False


def build_proposal_recall_pack(
    *,
    external_step_pairs: Path,
    candidate_pools: Path,
    output_dir: Path,
    max_rows: int | None = None,
    workers: int = 1,
    chunk_size: int = 512,
    derive_gt_from_pool: bool = False,
) -> dict[str, Any]:
    step_by_id = {}
    if not derive_gt_from_pool:
        steps = read_jsonl(external_step_pairs)
        step_by_id = {str(row.get("step_id") or row.get("external_step_id") or ""): row for row in steps}
    output_dir.mkdir(parents=True, exist_ok=True)
    pack_path = output_dir / "proposal_recall_pack.jsonl"
    audit_acc = _AuditAccumulator()
    count = 0
    with pack_path.open("w", encoding="utf-8") as fh:
        if int(workers or 1) > 1:
            for rows in _parallel_rows(
                candidate_pools,
                step_by_id=step_by_id,
                workers=workers,
                chunk_size=chunk_size,
                max_rows=max_rows,
                derive_gt_from_pool=derive_gt_from_pool,
            ):
                for row in rows:
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                    audit_acc.add(row)
                    count += 1
                    if max_rows is not None and count >= max_rows:
                        break
                if max_rows is not None and count >= max_rows:
                    break
        else:
            for pool in _iter_jsonl(candidate_pools):
                step = _step_from_pool(pool) if derive_gt_from_pool else step_by_id.get(str(pool.get("external_step_id") or ""))
                if step is None:
                    step = _step_from_pool(pool)
                row = proposal_recall_row(step, pool)
                if row:
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                    audit_acc.add(row)
                    count += 1
                if max_rows is not None and count >= max_rows:
                    break
    audit = audit_acc.to_dict()
    manifest = {
        "schema_version": PROPOSAL_RECALL_SCHEMA_VERSION,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "inputs": {
            "external_step_pairs": str(external_step_pairs),
            "candidate_pools": str(candidate_pools),
        },
        "files": {
            "proposal_recall_pack": str(pack_path),
            "manifest": str(output_dir / "manifest.json"),
            "report": str(output_dir / "report.md"),
        },
        "counts": {"rows": count},
        "audit": audit,
        "required_fields": list(REQUIRED_FIELDS),
        "workers": int(workers or 1),
        "chunk_size": int(chunk_size or 1),
        "derive_gt_from_pool": bool(derive_gt_from_pool),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "report.md").write_text(_report_markdown(manifest), encoding="utf-8")
    return manifest


def proposal_recall_row(step: dict[str, Any], pool: dict[str, Any]) -> dict[str, Any] | None:
    product = str(step.get("product") or pool.get("product") or pool.get("target_smiles") or "")
    rxn = str(step.get("reaction_smiles") or step.get("rxn_smiles") or "")
    gt_reactants = sorted(_canonical_set(step.get("reactants") or []) or _rxn_reactants(rxn))
    if not product or not gt_reactants:
        return None
    candidate_pool = [_normalize_candidate_item(item, default_product=product) for item in pool.get("candidates") or []]
    labels = label_candidate_pool_recall(
        product=product,
        gt_rxn=rxn,
        gt_reactants=gt_reactants,
        candidate_pool=candidate_pool,
    )
    return {
        "product": product,
        "gt_reactants": gt_reactants,
        "rxn_smiles": _can_reaction(rxn) or rxn,
        "reaction_type": step.get("reaction_type") or pool.get("reaction_type") or "",
        "ec": step.get("ec") or pool.get("ec") or "",
        "source": step.get("source") or pool.get("external_source") or pool.get("source") or "unknown",
        "conditions": _conditions_from_step(step),
        "candidate_pool": candidate_pool,
        "positive_candidate_indices": labels["positive_candidate_indices"],
        "hard_negative_type": _hard_negative_type(pool),
        "recall_labels": labels,
    }


def label_candidate_pool_recall(
    *,
    product: str,
    gt_rxn: str,
    gt_reactants: list[str] | tuple[str, ...] | set[str],
    candidate_pool: list[dict[str, Any]],
) -> dict[str, Any]:
    gt_rxn_can = _can_reaction(gt_rxn)
    gt_reactant_set = _canonical_set(gt_reactants)
    exact_indices: list[int] = []
    gt_reactant_indices: list[int] = []
    any_reactant_indices: list[int] = []
    for idx, candidate in enumerate(candidate_pool):
        cand_rxn = _can_reaction(candidate.get("rxn_smiles") or candidate.get("reaction_smiles") or "")
        cand_reactants = _candidate_reactants(candidate)
        if gt_rxn_can and cand_rxn == gt_rxn_can:
            exact_indices.append(idx)
        if gt_reactant_set and cand_reactants == gt_reactant_set:
            gt_reactant_indices.append(idx)
        if gt_reactant_set and cand_reactants & gt_reactant_set:
            any_reactant_indices.append(idx)
    positives = sorted(set(exact_indices) | set(gt_reactant_indices))
    return {
        "positive_candidate_indices": positives,
        "exact_reaction_indices": exact_indices,
        "gt_reactant_indices": gt_reactant_indices,
        "any_gt_reactant_indices": any_reactant_indices,
        "candidate_exact_reaction_in_pool": bool(exact_indices),
        "candidate_gt_reactant_in_pool": bool(gt_reactant_indices or any_reactant_indices),
    }


def audit_proposal_recall_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_domain: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_source[str(row.get("source") or "unknown")].append(row)
        by_domain[_domain(row)].append(row)
    return {
        "overall": _recall_summary(rows),
        "chemical": _recall_summary([row for row in rows if _domain(row) == "all_chemical"]),
        "enzymatic": _recall_summary([row for row in rows if _domain(row) == "all_enzymatic"]),
        "rhea_retrorules": _recall_summary([row for row in rows if str(row.get("source") or "").lower() in {"rhea", "retrorules"}]),
        "per_source": {source: _recall_summary(source_rows) for source, source_rows in sorted(by_source.items())},
        "per_domain": {domain: _recall_summary(domain_rows) for domain, domain_rows in sorted(by_domain.items())},
        "hard_negative_types": dict(Counter(row.get("hard_negative_type") or "unknown" for row in rows)),
    }


class _AuditAccumulator:
    def __init__(self) -> None:
        self.overall = _RecallCounter()
        self.by_source: dict[str, _RecallCounter] = defaultdict(_RecallCounter)
        self.by_domain: dict[str, _RecallCounter] = defaultdict(_RecallCounter)
        self.chemical = _RecallCounter()
        self.enzymatic = _RecallCounter()
        self.rhea_retrorules = _RecallCounter()
        self.hard_negative_types: Counter[str] = Counter()

    def add(self, row: dict[str, Any]) -> None:
        source = str(row.get("source") or "unknown")
        domain = _domain(row)
        self.overall.add(row)
        self.by_source[source].add(row)
        self.by_domain[domain].add(row)
        if domain == "all_chemical":
            self.chemical.add(row)
        if domain == "all_enzymatic":
            self.enzymatic.add(row)
        if source.lower() in {"rhea", "retrorules"}:
            self.rhea_retrorules.add(row)
        self.hard_negative_types.update([row.get("hard_negative_type") or "unknown"])

    def to_dict(self) -> dict[str, Any]:
        return {
            "overall": self.overall.to_dict(),
            "chemical": self.chemical.to_dict(),
            "enzymatic": self.enzymatic.to_dict(),
            "rhea_retrorules": self.rhea_retrorules.to_dict(),
            "per_source": {source: counter.to_dict() for source, counter in sorted(self.by_source.items())},
            "per_domain": {domain: counter.to_dict() for domain, counter in sorted(self.by_domain.items())},
            "hard_negative_types": dict(self.hard_negative_types),
        }


class _RecallCounter:
    def __init__(self) -> None:
        self.n = 0
        self.exact = 0
        self.gt = 0

    def add(self, row: dict[str, Any]) -> None:
        labels = row.get("recall_labels") or {}
        self.n += 1
        self.exact += int(bool(labels.get("candidate_exact_reaction_in_pool")))
        self.gt += int(bool(labels.get("candidate_gt_reactant_in_pool")))

    def to_dict(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "candidate_exact_reaction_in_pool": self.exact / max(self.n, 1),
            "candidate_gt_reactant_in_pool": self.gt / max(self.n, 1),
        }


def validate_proposal_recall_row(row: dict[str, Any]) -> None:
    missing = [field for field in REQUIRED_FIELDS if field not in row]
    if missing:
        raise ValueError(f"proposal recall row missing fields: {missing}")
    if not isinstance(row.get("candidate_pool"), list):
        raise ValueError("candidate_pool must be a list")
    if not isinstance(row.get("positive_candidate_indices"), list):
        raise ValueError("positive_candidate_indices must be a list")


def _recall_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    exact = sum(1 for row in rows if (row.get("recall_labels") or {}).get("candidate_exact_reaction_in_pool"))
    gt = sum(1 for row in rows if (row.get("recall_labels") or {}).get("candidate_gt_reactant_in_pool"))
    return {
        "n": n,
        "candidate_exact_reaction_in_pool": exact / max(n, 1),
        "candidate_gt_reactant_in_pool": gt / max(n, 1),
    }


def _iter_jsonl(path: Path):
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def _iter_chunks(path: Path, chunk_size: int, max_rows: int | None):
    chunk = []
    for idx, row in enumerate(_iter_jsonl(path)):
        if max_rows is not None and idx >= max_rows:
            break
        chunk.append(row)
        if len(chunk) >= chunk_size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def _parallel_rows(
    candidate_pools: Path,
    *,
    step_by_id: dict[str, dict[str, Any]],
    workers: int,
    chunk_size: int,
    max_rows: int | None,
    derive_gt_from_pool: bool,
):
    workers = max(1, int(workers or 1))
    chunk_size = max(1, int(chunk_size or 1))
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_worker,
        initargs=(step_by_id, derive_gt_from_pool),
    ) as pool:
        for rows in pool.map(
            _process_pool_chunk,
            _iter_chunks(candidate_pools, chunk_size, max_rows),
            chunksize=1,
        ):
            yield rows


def _init_worker(step_by_id: dict[str, dict[str, Any]], derive_gt_from_pool: bool) -> None:
    global _WORKER_STEP_BY_ID, _WORKER_DERIVE_GT_FROM_POOL
    _WORKER_STEP_BY_ID = step_by_id
    _WORKER_DERIVE_GT_FROM_POOL = bool(derive_gt_from_pool)


def _process_pool_chunk(pools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for pool in pools:
        step = _step_from_pool(pool) if _WORKER_DERIVE_GT_FROM_POOL else _WORKER_STEP_BY_ID.get(str(pool.get("external_step_id") or ""))
        if step is None:
            step = _step_from_pool(pool)
        row = proposal_recall_row(step, pool)
        if row:
            rows.append(row)
    return rows


def _can_smiles(smiles: str | None) -> str:
    key = str(smiles or "")
    cached = _CAN_SMILES_CACHE.get(key)
    if cached is None:
        cached = canonical_smiles(key)
        _CAN_SMILES_CACHE[key] = cached
    return cached


def _can_reaction(rxn: str | None) -> str:
    key = str(rxn or "")
    cached = _CAN_RXN_CACHE.get(key)
    if cached is None:
        cached = canonical_reaction(key)
        _CAN_RXN_CACHE[key] = cached
    return cached


def _rxn_reactants(rxn: str | None) -> set[str]:
    key = str(rxn or "")
    cached = _RXN_REACTANTS_CACHE.get(key)
    if cached is None:
        cached = reaction_reactants(key)
        _RXN_REACTANTS_CACHE[key] = cached
    return set(cached)


def _normalize_candidate_item(item: dict[str, Any], *, default_product: str) -> dict[str, Any]:
    candidate = dict((item or {}).get("candidate") or item or {})
    rxn = candidate.get("rxn_smiles") or candidate.get("reaction_smiles") or ""
    if not rxn and candidate.get("main_reactant"):
        lhs = ".".join([candidate.get("main_reactant"), *list(candidate.get("aux_reactants") or [])])
        rxn = f"{lhs}>>{default_product}"
    candidate["rxn_smiles"] = _can_reaction(rxn) or rxn
    candidate["reaction_smiles"] = candidate["rxn_smiles"]
    candidate.setdefault("source", (item or {}).get("source") or "unknown")
    candidate.setdefault("rank", (item or {}).get("rank") or candidate.get("rank") or 0)
    return candidate


def _candidate_reactants(candidate: dict[str, Any]) -> set[str]:
    out = set()
    if candidate.get("main_reactant"):
        out.add(_can_smiles(candidate["main_reactant"]))
    for smi in candidate.get("aux_reactants") or []:
        out.add(_can_smiles(smi))
    out.update(_rxn_reactants(candidate.get("rxn_smiles") or candidate.get("reaction_smiles") or ""))
    return {smi for smi in out if smi}


def _canonical_set(values: Any) -> set[str]:
    if isinstance(values, str):
        values = values.split(".")
    return {_can_smiles(str(value)) for value in (values or []) if _can_smiles(str(value))}


def _conditions_from_step(step: dict[str, Any]) -> dict[str, Any]:
    candidate = step.get("candidate") or {}
    return {
        "T": step.get("T", candidate.get("T")),
        "pH": step.get("pH", candidate.get("pH")),
        "solvent": step.get("solvent", candidate.get("solvent", "")),
        "catalyst": step.get("catalyst", candidate.get("catalyst", "")),
    }


def _hard_negative_type(pool: dict[str, Any]) -> str:
    labels = {
        str((item or {}).get("label_type") or "")
        for item in pool.get("candidates") or []
        if float((item or {}).get("label") or 0.0) <= 0.0
    }
    if "external_hard_negative" in labels:
        return "source_type_ec_hard_negative"
    return sorted(labels)[0] if labels else ""


def _domain(row: dict[str, Any]) -> str:
    source = str(row.get("source") or "").lower()
    ec = str(row.get("ec") or "")
    if source in {"uspto50k", "chemical", "retrochimera"} or not ec:
        return "all_chemical"
    if source in {"ecreact", "enzymatic_retro_data", "rhea", "retrorules"} or ec:
        return "all_enzymatic"
    return "chemoenzymatic"


def _step_from_pool(pool: dict[str, Any]) -> dict[str, Any]:
    positive = next((item for item in pool.get("candidates") or [] if float((item or {}).get("label") or 0.0) > 0.0), {})
    candidate = positive.get("candidate") or {}
    return {
        "product": pool.get("product") or pool.get("target_smiles") or "",
        "reactants": _candidate_reactants(candidate),
        "reaction_smiles": candidate.get("rxn_smiles") or candidate.get("reaction_smiles") or "",
        "reaction_type": pool.get("reaction_type") or candidate.get("reaction_type") or candidate.get("type") or "",
        "ec": pool.get("ec") or candidate.get("ec") or "",
        "source": pool.get("external_source") or candidate.get("source") or "",
        "candidate": candidate,
    }


def _report_markdown(manifest: dict[str, Any]) -> str:
    audit = manifest.get("audit") or {}
    overall = audit.get("overall") or {}
    return "\n".join(
        [
            "# Proposal Recall Pack",
            "",
            f"- rows: `{manifest.get('counts', {}).get('rows', 0)}`",
            f"- candidate exact reaction recall: `{overall.get('candidate_exact_reaction_in_pool')}`",
            f"- candidate GT reactant recall: `{overall.get('candidate_gt_reactant_in_pool')}`",
            "",
            "Rows follow the proposal recall schema and are suitable for source-specific ranker/gate training.",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build proposal recall pack from external step pairs and candidate pools")
    parser.add_argument("--external-step-pairs", default="results/shared/external_step_pairs/full_20260507/external_step_pairs.jsonl")
    parser.add_argument("--candidate-pools", default="results/shared/external_candidate_pools/hardneg_full_20260507/external_candidate_pools.jsonl")
    parser.add_argument("--output-dir", default="results/shared/proposal_recall/current")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--derive-gt-from-pool", action="store_true")
    args = parser.parse_args()
    manifest = build_proposal_recall_pack(
        external_step_pairs=Path(args.external_step_pairs),
        candidate_pools=Path(args.candidate_pools),
        output_dir=Path(args.output_dir),
        max_rows=args.max_rows,
        workers=args.workers,
        chunk_size=args.chunk_size,
        derive_gt_from_pool=args.derive_gt_from_pool,
    )
    print(json.dumps({"counts": manifest["counts"], "audit": manifest["audit"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
