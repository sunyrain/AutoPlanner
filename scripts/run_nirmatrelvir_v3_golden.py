"""Run the hash-bound Nirmatrelvir dual-route acceptance case without models."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cascade_planner.harness.deterministic_literature_registry import (  # noqa: E402
    build_deterministic_literature_resolvers,
    compile_deterministic_literature_step_registry,
)
from cascade_planner.harness.deterministic_resolver_cache import (  # noqa: E402
    DeterministicResolverCache,
)
from cascade_planner.runtime.run_metrics import (  # noqa: E402
    current_run_metrics,
    record_run_metrics,
    run_metric_stage,
)
from scripts.compile_source_route_portfolio import (  # noqa: E402
    compile_source_route_portfolio,
)
from scripts.replay_deterministic_literature_registry import (  # noqa: E402
    candidate_steps_from_blackboard,
    candidate_steps_from_manifest,
)


DEFAULT_GOLDEN = (
    ROOT / "config/examples/nirmatrelvir_v3_golden_acceptance.json"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--golden", type=Path, default=DEFAULT_GOLDEN)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results/shared/nirmatrelvir_v3_golden",
    )
    parser.add_argument("--timeout-s", type=float, default=30.0)
    args = parser.parse_args()

    summary = run_golden_case(
        golden_path=args.golden,
        output_dir=args.output_dir,
        timeout_s=max(1.0, float(args.timeout_s)),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


@record_run_metrics
def run_golden_case(
    *,
    golden_path: Path = DEFAULT_GOLDEN,
    output_dir: Path,
    timeout_s: float = 30.0,
    resolver_cache_root: Path | None = None,
) -> dict[str, Any]:
    """Reconstruct both sources, replay stock, and enforce expected metrics."""

    with run_metric_stage("golden.load_contract", category="input"):
        golden = _read_object(golden_path)
    if golden.get("schema_version") != "retrosynthesis_golden_acceptance.v1":
        raise SystemExit("unsupported golden acceptance schema")
    with run_metric_stage("golden.verify_sources", category="evidence"):
        _verify_source_artifacts(golden)
        manifest_path = _repo_path(golden.get("candidate_manifest"))
        stock_path = _repo_path(golden.get("stock_snapshots"))
        manifest = _read_object(manifest_path)

    output = output_dir.resolve()
    registry_root = output / "registries"
    source_summaries: list[dict[str, Any]] = []
    metrics = current_run_metrics()
    if metrics is not None:
        metrics.bind_case_id(str(golden.get("case_id") or ""))
        metrics.gauge("golden.source_count", len(manifest.get("sources") or []))
        metrics.gauge("model_invocations", 0)
    persistent_cache = DeterministicResolverCache(
        resolver_cache_root or output / ".autoplanner" / "artifacts",
        authority_id="autoplanner.opsin_pubchem_source_text.v8",
        opsin_base_url="https://opsin.ch.cam.ac.uk/opsin",
        pubchem_base_url="https://pubchem.ncbi.nlm.nih.gov/rest/pug",
    )
    structure_resolver, candidate_name_resolver = build_deterministic_literature_resolvers(
        timeout_s=timeout_s,
        persistent_cache=persistent_cache,
    )
    for index, raw_source in enumerate(manifest.get("sources") or []):
        if not isinstance(raw_source, dict):
            continue
        source = dict(raw_source)
        source_ref = str(source.get("source_ref") or "")
        source_dir = registry_root / f"source-{index + 1}"
        source_dir.mkdir(parents=True, exist_ok=True)
        with run_metric_stage(
            "golden.materialize_source_candidates",
            category="evidence",
            attributes={"source_index": index + 1, "source_ref": source_ref},
        ):
            if source.get("candidate_blackboard"):
                candidates = candidate_steps_from_blackboard(
                    _read_object(_repo_path(source["candidate_blackboard"])),
                    source_ref=source_ref,
                )
            else:
                candidates = candidate_steps_from_manifest(
                    manifest,
                    source_ref=source_ref,
                )
        if not candidates:
            raise SystemExit(f"no deterministic candidates for {source_ref}")
        with run_metric_stage(
            "golden.compile_source_registry",
            category="validation",
            attributes={
                "source_index": index + 1,
                "source_ref": source_ref,
                "candidate_count": len(candidates),
            },
        ):
            audit = compile_deterministic_literature_step_registry(
                candidates,
                registry_path=(
                    source_dir / "trusted_literature_step_registry.generated.json"
                ),
                audit_path=(
                    source_dir / "deterministic_literature_registry_audit.json"
                ),
                timeout_s=timeout_s,
                structure_resolver=structure_resolver,
                candidate_name_resolver=candidate_name_resolver,
            )
        source_summaries.append(
            {
                "source_ref": source_ref,
                "candidate_step_count": len(candidates),
                "approved_binding_count": int(
                    audit.get("approved_binding_count") or 0
                ),
                "rejected_step_count": int(
                    audit.get("rejected_step_count") or 0
                ),
            }
        )
    cache_flush = persistent_cache.flush()
    if metrics is not None:
        metrics.gauge(
            "resolver.persistent_cache_entry_count",
            int(cache_flush.get("entry_count") or 0),
        )

    with run_metric_stage("golden.compile_portfolio", category="portfolio"):
        portfolio_summary = compile_source_route_portfolio(
            candidate_manifest_path=manifest_path,
            registry_root=registry_root,
            stock_snapshots_path=stock_path,
            output_dir=output / "portfolio",
        )
    with run_metric_stage("golden.enforce_acceptance", category="acceptance"):
        _enforce_expected(portfolio_summary, dict(golden.get("expected") or {}))
    if portfolio_summary.get("accepted") is not True:
        raise SystemExit(
            "Nirmatrelvir golden acceptance failed: "
            + ",".join(portfolio_summary.get("reasons") or [])
        )
    summary = {
        "schema_version": "retrosynthesis_golden_run.v1",
        "case_id": str(golden.get("case_id") or ""),
        "accepted": True,
        "source_replays": source_summaries,
        "portfolio": portfolio_summary,
        "model_invocations": 0,
        "output_dir": str(output),
    }
    if metrics is not None:
        for source in source_summaries:
            metrics.increment(
                "golden.approved_source_bindings",
                int(source.get("approved_binding_count") or 0),
            )
        for key in (
            "hyperedge_count",
            "complete_route_count",
            "selected_route_count",
            "stock_terminal_count",
        ):
            if key in portfolio_summary:
                metrics.gauge(key, portfolio_summary[key])
    _write_json(output / "golden_run_summary.json", summary)
    return summary


def _verify_source_artifacts(golden: dict[str, Any]) -> None:
    for raw in golden.get("source_documents") or []:
        if not isinstance(raw, dict):
            continue
        source = dict(raw)
        for path_key, digest_key in (
            ("artifact_path", "artifact_sha256"),
            ("text_companion_path", "text_companion_sha256"),
        ):
            if not source.get(path_key):
                continue
            path = _repo_path(source[path_key])
            if not path.is_file():
                raise SystemExit(f"required local source artifact is missing: {path}")
            if _sha256_file(path) != str(source.get(digest_key) or "").lower():
                raise SystemExit(f"source artifact digest mismatch: {path}")


def _enforce_expected(
    summary: dict[str, Any],
    expected: dict[str, Any],
) -> None:
    field_map = {
        "approved_source_step_count": "approved_source_step_count",
        "unique_reaction_hyperedge_count": "hyperedge_count",
        "complete_route_count": "complete_route_count",
        "selected_route_count": "selected_route_count",
        "stock_terminal_count": "stock_terminal_count",
        "model_invocations": "model_invocations",
    }
    mismatches = [
        f"{expected_key}:expected={expected.get(expected_key)!r},"
        f"observed={summary.get(summary_key)!r}"
        for expected_key, summary_key in field_map.items()
        if expected_key in expected
        and expected.get(expected_key) != summary.get(summary_key)
    ]
    expected_groups = sorted(expected.get("independent_support_groups") or [])
    observed_groups = sorted(summary.get("independent_support_groups") or [])
    if expected_groups != observed_groups:
        mismatches.append(
            "independent_support_groups:"
            f"expected={expected_groups!r},observed={observed_groups!r}"
        )
    if mismatches:
        raise SystemExit("golden metrics changed: " + "; ".join(mismatches))


def _repo_path(value: Any) -> Path:
    path = Path(str(value or "")).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()


def _read_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.resolve().read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SystemExit(f"unable to read JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise SystemExit(f"JSON must be an object: {path}")
    return payload


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
