#!/usr/bin/env python3
"""Build the provisional literature-anchored retrosynthesis dataset.

The builder deliberately separates target-only planner inputs from evaluator-only
literature metadata.  It also records the SynthEx literature-145 cohort at the
cohort level without inventing row-level identities that are not publicly released.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import re
import urllib.request
from urllib.parse import quote
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


SYNTHATLAS_DATA_VERSION = "20260809-00e8823-5a1cf6"
SYNTHATLAS_INDEX_URL = (
    "https://data.synthatlas.xyz/"
    f"{SYNTHATLAS_DATA_VERSION}/index.json"
)
CROSSREF_QUERY_TERMS = {
    "total_synthesis": "total synthesis",
    "collective_synthesis": "collective synthesis",
    "biomimetic_synthesis": "biomimetic synthesis natural product",
    "bioinspired_synthesis": "bioinspired synthesis natural product",
    "concise_synthesis": "concise synthesis natural product",
}


def crossref_url(term: str) -> str:
    return (
        f"https://api.crossref.org/works?query.title={quote(term)}"
        "&filter=from-pub-date:2025-01-01,until-pub-date:2026-09-01,"
        "type:journal-article&rows=1000"
        "&select=DOI,title,container-title,published-online,published-print,"
        "URL,author,abstract"
    )
SYNTHEX_PAPER_URL = "https://arxiv.org/abs/2608.07454"
DOCUMENTED_MODEL_CUTOFF = "2026-02-16"
SYNTHEX_PREPRINT_DATE = "2026-08-09"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmarks/literature_strategy_rediscovery_v0_1"),
    )
    parser.add_argument(
        "--synthatlas-index-cache",
        type=Path,
        default=Path("tmp/synthatlas-index.json"),
    )
    parser.add_argument(
        "--crossref-cache",
        type=Path,
        default=Path("tmp/crossref-total-synthesis-2025-2026.json"),
    )
    return parser.parse_args()


def fetch_json(url: str, cache: Path) -> tuple[Any, str]:
    if not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "AutoPlanner-literature-benchmark/0.1"},
        )
        with urllib.request.urlopen(request, timeout=90) as response:
            cache.write_bytes(response.read())
    payload = cache.read_bytes()
    return json.loads(payload), hashlib.sha256(payload).hexdigest()


def clean_text(value: Any) -> str:
    if isinstance(value, list):
        value = " ".join(str(part) for part in value)
    value = html.unescape(str(value or ""))
    value = re.sub(r"<[^>]+>", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def date_from_crossref(item: dict[str, Any]) -> str:
    for field in ("published-online", "published-print"):
        parts = ((item.get(field) or {}).get("date-parts") or [[]])[0]
        if parts:
            year = int(parts[0])
            month = int(parts[1]) if len(parts) > 1 else 1
            day = int(parts[2]) if len(parts) > 2 else 1
            return f"{year:04d}-{month:02d}-{day:02d}"
    return ""


def normalize_title(title: str) -> str:
    title = title.lower().replace("−", "-").replace("–", "-")
    return re.sub(r"[^a-z0-9]+", " ", title).strip()


def classify_title(title: str) -> tuple[str, str]:
    low = normalize_title(title)
    if re.search(r"\b(correction|erratum|corrigendum)\b", low):
        return "exclude_correction", "correction or erratum"
    if re.search(
        r"\b(advances|advanced strategies|progress|recent highlights|recent update|"
        r"overview|review|applications?|assessment metric|tales of|improvements in)\b",
        low,
    ):
        return "exclude_review", "review, perspective, or methods overview"
    if re.search(r"\b(toward|towards|studies toward|approach toward)\b", low):
        return "exclude_incomplete", "toward rather than completed synthesis"
    if re.search(r"\b(total synthesis|formal synthesis)\b", low):
        if re.search(r"\b(protein|gene|nanoparticle|materials?)\b", low):
            return "manual_scope_review", "title may be outside small-molecule synthesis"
        return "needs_target_extraction", "primary synthesis candidate from title"
    if re.search(
        r"\b(collective|concise|divergent|enantioselective|asymmetric|bioinspired|"
        r"biomimetic|stereoselective|stereodivergent) synthesis of\b",
        low,
    ):
        return (
            "possible_omission_needs_scope_review",
            "completed-synthesis wording without an explicit total-synthesis claim",
        )
    return "manual_title_review", "query match without explicit total/formal synthesis"


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def stable_id(prefix: str, *parts: str) -> str:
    digest = hashlib.sha256("\0".join(parts).encode("utf-8")).hexdigest()[:16]
    return f"{prefix}-{digest}"


def load_strategy_candidates(repo_root: Path) -> list[dict[str, Any]]:
    base = repo_root / "tmp/pdfs/nature-2026-route-audit"
    rows: list[dict[str, Any]] = []
    for path in sorted(base.glob("strategy-screen-batch*.json")):
        for item in json.loads(path.read_text(encoding="utf-8")):
            rows.append(
                {
                    "benchmark_id": stable_id("target", item.get("smiles", "")),
                    "target_smiles": item.get("smiles", ""),
                    "cohort": "local_strategy_screen_candidate",
                    "source_record_id": item.get("slug", ""),
                    "source_url": f"https://doi.org/{item.get('doi', '')}",
                    "input_status": "runnable_target_only",
                    "notes": f"Recovered from {path.name}",
                    "_name": item.get("name", ""),
                    "_doi": item.get("doi", ""),
                }
            )
    return rows


def load_verified_cases(repo_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    snapshot_path = repo_root / "paper/arxiv/data/evidence_snapshot.json"
    if not snapshot_path.exists():
        return [], []
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    smiles_by_case: dict[str, str] = {}
    for path in (repo_root / "benchmarks").glob("*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
        for case in payload.get("cases", []):
            smiles_by_case[case.get("case_id", "")] = case.get("target_smiles", "")

    public_rows: list[dict[str, Any]] = []
    evaluator_rows: list[dict[str, Any]] = []
    for item in snapshot.get("cases", []):
        publication = item.get("publication", {})
        run = item.get("run", {})
        smiles = smiles_by_case.get(run.get("run_id", ""), "")
        benchmark_id = stable_id("target", smiles)
        public_rows.append(
            {
                "benchmark_id": benchmark_id,
                "target_smiles": smiles,
                "cohort": "local_literature_verified_case",
                "source_record_id": run.get("run_id", ""),
                "source_url": "",
                "input_status": "runnable_target_only" if smiles else "missing_target_smiles",
                "notes": "Reference metadata intentionally withheld from planner input",
            }
        )
        evaluator_rows.append(
            {
                "benchmark_id": benchmark_id,
                "target_name": item.get("label", ""),
                "target_smiles": smiles,
                "doi": publication.get("doi", ""),
                "publication_title": publication.get("title", ""),
                "journal": publication.get("journal", ""),
                "publication_date": publication.get("online_publication_date", ""),
                "after_documented_model_cutoff": publication.get(
                    "after_documented_model_cutoff", False
                ),
                "reference_strategy_summary": item.get("retrospective_match", {}).get(
                    "summary", ""
                ),
                "reference_match_level": item.get("retrospective_match", {}).get(
                    "level", ""
                ),
                "selection_is_post_hoc": item.get("retrospective_match", {}).get(
                    "selection_is_post_hoc_case_study", False
                ),
                "scientific_disposition": run.get("scientific_disposition", ""),
                "reaction_validated_skeletons": run.get(
                    "reaction_validated_skeletons", 0
                ),
                "source_url": f"https://doi.org/{publication.get('doi', '')}",
            }
        )
    return public_rows, evaluator_rows


def synthatlas_targets(index: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for route in index:
        grouped[str(route.get("target", ""))].append(route)
    rows: list[dict[str, Any]] = []
    for target_hash, routes in sorted(grouped.items()):
        first = routes[0]
        rows.append(
            {
                "benchmark_id": stable_id("target", first.get("target_smiles", "")),
                "target_smiles": first.get("target_smiles", ""),
                "cohort": "synthatlas_frontier_1098",
                "source_record_id": target_hash,
                "source_url": f"https://synthatlas.epfl.ch/#/routes/{first.get('id', '')}",
                "input_status": "runnable_target_only",
                "notes": (
                    f"{len(routes)} published SynthEx route artifact(s); "
                    "route artifacts are comparator outputs, not ground truth"
                ),
                "_name": first.get("name", ""),
                "_origin_type": first.get("origin_type", ""),
                "_route_count": len(routes),
                "_any_solved": any(bool(route.get("solved")) for route in routes),
            }
        )
    return rows


def crossref_candidates(payload: dict[str, Any]) -> list[dict[str, Any]]:
    items = payload.get("message", {}).get("items", [])
    rows: list[dict[str, Any]] = []
    seen_doi: set[str] = set()
    title_groups: dict[str, str] = {}
    for item in items:
        doi = clean_text(item.get("DOI", "")).lower()
        title = clean_text(item.get("title", ""))
        if not doi or not title or doi in seen_doi:
            continue
        seen_doi.add(doi)
        normalized = normalize_title(title)
        group = title_groups.setdefault(normalized, stable_id("title", normalized))
        status, reason = classify_title(title)
        journal = clean_text(item.get("container-title", ""))
        if journal.lower() == "synfacts":
            status = "exclude_secondary_summary"
            reason = "secondary Synfacts synthesis summary, not the primary report"
        published = date_from_crossref(item)
        authors = item.get("author") or []
        first_author = ""
        if authors:
            first_author = clean_text(
                " ".join(
                    part
                    for part in (authors[0].get("given", ""), authors[0].get("family", ""))
                    if part
                )
            )
        rows.append(
            {
                "candidate_id": stable_id("paper", doi),
                "doi": doi,
                "title": title,
                "journal": journal,
                "publication_date": published,
                "first_author": first_author,
                "source_url": clean_text(item.get("URL", "")) or f"https://doi.org/{doi}",
                "source_queries": ";".join(sorted(item.get("_source_queries", []))),
                "automated_screening_status": status,
                "automated_screening_reason": reason,
                "duplicate_title_group": group,
                "after_documented_model_cutoff": bool(
                    published and published > DOCUMENTED_MODEL_CUTOFF
                ),
                "after_synthex_preprint": bool(
                    published and published > SYNTHEX_PREPRINT_DATE
                ),
                "target_extraction_status": "not_started",
                "target_smiles": "",
                "human_review_status": "pending",
            }
        )
    rows.sort(key=lambda row: (row["publication_date"], row["doi"]))
    return rows


def merge_crossref_payloads(
    payloads: list[tuple[str, dict[str, Any]]],
) -> dict[str, Any]:
    merged: dict[str, dict[str, Any]] = {}
    for query_id, payload in payloads:
        for item in payload.get("message", {}).get("items", []):
            doi = clean_text(item.get("DOI", "")).lower()
            if not doi:
                continue
            if doi not in merged:
                merged[doi] = dict(item)
                merged[doi]["_source_queries"] = []
            merged[doi]["_source_queries"].append(query_id)
    return {"message": {"items": list(merged.values())}}


def select_new_since_preprint(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if (
            row["after_synthex_preprint"]
            and row["automated_screening_status"] == "needs_target_extraction"
        ):
            grouped[row["duplicate_title_group"]].append(row)

    selected: list[dict[str, Any]] = []
    for variants in grouped.values():
        # Wiley frequently registers the same article as German Angewandte and
        # English Angewandte International Edition records. Prefer the English DOI.
        variants.sort(
            key=lambda row: (
                0 if row["doi"].startswith("10.1002/anie.") else 1,
                row["doi"],
            )
        )
        chosen = dict(variants[0])
        chosen["duplicate_doi_variants"] = ";".join(
            variant["doi"] for variant in variants[1:]
        )
        selected.append(chosen)
    selected.sort(key=lambda row: (row["publication_date"], row["doi"]))
    return selected


def select_possible_omissions(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if (
            row["after_documented_model_cutoff"]
            and row["automated_screening_status"]
            == "possible_omission_needs_scope_review"
        ):
            grouped[row["duplicate_title_group"]].append(row)
    selected: list[dict[str, Any]] = []
    for variants in grouped.values():
        variants.sort(
            key=lambda row: (
                0 if row["doi"].startswith("10.1002/anie.") else 1,
                row["doi"],
            )
        )
        chosen = dict(variants[0])
        chosen["duplicate_doi_variants"] = ";".join(
            variant["doi"] for variant in variants[1:]
        )
        selected.append(chosen)
    selected.sort(key=lambda row: (row["publication_date"], row["doi"]))
    return selected


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    output_dir = (repo_root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    index, index_sha = fetch_json(SYNTHATLAS_INDEX_URL, repo_root / args.synthatlas_index_cache)
    crossref_payloads: list[tuple[str, dict[str, Any]]] = []
    crossref_sources: list[dict[str, Any]] = []
    for query_id, term in CROSSREF_QUERY_TERMS.items():
        url = crossref_url(term)
        cache = (
            repo_root / args.crossref_cache
            if query_id == "total_synthesis"
            else repo_root
            / args.crossref_cache.parent
            / f"crossref-{query_id}-2025-2026.json"
        )
        payload, source_sha = fetch_json(url, cache)
        crossref_payloads.append((query_id, payload))
        crossref_sources.append(
            {
                "id": f"crossref_{query_id}",
                "url": url,
                "sha256": source_sha,
                "query_window": "2025-01-01/2026-09-01",
            }
        )
    crossref = merge_crossref_payloads(crossref_payloads)

    frontier = synthatlas_targets(index)
    screen_rows = load_strategy_candidates(repo_root)
    verified_public, evaluator_rows = load_verified_cases(repo_root)
    target_membership_rows = frontier + screen_rows + verified_public
    target_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in target_membership_rows:
        target_groups[row["benchmark_id"]].append(row)
    target_rows: list[dict[str, Any]] = []
    for benchmark_id, memberships in sorted(target_groups.items()):
        target_rows.append(
            {
                "benchmark_id": benchmark_id,
                "target_smiles": memberships[0]["target_smiles"],
                "cohort": ";".join(
                    sorted({membership["cohort"] for membership in memberships})
                ),
                "input_status": (
                    "runnable_target_only"
                    if memberships[0]["target_smiles"]
                    else "missing_target_smiles"
                ),
            }
        )
    literature_rows = crossref_candidates(crossref)
    new_since_preprint = select_new_since_preprint(literature_rows)
    possible_omissions = select_possible_omissions(literature_rows)

    public_fields = ["benchmark_id", "target_smiles", "cohort", "input_status"]
    evaluator_fields = [
        "benchmark_id",
        "target_name",
        "target_smiles",
        "doi",
        "publication_title",
        "journal",
        "publication_date",
        "after_documented_model_cutoff",
        "reference_strategy_summary",
        "reference_match_level",
        "selection_is_post_hoc",
        "scientific_disposition",
        "reaction_validated_skeletons",
        "source_url",
        "source_queries",
    ]
    literature_fields = [
        "candidate_id",
        "doi",
        "title",
        "journal",
        "publication_date",
        "first_author",
        "source_url",
        "automated_screening_status",
        "automated_screening_reason",
        "duplicate_title_group",
        "after_documented_model_cutoff",
        "after_synthex_preprint",
        "target_extraction_status",
        "target_smiles",
        "human_review_status",
    ]
    write_csv(output_dir / "target_only.csv", target_rows, public_fields)
    evaluator_by_id = {row["benchmark_id"]: row for row in evaluator_rows}
    provenance_rows = []
    for row in target_membership_rows:
        evaluator = evaluator_by_id.get(row["benchmark_id"], {})
        provenance_rows.append(
            {
                "benchmark_id": row["benchmark_id"],
                "cohort": row["cohort"],
                "source_record_id": row.get("source_record_id", ""),
                "target_name": row.get("_name", "") or evaluator.get("target_name", ""),
                "publication_doi": row.get("_doi", "") or evaluator.get("doi", ""),
                "source_url": row.get("source_url", "") or evaluator.get("source_url", ""),
                "origin_type": row.get("_origin_type", ""),
                "published_route_artifact_count": row.get("_route_count", ""),
                "published_any_stock_solved": row.get("_any_solved", ""),
                "notes": row.get("notes", ""),
            }
        )
    provenance_fields = [
        "benchmark_id",
        "cohort",
        "source_record_id",
        "target_name",
        "publication_doi",
        "source_url",
        "origin_type",
        "published_route_artifact_count",
        "published_any_stock_solved",
        "notes",
    ]
    write_csv(output_dir / "target_provenance.csv", provenance_rows, provenance_fields)
    write_csv(output_dir / "evaluator_only.csv", evaluator_rows, evaluator_fields)
    write_csv(output_dir / "literature_candidates.csv", literature_rows, literature_fields)
    write_csv(
        output_dir / "new_since_synthex_preprint.csv",
        new_since_preprint,
        literature_fields + ["duplicate_doi_variants"],
    )
    write_csv(
        output_dir / "possible_omissions.csv",
        possible_omissions,
        literature_fields + ["duplicate_doi_variants"],
    )

    cohorts = [
        {
            "cohort_id": "synthex_literature_145",
            "reported_count": 145,
            "recovered_row_count": 0,
            "access_status": "aggregate_only_not_publicly_recovered",
            "intended_use": "literature-anchored strategy rediscovery",
            "source_url": SYNTHEX_PAPER_URL,
            "notes": (
                "Reported count is curated target/route entries; independent paper count "
                "is unknown. Do not invent identities."
            ),
        },
        {
            "cohort_id": "synthex_literature_usable_key_step_138",
            "reported_count": 138,
            "recovered_row_count": 0,
            "access_status": "aggregate_only_not_publicly_recovered",
            "intended_use": "human key-step reference",
            "source_url": SYNTHEX_PAPER_URL,
            "notes": "Subset of literature-145 with a reconstructable key step.",
        },
        {
            "cohort_id": "synthex_route_returned_70",
            "reported_count": 70,
            "recovered_row_count": 0,
            "access_status": "aggregate_only_not_publicly_recovered",
            "intended_use": "published SynthEx coverage comparator",
            "source_url": SYNTHEX_PAPER_URL,
            "notes": "Conditional subset; keep all 145 in the denominator.",
        },
        {
            "cohort_id": "synthex_strategy_congruent_47",
            "reported_count": 47,
            "recovered_row_count": 0,
            "access_status": "aggregate_only_not_publicly_recovered",
            "intended_use": "published shared-strategy key-step comparator",
            "source_url": SYNTHEX_PAPER_URL,
            "notes": "LLM-judged congruence; not a ground-truth label.",
        },
        {
            "cohort_id": "synthatlas_frontier_1098",
            "reported_count": 1098,
            "recovered_row_count": len(frontier),
            "access_status": "public_target_rows_recovered",
            "intended_use": "open-world natural-product reach",
            "source_url": "https://synthatlas.epfl.ch",
            "notes": "SynthEx routes are comparator artifacts, not experimental truth.",
        },
        {
            "cohort_id": "local_strategy_screen_candidate",
            "reported_count": len(screen_rows),
            "recovered_row_count": len(screen_rows),
            "access_status": "local_rows_recovered",
            "intended_use": "protocol development and target extraction",
            "source_url": "",
            "notes": "Includes reaction-method exemplars; not all are total syntheses.",
        },
        {
            "cohort_id": "local_literature_verified_case",
            "reported_count": len(verified_public),
            "recovered_row_count": len(verified_public),
            "access_status": "local_rows_recovered",
            "intended_use": "pilot evaluator and provenance checks",
            "source_url": "",
            "notes": "Post-hoc positive cases; not a hit-rate sample.",
        },
        {
            "cohort_id": "crossref_recent_total_synthesis_candidates",
            "reported_count": len(literature_rows),
            "recovered_row_count": len(literature_rows),
            "access_status": "public_metadata_recovered",
            "intended_use": "screening queue for omissions and newer reports",
            "source_url": crossref_url("total synthesis"),
            "notes": "Title-based candidate discovery requires human chemistry review.",
        },
    ]
    cohort_fields = [
        "cohort_id",
        "reported_count",
        "recovered_row_count",
        "access_status",
        "intended_use",
        "source_url",
        "notes",
    ]
    write_csv(output_dir / "cohorts.csv", cohorts, cohort_fields)

    status_counts: dict[str, int] = defaultdict(int)
    for row in literature_rows:
        status_counts[row["automated_screening_status"]] += 1
    manifest = {
        "schema_version": "literature_strategy_dataset.v0.1",
        "generated_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "claim_boundary": {
            "synthex_literature_145_row_level_recovered": False,
            "synthex_literature_145_count_unit": "reported_curated_targets_or_routes",
            "synthex_literature_145_independent_paper_count": None,
            "synthex_literature_145_public_row_identifiers_recovered": False,
            "synthatlas_routes_are_ground_truth": False,
            "crossref_candidates_are_human_curated": False,
            "local_verified_cases_are_post_hoc": True,
        },
        "counts": {
            "target_only_rows": len(target_rows),
            "target_only_distinct_smiles": len(target_rows),
            "target_cohort_memberships": len(target_membership_rows),
            "synthatlas_frontier_targets": len(frontier),
            "local_strategy_screen_candidates": len(screen_rows),
            "local_verified_cases": len(verified_public),
            "crossref_literature_candidates": len(literature_rows),
            "new_since_synthex_preprint_deduplicated": len(new_since_preprint),
            "possible_omissions_after_model_cutoff": len(possible_omissions),
            "crossref_screening_status": dict(sorted(status_counts.items())),
            "evaluator_reference_rows": len(evaluator_rows),
        },
        "sources": [
            {
                "id": "synthex_preprint",
                "url": SYNTHEX_PAPER_URL,
                "role": "aggregate cohort definitions and published evaluation protocol",
            },
            {
                "id": "synthatlas_index",
                "url": SYNTHATLAS_INDEX_URL,
                "sha256": index_sha,
                "data_version": SYNTHATLAS_DATA_VERSION,
            },
            *crossref_sources,
            {
                "id": "autoplanner_evidence_snapshot",
                "path": "paper/arxiv/data/evidence_snapshot.json",
                "role": "five local verified case metadata and claim boundaries",
            },
        ],
        "files": {
            "target_only": "target_only.csv",
            "target_provenance": "target_provenance.csv",
            "evaluator_only": "evaluator_only.csv",
            "literature_candidates": "literature_candidates.csv",
            "new_since_synthex_preprint": "new_since_synthex_preprint.csv",
            "possible_omissions": "possible_omissions.csv",
            "cohorts": "cohorts.csv",
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(output_dir)
    print(json.dumps(manifest["counts"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
