#!/usr/bin/env python3
"""Audit curated strategic disconnection source files."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


DEFAULT_DB_GLOB = "data/strategic_disconnections/strategic_disconnections*.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--db",
        type=Path,
        action="append",
        default=None,
        help=(
            "Strategic DB JSON file. May be repeated. "
            f"Default: audit files matching {DEFAULT_DB_GLOB!r}."
        ),
    )
    parser.add_argument("--json", action="store_true", help="Emit machine-readable audit JSON.")
    parser.add_argument("--output", type=Path, default=None, help="Optional output file.")
    parser.add_argument(
        "--fail-on-issues",
        action="store_true",
        help="Exit non-zero when duplicate IDs, missing coverage, or untraceable evidence are found.",
    )
    args = parser.parse_args()

    report = audit_databases(args.db)
    text = json.dumps(report, indent=2, ensure_ascii=False) if args.json else render_markdown(report)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    if args.fail_on_issues and not report["passed"]:
        raise SystemExit(1)


def audit_databases(paths: list[Path] | None = None) -> dict[str, Any]:
    db_paths = list(paths or sorted(Path().glob(DEFAULT_DB_GLOB)))
    if not db_paths:
        raise FileNotFoundError(DEFAULT_DB_GLOB)

    source_files: list[dict[str, Any]] = []
    records: dict[str, list[tuple[Path, dict[str, Any]]]] = {
        "families": [],
        "anchors": [],
        "disconnections": [],
    }
    source_notes = 0
    pptx_sources = 0
    for path in db_paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        counts = {
            "families": len(data.get("families", []) or []),
            "anchors": len(data.get("anchors", []) or []),
            "disconnections": len(data.get("disconnections", []) or []),
            "pptx_sources": len(data.get("pptx_sources", []) or []),
            "source_notes": len(data.get("source_notes", []) or []),
        }
        source_files.append(
            {
                "path": str(path),
                "schema_version": data.get("schema_version", ""),
                "status": data.get("status", ""),
                "counts": counts,
            }
        )
        source_notes += counts["source_notes"]
        pptx_sources += counts["pptx_sources"]
        for section in records:
            records[section].extend((path, item) for item in data.get(section, []) or [])

    duplicate_ids = _duplicate_ids(records)
    family_coverage = _family_coverage(records)
    coverage_issues = _coverage_issues(family_coverage)
    evidence = _evidence_audit(records["disconnections"])
    policy = _policy_audit(records["disconnections"])

    issue_count = (
        sum(len(items) for items in duplicate_ids.values())
        + len(coverage_issues)
        + len(evidence["disconnections_without_evidence"])
        + len(evidence["evidence_items_without_trace"])
    )
    return {
        "schema_version": "strategic_disconnection_source_audit.v1",
        "source_glob": DEFAULT_DB_GLOB if paths is None else "",
        "source_files": source_files,
        "totals": {
            "source_files": len(db_paths),
            "families": len(records["families"]),
            "anchors": len(records["anchors"]),
            "disconnections": len(records["disconnections"]),
            "pptx_sources": pptx_sources,
            "source_notes": source_notes,
        },
        "duplicate_ids": duplicate_ids,
        "family_coverage": family_coverage,
        "coverage_issues": coverage_issues,
        "evidence": evidence,
        "policy": policy,
        "issue_count": issue_count,
        "passed": issue_count == 0,
    }


def render_markdown(report: dict[str, Any]) -> str:
    totals = report["totals"]
    lines = [
        "# Strategic Disconnection Source Audit",
        "",
        f"- Passed: `{str(report['passed']).lower()}`",
        f"- Source files: `{totals['source_files']}`",
        f"- Families: `{totals['families']}`",
        f"- Anchors: `{totals['anchors']}`",
        f"- Disconnections: `{totals['disconnections']}`",
        f"- Issue count: `{report['issue_count']}`",
        "",
        "## Source Files",
    ]
    for source in report["source_files"]:
        counts = source["counts"]
        lines.append(
            "- `{path}` ({schema}): families={families}, anchors={anchors}, "
            "disconnections={disconnections}".format(
                path=source["path"],
                schema=source.get("schema_version") or "unknown",
                families=counts["families"],
                anchors=counts["anchors"],
                disconnections=counts["disconnections"],
            )
        )

    lines.extend(["", "## Coverage"])
    if report["coverage_issues"]:
        for issue in report["coverage_issues"]:
            lines.append(
                f"- `{issue['family_id']}`: families={issue['families']}, "
                f"anchors={issue['anchors']}, disconnections={issue['disconnections']}"
            )
    else:
        lines.append("- Every family has one family record, at least one anchor, and at least one disconnection.")

    evidence = report["evidence"]
    lines.extend(
        [
            "",
            "## Evidence",
            f"- Evidence items: `{evidence['items']}`",
            f"- Evidence URLs: `{evidence['urls']}`",
            f"- Local evidence refs: `{evidence['local_refs']}`",
            f"- Disconnections without evidence: `{len(evidence['disconnections_without_evidence'])}`",
            f"- Evidence items without URL: `{len(evidence['evidence_items_without_url'])}`",
            f"- Evidence items without URL or local file: `{len(evidence['evidence_items_without_trace'])}`",
        ]
    )
    if evidence["url_domains"]:
        lines.append("- URL domains:")
        for domain, count in evidence["url_domains"]:
            lines.append(f"  - `{domain}`: {count}")

    policy = report["policy"]
    lines.extend(
        [
            "",
            "## Policy",
            f"- Disconnections with use policy: `{policy['with_use_policy']}`",
            f"- Compliance-gated disconnections: `{len(policy['compliance_gated_disconnections'])}`",
        ]
    )
    for item in policy["compliance_gated_disconnections"]:
        lines.append(f"- `{item['id']}` ({item['family_id']}): {item['compliance_gate']}")

    return "\n".join(lines)


def _duplicate_ids(records: dict[str, list[tuple[Path, dict[str, Any]]]]) -> dict[str, list[dict[str, Any]]]:
    key_by_section = {
        "families": "family_id",
        "anchors": "anchor_id",
        "disconnections": "id",
    }
    duplicates: dict[str, list[dict[str, Any]]] = {}
    for section, key_name in key_by_section.items():
        rows_by_id: dict[str, list[str]] = defaultdict(list)
        for path, item in records[section]:
            key = str(item.get(key_name) or "")
            if key:
                rows_by_id[key].append(str(path))
        duplicates[section] = [
            {"id": key, "files": files}
            for key, files in sorted(rows_by_id.items())
            if len(files) > 1
        ]
    return duplicates


def _family_coverage(records: dict[str, list[tuple[Path, dict[str, Any]]]]) -> list[dict[str, Any]]:
    coverage: dict[str, dict[str, Any]] = {}
    for section in ("families", "anchors", "disconnections"):
        for path, item in records[section]:
            family_id = str(item.get("family_id") or item.get("id") or "")
            if section == "families":
                family_id = str(item.get("family_id") or "")
            if not family_id:
                family_id = "<missing>"
            entry = coverage.setdefault(
                family_id,
                {"family_id": family_id, "families": 0, "anchors": 0, "disconnections": 0, "files": []},
            )
            entry[section] += 1
            if str(path) not in entry["files"]:
                entry["files"].append(str(path))
    return [coverage[key] for key in sorted(coverage)]


def _coverage_issues(family_coverage: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        item
        for item in family_coverage
        if item["families"] != 1 or item["anchors"] < 1 or item["disconnections"] < 1
    ]


def _evidence_audit(disconnections: list[tuple[Path, dict[str, Any]]]) -> dict[str, Any]:
    items = 0
    urls = 0
    local_refs = 0
    domains: Counter[str] = Counter()
    without_evidence = []
    without_url = []
    without_trace = []
    for path, disconnection in disconnections:
        evidence = disconnection.get("evidence") or []
        if not evidence:
            without_evidence.append({"id": disconnection.get("id"), "file": str(path)})
            continue
        for index, item in enumerate(evidence):
            items += 1
            url = str(item.get("url") or "")
            local_file = str(item.get("file") or "")
            if not url:
                without_url.append(
                    {
                        "id": disconnection.get("id"),
                        "file": str(path),
                        "evidence_index": index,
                    }
                )
            else:
                urls += 1
                domain = urlparse(url).netloc.lower()
                domains[domain or "<missing>"] += 1
            if local_file:
                local_refs += 1
            if not url and not local_file:
                without_trace.append(
                    {
                        "id": disconnection.get("id"),
                        "file": str(path),
                        "evidence_index": index,
                    }
                )
    return {
        "items": items,
        "urls": urls,
        "local_refs": local_refs,
        "url_domains": sorted(domains.items(), key=lambda row: (-row[1], row[0])),
        "disconnections_without_evidence": without_evidence,
        "evidence_items_without_url": without_url,
        "evidence_items_without_trace": without_trace,
    }


def _policy_audit(disconnections: list[tuple[Path, dict[str, Any]]]) -> dict[str, Any]:
    with_use_policy = 0
    compliance_gated = []
    proposal_sources: Counter[str] = Counter()
    for path, disconnection in disconnections:
        use_policy = disconnection.get("use_policy") or {}
        if use_policy:
            with_use_policy += 1
        proposal_source = str(use_policy.get("proposal_source") or "")
        if proposal_source:
            proposal_sources[proposal_source] += 1
        compliance_gate = str(use_policy.get("compliance_gate") or "")
        if compliance_gate:
            compliance_gated.append(
                {
                    "id": disconnection.get("id"),
                    "family_id": disconnection.get("family_id"),
                    "file": str(path),
                    "compliance_gate": compliance_gate,
                }
            )
    return {
        "with_use_policy": with_use_policy,
        "proposal_sources": sorted(proposal_sources.items(), key=lambda row: (-row[1], row[0])),
        "compliance_gated_disconnections": compliance_gated,
    }


if __name__ == "__main__":
    main()
