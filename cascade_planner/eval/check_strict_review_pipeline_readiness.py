"""Check readiness of the strict route/block review pipeline."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import tarfile
import time
from collections import Counter
from pathlib import Path
from typing import Any

from AUTOPLANNRELLM.deepseek_client import is_placeholder_deepseek_key, normalize_deepseek_key_value


SCHEMA_VERSION = "strict_review_pipeline_readiness.v1"


DEFAULTS = {
    "value_pack": "results/shared/model_strengthening_20260519_route_block_value_runtime_train_provenance/route_block_value_pack.jsonl",
    "worklist_120": "results/shared/model_strengthening_20260519_strict_model_review_worklist/strict_model_control_disagreement_review.jsonl",
    "prompts_120": "results/shared/model_strengthening_20260519_strict_model_review_worklist/dryrun_pipeline/route_pool_evidence_review_prompts.jsonl",
    "dryrun_merge_report": "results/shared/model_strengthening_20260519_strict_model_review_worklist/dryrun_pipeline/strict_model_dryrun_merged_route_block_value_pack_report.json",
    "worklist_300": "results/shared/model_strengthening_20260519_strict_model_review_worklist_extended/strict_model_control_disagreement_review_300.jsonl",
    "prompts_300": "results/shared/model_strengthening_20260519_strict_model_review_worklist_extended/strict_model_control_disagreement_prompts_300.jsonl",
    "packet_120_csv": "results/shared/model_strengthening_20260519_strict_model_review_packet/route_pool_evidence_review_calibration_subset_TO_FILL.csv",
    "packet_300_csv": "results/shared/model_strengthening_20260519_strict_model_review_packet_300/route_pool_evidence_review_calibration_subset_TO_FILL.csv",
    "packet_120_json": "results/shared/model_strengthening_20260519_strict_model_review_packet/route_pool_review_calibration_packet.json",
    "packet_300_json": "results/shared/model_strengthening_20260519_strict_model_review_packet_300/route_pool_review_calibration_packet.json",
    "packet_archive": "results/shared/model_strengthening_20260519_strict_review_packets.tar.gz",
    "packet_archive_sha256": "results/shared/model_strengthening_20260519_strict_review_packets.tar.gz.sha256",
    "real_merge_report": "results/shared/model_strengthening_20260519_strict_model_review_worklist/real_review_pipeline/strict_model_real_merged_route_block_value_pack_report.json",
    "real_merge_report_300": "results/shared/model_strengthening_20260519_strict_model_review_worklist_extended/real_review_pipeline/strict_model_real_300_merged_route_block_value_pack_report.json",
    "human_merge_report": "results/shared/model_strengthening_20260519_strict_model_review_packet/csv_ingest_pipeline/strict_model_human_merged_route_block_value_pack_report.json",
    "human_merge_report_300": "results/shared/model_strengthening_20260519_strict_model_review_packet_300/csv_ingest_pipeline/strict_model_human_300_merged_route_block_value_pack_report.json",
}


REQUIRED_PACKET_ARCHIVE_MEMBERS = (
    "results/shared/model_strengthening_20260519_strict_review_PACKET_README.md",
    DEFAULTS["packet_120_csv"],
    DEFAULTS["packet_120_json"],
    "results/shared/model_strengthening_20260519_strict_model_review_packet/README.md",
    DEFAULTS["packet_300_csv"],
    DEFAULTS["packet_300_json"],
    "results/shared/model_strengthening_20260519_strict_model_review_packet_300/README.md",
    "results/shared/model_strengthening_20260519_strict_model_review_worklist/strict_model_control_disagreement_review.csv",
    "results/shared/model_strengthening_20260519_strict_model_review_worklist_extended/strict_model_control_disagreement_review_300.csv",
    "results/shared/model_strengthening_20260519_strict_model_review_worklist/strict_model_control_disagreement_prompts.jsonl",
    "results/shared/model_strengthening_20260519_strict_model_review_worklist_extended/strict_model_control_disagreement_prompts_300.jsonl",
)
REQUIRED_PACKET_CONTEXT_FIELDS = ("target_id", "route_id", "source_value_pack", "value_split")


def check_strict_review_pipeline_readiness(*, root: Path = Path(".")) -> dict[str, Any]:
    root = Path(root)
    paths = {name: root / rel for name, rel in DEFAULTS.items()}
    value_summary = _jsonl_summary(paths["value_pack"], required_fields=("route_id", "target_id", "split"))
    worklist_120 = _jsonl_summary(paths["worklist_120"], required_fields=("review_id", "route_id", "target_id"))
    prompts_120 = _jsonl_summary(paths["prompts_120"], required_fields=("review_id", "route_id", "target_id"))
    worklist_300 = _jsonl_summary(paths["worklist_300"], required_fields=("review_id", "route_id", "target_id"))
    prompts_300 = _jsonl_summary(paths["prompts_300"], required_fields=("review_id", "route_id", "target_id"))
    dryrun_merge = _json_summary(paths["dryrun_merge_report"])
    real_merge = _json_summary(paths["real_merge_report"])
    real_merge_300 = _json_summary(paths["real_merge_report_300"])
    human_merge = _json_summary(paths["human_merge_report"])
    human_merge_300 = _json_summary(paths["human_merge_report_300"])
    packet_120_csv = _csv_expert_fill_summary(paths["packet_120_csv"])
    packet_300_csv = _csv_expert_fill_summary(paths["packet_300_csv"])
    packet_120 = _packet_metadata_summary(paths["packet_120_json"])
    packet_300 = _packet_metadata_summary(paths["packet_300_json"])
    packet_archive = _archive_sha256_summary(
        paths["packet_archive"],
        paths["packet_archive_sha256"],
        required_members=REQUIRED_PACKET_ARCHIVE_MEMBERS,
    )
    key_status = _key_status(root)

    strict_ready = (
        value_summary["exists"]
        and worklist_120["rows"] == 120
        and worklist_120["rows_with_required_fields"] == 120
        and prompts_120["rows"] == 120
        and prompts_120["rows_with_required_fields"] == 120
        and dryrun_merge["exists"]
        and (dryrun_merge["payload"].get("counts") or {}).get("matched_route_ids") == 120
    )
    fallback_ready = (
        worklist_300["rows"] == 300
        and worklist_300["rows_with_required_fields"] == 300
        and prompts_300["rows"] == 300
        and prompts_300["rows_with_required_fields"] == 300
    )
    external_packet_ready = (
        paths["packet_120_csv"].exists()
        and paths["packet_300_csv"].exists()
        and packet_120_csv["context_fields_ok"]
        and packet_300_csv["context_fields_ok"]
        and packet_120["ready"]
        and packet_300["ready"]
        and packet_archive["ok"]
    )
    real_review_ready = strict_ready and key_status["configured"]
    ready_for_expert_value_training = any(
        _merge_ready(payload)
        for payload in (
            real_merge["payload"],
            real_merge_300["payload"],
            human_merge["payload"],
            human_merge_300["payload"],
        )
    )
    human_training_ready = _merge_ready(human_merge["payload"]) or _merge_ready(human_merge_300["payload"])
    filled_expert_csv_rows = packet_120_csv["filled_expert_rows"] + packet_300_csv["filled_expert_rows"]
    filled_expert_csv_available = filled_expert_csv_rows > 0

    blockers = []
    if not strict_ready:
        blockers.append("strict 120-row review pipeline artifacts are incomplete or missing route identity")
    if not fallback_ready:
        blockers.append("strict 300-row fallback artifacts are incomplete or missing route identity")
    if not external_packet_ready:
        blockers.append(
            "human/external review packets are missing or failed metadata/checksum/member/context-column validation"
        )
    if not key_status["configured"] and not human_training_ready:
        blockers.append("DEEPSEEK_API_KEY is not configured")
        if not filled_expert_csv_available:
            blockers.append("no filled expert CSV rows are available")
    if not ready_for_expert_value_training:
        blockers.append("no merged review value pack has passed expert-training gate")

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "paths": {name: str(path) for name, path in paths.items()},
        "summaries": {
            "value_pack": value_summary,
            "worklist_120": worklist_120,
            "prompts_120": prompts_120,
            "dryrun_merge": _compact_merge(dryrun_merge["payload"]),
            "worklist_300": worklist_300,
            "prompts_300": prompts_300,
            "external_packets": {
                "packet_120_csv_exists": paths["packet_120_csv"].exists(),
                "packet_300_csv_exists": paths["packet_300_csv"].exists(),
                "packet_120_csv": packet_120_csv,
                "packet_300_csv": packet_300_csv,
                "packet_120_metadata": packet_120,
                "packet_300_metadata": packet_300,
                "packet_archive": packet_archive,
            },
            "real_merge": _compact_merge(real_merge["payload"]),
            "real_merge_300": _compact_merge(real_merge_300["payload"]),
            "human_merge": _compact_merge(human_merge["payload"]),
            "human_merge_300": _compact_merge(human_merge_300["payload"]),
            "key_status": key_status,
        },
        "decision": {
            "strict_120_ready_for_real_review": strict_ready,
            "strict_300_fallback_ready": fallback_ready,
            "external_packet_ready": external_packet_ready,
            "real_review_can_run_now": real_review_ready,
            "filled_expert_csv_available": filled_expert_csv_available,
            "filled_expert_csv_rows": filled_expert_csv_rows,
            "ready_for_expert_value_training": ready_for_expert_value_training,
            "blockers": blockers,
        },
    }


def _merge_ready(payload: dict[str, Any]) -> bool:
    return bool((payload.get("decision") or {}).get("ready_for_expert_training"))


def _compact_merge(payload: dict[str, Any]) -> dict[str, Any]:
    if not payload:
        return {"exists": False}
    return {
        "exists": True,
        "counts": payload.get("counts") or {},
        "decision": payload.get("decision") or {},
    }


def _json_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False, "payload": {}}
    try:
        return {"exists": True, "payload": json.loads(path.read_text(encoding="utf-8"))}
    except (OSError, json.JSONDecodeError):
        return {"exists": True, "payload": {}, "error": "unreadable_json"}


def _csv_expert_fill_summary(path: Path) -> dict[str, Any]:
    expert_decision_fields = (
        "expert_route_plausible",
        "expert_block_transform_correct",
        "expert_support_precedent_relevant",
        "expert_cascade_coherent",
        "expert_priority",
    )
    expert_fields = (
        "expert_route_plausible",
        "expert_block_transform_correct",
        "expert_support_precedent_relevant",
        "expert_cascade_coherent",
        "expert_priority",
        "expert_reject_reason",
        "expert_risk_tags",
        "expert_comments",
    )
    if not path.exists():
        return {
            "exists": False,
            "rows": 0,
            "fieldnames": [],
            "context_fields": list(REQUIRED_PACKET_CONTEXT_FIELDS),
            "missing_context_fields": list(REQUIRED_PACKET_CONTEXT_FIELDS),
            "context_fields_ok": False,
            "expert_fields": list(expert_fields),
            "expert_decision_fields": list(expert_decision_fields),
            "value_split_counts": {},
            "filled_expert_rows_by_split": {},
            "filled_expert_rows": 0,
        }
    rows = 0
    filled = 0
    fieldnames: list[str] = []
    value_split_counts: Counter[str] = Counter()
    filled_by_split: Counter[str] = Counter()
    try:
        with path.open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            fieldnames = list(reader.fieldnames or [])
            for row in reader:
                rows += 1
                split = str(row.get("value_split") or "")
                if split:
                    value_split_counts[split] += 1
                if any(str(row.get(field) or "").strip() for field in expert_decision_fields):
                    filled += 1
                    if split:
                        filled_by_split[split] += 1
    except (OSError, UnicodeDecodeError, csv.Error):
        missing_context = [field for field in REQUIRED_PACKET_CONTEXT_FIELDS if field not in fieldnames]
        return {
            "exists": True,
            "rows": rows,
            "fieldnames": fieldnames,
            "context_fields": list(REQUIRED_PACKET_CONTEXT_FIELDS),
            "missing_context_fields": missing_context,
            "context_fields_ok": not missing_context,
            "expert_fields": list(expert_fields),
            "expert_decision_fields": list(expert_decision_fields),
            "value_split_counts": dict(sorted(value_split_counts.items())),
            "filled_expert_rows_by_split": dict(sorted(filled_by_split.items())),
            "filled_expert_rows": filled,
            "error": "unreadable_csv",
        }
    missing_context = [field for field in REQUIRED_PACKET_CONTEXT_FIELDS if field not in fieldnames]
    return {
        "exists": True,
        "rows": rows,
        "fieldnames": fieldnames,
        "context_fields": list(REQUIRED_PACKET_CONTEXT_FIELDS),
        "missing_context_fields": missing_context,
        "context_fields_ok": not missing_context,
        "expert_fields": list(expert_fields),
        "expert_decision_fields": list(expert_decision_fields),
        "value_split_counts": dict(sorted(value_split_counts.items())),
        "filled_expert_rows_by_split": dict(sorted(filled_by_split.items())),
        "filled_expert_rows": filled,
    }


def _packet_metadata_summary(path: Path) -> dict[str, Any]:
    summary = _json_summary(path)
    payload = summary.get("payload") or {}
    command = str(payload.get("pipeline_command") or "")
    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    min_evidence_classes = metadata.get("min_evidence_classes")
    has_strict_evidence_class_gate = "--min-evidence-classes 1" in command
    ready = bool(summary.get("exists")) and has_strict_evidence_class_gate and min_evidence_classes == 1
    return {
        "exists": bool(summary.get("exists")),
        "ready": ready,
        "min_evidence_classes": min_evidence_classes,
        "pipeline_command_has_min_evidence_classes_1": has_strict_evidence_class_gate,
        "error": summary.get("error"),
    }


def _archive_sha256_summary(
    archive_path: Path,
    sha256_path: Path,
    *,
    required_members: tuple[str, ...] = (),
) -> dict[str, Any]:
    archive_exists = archive_path.exists()
    sha256_exists = sha256_path.exists()
    expected = ""
    actual = ""
    if sha256_exists:
        try:
            expected = sha256_path.read_text(encoding="utf-8").split()[0]
        except (IndexError, OSError, UnicodeDecodeError):
            expected = ""
    if archive_exists:
        digest = hashlib.sha256()
        with archive_path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                digest.update(chunk)
        actual = digest.hexdigest()
    checksum_ok = bool(archive_exists and sha256_exists and expected and expected == actual)
    missing_required_members: list[str] = []
    archive_error = None
    required_members_ok = True
    if required_members:
        required_members_ok = False
        if archive_exists:
            try:
                with tarfile.open(archive_path, "r:*") as tf:
                    members = set(tf.getnames())
                missing_required_members = [
                    member for member in required_members if member not in members
                ]
                required_members_ok = not missing_required_members
            except (OSError, tarfile.TarError):
                archive_error = "unreadable_archive"
                missing_required_members = list(required_members)
        else:
            missing_required_members = list(required_members)
    return {
        "exists": archive_exists,
        "sha256_exists": sha256_exists,
        "sha256_ok": checksum_ok,
        "required_members_ok": required_members_ok,
        "required_members": list(required_members),
        "missing_required_members": missing_required_members,
        "ok": checksum_ok and required_members_ok,
        "expected_sha256": expected,
        "actual_sha256": actual,
        "error": archive_error,
    }


def _jsonl_summary(path: Path, *, required_fields: tuple[str, ...]) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False, "rows": 0, "rows_with_required_fields": 0}
    rows = 0
    rows_with_required = 0
    split_counts: dict[str, int] = {}
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            rows += 1
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if all(payload.get(field) not in (None, "") for field in required_fields):
                rows_with_required += 1
            split = payload.get("split")
            if split:
                split_counts[str(split)] = split_counts.get(str(split), 0) + 1
    return {
        "exists": True,
        "rows": rows,
        "rows_with_required_fields": rows_with_required,
        "required_fields": list(required_fields),
        "split_counts": split_counts,
    }


def _key_status(root: Path) -> dict[str, Any]:
    env_value = os.environ.get("DEEPSEEK_API_KEY")
    if _has_deepseek_key_value(env_value):
        return {
            "configured": _usable_deepseek_key(env_value),
            "source": "env" if _usable_deepseek_key(env_value) else None,
            "placeholder_source": "env" if not _usable_deepseek_key(env_value) else None,
        }
    configured = False
    source = None
    placeholder_source = None
    for filename in (".env.local", ".env"):
        path = root / filename
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            if not line.startswith("DEEPSEEK_API_KEY="):
                continue
            value = line.split("=", 1)[1]
            configured = _usable_deepseek_key(value)
            source = filename if configured else None
            placeholder_source = filename if not configured and _has_deepseek_key_value(value) else None
            break
        if configured or placeholder_source:
            break
    return {
        "configured": configured,
        "source": source,
        "placeholder_source": placeholder_source,
    }


def _has_deepseek_key_value(value: str | None) -> bool:
    return bool(normalize_deepseek_key_value(value))


def _usable_deepseek_key(value: str | None) -> bool:
    normalized = normalize_deepseek_key_value(value)
    return bool(normalized and not is_placeholder_deepseek_key(normalized))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()
    report = check_strict_review_pipeline_readiness(root=Path(args.root))
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report["decision"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
