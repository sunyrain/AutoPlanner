import hashlib
import json
import os
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from cascade_planner.eval.check_strict_review_pipeline_readiness import (
    DEFAULTS,
    REQUIRED_PACKET_ARCHIVE_MEMBERS,
    REQUIRED_PACKET_CONTEXT_FIELDS,
    check_strict_review_pipeline_readiness,
)


class CheckStrictReviewPipelineReadinessTest(unittest.TestCase):
    def test_reports_ready_for_review_but_blocked_without_key(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write_jsonl(root / DEFAULTS["value_pack"], [_row(split="train")])
            _write_jsonl(root / DEFAULTS["worklist_120"], [_review_row(i) for i in range(120)])
            _write_jsonl(root / DEFAULTS["prompts_120"], [_review_row(i) for i in range(120)])
            _write_jsonl(root / DEFAULTS["worklist_300"], [_review_row(i) for i in range(300)])
            _write_jsonl(root / DEFAULTS["prompts_300"], [_review_row(i) for i in range(300)])
            _write_json(
                root / DEFAULTS["dryrun_merge_report"],
                {"counts": {"matched_route_ids": 120}, "decision": {"ready_for_expert_training": False}},
            )
            _write_external_packet_artifacts(root)

            report = check_strict_review_pipeline_readiness(root=root)

        self.assertTrue(report["decision"]["strict_120_ready_for_real_review"])
        self.assertTrue(report["decision"]["strict_300_fallback_ready"])
        self.assertTrue(report["decision"]["external_packet_ready"])
        self.assertFalse(report["decision"]["real_review_can_run_now"])
        self.assertFalse(report["decision"]["filled_expert_csv_available"])
        self.assertEqual(report["decision"]["filled_expert_csv_rows"], 0)
        self.assertFalse(report["decision"]["ready_for_expert_value_training"])
        self.assertTrue(report["summaries"]["external_packets"]["packet_120_csv"]["context_fields_ok"])
        self.assertEqual(report["summaries"]["external_packets"]["packet_120_csv"]["missing_context_fields"], [])
        self.assertIn("DEEPSEEK_API_KEY is not configured", report["decision"]["blockers"])
        self.assertIn("no filled expert CSV rows are available", report["decision"]["blockers"])

    def test_env_local_key_unblocks_real_review_check(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write_jsonl(root / DEFAULTS["value_pack"], [_row(split="train")])
            _write_jsonl(root / DEFAULTS["worklist_120"], [_review_row(i) for i in range(120)])
            _write_jsonl(root / DEFAULTS["prompts_120"], [_review_row(i) for i in range(120)])
            _write_jsonl(root / DEFAULTS["worklist_300"], [_review_row(i) for i in range(300)])
            _write_jsonl(root / DEFAULTS["prompts_300"], [_review_row(i) for i in range(300)])
            _write_json(
                root / DEFAULTS["dryrun_merge_report"],
                {"counts": {"matched_route_ids": 120}, "decision": {"ready_for_expert_training": False}},
            )
            _write_external_packet_artifacts(root)
            (root / ".env.local").write_text("DEEPSEEK_API_KEY=test-key\n", encoding="utf-8")

            report = check_strict_review_pipeline_readiness(root=root)

        self.assertTrue(report["summaries"]["key_status"]["configured"])
        self.assertEqual(report["summaries"]["key_status"]["source"], ".env.local")
        self.assertTrue(report["decision"]["real_review_can_run_now"])

    def test_placeholder_key_does_not_unblock_real_review_check(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write_jsonl(root / DEFAULTS["value_pack"], [_row(split="train")])
            _write_jsonl(root / DEFAULTS["worklist_120"], [_review_row(i) for i in range(120)])
            _write_jsonl(root / DEFAULTS["prompts_120"], [_review_row(i) for i in range(120)])
            _write_jsonl(root / DEFAULTS["worklist_300"], [_review_row(i) for i in range(300)])
            _write_jsonl(root / DEFAULTS["prompts_300"], [_review_row(i) for i in range(300)])
            _write_json(
                root / DEFAULTS["dryrun_merge_report"],
                {"counts": {"matched_route_ids": 120}, "decision": {"ready_for_expert_training": False}},
            )
            _write_external_packet_artifacts(root)
            (root / ".env.local").write_text(
                "DEEPSEEK_API_KEY='  replace_with_your_deepseek_key  '\n",
                encoding="utf-8",
            )

            report = check_strict_review_pipeline_readiness(root=root)

        self.assertFalse(report["summaries"]["key_status"]["configured"])
        self.assertEqual(report["summaries"]["key_status"]["placeholder_source"], ".env.local")
        self.assertFalse(report["decision"]["real_review_can_run_now"])
        self.assertIn("DEEPSEEK_API_KEY is not configured", report["decision"]["blockers"])

    def test_env_placeholder_key_does_not_unblock_real_review_check(self):
        with tempfile.TemporaryDirectory() as td, patch.dict(
            os.environ,
            {"DEEPSEEK_API_KEY": "\"  replace_with_your_deepseek_key  \""},
        ):
            root = Path(td)
            _write_jsonl(root / DEFAULTS["value_pack"], [_row(split="train")])
            _write_jsonl(root / DEFAULTS["worklist_120"], [_review_row(i) for i in range(120)])
            _write_jsonl(root / DEFAULTS["prompts_120"], [_review_row(i) for i in range(120)])
            _write_jsonl(root / DEFAULTS["worklist_300"], [_review_row(i) for i in range(300)])
            _write_jsonl(root / DEFAULTS["prompts_300"], [_review_row(i) for i in range(300)])
            _write_json(
                root / DEFAULTS["dryrun_merge_report"],
                {"counts": {"matched_route_ids": 120}, "decision": {"ready_for_expert_training": False}},
            )
            _write_external_packet_artifacts(root)

            report = check_strict_review_pipeline_readiness(root=root)

        self.assertFalse(report["summaries"]["key_status"]["configured"])
        self.assertEqual(report["summaries"]["key_status"]["placeholder_source"], "env")
        self.assertFalse(report["decision"]["real_review_can_run_now"])
        self.assertIn("DEEPSEEK_API_KEY is not configured", report["decision"]["blockers"])

    def test_env_local_placeholder_takes_precedence_over_env_file_key(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write_jsonl(root / DEFAULTS["value_pack"], [_row(split="train")])
            _write_jsonl(root / DEFAULTS["worklist_120"], [_review_row(i) for i in range(120)])
            _write_jsonl(root / DEFAULTS["prompts_120"], [_review_row(i) for i in range(120)])
            _write_jsonl(root / DEFAULTS["worklist_300"], [_review_row(i) for i in range(300)])
            _write_jsonl(root / DEFAULTS["prompts_300"], [_review_row(i) for i in range(300)])
            _write_json(
                root / DEFAULTS["dryrun_merge_report"],
                {"counts": {"matched_route_ids": 120}, "decision": {"ready_for_expert_training": False}},
            )
            _write_external_packet_artifacts(root)
            (root / ".env.local").write_text(
                "DEEPSEEK_API_KEY=replace_with_your_deepseek_key\n",
                encoding="utf-8",
            )
            (root / ".env").write_text("DEEPSEEK_API_KEY=test-key\n", encoding="utf-8")

            report = check_strict_review_pipeline_readiness(root=root)

        self.assertFalse(report["summaries"]["key_status"]["configured"])
        self.assertEqual(report["summaries"]["key_status"]["placeholder_source"], ".env.local")
        self.assertFalse(report["decision"]["real_review_can_run_now"])

    def test_reports_filled_expert_csv_rows(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write_jsonl(root / DEFAULTS["value_pack"], [_row(split="train")])
            _write_jsonl(root / DEFAULTS["worklist_120"], [_review_row(i) for i in range(120)])
            _write_jsonl(root / DEFAULTS["prompts_120"], [_review_row(i) for i in range(120)])
            _write_jsonl(root / DEFAULTS["worklist_300"], [_review_row(i) for i in range(300)])
            _write_jsonl(root / DEFAULTS["prompts_300"], [_review_row(i) for i in range(300)])
            _write_json(
                root / DEFAULTS["dryrun_merge_report"],
                {"counts": {"matched_route_ids": 120}, "decision": {"ready_for_expert_training": False}},
            )
            _write_external_packet_artifacts(root)
            (root / DEFAULTS["packet_120_csv"]).write_text(
                _packet_csv_header() + "\n"
                "review-1,target1,route1,pack.jsonl,train,yes,looks plausible\n"
                "review-2,target2,route2,pack.jsonl,val,,\n"
                "review-3,target3,route3,pack.jsonl,test,,comment-only rows are not ready\n",
                encoding="utf-8",
            )

            report = check_strict_review_pipeline_readiness(root=root)

        external = report["summaries"]["external_packets"]
        self.assertTrue(report["decision"]["filled_expert_csv_available"])
        self.assertEqual(report["decision"]["filled_expert_csv_rows"], 1)
        self.assertIn("expert_route_plausible", external["packet_120_csv"]["expert_decision_fields"])
        self.assertEqual(external["packet_120_csv"]["filled_expert_rows"], 1)
        self.assertEqual(external["packet_120_csv"]["value_split_counts"], {"test": 1, "train": 1, "val": 1})
        self.assertEqual(external["packet_120_csv"]["filled_expert_rows_by_split"], {"train": 1})
        self.assertEqual(external["packet_300_csv"]["filled_expert_rows"], 0)
        self.assertNotIn("no filled expert CSV rows are available", report["decision"]["blockers"])

    def test_dot_env_key_unblocks_real_review_check(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write_jsonl(root / DEFAULTS["value_pack"], [_row(split="train")])
            _write_jsonl(root / DEFAULTS["worklist_120"], [_review_row(i) for i in range(120)])
            _write_jsonl(root / DEFAULTS["prompts_120"], [_review_row(i) for i in range(120)])
            _write_jsonl(root / DEFAULTS["worklist_300"], [_review_row(i) for i in range(300)])
            _write_jsonl(root / DEFAULTS["prompts_300"], [_review_row(i) for i in range(300)])
            _write_json(
                root / DEFAULTS["dryrun_merge_report"],
                {"counts": {"matched_route_ids": 120}, "decision": {"ready_for_expert_training": False}},
            )
            _write_external_packet_artifacts(root)
            (root / ".env").write_text("DEEPSEEK_API_KEY=test-key\n", encoding="utf-8")

            report = check_strict_review_pipeline_readiness(root=root)

        self.assertTrue(report["summaries"]["key_status"]["configured"])
        self.assertEqual(report["summaries"]["key_status"]["source"], ".env")
        self.assertTrue(report["decision"]["real_review_can_run_now"])

    def test_human_csv_merge_can_satisfy_expert_training_gate_without_key(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write_jsonl(root / DEFAULTS["value_pack"], [_row(split="train")])
            _write_jsonl(root / DEFAULTS["worklist_120"], [_review_row(i) for i in range(120)])
            _write_jsonl(root / DEFAULTS["prompts_120"], [_review_row(i) for i in range(120)])
            _write_jsonl(root / DEFAULTS["worklist_300"], [_review_row(i) for i in range(300)])
            _write_jsonl(root / DEFAULTS["prompts_300"], [_review_row(i) for i in range(300)])
            _write_json(
                root / DEFAULTS["dryrun_merge_report"],
                {"counts": {"matched_route_ids": 120}, "decision": {"ready_for_expert_training": False}},
            )
            _write_external_packet_artifacts(root)
            _write_json(
                root / DEFAULTS["human_merge_report"],
                {
                    "counts": {"usable_positive_rows": 30, "usable_negative_rows": 30},
                    "decision": {"ready_for_expert_training": True},
                },
            )

            report = check_strict_review_pipeline_readiness(root=root)

        self.assertTrue(report["summaries"]["human_merge"]["exists"])
        self.assertTrue(report["decision"]["ready_for_expert_value_training"])
        self.assertNotIn("DEEPSEEK_API_KEY is not configured", report["decision"]["blockers"])

    def test_external_packet_requires_metadata_and_matching_checksum(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write_jsonl(root / DEFAULTS["value_pack"], [_row(split="train")])
            _write_jsonl(root / DEFAULTS["worklist_120"], [_review_row(i) for i in range(120)])
            _write_jsonl(root / DEFAULTS["prompts_120"], [_review_row(i) for i in range(120)])
            _write_jsonl(root / DEFAULTS["worklist_300"], [_review_row(i) for i in range(300)])
            _write_jsonl(root / DEFAULTS["prompts_300"], [_review_row(i) for i in range(300)])
            _write_json(
                root / DEFAULTS["dryrun_merge_report"],
                {"counts": {"matched_route_ids": 120}, "decision": {"ready_for_expert_training": False}},
            )
            (root / DEFAULTS["packet_120_csv"]).parent.mkdir(parents=True, exist_ok=True)
            (root / DEFAULTS["packet_120_csv"]).write_text("review_id\n", encoding="utf-8")
            (root / DEFAULTS["packet_300_csv"]).parent.mkdir(parents=True, exist_ok=True)
            (root / DEFAULTS["packet_300_csv"]).write_text("review_id\n", encoding="utf-8")
            _write_json(
                root / DEFAULTS["packet_120_json"],
                {"metadata": {"min_evidence_classes": 2}, "pipeline_command": "run"},
            )
            _write_json(
                root / DEFAULTS["packet_300_json"],
                {"metadata": {"min_evidence_classes": 1}, "pipeline_command": "run --min-evidence-classes 1"},
            )
            _write_archive_with_checksum(root, checksum_matches=False)

            report = check_strict_review_pipeline_readiness(root=root)

        self.assertFalse(report["decision"]["external_packet_ready"])
        self.assertFalse(
            report["summaries"]["external_packets"]["packet_120_metadata"][
                "pipeline_command_has_min_evidence_classes_1"
            ]
        )
        self.assertFalse(report["summaries"]["external_packets"]["packet_archive"]["ok"])

    def test_external_packet_requires_context_columns(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write_jsonl(root / DEFAULTS["value_pack"], [_row(split="train")])
            _write_jsonl(root / DEFAULTS["worklist_120"], [_review_row(i) for i in range(120)])
            _write_jsonl(root / DEFAULTS["prompts_120"], [_review_row(i) for i in range(120)])
            _write_jsonl(root / DEFAULTS["worklist_300"], [_review_row(i) for i in range(300)])
            _write_jsonl(root / DEFAULTS["prompts_300"], [_review_row(i) for i in range(300)])
            _write_json(
                root / DEFAULTS["dryrun_merge_report"],
                {"counts": {"matched_route_ids": 120}, "decision": {"ready_for_expert_training": False}},
            )
            _write_external_packet_artifacts(root)
            (root / DEFAULTS["packet_120_csv"]).write_text("review_id,route_id\nreview-1,route1\n", encoding="utf-8")

            report = check_strict_review_pipeline_readiness(root=root)

        packet = report["summaries"]["external_packets"]["packet_120_csv"]
        self.assertFalse(report["decision"]["external_packet_ready"])
        self.assertFalse(packet["context_fields_ok"])
        self.assertIn("target_id", packet["missing_context_fields"])
        self.assertEqual(list(REQUIRED_PACKET_CONTEXT_FIELDS), packet["context_fields"])

    def test_external_packet_requires_archive_members(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write_jsonl(root / DEFAULTS["value_pack"], [_row(split="train")])
            _write_jsonl(root / DEFAULTS["worklist_120"], [_review_row(i) for i in range(120)])
            _write_jsonl(root / DEFAULTS["prompts_120"], [_review_row(i) for i in range(120)])
            _write_jsonl(root / DEFAULTS["worklist_300"], [_review_row(i) for i in range(300)])
            _write_jsonl(root / DEFAULTS["prompts_300"], [_review_row(i) for i in range(300)])
            _write_json(
                root / DEFAULTS["dryrun_merge_report"],
                {"counts": {"matched_route_ids": 120}, "decision": {"ready_for_expert_training": False}},
            )
            _write_external_packet_artifacts(
                root,
                omitted_archive_members={DEFAULTS["packet_300_csv"]},
            )

            report = check_strict_review_pipeline_readiness(root=root)

        archive = report["summaries"]["external_packets"]["packet_archive"]
        self.assertFalse(report["decision"]["external_packet_ready"])
        self.assertTrue(archive["sha256_ok"])
        self.assertFalse(archive["required_members_ok"])
        self.assertIn(DEFAULTS["packet_300_csv"], archive["missing_required_members"])


def _row(*, split):
    return {"route_id": "route1", "target_id": "target1", "split": split}


def _review_row(i):
    return {"review_id": f"review{i}", "route_id": f"route{i}", "target_id": f"target{i}"}


def _write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_external_packet_artifacts(root: Path, *, omitted_archive_members=None):
    omitted_archive_members = set(omitted_archive_members or ())
    (root / DEFAULTS["packet_120_csv"]).parent.mkdir(parents=True, exist_ok=True)
    (root / DEFAULTS["packet_120_csv"]).write_text(_packet_csv_header() + "\n", encoding="utf-8")
    (root / DEFAULTS["packet_300_csv"]).parent.mkdir(parents=True, exist_ok=True)
    (root / DEFAULTS["packet_300_csv"]).write_text(_packet_csv_header() + "\n", encoding="utf-8")
    _write_json(
        root / DEFAULTS["packet_120_json"],
        {"metadata": {"min_evidence_classes": 1}, "pipeline_command": "run --min-evidence-classes 1"},
    )
    _write_json(
        root / DEFAULTS["packet_300_json"],
        {"metadata": {"min_evidence_classes": 1}, "pipeline_command": "run --min-evidence-classes 1"},
    )
    _write_archive_with_checksum(
        root,
        checksum_matches=True,
        omitted_members=omitted_archive_members,
    )


def _packet_csv_header():
    return (
        "review_id,target_id,route_id,source_value_pack,value_split,"
        "expert_route_plausible,expert_comments"
    )


def _write_archive_with_checksum(root: Path, *, checksum_matches: bool, omitted_members=None):
    omitted_members = set(omitted_members or ())
    for member in REQUIRED_PACKET_ARCHIVE_MEMBERS:
        path = root / member
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_text(f"{member}\n", encoding="utf-8")
    archive = root / DEFAULTS["packet_archive"]
    archive.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "w:gz") as tf:
        for member in REQUIRED_PACKET_ARCHIVE_MEMBERS:
            if member in omitted_members:
                continue
            tf.add(root / member, arcname=member)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    expected = digest if checksum_matches else "0" * 64
    sha_path = root / DEFAULTS["packet_archive_sha256"]
    sha_path.write_text(f"{expected}  {DEFAULTS['packet_archive']}\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
