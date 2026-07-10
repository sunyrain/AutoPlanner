from __future__ import annotations

import asyncio
from pathlib import Path
import tempfile
import unittest

from cascade_planner.application.frontier_scheduler import (
    FrontierExecutor,
    FrontierJob,
    FrontierJobState,
    FrontierLeaseError,
    FrontierQueueError,
    FrontierScheduler,
    PersistentFrontierQueue,
    assess_frontier_completeness,
)
from cascade_planner.providers.stock import SnapshotStockProvider, stock_snapshot_sha256


NOW = "2026-07-10T00:00:00.000000Z"


def _snapshot(supplier: str, catalog_number: str) -> dict[str, object]:
    return {
        "schema_version": "stock_offer_snapshot.v1",
        "smiles": "CCO",
        "supplier": supplier,
        "catalog_number": catalog_number,
        "checked_at": NOW,
        "available": True,
    }


def _offer(supplier: str, catalog_number: str) -> dict[str, object]:
    snapshot = _snapshot(supplier, catalog_number)
    return {**snapshot, "snapshot_sha256": stock_snapshot_sha256(snapshot)}


class FrontierSchedulerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.queue = PersistentFrontierQueue(Path(self.temp.name))
        self.scheduler = FrontierScheduler(
            self.queue,
            SnapshotStockProvider(
                trusted_snapshots=[
                    _snapshot("test-supplier", "E-1"),
                    _snapshot("test", "1"),
                ]
            ),
        )

    def submit(self, smiles: str, key: str, **kwargs: object) -> FrontierJob:
        return self.scheduler.submit(
            run_id="run-1",
            case_id="paclitaxel",
            frontier_smiles=smiles,
            frontier_node_id=f"molecule:{key}",
            idempotency_key=key,
            now=NOW,
            **kwargs,
        )

    def test_stock_audit_closes_terminal_before_agent_work(self) -> None:
        job = self.submit(
            "CCO",
            "ethanol",
            stock_request={
                "offers": [
                    _offer("test-supplier", "E-1")
                ]
            },
        )
        self.assertEqual(job.state, FrontierJobState.SUCCEEDED)
        self.assertEqual(job.closure_kind, "stock_boundary")
        self.assertEqual(self.queue.claim("run-1", worker_id="worker"), [])
        self.assertTrue(job.metadata["stock_audit_preceded_agent_work"])

    def test_priority_dependencies_leases_and_idempotency(self) -> None:
        low = self.submit("CC", "low", proof_deficit=1, closure_probability=0.1)
        high = self.submit("CCC", "high", proof_deficit=4, closure_probability=0.9)
        dependent = self.submit("CCCC", "dep", dependency_ids=[low.job_id])
        self.assertEqual(
            self.submit("CCC", "high", proof_deficit=4, closure_probability=0.9),
            high,
        )

        claimed = self.queue.claim(
            "run-1", worker_id="worker", limit=3, lease_seconds=60, now=NOW
        )
        self.assertEqual([row.job_id for row in claimed], [high.job_id, low.job_id])
        with self.assertRaises(FrontierLeaseError):
            self.queue.complete(
                "run-1",
                high.job_id,
                lease_token="stale",
                result_ref="artifact:bad",
            )
        completed = self.queue.complete(
            "run-1",
            high.job_id,
            lease_token=claimed[0].lease_token,
            result_ref="artifact:high",
        )
        replay = self.queue.complete(
            "run-1",
            high.job_id,
            lease_token=claimed[0].lease_token,
            result_ref="artifact:high",
        )
        self.assertEqual(completed, replay)
        self.queue.complete(
            "run-1",
            low.job_id,
            lease_token=claimed[1].lease_token,
            result_ref="artifact:low",
        )
        self.assertEqual(
            self.queue.claim("run-1", worker_id="worker", now=NOW)[0].job_id,
            dependent.job_id,
        )

    def test_expired_lease_recovers_and_retry_is_bounded(self) -> None:
        job = self.submit("CCN", "retry", max_attempts=2)
        first = self.queue.claim(
            "run-1", worker_id="worker", lease_seconds=1, now=NOW
        )[0]
        recovered = self.queue.recover_expired(
            "run-1",
            retry_base_seconds=0,
            now="2026-07-10T00:00:02.000000Z",
        )
        self.assertEqual(recovered[0].state, FrontierJobState.RETRY_WAIT)
        second = self.queue.claim(
            "run-1",
            worker_id="worker-2",
            lease_seconds=1,
            now="2026-07-10T00:00:02.000000Z",
        )[0]
        self.assertNotEqual(first.lease_token, second.lease_token)
        terminal = self.queue.fail(
            "run-1",
            job.job_id,
            lease_token=second.lease_token,
            reason="still_open",
            now="2026-07-10T00:00:02.000000Z",
        )
        self.assertEqual(terminal.state, FrontierJobState.FAILED)
        self.assertEqual(terminal.attempt, 2)

    def test_empty_queue_is_not_route_completion(self) -> None:
        report = assess_frontier_completeness(["CCO"], [])
        self.assertFalse(report.complete)
        self.assertEqual(report.unresolved_frontiers[0]["reason"], "no_frontier_job")
        self.assertTrue(report.to_dict()["queue_empty_is_not_completion"])

    def test_completion_requires_stock_or_reaction_level_two_and_no_open_proof(self) -> None:
        stock = self.submit(
            "CCO",
            "stock",
            stock_request={
                "offers": [
                    _offer("test", "1")
                ]
            },
        )
        reaction = self.submit("CC", "reaction")
        lease = self.queue.claim("run-1", worker_id="worker", now=NOW)[0]
        reaction = self.queue.complete(
            "run-1",
            reaction.job_id,
            lease_token=lease.lease_token,
            result_ref="proof:cc",
            achieved_proof_level=2,
        )
        closed = assess_frontier_completeness(["CCO", "CC"], [stock, reaction])
        self.assertTrue(closed.complete)
        open_report = assess_frontier_completeness(
            ["CCO", "CC"], [stock, reaction], open_proof_frontiers=["step:x"]
        )
        self.assertFalse(open_report.complete)

    def test_bounded_async_executor_persists_success_and_failure(self) -> None:
        first = self.submit("CC", "first")
        second = self.submit("CCC", "second")
        active = 0
        peak = 0

        async def handler(job: FrontierJob) -> dict[str, object]:
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(0)
            active -= 1
            if job.job_id == second.job_id:
                raise RuntimeError("provider unavailable")
            return {"result_ref": "proof:first", "achieved_proof_level": 2}

        executor = FrontierExecutor(self.queue, worker_id="async", max_concurrency=2)
        result = asyncio.run(executor.run_ready("run-1", handler))
        by_id = {row.job_id: row for row in result}
        self.assertEqual(by_id[first.job_id].state, FrontierJobState.SUCCEEDED)
        self.assertEqual(by_id[second.job_id].state, FrontierJobState.RETRY_WAIT)
        self.assertLessEqual(peak, 2)

    def test_queue_rejects_content_digest_tampering(self) -> None:
        self.submit("CC", "digest")
        path = next(Path(self.temp.name).glob("frontiers-*.json"))
        payload = path.read_text(encoding="utf-8").replace('"frontier_smiles":"CC"', '"frontier_smiles":"CCC"')
        path.write_text(payload, encoding="utf-8")

        with self.assertRaisesRegex(FrontierQueueError, "content digest mismatch"):
            self.queue.snapshot("run-1")

    def test_invalid_succeeded_result_is_requeued_fail_closed(self) -> None:
        job = self.submit("CC", "invalid-result", max_attempts=2)
        leased = self.queue.claim("run-1", worker_id="worker", now=NOW)[0]
        completed = self.queue.complete(
            "run-1",
            job.job_id,
            lease_token=leased.lease_token,
            result_ref="commit:corrupt",
            closure_kind="proposal_expansion",
            achieved_proof_level=0,
            now=NOW,
        )

        invalidated = self.queue.invalidate_succeeded_result(
            "run-1",
            job.job_id,
            expected_result_ref=completed.result_ref,
            reason="expansion_commit_payload_digest_invalid",
            now=NOW,
        )

        self.assertEqual(invalidated.state, FrontierJobState.RETRY_WAIT)
        self.assertEqual(invalidated.result_ref, "")
        self.assertEqual(invalidated.closure_kind, "")
        self.assertIn("expansion_commit_payload_digest_invalid", invalidated.failure_reasons)

    def test_succeeded_result_can_be_rebound_to_immutable_migration_commit(self) -> None:
        job = self.submit("CC", "migrate-result")
        leased = self.queue.claim("run-1", worker_id="worker", now=NOW)[0]
        completed = self.queue.complete(
            "run-1",
            job.job_id,
            lease_token=leased.lease_token,
            result_ref="legacy:team-report.json",
            closure_kind="proposal_expansion",
            achieved_proof_level=0,
            now=NOW,
        )

        rebound = self.queue.rebind_succeeded_result(
            "run-1",
            job.job_id,
            expected_result_ref=completed.result_ref,
            result_ref="campaign_commits/immutable.json",
            metadata_updates={"legacy_result_migrated": True},
            now=NOW,
        )

        self.assertEqual(rebound.result_ref, "campaign_commits/immutable.json")
        self.assertTrue(rebound.metadata["legacy_result_migrated"])
        self.assertEqual(rebound.state, FrontierJobState.SUCCEEDED)

    def test_async_executor_heartbeats_long_handler(self) -> None:
        job = self.submit("CC", "heartbeat")
        recovered: list[FrontierJob] = []

        async def handler(_: FrontierJob) -> dict[str, object]:
            await asyncio.sleep(0.18)
            recovered.extend(await asyncio.to_thread(self.queue.recover_expired, "run-1"))
            return {"result_ref": "proof:heartbeat", "achieved_proof_level": 2}

        executor = FrontierExecutor(
            self.queue,
            worker_id="heartbeat-worker",
            max_concurrency=1,
            lease_seconds=0.06,
        )
        result = asyncio.run(executor.run_ready("run-1", handler))

        self.assertEqual(recovered, [])
        self.assertEqual(result[0].job_id, job.job_id)
        self.assertEqual(result[0].state, FrontierJobState.SUCCEEDED)

    def test_async_executor_fences_result_after_lease_theft(self) -> None:
        job = self.submit("CC", "fenced")

        async def handler(_: FrontierJob) -> dict[str, object]:
            self.queue.recover_expired(
                "run-1",
                retry_base_seconds=0,
                now="2099-01-01T00:00:00.000000Z",
            )
            stolen = self.queue.claim(
                "run-1",
                worker_id="new-owner",
                lease_seconds=60,
                now="2099-01-01T00:00:00.000000Z",
            )
            self.assertEqual(stolen[0].job_id, job.job_id)
            return {"result_ref": "proof:stale-owner", "achieved_proof_level": 4}

        executor = FrontierExecutor(
            self.queue,
            worker_id="old-owner",
            max_concurrency=1,
            lease_seconds=0.06,
        )
        result = asyncio.run(executor.run_ready("run-1", handler))
        current = self.queue.get("run-1", job.job_id)

        self.assertIsNotNone(current)
        self.assertEqual(current.lease_owner, "new-owner")
        self.assertEqual(current.result_ref, "")
        self.assertEqual(result[0], current)


if __name__ == "__main__":
    unittest.main()
