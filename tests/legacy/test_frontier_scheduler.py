from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path
import tempfile
import unittest

from cascade_planner.legacy.application_runtime.frontier_scheduler import (
    FrontierExecutor,
    FrontierJob,
    FrontierJobState,
    FrontierLeaseError,
    FrontierQueueError,
    FrontierScheduler,
    PersistentFrontierQueue,
    assess_frontier_completeness,
)
from cascade_planner.providers.stock import (
    BenchmarkCatalogStockProvider,
    SnapshotStockProvider,
    stock_snapshot_sha256,
)


NOW = "2026-07-10T00:00:00.000000Z"
LATER = "2026-07-11T00:00:00.000000Z"


def _snapshot(
    supplier: str,
    catalog_number: str,
    *,
    available: bool = True,
    checked_at: str = NOW,
) -> dict[str, object]:
    return {
        "schema_version": "stock_offer_snapshot.v1",
        "smiles": "CCO",
        "supplier": supplier,
        "catalog_number": catalog_number,
        "checked_at": checked_at,
        "available": available,
    }


def _offer(
    supplier: str,
    catalog_number: str,
    *,
    available: bool = True,
    checked_at: str = NOW,
) -> dict[str, object]:
    snapshot = _snapshot(
        supplier,
        catalog_number,
        available=available,
        checked_at=checked_at,
    )
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
        now = str(kwargs.pop("now", NOW))
        return self.scheduler.submit(
            run_id="run-1",
            case_id="paclitaxel",
            frontier_smiles=smiles,
            frontier_node_id=f"molecule:{key}",
            idempotency_key=key,
            now=now,
            **kwargs,
        )

    def trusted_stock_providers(self) -> dict[str, object]:
        return {
            provider.descriptor.provider_id: provider
            for provider in self.scheduler.stock_providers
        }

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
        self.assertEqual(job.state, FrontierJobState.PENDING)
        self.assertEqual(job.closure_kind, "")
        self.assertEqual(job.achieved_proof_level, 0)
        self.assertEqual(
            job.metadata["stock_boundary_authority"], "procurement_boundary"
        )
        self.assertEqual(
            self.queue.claim(
                "run-1",
                worker_id="worker",
                trusted_stock_provider_instances=self.trusted_stock_providers(),
            ),
            [],
        )
        self.assertTrue(job.metadata["stock_audit_preceded_agent_work"])
        self.assertTrue(job.metadata["stock_observation_current_closed"])
        self.assertEqual(
            job.metadata["stock_observations"]["schema_version"],
            "stock_observation_state.v1",
        )

    def test_pending_frontier_refreshes_to_available_without_becoming_work_success(
        self,
    ) -> None:
        pending = self.submit("CCO", "refresh-available")
        self.assertEqual(pending.state, FrontierJobState.PENDING)
        self.assertFalse(pending.metadata["stock_observation_current_closed"])

        refreshed = self.submit(
            "CCO",
            "refresh-available",
            stock_request={"offers": [_offer("test-supplier", "E-1")]},
            now=LATER,
        )

        self.assertEqual(refreshed.job_id, pending.job_id)
        self.assertEqual(refreshed.state, FrontierJobState.PENDING)
        self.assertEqual(refreshed.closure_kind, "")
        self.assertTrue(refreshed.metadata["stock_observation_current_closed"])
        self.assertEqual(
            len(refreshed.metadata["stock_observations"]["history"]), 2
        )
        self.assertEqual(
            self.queue.claim(
                "run-1",
                worker_id="worker",
                trusted_stock_provider_instances=self.trusted_stock_providers(),
            ),
            [],
        )

    def test_serialized_positive_stock_observation_cannot_suppress_work_without_replay(
        self,
    ) -> None:
        job = self.submit(
            "CCO",
            "host-replay-required",
            stock_request={"offers": [_offer("test-supplier", "E-1")]},
        )

        self.assertEqual(
            self.queue.claim(
                "run-1",
                worker_id="trusted-worker",
                trusted_stock_provider_instances=self.trusted_stock_providers(),
            ),
            [],
        )
        claimed = self.queue.claim("run-1", worker_id="fail-open-worker")

        self.assertEqual([row.job_id for row in claimed], [job.job_id])

    def test_available_frontier_can_be_revoked_and_become_claimable(self) -> None:
        available = self.submit(
            "CCO",
            "refresh-revoked",
            stock_request={"offers": [_offer("test-supplier", "E-1")]},
        )
        self.assertTrue(available.metadata["stock_observation_current_closed"])
        unavailable_snapshot = _snapshot(
            "test-supplier",
            "E-1",
            available=False,
            checked_at=LATER,
        )
        self.scheduler = FrontierScheduler(
            self.queue,
            SnapshotStockProvider(trusted_snapshots=[unavailable_snapshot]),
        )

        revoked = self.submit(
            "CCO",
            "refresh-revoked",
            stock_request={
                "offers": [
                    {
                        **unavailable_snapshot,
                        "snapshot_sha256": stock_snapshot_sha256(
                            unavailable_snapshot
                        ),
                    }
                ]
            },
            now=LATER,
        )

        self.assertFalse(revoked.metadata["stock_observation_current_closed"])
        self.assertEqual(
            len(revoked.metadata["stock_observations"]["history"]), 2
        )
        claimed = self.queue.claim("run-1", worker_id="worker", now=LATER)
        self.assertEqual([row.job_id for row in claimed], [revoked.job_id])

    def test_provider_set_preserves_benchmark_and_adds_commercial_authority(
        self,
    ) -> None:
        catalog = Path(self.temp.name) / "stock.smi"
        catalog.write_text("CCO\n", encoding="utf-8")
        benchmark = BenchmarkCatalogStockProvider(
            catalog_artifact=catalog,
            catalog_sha256=hashlib.sha256(catalog.read_bytes()).hexdigest(),
            catalog_name="fixture",
        )
        self.scheduler = FrontierScheduler(self.queue, benchmark)
        benchmark_only = self.submit("CCO", "provider-set")
        self.assertEqual(
            benchmark_only.metadata["stock_boundary_authority"],
            "benchmark_membership_only",
        )

        snapshot = _snapshot("test-supplier", "E-1")
        commercial = SnapshotStockProvider(trusted_snapshots=[snapshot])
        self.scheduler = FrontierScheduler(
            self.queue,
            [benchmark, commercial],
        )
        upgraded = self.submit(
            "CCO",
            "provider-set",
            stock_request={"offers": [_offer("test-supplier", "E-1")]},
            now=LATER,
        )

        current = upgraded.metadata["stock_observations"]["current"]
        self.assertEqual(len(current), 2)
        self.assertEqual(
            upgraded.metadata["stock_boundary_authority"],
            "procurement_boundary",
        )
        self.assertEqual(
            {
                row["provider_result"]["payload"]["boundary_type"]
                for row in current
            },
            {"benchmark_stock", "commercially_orderable"},
        )

    def test_proposal_success_survives_stock_refresh_and_revocation(self) -> None:
        job = self.submit("CCO", "proposal-retained")
        lease = self.queue.claim(
            "run-1",
            worker_id="worker",
            now=NOW,
            trusted_stock_provider_instances=self.trusted_stock_providers(),
        )[0]
        completed = self.queue.complete(
            "run-1",
            job.job_id,
            lease_token=lease.lease_token,
            result_ref="campaign_commits/proposal.json",
            closure_kind="proposal_expansion",
            achieved_proof_level=0,
            now=NOW,
        )
        self.assertEqual(completed.closure_kind, "proposal_expansion")

        refreshed = self.submit(
            "CCO",
            "proposal-retained",
            stock_request={"offers": [_offer("test-supplier", "E-1")]},
            now=LATER,
        )

        self.assertEqual(refreshed.state, FrontierJobState.SUCCEEDED)
        self.assertEqual(refreshed.closure_kind, "proposal_expansion")
        self.assertEqual(refreshed.result_ref, "campaign_commits/proposal.json")
        self.assertTrue(refreshed.metadata["stock_observation_current_closed"])

    def test_new_inbound_parent_edge_is_monotonically_merged_before_unlock(
        self,
    ) -> None:
        identity = "a" * 64
        policy = "b" * 64
        job = self.submit(
            "CC",
            "parent-edge-merge",
            metadata={
                "depth": 1,
                "campaign_identity_sha256": identity,
                "campaign_policy_sha256": policy,
                "campaign_root_smiles": "CCO",
                "parent_step_ids": ["step:old"],
                "proposal_expansion_allowed": False,
            },
        )

        merged = self.queue.merge_parent_step_ids(
            "run-1",
            job.job_id,
            parent_step_ids=["step:new", "step:old"],
            campaign_identity_sha256=identity,
            campaign_policy_sha256=policy,
            campaign_root_smiles="CCO",
            now=LATER,
        )
        self.assertEqual(
            merged.metadata["parent_step_ids"],
            ["step:new", "step:old"],
        )

        enabled = self.queue.enable_proposal_expansion(
            "run-1",
            job.job_id,
            validated_parent_step_ids=["step:new"],
            campaign_identity_sha256=identity,
            campaign_root_smiles="CCO",
            now=LATER,
        )
        self.assertTrue(enabled.metadata["proposal_expansion_allowed"])
        self.assertEqual(
            enabled.metadata["proposal_expansion_gate"][
                "validated_parent_step_ids"
            ],
            ["step:new"],
        )

    def test_legacy_benchmark_level_four_is_downgraded_without_losing_boundary(
        self,
    ) -> None:
        legacy = FrontierJob(
            run_id="run-1",
            job_id="frontier:legacy-benchmark",
            idempotency_key="legacy-benchmark",
            frontier_smiles="CC",
            frontier_node_id="molecule:legacy-benchmark",
            state=FrontierJobState.SUCCEEDED,
            closure_kind="stock_boundary",
            achieved_proof_level=4,
            result_ref="provider-result:sha256:" + "a" * 64,
            created_at=NOW,
            updated_at=NOW,
            metadata={
                "stock_audit": {
                    "payload": {
                        "schema_version": "stock_boundary.v1",
                        "accepted": True,
                        "boundary_type": "benchmark_stock",
                        "canonical_smiles": "CC",
                    }
                }
            },
        )
        self.queue.enqueue(legacy)

        changed = self.queue.migrate_legacy_benchmark_stock_authority(
            "run-1", now=NOW
        )
        migrated = self.queue.get("run-1", legacy.job_id)

        self.assertEqual(len(changed), 1)
        self.assertIsNotNone(migrated)
        self.assertEqual(migrated.closure_kind, "stock_boundary")
        self.assertEqual(migrated.achieved_proof_level, 0)
        self.assertEqual(
            migrated.metadata["stock_boundary_authority"],
            "benchmark_membership_only",
        )
        self.assertEqual(
            self.queue.migrate_legacy_benchmark_stock_authority("run-1", now=NOW),
            [],
        )

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

    def test_queue_reaction_work_cannot_impersonate_terminal_closure(self) -> None:
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
        lease = self.queue.claim(
            "run-1",
            worker_id="worker",
            now=NOW,
            trusted_stock_provider_instances=self.trusted_stock_providers(),
        )[0]
        reaction = self.queue.complete(
            "run-1",
            reaction.job_id,
            lease_token=lease.lease_token,
            result_ref="proof:cc",
            achieved_proof_level=2,
        )
        trusted = {
            provider.descriptor.provider_id: provider
            for provider in self.scheduler.stock_providers
        }
        closed = assess_frontier_completeness(
            ["CCO", "CC"],
            [stock, reaction],
            trusted_stock_provider_instances=trusted,
        )
        self.assertFalse(closed.complete)
        self.assertTrue(
            closed.unresolved_frontiers[0][
                "queue_work_cannot_authorize_reaction_closure"
            ]
        )
        open_report = assess_frontier_completeness(
            ["CCO", "CC"],
            [stock, reaction],
            open_proof_frontiers=["step:x"],
            trusted_stock_provider_instances=trusted,
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
