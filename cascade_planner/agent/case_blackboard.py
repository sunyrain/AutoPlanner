"""Append-only case blackboard and artifact store."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


CASE_BLACKBOARD_SCHEMA = "case_blackboard.v1"
BLACKBOARD_ARTIFACT_SCHEMA = "blackboard_artifact.v1"
ARTIFACT_REJECTION_SCHEMA = "artifact_rejection.v1"

CaseId = str
RunId = str
ArtifactId = str
TraceId = str


@dataclass
class BlackboardArtifact:
    artifact_id: ArtifactId
    case_id: CaseId
    artifact_type: str
    payload: dict[str, Any]
    source: str
    trace_id: TraceId
    run_id: RunId = ""
    created_at: str = ""
    validation_status: str = "accepted"
    parent_refs: list[ArtifactId] = field(default_factory=list)
    evidence_refs: list[str] = field(default_factory=list)
    route_id: str = ""
    step_id: str = ""
    molecule_id: str = ""
    schema_version: str = BLACKBOARD_ARTIFACT_SCHEMA

    def __post_init__(self) -> None:
        if not self.created_at:
            self.created_at = _utc_now()
        if self.validation_status not in {"draft", "accepted", "validated", "rejected", "draft_only"}:
            raise ValueError("invalid_validation_status")
        if not self.artifact_id:
            raise ValueError("missing_artifact_id")
        if not self.case_id:
            raise ValueError("missing_case_id")
        if not self.trace_id:
            raise ValueError("missing_trace_id")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ArtifactRejection:
    rejection_id: str
    case_id: CaseId
    artifact_id: ArtifactId
    trace_id: TraceId
    reasons: list[str]
    source: str = "artifact_validator"
    created_at: str = ""
    schema_version: str = ARTIFACT_REJECTION_SCHEMA

    def __post_init__(self) -> None:
        if not self.created_at:
            self.created_at = _utc_now()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CaseBlackboard:
    case_id: CaseId
    target: dict[str, Any] = field(default_factory=dict)
    artifacts: list[BlackboardArtifact] = field(default_factory=list)
    rejections: list[ArtifactRejection] = field(default_factory=list)
    created_at: str = ""
    schema_version: str = CASE_BLACKBOARD_SCHEMA

    def __post_init__(self) -> None:
        if not self.case_id:
            raise ValueError("missing_case_id")
        if not self.created_at:
            self.created_at = _utc_now()

    def append_artifact(self, artifact: BlackboardArtifact) -> None:
        if artifact.case_id != self.case_id:
            raise ValueError("artifact_case_id_mismatch")
        if any(item.artifact_id == artifact.artifact_id for item in self.artifacts):
            raise ValueError(f"duplicate_artifact_id:{artifact.artifact_id}")
        self.artifacts.append(artifact)

    def reject_artifact(self, artifact_id: ArtifactId, *, trace_id: TraceId, reasons: list[str], source: str = "artifact_validator") -> ArtifactRejection:
        artifact = self.get_artifact(artifact_id)
        if artifact is not None:
            artifact.validation_status = "rejected"
        rejection = ArtifactRejection(
            rejection_id=f"reject_{len(self.rejections) + 1}_{artifact_id}",
            case_id=self.case_id,
            artifact_id=artifact_id,
            trace_id=trace_id,
            reasons=[str(reason) for reason in reasons],
            source=source,
        )
        self.rejections.append(rejection)
        return rejection

    def accepted_artifacts(self, artifact_type: str | None = None) -> list[BlackboardArtifact]:
        rows = [
            artifact for artifact in self.artifacts
            if artifact.validation_status in {"accepted", "validated"}
        ]
        if artifact_type:
            rows = [artifact for artifact in rows if artifact.artifact_type == artifact_type]
        return rows

    def artifacts_by_type(self, artifact_type: str) -> list[BlackboardArtifact]:
        return [artifact for artifact in self.artifacts if artifact.artifact_type == artifact_type]

    def artifacts_by_route_id(self, route_id: str) -> list[BlackboardArtifact]:
        return [artifact for artifact in self.artifacts if artifact.route_id == route_id]

    def artifacts_by_step_id(self, step_id: str) -> list[BlackboardArtifact]:
        return [artifact for artifact in self.artifacts if artifact.step_id == step_id]

    def artifacts_by_molecule_id(self, molecule_id: str) -> list[BlackboardArtifact]:
        return [artifact for artifact in self.artifacts if artifact.molecule_id == molecule_id]

    def get_artifact(self, artifact_id: ArtifactId) -> BlackboardArtifact | None:
        for artifact in self.artifacts:
            if artifact.artifact_id == artifact_id:
                return artifact
        return None

    def current_summary(self) -> dict[str, Any]:
        counts: dict[str, int] = {}
        for artifact in self.accepted_artifacts():
            counts[artifact.artifact_type] = counts.get(artifact.artifact_type, 0) + 1
        return {
            "schema_version": "case_blackboard_summary.v1",
            "case_id": self.case_id,
            "target": dict(self.target),
            "accepted_artifact_counts": counts,
            "artifact_count": len(self.artifacts),
            "rejection_count": len(self.rejections),
            "trace_ids": sorted({artifact.trace_id for artifact in self.artifacts} | {rej.trace_id for rej in self.rejections}),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "case_id": self.case_id,
            "target": dict(self.target),
            "created_at": self.created_at,
            "artifacts": [artifact.to_dict() for artifact in self.artifacts],
            "rejections": [rejection.to_dict() for rejection in self.rejections],
        }


def create_case(case_id: CaseId, *, target: dict[str, Any] | None = None) -> CaseBlackboard:
    return CaseBlackboard(case_id=case_id, target=dict(target or {}))


def write_blackboard(blackboard: CaseBlackboard, path: str | Path) -> None:
    Path(path).write_text(
        json.dumps(blackboard.to_dict(), indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )


def load_blackboard(path: str | Path) -> CaseBlackboard:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    board = CaseBlackboard(
        case_id=str(data.get("case_id") or ""),
        target=dict(data.get("target") or {}),
        created_at=str(data.get("created_at") or ""),
        schema_version=str(data.get("schema_version") or CASE_BLACKBOARD_SCHEMA),
    )
    for item in data.get("artifacts") or []:
        board.append_artifact(BlackboardArtifact(
            artifact_id=str(item.get("artifact_id") or ""),
            case_id=str(item.get("case_id") or board.case_id),
            artifact_type=str(item.get("artifact_type") or ""),
            payload=dict(item.get("payload") or {}),
            source=str(item.get("source") or ""),
            trace_id=str(item.get("trace_id") or ""),
            run_id=str(item.get("run_id") or ""),
            created_at=str(item.get("created_at") or ""),
            validation_status=str(item.get("validation_status") or "accepted"),
            parent_refs=[str(ref) for ref in item.get("parent_refs") or []],
            evidence_refs=[str(ref) for ref in item.get("evidence_refs") or []],
            route_id=str(item.get("route_id") or ""),
            step_id=str(item.get("step_id") or ""),
            molecule_id=str(item.get("molecule_id") or ""),
            schema_version=str(item.get("schema_version") or BLACKBOARD_ARTIFACT_SCHEMA),
        ))
    for item in data.get("rejections") or []:
        board.rejections.append(ArtifactRejection(
            rejection_id=str(item.get("rejection_id") or ""),
            case_id=str(item.get("case_id") or board.case_id),
            artifact_id=str(item.get("artifact_id") or ""),
            trace_id=str(item.get("trace_id") or ""),
            reasons=[str(reason) for reason in item.get("reasons") or []],
            source=str(item.get("source") or "artifact_validator"),
            created_at=str(item.get("created_at") or ""),
            schema_version=str(item.get("schema_version") or ARTIFACT_REJECTION_SCHEMA),
        ))
    return board


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
