"""Minimal case trace and RouteStatus support for P1a."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


CASE_BUNDLE_SCHEMA = "case_bundle.v1"
ARTIFACT_RECORD_SCHEMA = "case_artifact_record.v1"
FAILURE_EVENT_SCHEMA = "failure_event.v1"


class RouteStatus(str, Enum):
    SOLVED = "solved"
    SEMISYNTHESIS_CLOSED = "semisynthesis_closed"
    PARTIAL_ANCHOR = "partial_anchor"
    FAKE_CLOSED_REJECTED = "fake_closed_rejected"
    UNRESOLVED = "unresolved"


@dataclass
class FailureEvent:
    failure_id: str
    case_id: str
    reason: str
    severity: str = "medium"
    source_artifact_id: str = ""
    details: dict[str, Any] = field(default_factory=dict)
    schema_version: str = FAILURE_EVENT_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ArtifactRecord:
    artifact_id: str
    case_id: str
    artifact_type: str
    payload: Any
    source: str = "smiles_first_workflow"
    validation_status: str = "accepted"
    input_refs: list[str] = field(default_factory=list)
    evidence_refs: list[str] = field(default_factory=list)
    schema_version: str = ARTIFACT_RECORD_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CaseBundle:
    case_id: str
    route_status: RouteStatus = RouteStatus.UNRESOLVED
    artifacts: list[ArtifactRecord] = field(default_factory=list)
    failure_events: list[FailureEvent] = field(default_factory=list)
    schema_version: str = CASE_BUNDLE_SCHEMA

    def append_artifact(self, artifact: ArtifactRecord) -> None:
        if artifact.case_id != self.case_id:
            raise ValueError("artifact case_id does not match bundle case_id")
        if any(existing.artifact_id == artifact.artifact_id for existing in self.artifacts):
            raise ValueError(f"duplicate artifact_id: {artifact.artifact_id}")
        self.artifacts.append(artifact)

    def append_failure_event(self, event: FailureEvent) -> None:
        if event.case_id != self.case_id:
            raise ValueError("failure event case_id does not match bundle case_id")
        if any(existing.failure_id == event.failure_id for existing in self.failure_events):
            raise ValueError(f"duplicate failure_id: {event.failure_id}")
        self.failure_events.append(event)

    def accepted_artifacts(self, artifact_type: str | None = None) -> list[ArtifactRecord]:
        rows = [item for item in self.artifacts if item.validation_status == "accepted"]
        if artifact_type:
            rows = [item for item in rows if item.artifact_type == artifact_type]
        return rows

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "case_id": self.case_id,
            "route_status": self.route_status.value,
            "artifacts": [artifact.to_dict() for artifact in self.artifacts],
            "failure_events": [event.to_dict() for event in self.failure_events],
        }


def case_bundle_from_p0_outputs(
    *,
    route_package: dict[str, Any],
    validation: dict[str, Any],
    evidence_cards: list[dict[str, Any]],
    candidate_cards: list[dict[str, Any]],
    summary_md: str,
    strategic_disconnection_cards: list[dict[str, Any]] | None = None,
) -> CaseBundle:
    case_id = str(route_package.get("case_id") or validation.get("case_id") or "case")
    status = route_status_from_p0_validation(validation)
    bundle = CaseBundle(case_id=case_id, route_status=status)
    bundle.append_artifact(ArtifactRecord(
        artifact_id="hybrid_route_package",
        case_id=case_id,
        artifact_type="HybridRoutePackage",
        payload=route_package,
        evidence_refs=list(route_package.get("literature_evidence_refs") or []),
    ))
    bundle.append_artifact(ArtifactRecord(
        artifact_id="route_package_validation",
        case_id=case_id,
        artifact_type="RoutePackageValidation",
        payload=validation,
        validation_status="accepted" if validation.get("accepted") else "rejected",
    ))
    bundle.append_artifact(ArtifactRecord(
        artifact_id="evidence_cards",
        case_id=case_id,
        artifact_type="EvidenceCardList",
        payload=evidence_cards,
    ))
    bundle.append_artifact(ArtifactRecord(
        artifact_id="literature_candidates",
        case_id=case_id,
        artifact_type="LiteratureCandidateList",
        payload=candidate_cards,
        evidence_refs=sorted({
            ref
            for candidate in candidate_cards
            for ref in candidate.get("evidence_refs", [])
        }),
    ))
    if strategic_disconnection_cards is not None:
        bundle.append_artifact(ArtifactRecord(
            artifact_id="strategic_disconnection_cards",
            case_id=case_id,
            artifact_type="StrategicDisconnectionCardList",
            payload=strategic_disconnection_cards,
            evidence_refs=sorted({
                ref
                for card in strategic_disconnection_cards
                for ref in card.get("evidence_refs", [])
            }),
        ))
    bundle.append_artifact(ArtifactRecord(
        artifact_id="summary_md",
        case_id=case_id,
        artifact_type="SummaryMarkdown",
        payload={"text": summary_md},
    ))
    for event in failure_events_from_p0(validation, route_package):
        bundle.append_failure_event(event)
    return bundle


def load_p0_outputs_as_case_bundle(output_dir: str | Path) -> CaseBundle:
    output = Path(output_dir)
    validation = json.loads((output / "validation.json").read_text(encoding="utf-8"))
    package_path = _first(output.glob("*_hybrid_retrosynthesis_route.json"))
    candidate_path = _first(output.glob("*_literature_rxn_candidates.jsonl"))
    disconnection_paths = list(output.glob("*_strategic_disconnection_cards.jsonl"))
    route_package = json.loads(package_path.read_text(encoding="utf-8"))
    evidence_cards = _read_jsonl(output / "evidence_cards.jsonl")
    candidate_cards = _read_jsonl(candidate_path)
    strategic_disconnection_cards = _read_jsonl(disconnection_paths[0]) if disconnection_paths else None
    summary_md = (output / "summary.md").read_text(encoding="utf-8")
    return case_bundle_from_p0_outputs(
        route_package=route_package,
        validation=validation,
        evidence_cards=evidence_cards,
        candidate_cards=candidate_cards,
        summary_md=summary_md,
        strategic_disconnection_cards=strategic_disconnection_cards,
    )


def route_status_from_p0_validation(validation: dict[str, Any]) -> RouteStatus:
    if not validation.get("accepted"):
        route_status = validation.get("route_status")
        if route_status == "invalid_package":
            return RouteStatus.FAKE_CLOSED_REJECTED
        return RouteStatus.UNRESOLVED
    route_status = validation.get("route_status")
    if route_status == "partial_anchor":
        return RouteStatus.PARTIAL_ANCHOR
    if route_status == "ready_for_guided_rerun":
        return RouteStatus.UNRESOLVED
    if route_status == "literature_gap":
        return RouteStatus.UNRESOLVED
    return RouteStatus.UNRESOLVED


def failure_events_from_p0(validation: dict[str, Any], route_package: dict[str, Any]) -> list[FailureEvent]:
    case_id = str(validation.get("case_id") or route_package.get("case_id") or "case")
    events: list[FailureEvent] = []
    reasons = [str(reason) for reason in validation.get("reasons") or []]
    package_status = str(validation.get("route_status") or route_package.get("route_status") or "")
    frontier = route_package.get("frontier") or {}
    frontier_flags = [str(flag) for flag in frontier.get("flags") or []]
    if package_status in {"literature_gap", "invalid_package"}:
        for idx, reason in enumerate(reasons or [package_status], start=1):
            events.append(FailureEvent(
                failure_id=f"p0_failure_{idx}",
                case_id=case_id,
                reason=reason,
                severity="high" if package_status == "invalid_package" else "medium",
                source_artifact_id="route_package_validation",
                details={"route_status": package_status},
            ))
    unresolved_flags = [
        flag for flag in frontier_flags
        if flag in {"unresolved_core", "advanced_same_scaffold", "no_complexity_drop", "ordinary_decoration_only"}
    ]
    for idx, flag in enumerate(unresolved_flags, start=1):
        events.append(FailureEvent(
            failure_id=f"frontier_{idx}_{flag}",
            case_id=case_id,
            reason=flag,
            severity="medium",
            source_artifact_id="hybrid_route_package",
            details={"frontier_smiles": frontier.get("frontier_smiles")},
        ))
    return events


def write_case_bundle(bundle: CaseBundle, path: str | Path) -> None:
    Path(path).write_text(json.dumps(bundle.to_dict(), indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")


def load_case_bundle(path: str | Path) -> CaseBundle:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    bundle = CaseBundle(
        case_id=str(data.get("case_id") or ""),
        route_status=RouteStatus(str(data.get("route_status") or RouteStatus.UNRESOLVED.value)),
    )
    for item in data.get("artifacts") or []:
        bundle.append_artifact(ArtifactRecord(
            artifact_id=str(item.get("artifact_id") or ""),
            case_id=str(item.get("case_id") or bundle.case_id),
            artifact_type=str(item.get("artifact_type") or ""),
            payload=item.get("payload"),
            source=str(item.get("source") or "loaded"),
            validation_status=str(item.get("validation_status") or "accepted"),
            input_refs=list(item.get("input_refs") or []),
            evidence_refs=list(item.get("evidence_refs") or []),
            schema_version=str(item.get("schema_version") or ARTIFACT_RECORD_SCHEMA),
        ))
    for item in data.get("failure_events") or []:
        bundle.append_failure_event(FailureEvent(
            failure_id=str(item.get("failure_id") or ""),
            case_id=str(item.get("case_id") or bundle.case_id),
            reason=str(item.get("reason") or ""),
            severity=str(item.get("severity") or "medium"),
            source_artifact_id=str(item.get("source_artifact_id") or ""),
            details=dict(item.get("details") or {}),
            schema_version=str(item.get("schema_version") or FAILURE_EVENT_SCHEMA),
        ))
    return bundle


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _first(paths: Any) -> Path:
    for path in paths:
        return path
    raise FileNotFoundError("expected matching P0 artifact")
