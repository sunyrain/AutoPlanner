"""Fail-closed contracts for target-only retrosynthesis benchmarks.

Blindness is an input property, not a claim made by a successful model run.
This module validates a deliberately tiny manifest and audits the tracked
repository before a benchmark starts.  It has no route-generation behavior and
therefore cannot grant chemistry or acceptance authority.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import re
import subprocess
from typing import Any, Iterable, Mapping

from rdkit import Chem, RDLogger


RDLogger.DisableLog("rdApp.*")
BLIND_CASE_SCHEMA = "blind_retrosynthesis_case.v1"
BLIND_MANIFEST_SCHEMA = "blind_retrosynthesis_manifest.v1"
BLIND_PREFLIGHT_SCHEMA = "blind_retrosynthesis_preflight.v1"
_CASE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$")
_BOUNDARIES = frozenset({"benchmark_search", "procurement", "in_house"})
_CASE_FIELDS = frozenset(
    {
        "schema_version",
        "case_id",
        "target_name",
        "target_smiles",
        "acceptance",
        "budget",
    }
)
_ACCEPTANCE_FIELDS = frozenset(
    {
        "minimum_complete_routes",
        "minimum_edge_proof_level",
        "minimum_independent_source_groups",
        "minimum_planning_route_steps",
        "stock_boundary",
    }
)
_BUDGET_FIELDS = frozenset(
    {
        "max_model_invocations",
        "max_total_input_tokens",
        "max_total_output_tokens",
        "max_total_wall_time_s",
        "max_accepted_expansions",
        "max_attempt_runs",
        "max_native_search_invocations",
        "min_target_native_search_invocations",
        "max_frontier_native_search_invocations",
        "allow_frontier_native_search_borrowing",
        "max_prompt_context_bytes",
    }
)
_FORBIDDEN_KEYS = frozenset(
    {
        "answer",
        "dossier",
        "evidence",
        "exact_rows",
        "fixture",
        "global_plan",
        "inventory",
        "literature_sources",
        "mapped_reaction_smiles",
        "precursor_smiles",
        "precursors",
        "reaction_smiles",
        "replay",
        "route",
        "routes",
        "source_refs",
        "sources",
        "stock_offers",
        "templates",
    }
)
_SKIP_PARTS = frozenset(
    {
        ".git",
        ".pytest_cache",
        ".ruff_cache",
        ".venv",
        "__pycache__",
        ".autoplanner",
        "artifacts",
        "data_external",
        "node_modules",
        "results",
        "runtime",
        "runs",
    }
)


class BlindBenchmarkError(ValueError):
    """The requested benchmark is not demonstrably target-only and fresh."""


@dataclass(frozen=True, slots=True)
class BlindCase:
    case_id: str
    target_name: str
    target_smiles: str
    acceptance: Mapping[str, Any] = field(default_factory=dict)
    budget: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = BLIND_CASE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != BLIND_CASE_SCHEMA:
            raise BlindBenchmarkError("blind_case_schema_invalid")
        if not _CASE_ID.fullmatch(self.case_id):
            raise BlindBenchmarkError("blind_case_id_invalid")
        if not self.target_name.strip():
            raise BlindBenchmarkError("blind_target_name_missing")
        canonical = canonical_smiles(self.target_smiles)
        if not canonical:
            raise BlindBenchmarkError("blind_target_smiles_invalid")
        if canonical != self.target_smiles:
            raise BlindBenchmarkError("blind_target_smiles_not_canonical")
        _validate_options(self.acceptance, _ACCEPTANCE_FIELDS, "acceptance")
        _validate_options(self.budget, _BUDGET_FIELDS, "budget")
        boundary = str(self.acceptance.get("stock_boundary") or "")
        if boundary and boundary not in _BOUNDARIES:
            raise BlindBenchmarkError("blind_stock_boundary_invalid")
        planning_depth = self.acceptance.get("minimum_planning_route_steps", 0)
        if (
            isinstance(planning_depth, bool)
            or not isinstance(planning_depth, int)
            or not 0 <= planning_depth <= 24
        ):
            raise BlindBenchmarkError("blind_minimum_planning_route_steps_invalid")
        if _forbidden_paths(self.to_dict()):
            raise BlindBenchmarkError("blind_case_contains_forbidden_route_material")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "BlindCase":
        row = dict(value)
        unknown = sorted(set(row) - _CASE_FIELDS)
        if unknown:
            raise BlindBenchmarkError("blind_case_fields_forbidden:" + ",".join(unknown))
        return cls(
            case_id=str(row.get("case_id") or ""),
            target_name=str(row.get("target_name") or ""),
            target_smiles=str(row.get("target_smiles") or ""),
            acceptance=dict(row.get("acceptance") or {}),
            budget=dict(row.get("budget") or {}),
            schema_version=str(row.get("schema_version") or ""),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "case_id": self.case_id,
            "target_name": self.target_name,
            "target_smiles": self.target_smiles,
            "acceptance": dict(self.acceptance),
            "budget": dict(self.budget),
        }


def load_blind_manifest(path: str | Path) -> tuple[BlindCase, ...]:
    manifest_path = Path(path).resolve()
    try:
        value = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BlindBenchmarkError("blind_manifest_not_readable") from exc
    if not isinstance(value, Mapping):
        raise BlindBenchmarkError("blind_manifest_not_object")
    unknown = sorted(set(value) - {"schema_version", "cases"})
    if unknown:
        raise BlindBenchmarkError("blind_manifest_fields_forbidden:" + ",".join(unknown))
    if value.get("schema_version") != BLIND_MANIFEST_SCHEMA:
        raise BlindBenchmarkError("blind_manifest_schema_invalid")
    raw_cases = value.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise BlindBenchmarkError("blind_manifest_cases_missing")
    cases = tuple(
        BlindCase.from_dict(row) if isinstance(row, Mapping) else _raise_case_object()
        for row in raw_cases
    )
    case_ids = [case.case_id for case in cases]
    if len(case_ids) != len(set(case_ids)):
        raise BlindBenchmarkError("blind_manifest_case_id_duplicate")
    targets = [case.target_smiles for case in cases]
    if len(targets) != len(set(targets)):
        raise BlindBenchmarkError("blind_manifest_target_duplicate")
    return cases


def audit_blind_preflight(
    case: BlindCase,
    *,
    repository_root: str | Path,
    run_dir: str | Path,
    manifest_path: str | Path | None = None,
    additional_allowed_paths: Iterable[str | Path] = (),
    additional_leakage_needles: Mapping[str, Iterable[str]] | None = None,
    target_synonym_not_applicable_reason: str = "",
) -> dict[str, Any]:
    """Prove target absence in the tracked tree and require a fresh run path."""

    root = Path(repository_root).resolve()
    destination = Path(run_dir).resolve()
    reasons: list[str] = []
    if not root.is_dir():
        reasons.append("repository_root_missing")
    if destination.exists() and any(destination.iterdir()):
        reasons.append("blind_run_directory_not_fresh")
    allowed = {Path(value).resolve() for value in additional_allowed_paths}
    if manifest_path is not None:
        allowed.add(Path(manifest_path).resolve())
    target_molecule = Chem.MolFromSmiles(case.target_smiles)
    target_inchikey = Chem.MolToInchiKey(target_molecule) if target_molecule else ""
    needles = {
        "target_smiles": case.target_smiles,
        "target_inchikey": target_inchikey,
    }
    target_name = case.target_name.strip()
    if target_name.casefold() not in {
        "blind target",
        "blind molecule",
        "opaque target",
        "target",
        "unknown target",
    }:
        needles["target_name"] = target_name
    extra_needles: dict[str, list[str]] = {}
    for kind, values in dict(additional_leakage_needles or {}).items():
        if kind not in {
            "target_synonym",
            "key_intermediate_smiles",
            "key_intermediate_inchikey",
        }:
            raise BlindBenchmarkError(f"blind_leakage_needle_kind_invalid:{kind}")
        extra_needles[kind] = sorted(
            {str(value).strip() for value in values if len(str(value).strip()) >= 5}
        )
    synonym_not_applicable = str(target_synonym_not_applicable_reason or "").strip()
    if synonym_not_applicable and len(synonym_not_applicable) < 8:
        raise BlindBenchmarkError("blind_synonym_not_applicable_reason_invalid")
    opaque_identity = target_name.casefold() in {
        "blind target",
        "blind molecule",
        "opaque target",
        "target",
        "unknown target",
    } or target_name.casefold().startswith("opaque benchmark target ")
    if synonym_not_applicable and not opaque_identity:
        reasons.append("target_synonym_audit_not_applicable_for_named_target")
    matches: list[dict[str, Any]] = []
    if root.is_dir():
        for path in _tracked_files(root):
            resolved = path.resolve()
            if resolved in allowed or any(part in _SKIP_PARTS for part in path.parts):
                continue
            if path.stat().st_size > 2_000_000:
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            lowered = text.casefold()
            scan_values = [
                *needles.items(),
                *((kind, needle) for kind, values in extra_needles.items() for needle in values),
            ]
            for kind, needle in scan_values:
                if len(needle) < 5:
                    continue
                exact_text = kind in {"target_smiles", "key_intermediate_smiles"}
                haystack = text if exact_text else lowered
                query = needle if exact_text else needle.casefold()
                if query in haystack:
                    matches.append(
                        {
                            "kind": kind,
                            "needle_sha256": hashlib.sha256(needle.encode("utf-8")).hexdigest(),
                            "path": path.relative_to(root).as_posix(),
                            "content_sha256": _file_digest(path),
                        }
                    )
    if matches:
        if any(str(row.get("kind") or "").startswith("key_intermediate") for row in matches):
            reasons.append("evaluator_answer_material_already_present_in_repository")
        if any(not str(row.get("kind") or "").startswith("key_intermediate") for row in matches):
            reasons.append("target_material_already_present_in_repository")
    payload = {
        "schema_version": BLIND_PREFLIGHT_SCHEMA,
        "case": case.to_dict(),
        "repository_root": str(root),
        "run_dir": str(destination),
        "fresh_run_directory": "blind_run_directory_not_fresh" not in reasons,
        "repository_absence_attested": not matches and root.is_dir(),
        "repository_matches": sorted(matches, key=lambda row: (row["path"], row["kind"])),
        "additional_leakage_needle_counts": {
            kind: len(values) for kind, values in sorted(extra_needles.items())
        },
        "target_synonym_not_applicable_reason": synonym_not_applicable,
        "forbidden_input_fields": _forbidden_paths(case.to_dict()),
        "accepted": not reasons,
        "reasons": sorted(set(reasons)),
        "semantics": {
            "target_only_input": True,
            "old_run_cache_forbidden": True,
            "absence_is_checked_before_model_work": True,
            "target_name_smiles_and_inchikey_checked": True,
            "synonym_and_intermediate_needles_require_an_evaluator_only_pack": True,
            "target_synonym_needles_checked": bool(extra_needles.get("target_synonym"))
            or bool(synonym_not_applicable),
            "target_synonym_audit_not_applicable": bool(synonym_not_applicable),
            "key_intermediate_needles_checked": bool(
                extra_needles.get("key_intermediate_smiles")
                or extra_needles.get("key_intermediate_inchikey")
            ),
            "additional_needle_values_are_not_emitted": True,
            "preflight_grants_no_chemistry_authority": True,
        },
    }
    payload["content_sha256"] = _digest(payload)
    return payload


def canonical_smiles(value: Any) -> str:
    molecule = Chem.MolFromSmiles(str(value or "").strip())
    if molecule is None:
        return ""
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)


def _validate_options(
    values: Mapping[str, Any],
    allowed: frozenset[str],
    label: str,
) -> None:
    if not isinstance(values, Mapping):
        raise BlindBenchmarkError(f"blind_{label}_not_object")
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise BlindBenchmarkError(f"blind_{label}_fields_forbidden:" + ",".join(unknown))


def _forbidden_paths(value: Any, *, path: str = "$") -> list[str]:
    found: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_path = f"{path}.{key}"
            if str(key).casefold() in _FORBIDDEN_KEYS:
                found.append(child_path)
            found.extend(_forbidden_paths(child, path=child_path))
    elif isinstance(value, list | tuple):
        for index, child in enumerate(value):
            found.extend(_forbidden_paths(child, path=f"{path}[{index}]"))
    return sorted(set(found))


def _tracked_files(root: Path) -> list[Path]:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "ls-files", "-z"],
            check=True,
            capture_output=True,
            timeout=15,
        )
        values = result.stdout.decode("utf-8", errors="strict").split("\0")
        return [root / value for value in values if value and (root / value).is_file()]
    except (OSError, subprocess.SubprocessError, UnicodeDecodeError):
        return [
            path
            for path in root.rglob("*")
            if path.is_file() and not any(part in _SKIP_PARTS for part in path.parts)
        ]


def _raise_case_object() -> BlindCase:
    raise BlindBenchmarkError("blind_manifest_case_not_object")


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _digest(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


__all__ = [
    "BLIND_CASE_SCHEMA",
    "BLIND_MANIFEST_SCHEMA",
    "BLIND_PREFLIGHT_SCHEMA",
    "BlindBenchmarkError",
    "BlindCase",
    "audit_blind_preflight",
    "canonical_smiles",
    "load_blind_manifest",
]
