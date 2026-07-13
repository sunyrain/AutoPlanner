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
        "artifacts",
        "node_modules",
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
) -> dict[str, Any]:
    """Prove target absence in the tracked tree and require a fresh run path."""

    root = Path(repository_root).resolve()
    destination = Path(run_dir).resolve()
    reasons: list[str] = []
    if not root.is_dir():
        reasons.append("repository_root_missing")
    if destination.exists() and any(destination.iterdir()):
        reasons.append("blind_run_directory_not_fresh")
    allowed = {
        Path(value).resolve()
        for value in additional_allowed_paths
    }
    if manifest_path is not None:
        allowed.add(Path(manifest_path).resolve())
    needles = {
        "target_smiles": case.target_smiles,
        "target_name": case.target_name.strip(),
    }
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
            for kind, needle in needles.items():
                if len(needle) < 5:
                    continue
                haystack = text if kind == "target_smiles" else lowered
                query = needle if kind == "target_smiles" else needle.casefold()
                if query in haystack:
                    matches.append(
                        {
                            "kind": kind,
                            "path": path.relative_to(root).as_posix(),
                            "content_sha256": _file_digest(path),
                        }
                    )
    if matches:
        reasons.append("target_material_already_present_in_repository")
    payload = {
        "schema_version": BLIND_PREFLIGHT_SCHEMA,
        "case": case.to_dict(),
        "repository_root": str(root),
        "run_dir": str(destination),
        "fresh_run_directory": "blind_run_directory_not_fresh" not in reasons,
        "repository_absence_attested": not matches and root.is_dir(),
        "repository_matches": sorted(matches, key=lambda row: (row["path"], row["kind"])),
        "forbidden_input_fields": _forbidden_paths(case.to_dict()),
        "accepted": not reasons,
        "reasons": sorted(set(reasons)),
        "semantics": {
            "target_only_input": True,
            "old_run_cache_forbidden": True,
            "absence_is_checked_before_model_work": True,
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
