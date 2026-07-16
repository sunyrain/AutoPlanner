"""Deterministic audit of the tracked current repository tree.

The audit never traverses Git history and never deletes files.  Findings are
explicit candidates for review, not proof that a scientific fixture is unused.
"""
from __future__ import annotations

import ast
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Iterable

from cascade_planner.runtime.ast_audit import type_checking_import_lines


REPOSITORY_AUDIT_SCHEMA = "autoplanner_repository_audit.v1"
_ASSET_SUFFIXES = {
    ".gif",
    ".html",
    ".jpeg",
    ".jpg",
    ".pdf",
    ".png",
    ".pptx",
    ".svg",
    ".webp",
}
_GENERATED_SUFFIXES = {
    ".aux",
    ".bbl",
    ".blg",
    ".ckpt",
    ".fls",
    ".log",
    ".npy",
    ".npz",
    ".out",
    ".pid",
    ".pth",
    ".pt",
    ".sqlite3",
    ".synctex.gz",
}
_SECRET_NAMES = {
    ".env",
    "id_ed25519",
    "id_rsa",
    "key.txt",
    "psaaword.txt",
}


def audit_repository(
    repository_root: str | Path,
    *,
    large_blob_bytes: int = 1_000_000,
    tracked_paths: Iterable[str] | None = None,
) -> dict[str, Any]:
    root = Path(repository_root).expanduser().resolve()
    paths = sorted(
        set(tracked_paths if tracked_paths is not None else _git_tracked_paths(root))
    )
    files: list[tuple[str, Path, int]] = []
    missing: list[str] = []
    for relative in paths:
        path = (root / relative).resolve()
        try:
            path.relative_to(root)
        except ValueError:
            missing.append(relative)
            continue
        if not path.is_file():
            missing.append(relative)
            continue
        files.append((relative.replace("\\", "/"), path, path.stat().st_size))

    total_bytes = sum(size for _relative, _path, size in files)
    suffix_counts = Counter(_suffix(relative) or "<none>" for relative, _, _ in files)
    large = [
        {"path": relative, "size_bytes": size}
        for relative, _path, size in files
        if size >= max(1, int(large_blob_bytes))
    ]
    generated = [
        {"path": relative, "reason": _generated_reason(relative)}
        for relative, _path, _size in files
        if _generated_reason(relative)
    ]
    credentials = [
        relative
        for relative, _path, _size in files
        if _credential_candidate(relative)
    ]
    action_files = [
        relative
        for relative, _path, _size in files
        if relative.startswith(".github/workflows/")
    ]
    duplicates = _duplicate_assets(files)
    dead_imports, parse_errors = _dead_import_candidates(files)
    launchers = _script_launchers(files)
    active_docs = [
        relative
        for relative, _path, _size in files
        if relative.startswith("docs/")
        and "/archive/" not in relative
        and relative.count("/") <= 2
    ]
    historical = [
        relative
        for relative, _path, _size in files
        if relative.startswith("archive/") or relative.startswith("docs/archive/")
    ]
    checks = {
        "no_tracked_credentials": not credentials,
        "no_github_actions": not action_files,
        "no_generated_artifacts": not generated,
        "no_missing_tracked_files": not missing,
    }
    report: dict[str, Any] = {
        "schema_version": REPOSITORY_AUDIT_SCHEMA,
        "repository_root": str(root),
        "status": "clean" if all(checks.values()) else "attention_required",
        "checks": checks,
        "tracked": {
            "file_count": len(files),
            "total_bytes": total_bytes,
            "large_blob_threshold_bytes": max(1, int(large_blob_bytes)),
            "large_blobs": sorted(large, key=lambda row: (-row["size_bytes"], row["path"])),
            "suffix_counts": dict(sorted(suffix_counts.items())),
            "missing": missing,
        },
        "duplicate_assets": duplicates,
        "dead_import_candidates": dead_imports,
        "python_parse_errors": parse_errors,
        "generated_artifact_candidates": generated,
        "credential_candidates": credentials,
        "github_action_files": action_files,
        "script_launchers": launchers,
        "documentation": {
            "active_document_count": len(active_docs),
            "historical_tracked_file_count": len(historical),
            "active_documents": active_docs,
        },
        "semantics": {
            "current_tree_only": True,
            "read_only": True,
            "dead_imports_are_review_candidates": True,
            "history_rewrite_not_performed": True,
        },
    }
    report["content_sha256"] = _digest(report)
    return report


def _git_tracked_paths(root: Path) -> list[str]:
    completed = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=root,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode != 0:
        reason = completed.stderr.decode("utf-8", errors="replace").strip()
        raise ValueError(f"git_ls_files_failed:{reason}")
    return [
        item.decode("utf-8", errors="strict")
        for item in completed.stdout.split(b"\0")
        if item
    ]


def _duplicate_assets(files: list[tuple[str, Path, int]]) -> list[dict[str, Any]]:
    by_digest: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for relative, path, size in files:
        if _suffix(relative) not in _ASSET_SUFFIXES:
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        by_digest[digest].append({"path": relative, "size_bytes": size})
    rows = []
    for digest, members in by_digest.items():
        if len(members) < 2:
            continue
        ordered = sorted(members, key=lambda row: row["path"])
        rows.append(
            {
                "sha256": digest,
                "copy_count": len(ordered),
                "duplicate_bytes": ordered[0]["size_bytes"] * (len(ordered) - 1),
                "files": ordered,
            }
        )
    return sorted(rows, key=lambda row: (-row["duplicate_bytes"], row["sha256"]))


def _dead_import_candidates(
    files: list[tuple[str, Path, int]],
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    candidates: list[dict[str, Any]] = []
    parse_errors: list[dict[str, str]] = []
    for relative, path, _size in files:
        if _suffix(relative) != ".py" or path.name == "__init__.py":
            continue
        if not relative.startswith(("cascade_planner/", "scripts/", "tests/")):
            continue
        source = path.read_text(encoding="utf-8")
        try:
            tree = ast.parse(source, filename=relative)
        except SyntaxError as exc:
            parse_errors.append({"path": relative, "reason": str(exc)})
            continue
        loaded = {
            node.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
        }
        loaded.update(_exported_names(tree))
        type_checking_lines = type_checking_import_lines(tree)
        lines = source.splitlines()
        for node in ast.walk(tree):
            if getattr(node, "lineno", -1) in type_checking_lines:
                continue
            bindings: list[tuple[str, str]] = []
            if isinstance(node, ast.Import):
                bindings = [
                    (alias.asname or alias.name.split(".")[0], alias.name)
                    for alias in node.names
                ]
            elif isinstance(node, ast.ImportFrom):
                if node.module == "__future__":
                    continue
                bindings = [
                    (alias.asname or alias.name, f"{node.module or ''}.{alias.name}")
                    for alias in node.names
                    if alias.name != "*" and alias.asname != alias.name
                ]
            if not bindings:
                continue
            line = lines[node.lineno - 1] if node.lineno <= len(lines) else ""
            if "noqa" in line.lower():
                continue
            for binding, imported in bindings:
                if binding not in loaded:
                    candidates.append(
                        {
                            "path": relative,
                            "line": node.lineno,
                            "binding": binding,
                            "imported": imported,
                        }
                    )
    return (
        sorted(candidates, key=lambda row: (row["path"], row["line"], row["binding"])),
        parse_errors,
    )
def _exported_names(tree: ast.AST) -> set[str]:
    """Treat explicit ``__all__`` re-exports as intentional import uses."""

    exported: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if not any(isinstance(target, ast.Name) and target.id == "__all__" for target in targets):
            continue
        value = node.value
        if isinstance(value, (ast.List, ast.Tuple, ast.Set)):
            exported.update(
                item.value
                for item in value.elts
                if isinstance(item, ast.Constant) and isinstance(item.value, str)
            )
    return exported


def _script_launchers(files: list[tuple[str, Path, int]]) -> list[dict[str, Any]]:
    rows = []
    for relative, path, size in files:
        if not relative.startswith("scripts/") or _suffix(relative) != ".py":
            continue
        source = path.read_text(encoding="utf-8")
        if "if __name__" in source and "__main__" in source:
            rows.append({"path": relative, "size_bytes": size})
    return sorted(rows, key=lambda row: row["path"])


def _generated_reason(relative: str) -> str:
    lowered = relative.lower()
    suffix = _suffix(lowered)
    if suffix in _GENERATED_SUFFIXES:
        return "generated_suffix"
    if lowered.startswith(("results/", "releases/", "workspace/")):
        return "runtime_output_tree"
    if lowered.startswith("docs/archive/") and suffix in _ASSET_SUFFIXES:
        return "archived_rendered_report"
    if "/report_" in lowered and suffix in {".html", ".pdf", ".png"}:
        return "rendered_report"
    return ""


def _credential_candidate(relative: str) -> bool:
    path = Path(relative)
    lowered = path.name.lower()
    structured_secret = path.suffix.lower() in {
        ".json",
        ".toml",
        ".yaml",
        ".yml",
    }
    return (
        lowered in _SECRET_NAMES
        or ("credential" in lowered and structured_secret)
        or lowered.endswith((".pem", ".p12", ".pfx", ".key"))
    )


def _suffix(relative: str) -> str:
    lowered = relative.lower()
    if lowered.endswith(".synctex.gz"):
        return ".synctex.gz"
    return Path(lowered).suffix


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


__all__ = ["REPOSITORY_AUDIT_SCHEMA", "audit_repository"]
