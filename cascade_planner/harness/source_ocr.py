"""Bounded, hash-bound local OCR for image-only primary-source pages."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Any, Callable, Iterable, Mapping


LOCAL_OCR_MATERIALIZATION_SCHEMA = "local_source_ocr_materialization.v1"
HASH_BOUND_OCR_FORMAT = "hash_bound_ocr_pages.v1"
SUPPORTED_LOCAL_OCR_ENGINES = frozenset({"tesseract"})
OcrRunner = Callable[[Path, int, "LocalOcrConfig"], Mapping[str, Any]]


@dataclass(frozen=True, slots=True)
class LocalOcrConfig:
    max_pages: int = 12
    min_native_text_chars: int = 80
    min_ocr_text_chars: int = 20
    max_text_chars_per_page: int = 64_000
    timeout_s_per_page: float = 12.0
    languages: tuple[str, ...] = ("eng",)
    page_segmentation_mode: int = 6

    def __post_init__(self) -> None:
        if not 1 <= self.max_pages <= 80:
            raise ValueError("local_ocr_page_limit_invalid")
        if self.min_native_text_chars < 0 or self.min_ocr_text_chars < 1:
            raise ValueError("local_ocr_text_threshold_invalid")
        if self.max_text_chars_per_page < 1_000:
            raise ValueError("local_ocr_text_limit_invalid")
        if self.timeout_s_per_page <= 0:
            raise ValueError("local_ocr_timeout_invalid")
        if not self.languages or any(not str(value).strip() for value in self.languages):
            raise ValueError("local_ocr_languages_invalid")
        if not 3 <= self.page_segmentation_mode <= 13:
            raise ValueError("local_ocr_page_segmentation_mode_invalid")


def materialize_local_ocr_companion(
    *,
    pdf_path: str | Path,
    source_ref: str,
    rendered_pages: Iterable[Mapping[str, Any]],
    output_dir: str | Path,
    target_terms: Iterable[str] = (),
    config: LocalOcrConfig | None = None,
    runner: OcrRunner | None = None,
) -> dict[str, Any]:
    """OCR only low-text pages and emit a replayable companion specification."""

    active = config or LocalOcrConfig()
    source_pdf = Path(pdf_path).expanduser().resolve()
    out = Path(output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    source_key = str(source_ref or "").strip().lower()
    if not source_pdf.is_file() or not source_key:
        return _result(
            status="failed",
            reasons=["local_ocr_source_binding_invalid"],
        )
    pdf_sha256 = _sha256(source_pdf)
    native_counts = _native_text_counts(source_pdf)
    images = {
        int(row.get("page_number") or 0): dict(row)
        for row in rendered_pages
        if isinstance(row, Mapping) and int(row.get("page_number") or 0) > 0
    }
    low_text_pages = [
        page_number
        for page_number, count in sorted(native_counts.items())
        if count < active.min_native_text_chars and page_number in images
    ]
    selected = _stratified_page_selection(low_text_pages, limit=active.max_pages)
    if not selected:
        return _result(
            status="not_needed",
            source_pdf_sha256=pdf_sha256,
            native_text_page_count=sum(
                count >= active.min_native_text_chars for count in native_counts.values()
            ),
            low_text_page_count=len(low_text_pages),
        )
    execute = runner or _tesseract_runner
    if runner is None and not shutil.which("tesseract"):
        return _result(
            status="unavailable",
            reasons=["local_ocr_engine_unavailable:tesseract"],
            source_pdf_sha256=pdf_sha256,
            low_text_page_count=len(low_text_pages),
            selected_page_count=len(selected),
            selected_page_numbers=selected,
            coverage_truncated=len(low_text_pages) > len(selected),
            visual_candidate_pages=_visual_pages(selected, images),
        )

    page_specs: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    ocr_text_by_page: dict[int, str] = {}
    for page_number in selected:
        image = images[page_number]
        image_path = Path(str(image.get("image_path") or "")).expanduser().resolve()
        image_sha256 = str(image.get("sha256") or image.get("image_sha256") or "")
        if (
            not image_path.is_file()
            or not _is_sha256(image_sha256)
            or _sha256(image_path) != image_sha256
        ):
            failures.append({"page_number": page_number, "reason": "rendered_page_invalid"})
            continue
        try:
            raw = dict(execute(image_path, page_number, active))
        except (
            OSError,
            RuntimeError,
            TimeoutError,
            ValueError,
            subprocess.SubprocessError,
        ) as exc:
            failures.append(
                {
                    "page_number": page_number,
                    "reason": f"{type(exc).__name__}:{str(exc)[:300]}",
                }
            )
            continue
        text = _bounded_ocr_text(raw.get("text"), limit=active.max_text_chars_per_page)
        if len(re.sub(r"\s+", "", text)) < active.min_ocr_text_chars:
            failures.append({"page_number": page_number, "reason": "ocr_text_too_sparse"})
            continue
        engine_id = str(raw.get("engine_id") or "").strip()
        engine_version = str(raw.get("engine_version") or "").strip()
        if not engine_id or not engine_version:
            failures.append({"page_number": page_number, "reason": "ocr_engine_identity_missing"})
            continue
        if engine_id.casefold() not in SUPPORTED_LOCAL_OCR_ENGINES:
            failures.append(
                {"page_number": page_number, "reason": "ocr_engine_not_allowlisted"}
            )
            continue
        text_path = out / f"page_{page_number:03d}.ocr.txt"
        _write_text_atomic(text_path, text)
        text_sha256 = _sha256(text_path)
        ocr_text_by_page[page_number] = text
        page_specs.append(
            {
                "page_number": page_number,
                "image_path": str(image_path),
                "image_sha256": image_sha256,
                "text_path": str(text_path),
                "text_sha256": text_sha256,
                "engine_id": engine_id,
                "engine_version": engine_version,
            }
        )
    companion: dict[str, Any] = {}
    if page_specs:
        companion = {
            "schema_version": "trusted_source_text_companion.v1",
            "format": HASH_BOUND_OCR_FORMAT,
            "source_ref": source_key,
            "source_pdf_path": str(source_pdf),
            "source_pdf_sha256": pdf_sha256,
            "pages": page_specs,
        }
    focus_pages = _rank_focus_pages(ocr_text_by_page, target_terms=target_terms)
    visual_pages = _visual_pages(focus_pages or selected, images)
    result = _result(
        status="completed" if page_specs else "failed",
        reasons=[] if page_specs else ["local_ocr_produced_no_replayable_pages"],
        source_pdf_sha256=pdf_sha256,
        native_text_page_count=sum(
            count >= active.min_native_text_chars for count in native_counts.values()
        ),
        low_text_page_count=len(low_text_pages),
        selected_page_count=len(selected),
        selected_page_numbers=selected,
        ocr_page_count=len(page_specs),
        failure_count=len(failures),
        failures=failures,
        coverage_truncated=len(low_text_pages) > len(selected),
        focus_page_numbers=focus_pages,
        visual_candidate_pages=visual_pages,
        companion=companion,
    )
    manifest = out / "local-source-ocr-materialization.json"
    _write_json_atomic(manifest, result)
    return {**result, "manifest_path": str(manifest), "manifest_sha256": _sha256(manifest)}


def _tesseract_runner(
    image_path: Path,
    page_number: int,
    config: LocalOcrConfig,
) -> Mapping[str, Any]:
    del page_number
    executable = shutil.which("tesseract")
    if not executable:
        raise RuntimeError("tesseract_not_found")
    version = subprocess.run(
        [executable, "--version"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=min(10.0, config.timeout_s_per_page),
        check=False,
        creationflags=_no_window_flag(),
    )
    process = subprocess.run(
        [
            executable,
            str(image_path),
            "stdout",
            "-l",
            "+".join(config.languages),
            "--psm",
            str(config.page_segmentation_mode),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=config.timeout_s_per_page,
        check=False,
        creationflags=_no_window_flag(),
    )
    if process.returncode != 0:
        raise RuntimeError(f"tesseract_exit_{process.returncode}:{process.stderr[:300]}")
    first_line = next((line.strip() for line in version.stdout.splitlines() if line.strip()), "")
    return {
        "text": process.stdout,
        "engine_id": "tesseract",
        "engine_version": first_line or "tesseract:unknown",
    }


def _native_text_counts(path: Path) -> dict[int, int]:
    try:
        import fitz  # type: ignore
    except ImportError as exc:
        raise RuntimeError("pymupdf_unavailable") from exc
    document = fitz.open(str(path))
    try:
        return {
            index + 1: len(re.sub(r"\s+", "", document[index].get_text("text") or ""))
            for index in range(len(document))
        }
    finally:
        document.close()


def _rank_focus_pages(
    values: Mapping[int, str],
    *,
    target_terms: Iterable[str],
) -> list[int]:
    terms = {
        "preparation of": 8,
        "synthesis of": 8,
        "general procedure": 7,
        "was added": 4,
        "was stirred": 4,
        "yield": 2,
        "example": 2,
        "scheme": 2,
    }
    for value in target_terms:
        term = " ".join(str(value or "").casefold().split())
        if len(term) >= 3:
            terms[term] = 12
    scored = []
    for page_number, text in values.items():
        normalized = " ".join(text.casefold().split())
        score = sum(weight * min(3, normalized.count(term)) for term, weight in terms.items())
        scored.append((score, page_number))
    scored.sort(key=lambda row: (-row[0], row[1]))
    return [page for score, page in scored if score > 0][:8]


def _stratified_page_selection(values: Iterable[int], *, limit: int) -> list[int]:
    """Cover the front, body, and tail instead of OCRing only patent covers."""

    pages = sorted({int(value) for value in values if int(value) > 0})
    if len(pages) <= limit:
        return pages
    front_count = min(3, max(1, limit // 4))
    selected = list(pages[:front_count])
    remaining = limit - len(selected)
    for slot in range(1, remaining + 1):
        index = round((len(pages) - 1) * slot / remaining)
        page = pages[index]
        if page not in selected:
            selected.append(page)
    if len(selected) < limit:
        selected.extend(page for page in pages if page not in selected)
    return sorted(selected[:limit])


def _visual_pages(
    page_numbers: Iterable[int],
    images: Mapping[int, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for page_number in page_numbers:
        image = dict(images.get(int(page_number)) or {})
        path = str(image.get("image_path") or "")
        digest = str(image.get("sha256") or image.get("image_sha256") or "")
        if path and _is_sha256(digest):
            rows.append(
                {
                    "page_number": int(page_number),
                    "image_path": path,
                    "image_sha256": digest,
                }
            )
        if len(rows) >= 8:
            break
    return rows


def _bounded_ocr_text(value: Any, *, limit: int) -> str:
    text = str(value or "").replace("\x00", "")
    text = "\n".join(line.rstrip() for line in text.splitlines()).strip()
    return text[:limit]


def _result(status: str, *, reasons: Iterable[str] = (), **values: Any) -> dict[str, Any]:
    row = {
        "schema_version": LOCAL_OCR_MATERIALIZATION_SCHEMA,
        "status": status,
        "reasons": sorted(set(str(value) for value in reasons if str(value))),
        "model_invocations": 0,
        "visual_invocations": 0,
        "semantics": {
            "ocr_text_is_parser_input_not_scientific_authority": True,
            "page_image_pdf_and_text_hashes_are_replay_required": True,
            "visual_model_is_never_implicitly_invoked": True,
        },
        **values,
    }
    row["content_sha256"] = hashlib.sha256(
        json.dumps(
            row,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    return row


def _write_text_atomic(path: Path, value: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value + "\n", encoding="utf-8")
    temporary.replace(path)


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _no_window_flag() -> int:
    return int(getattr(subprocess, "CREATE_NO_WINDOW", 0)) if sys.platform == "win32" else 0


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _is_sha256(value: str) -> bool:
    return bool(re.fullmatch(r"[0-9a-f]{64}", str(value or "")))


__all__ = [
    "HASH_BOUND_OCR_FORMAT",
    "LOCAL_OCR_MATERIALIZATION_SCHEMA",
    "LocalOcrConfig",
    "OcrRunner",
    "SUPPORTED_LOCAL_OCR_ENGINES",
    "materialize_local_ocr_companion",
]
