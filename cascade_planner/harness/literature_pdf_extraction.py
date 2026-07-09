"""Local PDF/image evidence extraction for literature structure curation."""
from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


LITERATURE_PDF_EXTRACTION_SCHEMA = "literature_pdf_structure_evidence.v1"


def extract_literature_pdf_assets(
    *,
    pdf_path: str | Path | None = None,
    output_dir: str | Path,
    page_numbers: list[int] | None = None,
    render_zoom: float = 2.0,
    image_paths: list[str | Path] | None = None,
    scheme_crops: list[dict[str, Any]] | None = None,
    compound_labels: list[str] | None = None,
) -> dict[str, Any]:
    """Render/source-index literature assets without claiming route evidence.

    The output is an evidence manifest for a later vision/curator step. It may
    contain short text snippets and local image paths, but it deliberately does
    not emit SMILES or route steps by itself.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    pages_dir = out / "pages"
    crops_dir = out / "crops"
    pages_dir.mkdir(parents=True, exist_ok=True)
    crops_dir.mkdir(parents=True, exist_ok=True)

    source_pdf = Path(pdf_path).expanduser().resolve() if pdf_path else None
    reasons: list[str] = []
    warnings: list[str] = []
    rendered_pages: list[dict[str, Any]] = []
    fulltext_path = ""
    page_texts: list[dict[str, Any]] = []

    if source_pdf:
        if source_pdf.is_file():
            pdf_result = _render_pdf(
                source_pdf=source_pdf,
                pages_dir=pages_dir,
                page_numbers=page_numbers,
                zoom=max(0.5, float(render_zoom or 2.0)),
            )
            rendered_pages = pdf_result["rendered_pages"]
            page_texts = pdf_result["page_texts"]
            warnings.extend(pdf_result["warnings"])
            if page_texts:
                text = "\n\n".join(str(row.get("text") or "") for row in page_texts)
                text_file = out / "fulltext.txt"
                text_file.write_text(text, encoding="utf-8")
                fulltext_path = str(text_file)
        else:
            reasons.append("pdf_path_missing")

    indexed_images = _index_image_paths(image_paths or [])
    crop_rows = _extract_scheme_crops(
        scheme_crops or [],
        rendered_pages=rendered_pages,
        crops_dir=crops_dir,
        warnings=warnings,
    )
    snippets = _compound_text_snippets(page_texts, compound_labels or [])

    manifest = {
        "schema_version": LITERATURE_PDF_EXTRACTION_SCHEMA,
        "accepted": not reasons,
        "status": "completed" if not reasons else "incomplete",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_pdf_path": str(source_pdf) if source_pdf else "",
        "source_pdf_sha256": _sha256(source_pdf) if source_pdf and source_pdf.is_file() else "",
        "fulltext_path": fulltext_path,
        "rendered_pages": rendered_pages,
        "indexed_images": indexed_images,
        "scheme_crops": crop_rows,
        "compound_text_snippets": snippets,
        "summary": {
            "rendered_page_count": len(rendered_pages),
            "indexed_image_count": len(indexed_images),
            "scheme_crop_count": len(crop_rows),
            "compound_text_snippet_count": len(snippets),
        },
        "source_policy": {
            "route_evidence_until_structured_extraction": False,
            "emits_smiles": False,
            "full_text_content_for_local_evidence_only": bool(fulltext_path),
            "no_solved_claim": True,
            "production_write_blocked": True,
        },
        "warnings": warnings,
        "reasons": reasons,
    }
    path = out / "literature_pdf_structure_evidence.json"
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def _render_pdf(
    *,
    source_pdf: Path,
    pages_dir: Path,
    page_numbers: list[int] | None,
    zoom: float,
) -> dict[str, Any]:
    try:
        import fitz  # type: ignore
    except Exception as exc:  # pragma: no cover - environment dependent
        return {
            "rendered_pages": [],
            "page_texts": [],
            "warnings": [f"pymupdf_unavailable:{type(exc).__name__}"],
        }

    rendered: list[dict[str, Any]] = []
    texts: list[dict[str, Any]] = []
    warnings: list[str] = []
    doc = fitz.open(str(source_pdf))
    try:
        requested = _page_indices(page_numbers, page_count=len(doc))
        matrix = fitz.Matrix(float(zoom), float(zoom))
        for page_index in requested:
            page = doc.load_page(page_index)
            image_path = pages_dir / f"page_{page_index + 1:03d}_z{_zoom_label(zoom)}.png"
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            pix.save(str(image_path))
            rendered.append(
                {
                    "page_number": page_index + 1,
                    "image_path": str(image_path),
                    "width_px": int(pix.width),
                    "height_px": int(pix.height),
                    "render_zoom": float(zoom),
                    "sha256": _sha256(image_path),
                }
            )
            text = page.get_text("text") or ""
            texts.append({"page_number": page_index + 1, "text": text})
    finally:
        doc.close()
    return {"rendered_pages": rendered, "page_texts": texts, "warnings": warnings}


def _page_indices(page_numbers: list[int] | None, *, page_count: int) -> list[int]:
    if not page_numbers:
        return list(range(page_count))
    out: list[int] = []
    for value in page_numbers:
        index = int(value) - 1
        if 0 <= index < page_count and index not in out:
            out.append(index)
    return out


def _zoom_label(zoom: float) -> str:
    text = f"{float(zoom):.2f}".rstrip("0").rstrip(".")
    return text.replace(".", "p")


def _index_image_paths(paths: list[str | Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, raw in enumerate(paths, start=1):
        path = Path(raw).expanduser().resolve()
        exists = path.is_file()
        rows.append(
            {
                "image_id": f"provided_image_{idx}",
                "image_path": str(path),
                "exists": exists,
                "sha256": _sha256(path) if exists else "",
            }
        )
    return rows


def _extract_scheme_crops(
    crops: list[dict[str, Any]],
    *,
    rendered_pages: list[dict[str, Any]],
    crops_dir: Path,
    warnings: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    page_by_number = {int(row.get("page_number") or 0): row for row in rendered_pages}
    for idx, crop in enumerate(crops, start=1):
        crop_id = _safe_id(str(crop.get("crop_id") or crop.get("scheme_id") or f"crop_{idx}"))
        source_raw = str(crop.get("source_image_path") or "").strip()
        source_path = Path(source_raw).expanduser() if source_raw else Path()
        if (not source_raw or not source_path.is_file()) and crop.get("page_number"):
            source_path = Path(str((page_by_number.get(int(crop.get("page_number"))) or {}).get("image_path") or ""))
        bbox = crop.get("bbox_px") or crop.get("bbox")
        if not source_path.is_file() or not _valid_bbox(bbox):
            rows.append(
                {
                    "crop_id": crop_id,
                    "source_image_path": str(source_path) if str(source_path) else "",
                    "image_path": "",
                    "page_number": int(crop.get("page_number") or 0),
                    "bbox_px": bbox if isinstance(bbox, list) else [],
                    "status": "not_created",
                    "reason": "source_image_or_bbox_missing",
                }
            )
            continue
        try:
            from PIL import Image
        except Exception as exc:  # pragma: no cover - environment dependent
            warnings.append(f"pillow_unavailable:{type(exc).__name__}")
            rows.append(
                {
                    "crop_id": crop_id,
                    "source_image_path": str(source_path),
                    "image_path": "",
                    "page_number": int(crop.get("page_number") or 0),
                    "bbox_px": bbox,
                    "status": "not_created",
                    "reason": "pillow_unavailable",
                }
            )
            continue
        image = Image.open(source_path)
        x0, y0, x1, y1 = [int(v) for v in bbox]
        x0 = max(0, min(x0, image.width))
        x1 = max(0, min(x1, image.width))
        y0 = max(0, min(y0, image.height))
        y1 = max(0, min(y1, image.height))
        if x1 <= x0 or y1 <= y0:
            status = "not_created"
            crop_path = ""
            reason = "empty_bbox"
        else:
            crop_path_obj = crops_dir / f"{crop_id}.png"
            image.crop((x0, y0, x1, y1)).save(crop_path_obj)
            crop_path = str(crop_path_obj)
            status = "created"
            reason = ""
        rows.append(
            {
                "crop_id": crop_id,
                "scheme_id": str(crop.get("scheme_id") or ""),
                "source_image_path": str(source_path),
                "image_path": crop_path,
                "page_number": int(crop.get("page_number") or 0),
                "bbox_px": [x0, y0, x1, y1],
                "status": status,
                "reason": reason,
                "sha256": _sha256(Path(crop_path)) if crop_path else "",
                "evidence_refs": [str(item) for item in crop.get("evidence_refs") or [] if str(item).strip()],
            }
        )
    return rows


def _valid_bbox(value: Any) -> bool:
    return isinstance(value, list) and len(value) == 4 and all(str(item).strip() for item in value)


def _compound_text_snippets(page_texts: list[dict[str, Any]], labels: list[str]) -> list[dict[str, Any]]:
    if not labels:
        return []
    out: list[dict[str, Any]] = []
    seen: set[tuple[int, str, str]] = set()
    for row in page_texts:
        page_number = int(row.get("page_number") or 0)
        text = str(row.get("text") or "")
        for label in labels:
            pattern = re.compile(rf"\b(?:compound\s*)?{re.escape(str(label))}\b", flags=re.IGNORECASE)
            for match in pattern.finditer(text):
                start = max(0, match.start() - 180)
                end = min(len(text), match.end() + 260)
                snippet = _compact_ws(text[start:end])
                key = (page_number, str(label), snippet)
                if key in seen:
                    continue
                seen.add(key)
                out.append(
                    {
                        "compound_label": str(label),
                        "page_number": page_number,
                        "source_locator": f"page {page_number}",
                        "snippet": snippet,
                    }
                )
                if len(out) >= 200:
                    return out
    return out


def _compact_ws(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _safe_id(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "").strip()).strip("_")
    return text or "item"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
