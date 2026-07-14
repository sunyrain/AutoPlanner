from __future__ import annotations

import hashlib
from pathlib import Path
import subprocess

import fitz
from PIL import Image, ImageDraw

from cascade_planner.harness.source_ocr import (
    LocalOcrConfig,
    _stratified_page_selection,
    materialize_local_ocr_companion,
)
from cascade_planner.harness.source_text_companion import (
    materialize_source_text_companion_pages,
    source_text_companion_matches_page,
    validate_source_text_companion_binding,
)


SOURCE_REF = "patent:US1234567A1"


def test_ocr_page_selection_covers_patent_front_body_and_tail() -> None:
    selected = _stratified_page_selection(range(1, 81), limit=12)

    assert len(selected) == 12
    assert selected[:3] == [1, 2, 3]
    assert selected[-1] == 80
    assert any(30 <= page <= 50 for page in selected)


def _image_only_source(tmp_path: Path) -> tuple[Path, Path, str]:
    image_path = tmp_path / "source-page.png"
    image = Image.new("RGB", (1200, 1600), "white")
    draw = ImageDraw.Draw(image)
    draw.text(
        (60, 80),
        "Example 1 Preparation of ethyl acetate\n"
        "Ethanol and acetic acid were added and stirred.",
        fill="black",
    )
    image.save(image_path)
    document = fitz.open()
    page = document.new_page(width=600, height=800)
    page.insert_image(page.rect, filename=str(image_path))
    pdf_path = tmp_path / "image-only.pdf"
    document.save(pdf_path)
    document.close()
    return pdf_path, image_path, hashlib.sha256(image_path.read_bytes()).hexdigest()


def test_local_ocr_companion_is_hash_bound_and_replayable(tmp_path: Path) -> None:
    pdf_path, image_path, image_sha256 = _image_only_source(tmp_path)

    def runner(_path: Path, page_number: int, _config: LocalOcrConfig) -> dict:
        assert page_number == 1
        return {
            "text": (
                "Example 1 Preparation of ethyl acetate\n"
                "Ethanol and acetic acid were added. The reaction mixture "
                "was stirred to afford ethyl acetate."
            ),
            "engine_id": "tesseract",
            "engine_version": "tesseract fixture 1.0",
        }

    audit = materialize_local_ocr_companion(
        pdf_path=pdf_path,
        source_ref=SOURCE_REF,
        rendered_pages=[
            {
                "page_number": 1,
                "image_path": str(image_path),
                "sha256": image_sha256,
            }
        ],
        output_dir=tmp_path / "ocr",
        config=LocalOcrConfig(max_pages=1),
        runner=runner,
    )

    assert audit["status"] == "completed"
    assert audit["model_invocations"] == 0
    assert audit["visual_invocations"] == 0
    spec = audit["companion"]
    pages, binding, reasons = materialize_source_text_companion_pages(
        spec,
        source_ref=SOURCE_REF,
    )
    assert reasons == ()
    assert pages[0]["page_number"] == 1
    assert "ethyl acetate" in pages[0]["text"]
    assert validate_source_text_companion_binding(
        binding,
        expected_source_ref=SOURCE_REF,
    )
    assert source_text_companion_matches_page(
        binding,
        page_number=1,
        image_sha256=image_sha256,
        source_pdf_sha256=hashlib.sha256(pdf_path.read_bytes()).hexdigest(),
    )

    Path(binding["pages"][0]["text_path"]).write_text(
        "tampered\n",
        encoding="utf-8",
    )
    assert not validate_source_text_companion_binding(
        binding,
        expected_source_ref=SOURCE_REF,
    )


def test_local_ocr_rejects_wrong_source_and_non_allowlisted_engine(
    tmp_path: Path,
) -> None:
    pdf_path, image_path, image_sha256 = _image_only_source(tmp_path)
    audit = materialize_local_ocr_companion(
        pdf_path=pdf_path,
        source_ref=SOURCE_REF,
        rendered_pages=[
            {
                "page_number": 1,
                "image_path": str(image_path),
                "sha256": image_sha256,
            }
        ],
        output_dir=tmp_path / "ocr",
        runner=lambda *_args: {
            "text": "Example 1 Preparation of ethyl acetate. Reaction was stirred.",
            "engine_id": "remote-model-ocr",
            "engine_version": "1",
        },
    )

    assert audit["status"] == "failed"
    assert audit["companion"] == {}
    assert audit["failures"] == [
        {"page_number": 1, "reason": "ocr_engine_not_allowlisted"}
    ]

    trusted = materialize_local_ocr_companion(
        pdf_path=pdf_path,
        source_ref=SOURCE_REF,
        rendered_pages=[
            {
                "page_number": 1,
                "image_path": str(image_path),
                "sha256": image_sha256,
            }
        ],
        output_dir=tmp_path / "trusted-ocr",
        runner=lambda *_args: {
            "text": "Example 1 Preparation of ethyl acetate. Reaction was stirred.",
            "engine_id": "tesseract",
            "engine_version": "fixture",
        },
    )
    _, _, reasons = materialize_source_text_companion_pages(
        trusted["companion"],
        source_ref="patent:US0000000A1",
    )
    assert reasons == ("source_ocr_source_ref_mismatch",)


def test_local_ocr_engine_absence_is_explicit_and_model_free(
    tmp_path: Path,
    monkeypatch,
) -> None:
    pdf_path, image_path, image_sha256 = _image_only_source(tmp_path)
    monkeypatch.setattr(
        "cascade_planner.harness.source_ocr.shutil.which",
        lambda _name: None,
    )

    audit = materialize_local_ocr_companion(
        pdf_path=pdf_path,
        source_ref=SOURCE_REF,
        rendered_pages=[
            {
                "page_number": 1,
                "image_path": str(image_path),
                "sha256": image_sha256,
            }
        ],
        output_dir=tmp_path / "ocr",
    )

    assert audit["status"] == "unavailable"
    assert audit["reasons"] == ["local_ocr_engine_unavailable:tesseract"]
    assert audit["model_invocations"] == 0
    assert audit["visual_invocations"] == 0
    assert audit["visual_candidate_pages"][0]["image_sha256"] == image_sha256


def test_local_ocr_page_timeout_is_bounded_failure(tmp_path: Path) -> None:
    pdf_path, image_path, image_sha256 = _image_only_source(tmp_path)

    def timed_out(*_args):
        raise subprocess.TimeoutExpired(cmd="tesseract", timeout=0.01)

    audit = materialize_local_ocr_companion(
        pdf_path=pdf_path,
        source_ref=SOURCE_REF,
        rendered_pages=[
            {
                "page_number": 1,
                "image_path": str(image_path),
                "sha256": image_sha256,
            }
        ],
        output_dir=tmp_path / "ocr",
        runner=timed_out,
    )

    assert audit["status"] == "failed"
    assert audit["failure_count"] == 1
    assert audit["failures"][0]["reason"].startswith("TimeoutExpired:")
    assert audit["model_invocations"] == 0
