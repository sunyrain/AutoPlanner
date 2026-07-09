"""Local ASKCOS template_relevance model availability checks."""
from __future__ import annotations

from pathlib import Path
from zipfile import BadZipFile, ZipFile


DEFAULT_TEMPLATE_RELEVANCE_VENDOR_ROOT = Path("vendor/ChemEnzyRetroPlanner")
KNOWN_TEMPLATE_RELEVANCE_MODELS = (
    "bkms_metabolic",
    "pistachio",
    "pistachio_ringbreaker",
    "reaxys",
    "reaxys_biocatalysis",
    "cas",
    "uspto_higher_level",
)
DEFAULT_TEMPLATE_RELEVANCE_MODELS = (
    "template_relevance.bkms_metabolic",
    "template_relevance.pistachio_ringbreaker",
)


def template_relevance_model_dir(vendor_root: Path | str = DEFAULT_TEMPLATE_RELEVANCE_VENDOR_ROOT) -> Path:
    return (
        Path(vendor_root)
        / "retro_planner"
        / "packages"
        / "template_relevance"
        / "mars"
    )


def check_template_relevance(vendor_root: Path | str = DEFAULT_TEMPLATE_RELEVANCE_VENDOR_ROOT) -> dict:
    model_dir = template_relevance_model_dir(vendor_root)
    models = [_inspect_mar(model_dir, name) for name in KNOWN_TEMPLATE_RELEVANCE_MODELS]
    return {
        "model_dir": str(model_dir),
        "available_count": sum(1 for item in models if item["available"]),
        "models": models,
        "available_model_names": [
            f"template_relevance.{item['name']}" for item in models if item["available"]
        ],
        "missing_model_names": [
            f"template_relevance.{item['name']}" for item in models if not item["available"]
        ],
    }


def available_template_relevance_models(
    vendor_root: Path | str = DEFAULT_TEMPLATE_RELEVANCE_VENDOR_ROOT,
) -> list[str]:
    return list(check_template_relevance(vendor_root).get("available_model_names") or [])


def missing_template_relevance_models(
    models: list[str] | tuple[str, ...],
    vendor_root: Path | str = DEFAULT_TEMPLATE_RELEVANCE_VENDOR_ROOT,
) -> list[str]:
    available = set(available_template_relevance_models(vendor_root))
    out: list[str] = []
    for model in models:
        text = str(model or "")
        if text.startswith("template_relevance.") and text not in available:
            out.append(text)
    return out


def _inspect_mar(model_dir: Path, name: str) -> dict:
    path = model_dir / f"{name}.mar"
    if not path.exists():
        return {"name": name, "available": False, "reason": "missing_mar", "path": str(path)}
    item = {
        "name": name,
        "available": True,
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "size_mb": path.stat().st_size / 1024 / 1024,
    }
    try:
        with ZipFile(path) as archive:
            names = set(archive.namelist())
            required = {
                "templates.jsonl",
                "model_latest.pt",
                "templ_rel_handler.py",
                "MAR-INF/MANIFEST.json",
            }
            item["required_entries_present"] = sorted(required & names)
            item["required_entries_missing"] = sorted(required - names)
            item["valid_archive"] = not item["required_entries_missing"]
            if "templates.jsonl" in names:
                item["template_count"] = sum(1 for _ in archive.open("templates.jsonl"))
            if item["required_entries_missing"]:
                item["available"] = False
                item["reason"] = "missing_required_entries"
    except BadZipFile:
        item["available"] = False
        item["reason"] = "bad_zip_archive"
    return item
