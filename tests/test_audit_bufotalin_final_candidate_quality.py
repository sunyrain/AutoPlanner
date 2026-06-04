import json

from scripts.audit_bufotalin_final_candidate_quality import audit_final_candidate_quality


def test_audit_final_candidate_quality_accepts_clean_package(tmp_path):
    _write_final_candidate_package(tmp_path)

    report = audit_final_candidate_quality(tmp_path)

    assert report["passed"]
    checks = {check["name"]: check["passed"] for check in report["checks"]}
    assert checks["review_routes_min_steps"]
    assert checks["no_high_risk_warnings_selected"]
    assert report["metrics"]["selected_count"] == 3


def test_audit_final_candidate_quality_rejects_short_or_high_risk_review_route(tmp_path):
    _write_final_candidate_package(tmp_path, short_native=True, high_risk_stitched=True)

    report = audit_final_candidate_quality(tmp_path)

    assert not report["passed"]
    checks = {check["name"]: check["passed"] for check in report["checks"]}
    assert not checks["review_routes_min_steps"]
    assert not checks["no_high_risk_warnings_selected"]
    assert report["violations"]["short_review_routes"]
    assert report["violations"]["high_risk_warning_routes"]


def _write_final_candidate_package(root, *, short_native: bool = False, high_risk_stitched: bool = False):
    final = root / "final_candidates"
    figures = final / "figures"
    figures.mkdir(parents=True)
    for idx in range(1, 4):
        (figures / f"scheme_route_{idx:02d}.svg").write_text("<svg/>", encoding="utf-8")
        (figures / f"scheme_route_{idx:02d}.pdf").write_text("%PDF", encoding="utf-8")
    (figures / "manifest.json").write_text(
        json.dumps(
            {
                "figures": [
                    {"svg": f"scheme_route_{idx:02d}.svg", "pdf": f"scheme_route_{idx:02d}.pdf"}
                    for idx in range(1, 4)
                ]
            }
        ),
        encoding="utf-8",
    )
    stitched_warnings = ["rcr_condition_prediction_only"]
    if high_risk_stitched:
        stitched_warnings.append("strong_hydride_reagent_predicted")
    native_steps = 2 if short_native else 3
    selected = [
        {"confidence_tier": "high_confidence_source_supported", "presentation_ready": True, "warnings": []},
        {
            "confidence_tier": "stitched_semisynthesis_upstream_review_only",
            "presentation_ready": False,
            "n_steps": 4,
            "warnings": stitched_warnings,
        },
        {
            "confidence_tier": "native_model_candidate_review_only",
            "presentation_ready": False,
            "n_steps": native_steps,
            "warnings": ["rcr_condition_prediction_only"],
        },
    ]
    (final / "final_candidates.json").write_text(
        json.dumps(
            {
                "high_confidence_count": 1,
                "stitched_review_only_count": 1,
                "native_review_only_count": 1,
                "selected_count": 3,
                "selected": selected,
            }
        ),
        encoding="utf-8",
    )
    (final / "final_candidates_payload.json").write_text(
        json.dumps(
            {
                "routes": [
                    _route("high_confidence_source_supported", n_steps=1, source_supported=True),
                    _route(
                        "stitched_semisynthesis_upstream_review_only",
                        n_steps=4,
                        warnings=stitched_warnings,
                    ),
                    _route("native_model_candidate_review_only", n_steps=native_steps),
                ]
            }
        ),
        encoding="utf-8",
    )


def _route(
    confidence_tier: str,
    *,
    n_steps: int,
    source_supported: bool = False,
    warnings: list[str] | None = None,
):
    return {
        "n_steps": n_steps,
        "steps": [
            {
                "main_reactant": f"C{idx}",
                "condition_predictions": [{"condition_label": "RCR model prediction"}],
            }
            for idx in range(n_steps)
        ],
        "final_candidate": {
            "confidence_tier": confidence_tier,
            "presentation_ready": confidence_tier == "high_confidence_source_supported",
            "target_terminal": False,
            "exclusion_reasons": [],
            "source_supported_semisynthesis": source_supported,
            "warnings": list(warnings or ["rcr_condition_prediction_only"]),
        },
    }
