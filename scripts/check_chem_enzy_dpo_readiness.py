#!/usr/bin/env python3
"""Inspect ChemEnzy fine-tuning and DPO readiness.

The goal of this script is deliberately narrow: produce an auditable manifest
for what the vendored ChemEnzy stack can train today, what data is available,
and what is still missing before claiming cascade-mode DPO or LoRA fine-tuning.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import yaml
except Exception:  # pragma: no cover - PyYAML is present in the normal env.
    yaml = None


SCHEMA_VERSION = "chem_enzy_dpo_readiness.v1"

MODEL_FAMILY_CAPABILITIES = {
    "onmt_models": {
        "role": "sequence-to-sequence one-step proposer",
        "trainability": "supervised_continue_train_possible",
        "adapter_note": "Can resume OpenNMT training from checkpoints with src/tgt corpora; no local DPO or LoRA path was found.",
    },
    "graphfp_models": {
        "role": "template classifier one-step proposer",
        "trainability": "supervised_retrain_possible",
        "adapter_note": "Can retrain template classifier on template-indexed data; cascade-conditioned DPO requires new objective/features.",
    },
    "mlp_models": {
        "role": "fingerprint template classifier one-step proposer",
        "trainability": "supervised_retrain_possible",
        "adapter_note": "Can retrain template classifier on product-to-template data; not a generative DPO/LoRA target as-is.",
    },
    "template_relevance": {
        "role": "external template relevance service",
        "trainability": "not_locally_trainable_from_this_repo",
        "adapter_note": "Configured as HTTP service; no local training or DPO entrypoint is available in the vendored tree.",
    },
}


def main() -> None:
    args = _parse_args()
    result = check_readiness(
        vendor_root=args.vendor_root,
        config_path=args.config,
        preference_summary=args.preference_summary,
        preference_jsonl=args.preference_jsonl,
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(render_markdown(result), encoding="utf-8")
    print(json.dumps(result["summary"], indent=2, ensure_ascii=False))


def check_readiness(
    *,
    vendor_root: Path,
    config_path: Path | None = None,
    preference_summary: Path | None = None,
    preference_jsonl: Path | None = None,
) -> dict[str, Any]:
    vendor_root = Path(vendor_root)
    config_path = Path(config_path) if config_path else vendor_root / "retro_planner" / "config" / "config.yaml"
    retro_root = config_path.parents[1] if len(config_path.parents) >= 2 else vendor_root / "retro_planner"

    config = _load_config(config_path)
    one_step_configs = config.get("one_step_model_configs") if isinstance(config, dict) else {}
    if not isinstance(one_step_configs, dict):
        one_step_configs = {}

    configured_models = _configured_models(one_step_configs, retro_root)
    entrypoints = _training_entrypoints(retro_root)
    code_support = _code_support(vendor_root)
    preference = _preference_pack(preference_summary, preference_jsonl)

    missing = []
    if not vendor_root.exists():
        missing.append("vendor_root_missing")
    if not config_path.exists():
        missing.append("config_missing")
    if not configured_models:
        missing.append("no_configured_one_step_models")
    if not preference["available"]:
        missing.append("preference_pack_missing")
    if not code_support["dpo"]:
        missing.append("vendor_dpo_loss_missing")
    if not code_support["lora"]:
        missing.append("vendor_lora_adapter_missing")

    supervised_ready = any(
        model["family"] in {"onmt_models", "graphfp_models", "mlp_models"}
        and model["configured_artifacts"]["all_required_exist"]
        for model in configured_models
    )
    onmt_ready = any(
        model["family"] == "onmt_models" and model["configured_artifacts"]["all_required_exist"]
        for model in configured_models
    )
    dpo_ready = preference["available"] and code_support["dpo"]
    lora_ready = code_support["lora"] and any(model["family"] == "onmt_models" for model in configured_models)

    if dpo_ready:
        overall_status = "direct_dpo_entrypoint_detected"
    elif preference["available"] and supervised_ready:
        overall_status = "ready_for_supervised_adapter_manifest_not_direct_dpo"
    elif supervised_ready:
        overall_status = "trainable_vendor_detected_preference_pack_missing"
    else:
        overall_status = "not_ready_for_training_claim"

    result = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "vendor_root": str(vendor_root),
        "config_path": str(config_path),
        "retro_root": str(retro_root),
        "configured_models": configured_models,
        "training_entrypoints": entrypoints,
        "code_support": code_support,
        "preference_pack": preference,
        "capability_matrix": {
            "supervised_continue_train": {
                "status": "available" if supervised_ready else "blocked",
                "best_current_target": "onmt_models" if onmt_ready else "graphfp_or_mlp_template_classifier",
                "note": "This is not DPO; it needs converted src/tgt or template-indexed supervised corpora.",
            },
            "cascade_conditioned_generation": {
                "status": "requires_new_data_adapter_and_training_objective",
                "note": "The current ChemEnzy API accepts target molecules and model config, not a persistent cascade state ledger.",
            },
            "verifier_preference_dpo": {
                "status": "blocked_without_new_loss" if preference["available"] else "blocked_without_preference_pack",
                "note": "Verifier preference pairs exist, but the vendored stack has no DPO trainer/loss detected.",
            },
            "lora_adapter": {
                "status": "blocked_without_peft_or_local_lora" if not lora_ready else "detected",
                "note": "OpenNMT checkpoints are trainable, but no LoRA adapter implementation was found in the local vendor tree.",
            },
        },
        "blockers": missing,
        "recommended_next_steps": _recommended_next_steps(preference["available"], onmt_ready, code_support),
        "summary": {
            "overall_status": overall_status,
            "configured_model_count": len(configured_models),
            "configured_families": sorted({model["family"] for model in configured_models}),
            "preference_pairs": preference.get("n_pairs"),
            "preference_groups": preference.get("n_groups"),
            "supervised_vendor_training_ready": supervised_ready,
            "direct_dpo_ready": dpo_ready,
            "lora_ready": lora_ready,
            "blockers": missing,
        },
        "contract": "Readiness manifest only. It does not train ChemEnzy and must not be reported as completed DPO/LoRA fine-tuning.",
    }
    return result


def render_markdown(result: dict[str, Any]) -> str:
    summary = result["summary"]
    lines = [
        "# ChemEnzy Cascade/DPO Readiness",
        "",
        f"生成时间：{result['created_at']}",
        "",
        "## 结论",
        "",
        f"- overall_status: `{summary['overall_status']}`",
        f"- configured_model_count: {summary['configured_model_count']}",
        f"- configured_families: {', '.join(summary['configured_families']) or 'none'}",
        f"- preference_pairs: {summary.get('preference_pairs')}",
        f"- supervised_vendor_training_ready: {summary['supervised_vendor_training_ready']}",
        f"- direct_dpo_ready: {summary['direct_dpo_ready']}",
        f"- lora_ready: {summary['lora_ready']}",
        "",
        "解释：当前 vendor 可以作为 supervised continue-train/retrain 的基础，"
        "但没有检测到现成 DPO loss 或 LoRA adapter。因此当前状态不能宣称 ChemEnzy DPO/LoRA 已完成。",
        "",
        "## Configured Models",
        "",
        "| family | name | artifacts_ok | trainability | note |",
        "| --- | --- | ---: | --- | --- |",
    ]
    for model in result["configured_models"]:
        cap = model["capability"]
        lines.append(
            "| {family} | {name} | {ok} | {trainability} | {note} |".format(
                family=model["family"],
                name=model["name"],
                ok=str(model["configured_artifacts"]["all_required_exist"]),
                trainability=cap["trainability"],
                note=cap["adapter_note"],
            )
        )
    lines.extend([
        "",
        "## Capability Matrix",
        "",
        "| capability | status | note |",
        "| --- | --- | --- |",
    ])
    for name, row in result["capability_matrix"].items():
        lines.append(f"| {name} | {row['status']} | {row['note']} |")
    lines.extend([
        "",
        "## Blockers",
        "",
    ])
    blockers = result.get("blockers") or []
    if blockers:
        lines.extend(f"- `{item}`" for item in blockers)
    else:
        lines.append("- none")
    lines.extend([
        "",
        "## Recommended Next Steps",
        "",
    ])
    lines.extend(f"{idx}. {step}" for idx, step in enumerate(result["recommended_next_steps"], 1))
    lines.append("")
    return "\n".join(lines)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vendor-root", type=Path, default=Path("vendor/ChemEnzyRetroPlanner"))
    parser.add_argument("--config", type=Path)
    parser.add_argument("--preference-summary", type=Path)
    parser.add_argument("--preference-jsonl", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--markdown", type=Path)
    return parser.parse_args()


def _load_config(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        return {}
    if yaml is None:
        return {}
    data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    return data if isinstance(data, dict) else {}


def _configured_models(one_step_configs: dict[str, Any], retro_root: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for family, models in sorted(one_step_configs.items()):
        if not isinstance(models, dict):
            continue
        for name, config in sorted(models.items()):
            if not isinstance(config, dict):
                config = {}
            artifacts = _required_artifacts(family, config, retro_root)
            out.append(
                {
                    "family": family,
                    "name": name,
                    "full_name": f"{family}.{name}",
                    "weight": config.get("weight"),
                    "capability": MODEL_FAMILY_CAPABILITIES.get(
                        family,
                        {
                            "role": "unknown",
                            "trainability": "unknown",
                            "adapter_note": "No local capability rule for this model family.",
                        },
                    ),
                    "configured_artifacts": artifacts,
                }
            )
    return out


def _required_artifacts(family: str, config: dict[str, Any], retro_root: Path) -> dict[str, Any]:
    required: list[dict[str, Any]] = []
    if family == "onmt_models":
        model_paths = config.get("model_path") or []
        if isinstance(model_paths, (str, Path)):
            model_paths = [model_paths]
        for item in model_paths:
            required.append(_artifact_record("model_path", retro_root / str(item)))
    elif family == "graphfp_models":
        for key in ("graph_model_dumb", "graph_dataset_root"):
            if config.get(key):
                required.append(_artifact_record(key, retro_root / str(config[key])))
    elif family == "mlp_models":
        for key in ("mlp_templates", "mlp_model_dump"):
            if config.get(key):
                required.append(_artifact_record(key, retro_root / str(config[key])))
    elif family == "template_relevance":
        required.append({"key": "service_state_name", "path": str(config.get("state_name") or ""), "exists": bool(config.get("state_name"))})
    return {
        "required": required,
        "all_required_exist": bool(required) and all(row["exists"] for row in required),
    }


def _artifact_record(key: str, path: Path) -> dict[str, Any]:
    return {"key": key, "path": str(path), "exists": path.exists()}


def _training_entrypoints(retro_root: Path) -> dict[str, Any]:
    entrypoint_paths = {
        "onmt_preprocess": retro_root / "packages" / "onmt" / "onmt" / "bin" / "preprocess.py",
        "onmt_train": retro_root / "packages" / "onmt" / "onmt" / "bin" / "train.py",
        "graphfp_train": retro_root / "packages" / "graph_retrosyn" / "graph_retrosyn" / "graph_train.py",
        "mlp_train": retro_root / "packages" / "mlp_retrosyn" / "mlp_retrosyn" / "mlp_train.py",
    }
    return {
        key: {
            "path": str(path),
            "exists": path.exists(),
        }
        for key, path in entrypoint_paths.items()
    }


def _code_support(vendor_root: Path) -> dict[str, Any]:
    hits = _term_hits(vendor_root, terms=("DPO", "LoRA", "lora", "peft"))
    return {
        "dpo": bool(hits.get("DPO")),
        "lora": bool(hits.get("LoRA") or hits.get("lora") or hits.get("peft")),
        "term_hits": hits,
    }


def _term_hits(vendor_root: Path, *, terms: tuple[str, ...]) -> dict[str, list[str]]:
    hits = {term: [] for term in terms}
    if not vendor_root.exists():
        return hits
    allowed_suffixes = {".py", ".md", ".yaml", ".yml", ".sh", ".txt"}
    for path in vendor_root.rglob("*"):
        if ".git" in path.parts or not path.is_file() or path.suffix not in allowed_suffixes:
            continue
        try:
            if path.stat().st_size > 1_000_000:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for term in terms:
            if term in text and len(hits[term]) < 20:
                hits[term].append(str(path))
    return hits


def _preference_pack(preference_summary: Path | None, preference_jsonl: Path | None) -> dict[str, Any]:
    out: dict[str, Any] = {
        "available": False,
        "summary_path": str(preference_summary) if preference_summary else None,
        "jsonl_path": str(preference_jsonl) if preference_jsonl else None,
        "n_pairs": None,
        "n_groups": None,
        "reason_counts": {},
    }
    if preference_summary and preference_summary.exists():
        summary = json.loads(preference_summary.read_text(encoding="utf-8")).get("summary")
        if isinstance(summary, dict):
            out.update(
                {
                    "available": int(summary.get("n_pairs") or 0) > 0,
                    "n_pairs": int(summary.get("n_pairs") or 0),
                    "n_groups": int(summary.get("n_groups") or 0),
                    "reason_counts": summary.get("reason_counts") or {},
                    "schema_version": summary.get("schema_version"),
                }
            )
    if not out["available"] and preference_jsonl and preference_jsonl.exists():
        with preference_jsonl.open(encoding="utf-8") as handle:
            count = sum(1 for _ in handle)
        out.update({"available": count > 0, "n_pairs": count})
    return out


def _recommended_next_steps(preferences_available: bool, onmt_ready: bool, code_support: dict[str, Any]) -> list[str]:
    steps = []
    if preferences_available:
        steps.append("Freeze verifier-derived preference JSONL as DPO/rerank input, but label it as verifier preference, not expert preference.")
    else:
        steps.append("Generate verifier-derived preference JSONL before any DPO/rerank training claim.")
    if onmt_ready:
        steps.append("Build an OpenNMT supervised cascade-conditioned src/tgt corpus as the first ChemEnzy-compatible adapter target.")
    else:
        steps.append("Select a trainable local ChemEnzy model family and verify checkpoints/data before training.")
    if not code_support["dpo"]:
        steps.append("Implement a new DPO/pairwise loss wrapper; the vendor tree has no direct DPO trainer.")
    if not code_support["lora"]:
        steps.append("Do not claim LoRA until a PEFT/local adapter implementation is added and checkpoint loading is tested.")
    steps.append("Keep verifier as search value/rerank signal until generator fine-tuning is separately validated.")
    return steps


if __name__ == "__main__":
    main()
