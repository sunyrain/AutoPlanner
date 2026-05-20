"""Plan the full v4 trace -> pack -> train -> full100 evaluation pipeline.

The heavy ChemEnzy runs are intentionally sharded command steps.  This module
creates a reproducible command manifest; each command can be run by a scheduler
or shell without hand-editing paths.  Training commands use the v4 split files,
while full100 is only emitted as a locked evaluation command.
"""
from __future__ import annotations

import argparse
import json
import shlex
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PIPELINE_SCHEMA_VERSION = "v4_full_training_pipeline.v2"


@dataclass(frozen=True)
class TraceConfig:
    name: str
    extra_args: tuple[str, ...] = ()


TRACE_CONFIGS: dict[str, TraceConfig] = {
    "baseline": TraceConfig("baseline"),
    "graphfp_only": TraceConfig(
        "graphfp_only",
        ("--one-step-model", "graphfp_models.USPTO-full_remapped"),
    ),
    "bionav_only": TraceConfig(
        "bionav_only",
        ("--one-step-model", "onmt_models.bionav_one_step"),
    ),
    "cascade_rule_safe": TraceConfig(
        "cascade_rule_safe",
        (
            "--chem-enzy-cascade-cost",
            "--chem-enzy-cascade-context-from-row",
            "--chem-enzy-cascade-context-policy",
            "safe",
        ),
    ),
}


def plan_v4_full_training_pipeline(
    *,
    split_dir: Path,
    output_root: Path,
    vendor_root: Path = Path("vendor/ChemEnzyRetroPlanner"),
    benchmark_runner: list[str] | None = None,
    configs: list[str] | None = None,
    splits: list[str] | None = None,
    num_shards: int = 8,
    gpu: int = -1,
    iterations: int = 10,
    chem_enzy_max_depth: int = 6,
    expansion_topk: int = 50,
    cascade_expansion_budget: int = 100,
    action_epochs: int = 20,
    source_epochs: int = 20,
    transition_epochs: int = 8,
    pair_epochs: int = 20,
    cascade_pair_reward_weight: float = 0.35,
    cascade_pair_reward_mode: str = "additive",
    cascade_pair_reward_tie_epsilon: float = 0.0,
    enable_bootstrap_stage3: bool = True,
    bootstrap_splits: list[str] | None = None,
    bootstrap_action_epochs: int | None = None,
    bootstrap_source_epochs: int | None = None,
) -> dict[str, Any]:
    split_dir = Path(split_dir)
    output_root = Path(output_root)
    selected_configs = configs or ["baseline", "graphfp_only", "bionav_only", "cascade_rule_safe"]
    selected_splits = splits or ["train", "val", "test"]
    selected_bootstrap_splits = bootstrap_splits or ["train", "val"]
    _validate_configs(selected_configs)
    _validate_splits(selected_splits)
    _validate_splits(selected_bootstrap_splits)
    num_shards = max(1, int(num_shards))
    benchmark_runner = benchmark_runner or ["scripts/run_cascade_benchmark_chem_enzy_env.sh"]

    manifest_path = split_dir / "v4_trace_split_manifest.json"
    split_manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    all_benchmark = split_dir / "v4_trace_candidates_all.json"
    split_files = {
        "train": split_dir / "v4_trace_train.json",
        "val": split_dir / "v4_trace_val.json",
        "test": split_dir / "v4_trace_test.json",
    }

    commands: list[dict[str, Any]] = []
    trace_outputs: dict[str, dict[str, dict[str, Any]]] = {}
    for config_name in selected_configs:
        config = TRACE_CONFIGS[config_name]
        trace_outputs[config_name] = {}
        for split in selected_splits:
            runtime_paths = []
            expansion_paths = []
            cascade_trace_paths = []
            base_dir = output_root / "traces" / config_name / split
            for shard in range(num_shards):
                runtime = base_dir / f"runtime_shard{shard}of{num_shards}.json"
                expansion_base = base_dir / "chem_enzy_expansion.jsonl"
                cascade_base = base_dir / "cascade_trace.jsonl"
                runtime_paths.append(runtime)
                expansion_paths.append(_sharded_jsonl_path(expansion_base, shard, num_shards))
                cascade_trace_paths.append(_sharded_jsonl_path(cascade_base, shard, num_shards))
                commands.append(
                    {
                        "stage": "trace",
                        "config": config_name,
                        "split": split,
                        "shard_index": shard,
                        "num_shards": num_shards,
                        "outputs": {
                            "runtime": str(runtime),
                            "chem_enzy_expansion_trace": str(expansion_paths[-1]),
                            "cascade_trace": str(cascade_trace_paths[-1]),
                        },
                        "cmd": _join_cmd(
                            _benchmark_cmd(
                                benchmark_runner,
                                [
                                "--benchmark",
                                str(split_files[split]),
                                "--output",
                                str(runtime),
                                "--trace-output",
                                str(cascade_base),
                                "--chem-enzy-expansion-trace-output",
                                str(expansion_base),
                                "--include-route-outcomes",
                                "--vendor-root",
                                str(vendor_root),
                                "--iterations",
                                str(iterations),
                                "--chem-enzy-max-depth",
                                str(chem_enzy_max_depth),
                                "--expansion-topk",
                                str(expansion_topk),
                                "--cascade-expansion-budget",
                                str(cascade_expansion_budget),
                                "--gpu",
                                str(gpu),
                                "--num-shards",
                                str(num_shards),
                                "--shard-index",
                                str(shard),
                                *config.extra_args,
                                ],
                            )
                        ),
                    }
                )
            merged_runtime = output_root / "merged" / f"{config_name}_{split}_runtime.json"
            commands.append(
                {
                    "stage": "merge_runtime",
                    "config": config_name,
                    "split": split,
                    "outputs": {"runtime": str(merged_runtime)},
                    "cmd": _join_cmd(
                        _benchmark_cmd(
                            benchmark_runner,
                            [
                            "--merge",
                            *[str(path) for path in runtime_paths],
                            "--output",
                            str(merged_runtime),
                            ],
                        )
                    ),
                }
            )
            trace_outputs[config_name][split] = {
                "runtime_paths": [str(path) for path in runtime_paths],
                "merged_runtime": str(merged_runtime),
                "expansion_trace_paths": [str(path) for path in expansion_paths],
                "cascade_trace_paths": [str(path) for path in cascade_trace_paths],
            }

    train_val_expansion = _collect_paths(trace_outputs, configs=selected_configs, splits=["train", "val"], key="expansion_trace_paths")
    train_val_runtime = _collect_paths(trace_outputs, configs=selected_configs, splits=["train", "val"], key="merged_runtime")
    train_val_cascade_trace = _collect_paths(trace_outputs, configs=selected_configs, splits=["train", "val"], key="cascade_trace_paths")
    pair_pack_dir = output_root / "packs" / "v4_full_pair_compatibility_pack"
    pack_dir = output_root / "packs" / "v4_full_action_value_pack"
    transition_pack_dir = output_root / "packs" / "v4_full_transition_value_pack"
    models_dir = output_root / "models"
    reports_dir = output_root / "reports"

    commands.append(
        {
            "stage": "build_pair_pack",
            "outputs": {"pack_dir": str(pair_pack_dir)},
            "cmd": _join_cmd(
                [
                    "python",
                    "-m",
                    "cascade_planner.eval.build_cascade_pair_pack",
                    "--v4-jsonl",
                    "dataset_v4_release/cascade_v4_high_quality.jsonl",
                    "--benchmark",
                    "data/benchmark_v2_100.json",
                    "--output-dir",
                    str(pair_pack_dir),
                ]
            ),
        }
    )
    pair_model = models_dir / "cascade_pair_scorer.pt"
    commands.append(
        {
            "stage": "train_pair_scorer",
            "outputs": {"model": str(pair_model), "report": str(reports_dir / "cascade_pair_scorer.json")},
            "cmd": _join_cmd(
                [
                    "python",
                    "-m",
                    "cascade_planner.eval.train_cascade_pair_scorer",
                    "--pack-dir",
                    str(pair_pack_dir),
                    "--model-output",
                    str(pair_model),
                    "--report-output",
                    str(reports_dir / "cascade_pair_scorer.json"),
                    "--md-output",
                    str(reports_dir / "cascade_pair_scorer.md"),
                    "--epochs",
                    str(pair_epochs),
                ]
            ),
        }
    )

    commands.append(
        {
            "stage": "build_action_source_pack",
            "outputs": {"pack_dir": str(pack_dir)},
            "cmd": _join_cmd(
                [
                    "python",
                    "-m",
                    "cascade_planner.eval.build_cascade_action_value_pack",
                    *sum([["--trace", path] for path in train_val_expansion], []),
                    *sum([["--runtime", path] for path in train_val_runtime], []),
                    "--benchmark",
                    str(all_benchmark),
                    "--output-dir",
                    str(pack_dir),
                    "--preserve-benchmark-splits",
                ]
            ),
        }
    )
    commands.append(
        {
            "stage": "build_transition_pack",
            "outputs": {"pack_dir": str(transition_pack_dir)},
            "cmd": _join_cmd(
                [
                    "python",
                    "-m",
                    "cascade_planner.eval.build_cascade_transition_pack",
                    *sum([["--trace", path] for path in train_val_cascade_trace], []),
                    "--output-dir",
                    str(transition_pack_dir),
                ]
            ),
        }
    )
    action_model = models_dir / "cascade_action_value_route_quality.pt"
    source_model = models_dir / "cascade_source_value.pt"
    transition_model = models_dir / "cascade_transition_value.pt"
    final_action_model = action_model
    final_source_model = source_model
    bootstrap_trace_outputs: dict[str, dict[str, Any]] = {}
    commands.extend(
        [
            {
                "stage": "train_action_value",
                "outputs": {"model": str(action_model), "report": str(reports_dir / "cascade_action_value_route_quality.json")},
                "cmd": _join_cmd(
                    [
                        "python",
                        "-m",
                        "cascade_planner.eval.train_cascade_action_value",
                        "--pack-dir",
                        str(pack_dir),
                        "--model-output",
                        str(action_model),
                        "--report-output",
                        str(reports_dir / "cascade_action_value_route_quality.json"),
                        "--md-output",
                        str(reports_dir / "cascade_action_value_route_quality.md"),
                        "--label-name",
                        "route_quality_action_value",
                        "--loss-mode",
                        "pairwise",
                        "--selection-metric",
                        "pairwise_positive_state_accuracy",
                        "--epochs",
                        str(action_epochs),
                    ]
                ),
            },
            {
                "stage": "train_source_value",
                "outputs": {"model": str(source_model), "report": str(reports_dir / "cascade_source_value.json")},
                "cmd": _join_cmd(
                    [
                        "python",
                        "-m",
                        "cascade_planner.eval.train_cascade_source_value",
                        "--pack-dir",
                        str(pack_dir),
                        "--model-output",
                        str(source_model),
                        "--report-output",
                        str(reports_dir / "cascade_source_value.json"),
                        "--md-output",
                        str(reports_dir / "cascade_source_value.md"),
                        "--loss-mode",
                        "pairwise",
                        "--selection-metric",
                        "top1_positive_state_hit_rate",
                        "--epochs",
                        str(source_epochs),
                    ]
                ),
            },
            {
                "stage": "train_transition_value",
                "outputs": {"model": str(transition_model), "report": str(reports_dir / "cascade_transition_value.json")},
                "cmd": _join_cmd(
                    [
                        "python",
                        "-m",
                        "cascade_planner.eval.train_cascade_transition_value",
                        "--pack-dir",
                        str(transition_pack_dir),
                        "--model-output",
                        str(transition_model),
                        "--report-output",
                        str(reports_dir / "cascade_transition_value.json"),
                        "--md-output",
                        str(reports_dir / "cascade_transition_value.md"),
                        "--epochs",
                        str(transition_epochs),
                    ]
                ),
            },
        ]
    )

    if enable_bootstrap_stage3:
        bootstrap_pack_dir = output_root / "packs" / "v4_bootstrap_action_value_pack"
        bootstrap_action_model = models_dir / "cascade_action_value_route_outcome_bootstrap.pt"
        bootstrap_source_model = models_dir / "cascade_source_value_bootstrap.pt"
        final_action_model = bootstrap_action_model
        final_source_model = bootstrap_source_model
        bootstrap_action_epochs = bootstrap_action_epochs or action_epochs
        bootstrap_source_epochs = bootstrap_source_epochs or source_epochs
        bootstrap_runtime_paths = []
        bootstrap_expansion_paths = []
        bootstrap_cascade_trace_paths = []
        for split in selected_bootstrap_splits:
            runtime_paths = []
            expansion_paths = []
            cascade_trace_paths = []
            base_dir = output_root / "traces" / "bootstrap_stage3" / split
            for shard in range(num_shards):
                runtime = base_dir / f"runtime_shard{shard}of{num_shards}.json"
                expansion_base = base_dir / "chem_enzy_expansion.jsonl"
                cascade_base = base_dir / "cascade_trace.jsonl"
                runtime_paths.append(runtime)
                expansion_paths.append(_sharded_jsonl_path(expansion_base, shard, num_shards))
                cascade_trace_paths.append(_sharded_jsonl_path(cascade_base, shard, num_shards))
                commands.append(
                    {
                        "stage": "bootstrap_trace",
                        "config": "bootstrap_stage3",
                        "split": split,
                        "shard_index": shard,
                        "num_shards": num_shards,
                        "depends_on": [
                            "train_action_value",
                            "train_source_value",
                            "train_transition_value",
                            "train_pair_scorer",
                        ],
                        "outputs": {
                            "runtime": str(runtime),
                            "chem_enzy_expansion_trace": str(expansion_paths[-1]),
                            "cascade_trace": str(cascade_trace_paths[-1]),
                        },
                        "cmd": _join_cmd(
                            _benchmark_cmd(
                                benchmark_runner,
                                [
                                    "--benchmark",
                                    str(split_files[split]),
                                    "--output",
                                    str(runtime),
                                    "--trace-output",
                                    str(cascade_base),
                                    "--chem-enzy-expansion-trace-output",
                                    str(expansion_base),
                                    "--include-route-outcomes",
                                    "--vendor-root",
                                    str(vendor_root),
                                    "--iterations",
                                    str(iterations),
                                    "--chem-enzy-max-depth",
                                    str(chem_enzy_max_depth),
                                    "--expansion-topk",
                                    str(expansion_topk),
                                    "--cascade-expansion-budget",
                                    str(cascade_expansion_budget),
                                    "--gpu",
                                    str(gpu),
                                    "--num-shards",
                                    str(num_shards),
                                    "--shard-index",
                                    str(shard),
                                    "--chem-enzy-cascade-cost",
                                    "--chem-enzy-cascade-source-policy",
                                    "--chem-enzy-cascade-context-from-row",
                                    "--chem-enzy-cascade-context-policy",
                                    "safe",
                                    "--chem-enzy-cascade-source-policy-json",
                                    json.dumps(
                                        {
                                            "enabled": True,
                                            "source_value_model_path": str(source_model),
                                            "learned_topk_mode": "score_ratio",
                                            "learned_rule_combine": "max",
                                            "learned_min_topk_fraction": 0.25,
                                            "min_unpreferred_topk": 8,
                                        },
                                        separators=(",", ":"),
                                    ),
                                    "--chem-enzy-cascade-cost-json",
                                    json.dumps(
                                        {
                                            "enabled": True,
                                            "action_value_model_path": str(action_model),
                                            "weights": {
                                                "learned_action_value_score_reward": 0.35,
                                                "learned_source_value_score_reward": 0.10,
                                                "min_cost": 0.000001,
                                            },
                                        },
                                        separators=(",", ":"),
                                    ),
                                    "--cascade-transition-model",
                                    str(transition_model),
                                    "--cascade-pair-scorer",
                                    str(pair_model),
                                    "--cascade-pair-reward-weight",
                                    str(cascade_pair_reward_weight),
                                    "--cascade-pair-reward-mode",
                                    cascade_pair_reward_mode,
                                    "--cascade-pair-reward-tie-epsilon",
                                    str(cascade_pair_reward_tie_epsilon),
                                ],
                            )
                        ),
                    }
                )
            merged_runtime = output_root / "merged" / f"bootstrap_stage3_{split}_runtime.json"
            commands.append(
                {
                    "stage": "bootstrap_merge_runtime",
                    "config": "bootstrap_stage3",
                    "split": split,
                    "outputs": {"runtime": str(merged_runtime)},
                    "cmd": _join_cmd(
                        _benchmark_cmd(
                            benchmark_runner,
                            [
                                "--merge",
                                *[str(path) for path in runtime_paths],
                                "--output",
                                str(merged_runtime),
                            ],
                        )
                    ),
                }
            )
            bootstrap_trace_outputs[split] = {
                "runtime_paths": [str(path) for path in runtime_paths],
                "merged_runtime": str(merged_runtime),
                "expansion_trace_paths": [str(path) for path in expansion_paths],
                "cascade_trace_paths": [str(path) for path in cascade_trace_paths],
            }
            bootstrap_runtime_paths.append(str(merged_runtime))
            bootstrap_expansion_paths.extend(str(path) for path in expansion_paths)
            bootstrap_cascade_trace_paths.extend(str(path) for path in cascade_trace_paths)

        commands.extend(
            [
                {
                    "stage": "build_bootstrap_action_source_pack",
                    "outputs": {"pack_dir": str(bootstrap_pack_dir)},
                    "cmd": _join_cmd(
                        [
                            "python",
                            "-m",
                            "cascade_planner.eval.build_cascade_action_value_pack",
                            *sum([["--trace", path] for path in bootstrap_expansion_paths], []),
                            *sum([["--runtime", path] for path in bootstrap_runtime_paths], []),
                            "--benchmark",
                            str(all_benchmark),
                            "--output-dir",
                            str(bootstrap_pack_dir),
                            "--preserve-benchmark-splits",
                        ]
                    ),
                },
                {
                    "stage": "train_bootstrap_action_value",
                    "outputs": {
                        "model": str(bootstrap_action_model),
                        "report": str(reports_dir / "cascade_action_value_route_outcome_bootstrap.json"),
                    },
                    "cmd": _join_cmd(
                        [
                            "python",
                            "-m",
                            "cascade_planner.eval.train_cascade_action_value",
                            "--pack-dir",
                            str(bootstrap_pack_dir),
                            "--model-output",
                            str(bootstrap_action_model),
                            "--report-output",
                            str(reports_dir / "cascade_action_value_route_outcome_bootstrap.json"),
                            "--md-output",
                            str(reports_dir / "cascade_action_value_route_outcome_bootstrap.md"),
                            "--label-name",
                            "route_outcome_action_value",
                            "--loss-mode",
                            "pairwise",
                            "--selection-metric",
                            "pairwise_positive_state_accuracy",
                            "--epochs",
                            str(bootstrap_action_epochs),
                        ]
                    ),
                },
                {
                    "stage": "train_bootstrap_source_value",
                    "outputs": {
                        "model": str(bootstrap_source_model),
                        "report": str(reports_dir / "cascade_source_value_bootstrap.json"),
                    },
                    "cmd": _join_cmd(
                        [
                            "python",
                            "-m",
                            "cascade_planner.eval.train_cascade_source_value",
                            "--pack-dir",
                            str(bootstrap_pack_dir),
                            "--model-output",
                            str(bootstrap_source_model),
                            "--report-output",
                            str(reports_dir / "cascade_source_value_bootstrap.json"),
                            "--md-output",
                            str(reports_dir / "cascade_source_value_bootstrap.md"),
                            "--loss-mode",
                            "pairwise",
                            "--selection-metric",
                            "top1_positive_state_hit_rate",
                            "--epochs",
                            str(bootstrap_source_epochs),
                        ]
                    ),
                },
            ]
        )

    full100_eval = output_root / "full100_eval" / "learned_cascade_value_full100.json"
    commands.append(
        {
            "stage": "locked_full100_eval",
            "training_use": "forbidden",
            "outputs": {"runtime": str(full100_eval)},
            "cmd": _join_cmd(
                _benchmark_cmd(
                    benchmark_runner,
                    [
                    "--benchmark",
                    "data/benchmark_v2_100.json",
                    "--output",
                    str(full100_eval),
                    "--include-route-outcomes",
                    "--vendor-root",
                    str(vendor_root),
                    "--chem-enzy-cascade-cost",
                    "--chem-enzy-cascade-source-policy",
                    "--chem-enzy-cascade-context-from-row",
                    "--chem-enzy-cascade-context-policy",
                    "safe",
                    "--chem-enzy-cascade-source-policy-json",
                    json.dumps(
                        {
                            "enabled": True,
                            "source_value_model_path": str(final_source_model),
                            "learned_topk_mode": "score_ratio",
                            "learned_rule_combine": "max",
                            "learned_min_topk_fraction": 0.25,
                            "min_unpreferred_topk": 8,
                        },
                        separators=(",", ":"),
                    ),
                    "--chem-enzy-cascade-cost-json",
                    json.dumps(
                        {
                            "enabled": True,
                            "action_value_model_path": str(final_action_model),
                            "weights": {
                                "learned_action_value_score_reward": 0.5,
                                "learned_source_value_score_reward": 0.10,
                            },
                        },
                        separators=(",", ":"),
                    ),
                    "--cascade-transition-model",
                    str(transition_model),
                    "--cascade-pair-scorer",
                    str(pair_model),
                    "--cascade-pair-reward-weight",
                    str(cascade_pair_reward_weight),
                    "--cascade-pair-reward-mode",
                    cascade_pair_reward_mode,
                    "--cascade-pair-reward-tie-epsilon",
                    str(cascade_pair_reward_tie_epsilon),
                    "--iterations",
                    str(iterations),
                    "--chem-enzy-max-depth",
                    str(chem_enzy_max_depth),
                    "--expansion-topk",
                    str(expansion_topk),
                    "--gpu",
                    str(gpu),
                    ],
                )
            ),
        }
    )

    manifest = {
        "schema_version": PIPELINE_SCHEMA_VERSION,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "metadata": {
            "split_dir": str(split_dir),
            "output_root": str(output_root),
            "vendor_root": str(vendor_root),
            "configs": selected_configs,
            "splits": selected_splits,
            "bootstrap_stage3": {
                "enabled": enable_bootstrap_stage3,
                "splits": selected_bootstrap_splits if enable_bootstrap_stage3 else [],
                "action_label": "route_outcome_action_value",
            },
            "benchmark_runner": benchmark_runner,
            "num_shards": num_shards,
            "cascade_pair_reward": {
                "weight": cascade_pair_reward_weight,
                "mode": cascade_pair_reward_mode,
                "tie_epsilon": cascade_pair_reward_tie_epsilon,
            },
            "full100_training_use": "forbidden",
        },
        "split_manifest": split_manifest,
        "trace_outputs": trace_outputs,
        "bootstrap_trace_outputs": bootstrap_trace_outputs,
        "models": {
            "pair_scorer": str(pair_model),
            "initial_action_value": str(action_model),
            "initial_source_value": str(source_model),
            "final_action_value": str(final_action_model),
            "final_source_value": str(final_source_model),
            "transition_value": str(transition_model),
        },
        "commands": commands,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_file = output_root / "v4_full_training_pipeline_manifest.json"
    commands_file = output_root / "v4_full_training_commands.sh"
    manifest_file.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    commands_file.write_text(_commands_script(commands), encoding="utf-8")
    manifest["outputs"] = {
        "manifest": str(manifest_file),
        "commands": str(commands_file),
    }
    manifest_file.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def _collect_paths(
    trace_outputs: dict[str, dict[str, dict[str, Any]]],
    *,
    configs: list[str],
    splits: list[str],
    key: str,
) -> list[str]:
    out: list[str] = []
    for config in configs:
        for split in splits:
            value = trace_outputs.get(config, {}).get(split, {}).get(key)
            if isinstance(value, list):
                out.extend(value)
            elif value:
                out.append(str(value))
    return out


def _validate_configs(configs: list[str]) -> None:
    unknown = [config for config in configs if config not in TRACE_CONFIGS]
    if unknown:
        raise ValueError(f"unknown trace configs: {unknown}; allowed={sorted(TRACE_CONFIGS)}")


def _validate_splits(splits: list[str]) -> None:
    unknown = [split for split in splits if split not in {"train", "val", "test"}]
    if unknown:
        raise ValueError(f"unknown splits: {unknown}")


def _sharded_jsonl_path(path: Path, shard_index: int, num_shards: int) -> Path:
    if num_shards <= 1:
        return Path(path)
    path = Path(path)
    suffix = "".join(path.suffixes) if path.suffixes else path.suffix
    stem = path.name[:-len(suffix)] if suffix else path.name
    return path.with_name(f"{stem}_shard{shard_index}of{num_shards}{suffix}")


def _join_cmd(parts: list[Any]) -> str:
    return " ".join(shlex.quote(str(part)) for part in parts if str(part) != "")


def _benchmark_cmd(runner: list[str], args: list[Any]) -> list[Any]:
    if not runner:
        return ["python", "-m", "cascade_planner.eval.run_cascade_search_benchmark", *args]
    return [*runner, *args]


def _commands_script(commands: list[dict[str, Any]]) -> str:
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        "# Generated by cascade_planner.eval.run_v4_full_training_pipeline.",
        "# Run trace commands on compute shards first, then merge/pack/train commands.",
        "",
    ]
    for idx, command in enumerate(commands, start=1):
        lines.append(f"# [{idx}] {command.get('stage')} {command.get('config', '')} {command.get('split', '')}".rstrip())
        lines.append(str(command["cmd"]))
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Create full v4 training pipeline command manifest")
    ap.add_argument("--split-dir", required=True)
    ap.add_argument("--output-root", required=True)
    ap.add_argument("--vendor-root", default="vendor/ChemEnzyRetroPlanner")
    ap.add_argument(
        "--benchmark-runner",
        action="append",
        default=[],
        help=(
            "Command prefix used for ChemEnzy benchmark runs. Defaults to the "
            "isolated ChemEnzy runtime wrapper."
        ),
    )
    ap.add_argument("--config", action="append", choices=sorted(TRACE_CONFIGS), default=[])
    ap.add_argument("--split", action="append", choices=["train", "val", "test"], default=[])
    ap.add_argument("--bootstrap-split", action="append", choices=["train", "val", "test"], default=[])
    ap.add_argument("--no-bootstrap-stage3", action="store_true")
    ap.add_argument("--num-shards", type=int, default=8)
    ap.add_argument("--gpu", type=int, default=-1)
    ap.add_argument("--iterations", type=int, default=10)
    ap.add_argument("--chem-enzy-max-depth", type=int, default=6)
    ap.add_argument("--expansion-topk", type=int, default=50)
    ap.add_argument("--cascade-expansion-budget", type=int, default=100)
    ap.add_argument("--action-epochs", type=int, default=20)
    ap.add_argument("--source-epochs", type=int, default=20)
    ap.add_argument("--transition-epochs", type=int, default=8)
    ap.add_argument("--pair-epochs", type=int, default=20)
    ap.add_argument("--cascade-pair-reward-weight", type=float, default=0.35)
    ap.add_argument(
        "--cascade-pair-reward-mode",
        choices=["additive", "guarded_tie_break"],
        default="additive",
    )
    ap.add_argument("--cascade-pair-reward-tie-epsilon", type=float, default=0.0)
    ap.add_argument("--bootstrap-action-epochs", type=int, default=None)
    ap.add_argument("--bootstrap-source-epochs", type=int, default=None)
    args = ap.parse_args()
    manifest = plan_v4_full_training_pipeline(
        split_dir=Path(args.split_dir),
        output_root=Path(args.output_root),
        vendor_root=Path(args.vendor_root),
        benchmark_runner=args.benchmark_runner or None,
        configs=args.config or None,
        splits=args.split or None,
        bootstrap_splits=args.bootstrap_split or None,
        enable_bootstrap_stage3=not args.no_bootstrap_stage3,
        num_shards=args.num_shards,
        gpu=args.gpu,
        iterations=args.iterations,
        chem_enzy_max_depth=args.chem_enzy_max_depth,
        expansion_topk=args.expansion_topk,
        cascade_expansion_budget=args.cascade_expansion_budget,
        action_epochs=args.action_epochs,
        source_epochs=args.source_epochs,
        transition_epochs=args.transition_epochs,
        pair_epochs=args.pair_epochs,
        cascade_pair_reward_weight=args.cascade_pair_reward_weight,
        cascade_pair_reward_mode=args.cascade_pair_reward_mode,
        cascade_pair_reward_tie_epsilon=args.cascade_pair_reward_tie_epsilon,
        bootstrap_action_epochs=args.bootstrap_action_epochs,
        bootstrap_source_epochs=args.bootstrap_source_epochs,
    )
    print(json.dumps({
        "outputs": manifest["outputs"],
        "n_commands": len(manifest["commands"]),
        "configs": manifest["metadata"]["configs"],
        "splits": manifest["metadata"]["splits"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
