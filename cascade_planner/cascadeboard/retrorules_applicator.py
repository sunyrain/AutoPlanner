"""RetroRules/Rhea template applicator for recall-oriented candidate generation."""
from __future__ import annotations

import csv
import gzip
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from rdkit import Chem, RDLogger


RDLogger.DisableLog("rdApp.*")


DEFAULT_TEMPLATE_PATHS = (
    Path("data_external/retrorules/templates_rhea.csv.gz"),
    Path("data_external/retrorules/templates_metanetx.csv.gz"),
)

_ATOM_RE = re.compile(r"\[([A-Za-z#0-9]+)((?:;[^:\]]+)*):(\d+)\]")


@dataclass(frozen=True)
class RetroRuleTemplate:
    template_id: str
    template: str
    product_smarts: str
    dataset: str
    ecs: tuple[str, ...]
    ec1s: tuple[str, ...]
    score: float
    reactions_count: int


class RetroRulesApplicator:
    """Apply RetroRules-style retrosynthetic templates to a product molecule.

    The loader keeps a bounded, EC-indexed subset of valid templates. Templates
    are used only as candidate generators; they are not treated as labels.
    """

    def __init__(
        self,
        template_paths: Iterable[str | Path] = DEFAULT_TEMPLATE_PATHS,
        *,
        max_templates: int = 5000,
        max_per_ec1: int = 500,
        max_templates_per_query: int = 250,
        max_outcomes_per_template: int = 1,
        generalize: int = 0,
    ):
        self.template_paths = tuple(Path(path) for path in template_paths)
        self.max_templates = max(1, int(max_templates))
        self.max_per_ec1 = max(1, int(max_per_ec1))
        self.max_templates_per_query = max(1, int(max_templates_per_query))
        self.max_outcomes_per_template = max(1, int(max_outcomes_per_template))
        self.generalize = max(0, min(int(generalize), 2))
        self._loaded = False
        self._templates: list[RetroRuleTemplate] = []
        self._by_ec1: dict[str, list[RetroRuleTemplate]] = defaultdict(list)
        self._product_queries: dict[tuple[str, str], Chem.Mol] = {}

    @classmethod
    def from_env(cls) -> "RetroRulesApplicator":
        raw_paths = os.environ.get("AUTOPLANNER_RETRORULES_TEMPLATES", "")
        paths = [Path(p) for p in raw_paths.split(os.pathsep) if p] if raw_paths else DEFAULT_TEMPLATE_PATHS
        return cls(
            paths,
            max_templates=_env_int("AUTOPLANNER_RETRORULES_MAX_TEMPLATES", 5000),
            max_per_ec1=_env_int("AUTOPLANNER_RETRORULES_MAX_PER_EC1", 500),
            max_templates_per_query=_env_int("AUTOPLANNER_RETRORULES_MAX_PER_QUERY", 250),
            max_outcomes_per_template=_env_int("AUTOPLANNER_RETRORULES_OUTCOMES_PER_TEMPLATE", 1),
            generalize=_env_int("AUTOPLANNER_RETRORULES_GENERALIZE", 0),
        )

    @property
    def available(self) -> bool:
        self._load()
        return bool(self._templates)

    def predict(
        self,
        product_smiles: str,
        top_k: int = 10,
        *,
        ec_token: str = "",
        skel_type: str = "",
    ) -> list[dict[str, Any]]:
        self._load()
        product_mol = Chem.MolFromSmiles(product_smiles)
        if not self._templates or not product_smiles or product_mol is None:
            return []
        ec1 = _ec1(ec_token)
        templates = self._rank_templates(self._query_templates(ec1), product_mol, ec_token=ec_token, skel_type=skel_type)
        out: list[dict[str, Any]] = []
        seen: set[str] = set()
        applied_templates = 0
        for template_rank, item in enumerate(templates, start=1):
            product_query = self._product_queries.get(_template_key(item))
            if product_query is not None and not product_mol.HasSubstructMatch(product_query):
                continue
            applied_templates += 1
            if applied_templates > self.max_templates_per_query:
                break
            outcomes = apply_template_to_product(
                item.template,
                product_smiles,
                max_outcomes=max(1, min(top_k, self.max_outcomes_per_template)),
                generalize=self.generalize,
            )
            for outcome in outcomes:
                reactants = sorted(outcome)
                if not reactants:
                    continue
                rxn = ".".join(reactants) + ">>" + product_smiles
                if rxn in seen:
                    continue
                seen.add(rxn)
                ec = _best_ec(item.ecs, ec1)
                out.append({
                    "main_reactant": reactants[0],
                    "aux_reactants": reactants[1:],
                    "rxn_smiles": rxn,
                    "reaction_smiles": rxn,
                    "ec": ec,
                    "score": _template_score(item, template_rank),
                    "type": skel_type or "retrorules_template",
                    "reaction_type": skel_type or "retrorules_template",
                    "source": f"retrorules_{item.dataset or 'template'}",
                    "template_id": item.template_id,
                    "template_rank": template_rank,
                    "evidence": {
                        "template_id": item.template_id,
                        "dataset": item.dataset,
                        "ecs": ";".join(item.ecs),
                        "reactions_count": item.reactions_count,
                    },
                })
                if len(out) >= top_k:
                    return out
        return out

    def _load(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        rows: list[RetroRuleTemplate] = []
        for path in self.template_paths:
            if not path.exists():
                continue
            rows.extend(_iter_template_rows(path))
        rows = sorted(rows, key=lambda row: (row.score, row.reactions_count), reverse=True)
        ec_counts: dict[str, int] = defaultdict(int)
        for row in rows:
            if len(self._templates) >= self.max_templates:
                break
            product_query = Chem.MolFromSmarts(row.product_smarts)
            if product_query is None:
                continue
            ec1_keys = row.ec1s or ("",)
            if all(ec_counts[key] >= self.max_per_ec1 for key in ec1_keys):
                continue
            self._templates.append(row)
            self._product_queries[_template_key(row)] = product_query
            self._by_ec1[""].append(row)
            for key in ec1_keys:
                if key:
                    self._by_ec1[key].append(row)
                    ec_counts[key] += 1

    def _query_templates(self, ec1: str) -> list[RetroRuleTemplate]:
        if ec1 and self._by_ec1.get(ec1):
            specific = self._by_ec1[ec1]
            generic = [row for row in self._by_ec1.get("", []) if row not in specific]
            return [*specific, *generic]
        return self._templates

    def _rank_templates(
        self,
        templates: list[RetroRuleTemplate],
        product_mol: Chem.Mol,
        *,
        ec_token: str = "",
        skel_type: str = "",
    ) -> list[RetroRuleTemplate]:
        exact_ec = _normal_ec(ec_token)
        prefixes = _skeleton_ec_prefixes(skel_type)

        def key(item: RetroRuleTemplate) -> tuple[float, ...]:
            product_query = self._product_queries.get(_template_key(item))
            product_match = bool(product_query is not None and product_mol.HasSubstructMatch(product_query))
            exact_match = bool(exact_ec and exact_ec in item.ecs)
            prefix_match = _template_matches_prefix(item, prefixes)
            dataset_score = 1.0 if item.dataset == "rhea" else 0.5 if item.dataset == "metanetx" else 0.0
            atoms = float(product_query.GetNumAtoms() if product_query is not None else 0)
            support = float(min(item.reactions_count, 100))
            return (
                1.0 if product_match else 0.0,
                1.0 if exact_match else 0.0,
                1.0 if prefix_match else 0.0,
                dataset_score,
                atoms,
                float(item.score or 0.0),
                support,
            )

        return sorted(templates, key=key, reverse=True)


def retrorules_enabled() -> bool:
    return str(os.environ.get("AUTOPLANNER_ENABLE_RETRORULES") or "").lower() in {"1", "true", "yes", "on"}


def _iter_template_rows(path: Path) -> Iterable[RetroRuleTemplate]:
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8", errors="replace", newline="") as fh:
        for row in csv.DictReader(fh):
            if str(row.get("VALID") or "").lower() != "true":
                continue
            template = row.get("TEMPLATE") or ""
            if ">>" not in template:
                continue
            ecs = tuple(ec for ec in (row.get("ECS") or "").split(";") if ec)
            ec1s = tuple(sorted({_ec1(ec) for ec in ecs if _ec1(ec)}))
            yield RetroRuleTemplate(
                template_id=row.get("TEMPLATE_ID") or "",
                template=template,
                product_smarts=template.split(">>", 1)[0],
                dataset=row.get("DATASETS") or path.stem,
                ecs=ecs,
                ec1s=ec1s,
                score=_safe_float(row.get("SCORE"), 0.0),
                reactions_count=int(_safe_float(row.get("REACTIONS_COUNT"), 0.0)),
            )


def _template_score(template: RetroRuleTemplate, rank: int) -> float:
    support = min(template.reactions_count, 100) / 100.0
    return float(template.score or 0.0) + 0.25 * support + 1.0 / max(rank, 1)


def _template_key(template: RetroRuleTemplate) -> tuple[str, str]:
    return (template.template_id, template.template)


def _best_ec(ecs: tuple[str, ...], ec1: str) -> str:
    if ec1:
        for ec in ecs:
            if _ec1(ec) == ec1:
                return ec
    return ecs[0] if ecs else ""


def _normal_ec(value: Any) -> str:
    text = str(value or "").strip()
    parts = text.split(".")
    if len(parts) < 2:
        return ""
    return text if parts[0].isdigit() else ""


def _ec1(value: Any) -> str:
    text = str(value or "")
    first = text.split(".", 1)[0]
    return first if first.isdigit() else ""


def _skeleton_ec_prefixes(skel_type: str) -> tuple[str, ...]:
    key = str(skel_type or "").strip().lower()
    mapping = {
        "phosphorylation": ("2.7",),
        "glycosylation": ("2.4",),
        "hydrolysis": ("3",),
        "isomerization": ("5",),
        "oxidation": ("1",),
        "reduction": ("1",),
        "dehydrogenation": ("1",),
        "methylation": ("2.1",),
        "acylation": ("2.3",),
        "amination": ("2.6",),
        "transamination": ("2.6",),
        "decarboxylation": ("4.1",),
        "c-c coupling": ("4", "5"),
        "c_c_coupling": ("4", "5"),
    }
    return mapping.get(key, ())


def _template_matches_prefix(template: RetroRuleTemplate, prefixes: tuple[str, ...]) -> bool:
    if not prefixes or not template.ecs:
        return False
    for prefix in prefixes:
        prefix = str(prefix).strip(".")
        if not prefix:
            continue
        for ec in template.ecs:
            if ec == prefix or ec.startswith(prefix + "."):
                return True
    return False


def _safe_float(value: Any, default: float) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if out == out else default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def apply_template_to_product(
    template: str,
    product_smi: str,
    max_outcomes: int = 5,
    generalize: int = 0,
) -> list[frozenset[str]]:
    """Apply an rdchiral retrosynthetic template without importing training code."""
    from rdchiral.main import rdchiralRunText

    out: list[frozenset[str]] = []
    if generalize:
        template = _generalize_template(template, generalize)
    try:
        outcomes = rdchiralRunText(template, product_smi)
    except Exception:
        return out
    seen: set[frozenset[str]] = set()
    for outcome in outcomes[:max_outcomes]:
        try:
            canonical = _canon_set(outcome)
            if canonical and canonical not in seen:
                seen.add(canonical)
                out.append(canonical)
        except Exception:
            continue
    return out


def _generalize_template(template: str, level: int = 1) -> str:
    if level <= 0:
        return template

    def _sub(match: re.Match[str]) -> str:
        element, _props, map_number = match.group(1), match.group(2), match.group(3)
        if level >= 2:
            return f"[*:{map_number}]"
        return f"[{element}:{map_number}]"

    return _ATOM_RE.sub(_sub, template)


def _canon_set(smiles_dot: str) -> frozenset[str]:
    parts = []
    for smiles in str(smiles_dot or "").split("."):
        text = smiles.strip()
        if not text:
            continue
        mol = Chem.MolFromSmiles(text)
        if mol is None:
            return frozenset()
        parts.append(Chem.MolToSmiles(mol))
    return frozenset(parts)
