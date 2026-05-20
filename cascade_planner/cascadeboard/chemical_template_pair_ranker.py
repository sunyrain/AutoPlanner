"""Pairwise neural ranker for USPTO chemical templates."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from rdkit import Chem, DataStructs

from cascade_planner.vnext.features import morgan_fp


DEFAULT_PAIR_RANKER_DIR = Path("results/shared/chemical_template_preselector/uspto_pair_hardneg_mlp_20260507")


class ChemicalTemplatePairRankerModel(nn.Module):
    def __init__(self, n_bits: int, numeric_dim: int = 3, hidden: int = 512, dropout: float = 0.15):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_bits * 2 + numeric_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class ChemicalTemplatePairRanker:
    def __init__(self, artifact_dir: str | Path = DEFAULT_PAIR_RANKER_DIR, *, device: str | None = None):
        self.artifact_dir = Path(artifact_dir)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.available = False
        self.n_bits = 256
        self.hidden = 512
        self.model: ChemicalTemplatePairRankerModel | None = None
        self._template_fp_cache: dict[str, np.ndarray] = {}
        self._load()

    @classmethod
    def from_env(cls) -> "ChemicalTemplatePairRanker":
        return cls(os.environ.get("AUTOPLANNER_CHEM_TEMPLATE_PAIR_RANKER_DIR", DEFAULT_PAIR_RANKER_DIR))

    def score_templates(self, product_smiles: str, templates: list[Any]) -> dict[str, float]:
        if not self.available or self.model is None or not templates:
            return {}
        product_fp = morgan_fp(product_smiles, n_bits=self.n_bits)
        product_mol = Chem.MolFromSmiles(product_smiles or "")
        rows = []
        ids = []
        for item in templates:
            tid = str(getattr(item, "template_id", ""))
            template = str(getattr(item, "template", ""))
            query_smarts = str(getattr(item, "product_smarts", ""))
            query = Chem.MolFromSmarts(query_smarts) if query_smarts else None
            match = float(product_mol is not None and query is not None and product_mol.HasSubstructMatch(query))
            atoms = float(query.GetNumAtoms() if query is not None else 0.0) / 64.0
            support = min(float(getattr(item, "reactions_count", 0) or 0.0), 100.0) / 100.0
            rows.append(np.concatenate([
                product_fp,
                self._template_fp(template, query_smarts),
                np.asarray([match, atoms, support], dtype=np.float32),
            ]))
            ids.append(tid)
        x = torch.from_numpy(np.asarray(rows, dtype=np.float32)).to(self.device)
        with torch.no_grad():
            scores = torch.sigmoid(self.model(x)).detach().cpu().numpy()
        return {tid: float(score) for tid, score in zip(ids, scores)}

    def _template_fp(self, template: str, product_smarts: str) -> np.ndarray:
        key = f"{self.n_bits}:{template}"
        cached = self._template_fp_cache.get(key)
        if cached is not None:
            return cached
        arr = np.zeros(self.n_bits, dtype=np.float32)
        mol = Chem.MolFromSmarts(product_smarts or "")
        if mol is not None:
            fp = Chem.PatternFingerprint(mol, fpSize=self.n_bits)
            DataStructs.ConvertToNumpyArray(fp, arr)
        self._template_fp_cache[key] = arr
        return arr

    def _load(self) -> None:
        meta_path = self.artifact_dir / "chemical_template_pair_ranker.json"
        model_path = self.artifact_dir / "chemical_template_pair_ranker.pt"
        if not meta_path.exists() or not model_path.exists():
            return
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            self.n_bits = int(meta.get("n_bits") or 256)
            self.hidden = int(meta.get("hidden") or 512)
            model = ChemicalTemplatePairRankerModel(self.n_bits, hidden=self.hidden, dropout=0.0)
            state = torch.load(model_path, map_location=self.device)
            model.load_state_dict(state)
            model.to(self.device)
            model.eval()
        except Exception:
            return
        self.model = model
        self.available = True


def pair_ranker_enabled() -> bool:
    raw = os.environ.get("AUTOPLANNER_ENABLE_CHEM_TEMPLATE_PAIR_RANKER")
    if raw is None:
        return DEFAULT_PAIR_RANKER_DIR.joinpath("chemical_template_pair_ranker.pt").exists()
    return str(raw).lower() in {"1", "true", "yes", "on"}
