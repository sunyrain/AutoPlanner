"""Neural USPTO template preselector for chemical retrosynthesis templates."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from cascade_planner.vnext.features import morgan_fp


DEFAULT_PRESELECTOR_DIR = Path("results/shared/chemical_template_preselector/uspto_product_mlp_20260507")


class ChemicalTemplatePreselectorModel(nn.Module):
    def __init__(self, n_bits: int, n_templates: int, hidden: int = 512, dropout: float = 0.15):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_bits, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_templates),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ChemicalTemplatePreselector:
    """Load a trained product-to-template classifier and score template IDs."""

    def __init__(self, artifact_dir: str | Path = DEFAULT_PRESELECTOR_DIR, *, device: str | None = None):
        self.artifact_dir = Path(artifact_dir)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.available = False
        self.n_bits = 512
        self.template_ids: list[str] = []
        self.template_to_index: dict[str, int] = {}
        self.model: ChemicalTemplatePreselectorModel | None = None
        self._load()

    @classmethod
    def from_env(cls) -> "ChemicalTemplatePreselector":
        return cls(os.environ.get("AUTOPLANNER_CHEM_TEMPLATE_PRESELECTOR_DIR", DEFAULT_PRESELECTOR_DIR))

    def score_template_ids(self, product_smiles: str, template_ids: list[str]) -> dict[str, float]:
        if not self.available or self.model is None or not template_ids:
            return {}
        x = torch.from_numpy(morgan_fp(product_smiles, n_bits=self.n_bits)).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(x).squeeze(0)
            probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()
        out: dict[str, float] = {}
        for tid in template_ids:
            idx = self.template_to_index.get(tid)
            if idx is not None:
                out[tid] = float(probs[idx])
        return out

    def top_template_ids(self, product_smiles: str, top_k: int = 100) -> list[tuple[str, float]]:
        if not self.available or self.model is None:
            return []
        x = torch.from_numpy(morgan_fp(product_smiles, n_bits=self.n_bits)).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(x).squeeze(0)
            k = min(max(1, int(top_k)), len(self.template_ids))
            scores, indices = torch.topk(torch.softmax(logits, dim=-1), k=k)
        return [(self.template_ids[int(idx)], float(score)) for score, idx in zip(scores.detach().cpu(), indices.detach().cpu())]

    def _load(self) -> None:
        meta_path = self.artifact_dir / "chemical_template_preselector.json"
        model_path = self.artifact_dir / "chemical_template_preselector.pt"
        if not meta_path.exists() or not model_path.exists():
            return
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            template_ids = list(meta.get("template_ids") or [])
            n_bits = int(meta.get("n_bits") or 512)
            hidden = int(meta.get("hidden") or 512)
            if not template_ids:
                return
            model = ChemicalTemplatePreselectorModel(n_bits, len(template_ids), hidden=hidden, dropout=0.0)
            state = torch.load(model_path, map_location=self.device)
            model.load_state_dict(state)
            model.to(self.device)
            model.eval()
        except Exception:
            return
        self.n_bits = n_bits
        self.template_ids = template_ids
        self.template_to_index = {tid: idx for idx, tid in enumerate(template_ids)}
        self.model = model
        self.available = True


def preselector_enabled() -> bool:
    raw = os.environ.get("AUTOPLANNER_ENABLE_CHEM_TEMPLATE_PRESELECTOR")
    if raw is None:
        return DEFAULT_PRESELECTOR_DIR.joinpath("chemical_template_preselector.pt").exists()
    return str(raw).lower() in {"1", "true", "yes", "on"}
