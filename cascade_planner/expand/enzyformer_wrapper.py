"""Enzyformer wrapper for enzymatic retrosynthesis.

Wraps the pretrained Enzyformer model (Chemformer-based, EC-conditioned)
for use as the enzymatic single-step model in CascadeBoard planning.

Enzyformer uses a two-stage pretrained BART encoder-decoder with R-SMILES
representation. The EC number is prepended as a conditioning token to the
product SMILES input.

Paper: https://link.springer.com/article/10.1186/s13321-026-01164-y
Repo:  https://github.com/Tiantao2000/Enzyformer

COMPATIBILITY NOTES:
    - Enzyformer was developed with Python 3.7, PyTorch 1.9, PL 1.x
    - This project uses Python 3.11, PyTorch 2.3, PL 2.6
    - We handle loading via raw state_dict + manual model construction
      to avoid PL version incompatibilities.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

import torch
from rdkit import Chem

logger = logging.getLogger(__name__)

# Default paths (relative to project root)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_CKPT = _PROJECT_ROOT / "data_external" / "enzyformer" / "checkpoints" / "enzyformer_retro.ckpt"
_ENZYFORMER_REPO = _PROJECT_ROOT / "data_external" / "enzyformer"


def _canonicalize(smi: str) -> str:
    """Canonicalize SMILES, return empty string on failure."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return ""
    return Chem.MolToSmiles(mol)


def _add_enzyformer_to_path():
    """Add Enzyformer repo to sys.path if not already importable."""
    repo_str = str(_ENZYFORMER_REPO)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)
    # Also add Chemformer if bundled
    for subdir in ["Chemformer", "chemformer"]:
        p = _ENZYFORMER_REPO / subdir
        if p.is_dir() and str(p) not in sys.path:
            sys.path.insert(0, str(p))
    # External Chemformer clone
    ext_chemformer = _PROJECT_ROOT / "data_external" / "chemformer"
    if ext_chemformer.is_dir() and str(ext_chemformer) not in sys.path:
        sys.path.insert(0, str(ext_chemformer))


class EnzyformerWrapper:
    """Lazy-loading wrapper around the Enzyformer pretrained model.

    Usage:
        wrapper = EnzyformerWrapper()
        results = wrapper.predict("CC(=O)OC1=CC=CC=C1C(=O)O", ec_token="3.1.1", top_k=5)
    """

    def __init__(self, checkpoint_path: Optional[str] = None, device: Optional[str] = None):
        """Initialize wrapper (model loaded lazily on first predict call).

        Args:
            checkpoint_path: Path to .ckpt file. Defaults to
                data_external/enzyformer/checkpoints/enzyformer_retro.ckpt
            device: 'cuda' or 'cpu'. Auto-detected if None.
        """
        self._ckpt_path = Path(checkpoint_path) if checkpoint_path else _DEFAULT_CKPT
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model = None
        self._tokenizer = None
        self._loaded = False
        self._available = None  # None = not checked yet

    @property
    def available(self) -> bool:
        """Check if Enzyformer can be loaded (checkpoint exists, deps OK)."""
        if self._available is not None:
            return self._available
        if not self._ckpt_path.exists():
            logger.warning(
                "Enzyformer checkpoint not found at %s. "
                "Run: bash scripts/setup_enzyformer.sh",
                self._ckpt_path,
            )
            self._available = False
            return False
        # Try importing required modules
        try:
            _add_enzyformer_to_path()
            self._available = True
        except Exception as e:
            logger.warning("Enzyformer dependencies not available: %s", e)
            self._available = False
        return self._available

    def _load_model(self):
        """Load the Enzyformer model from checkpoint.

        Strategy:
        1. Try our own fine-tuned EnzyformerBART (finetune_enzyformer.py)
        2. Fall back to native Chemformer loading
        3. Fall back to HF BART reconstruction (last resort)
        """
        if self._loaded:
            return
        if not self.available:
            raise RuntimeError("Enzyformer not available. Run: bash scripts/setup_enzyformer.sh")

        _add_enzyformer_to_path()

        # --- Strategy 0: Use our own fine-tuned EnzyformerBART ---
        # This is the preferred path for results/shared/enzyformer_retro_*.pt
        try:
            self._load_via_finetune()
            self._loaded = True
            logger.info("Enzyformer loaded via finetune_enzyformer.EnzyformerBART.")
            return
        except Exception as e:
            logger.debug("finetune_enzyformer load failed: %s", e)

        # --- Strategy 1: Try native Enzyformer/Chemformer loading ---
        try:
            self._load_via_enzyformer_native()
            self._loaded = True
            logger.info("Enzyformer loaded via native Enzyformer/Chemformer code.")
            return
        except Exception as e:
            logger.debug("Native Enzyformer load failed: %s", e)

        # --- Strategy 2: Try Chemformer's BARTModel directly ---
        try:
            self._load_via_chemformer()
            self._loaded = True
            logger.info("Enzyformer loaded via Chemformer BARTModel.")
            return
        except Exception as e:
            logger.debug("Chemformer BARTModel load failed: %s", e)

        # --- Strategy 3: Minimal standalone BART (last resort) ---
        try:
            self._load_standalone()
            self._loaded = True
            logger.info("Enzyformer loaded via standalone BART reconstruction.")
            return
        except Exception as e:
            logger.error("All Enzyformer loading strategies failed: %s", e)
            raise RuntimeError(
                f"Cannot load Enzyformer checkpoint. Last error: {e}\n"
                "Ensure the checkpoint is valid and dependencies are installed."
            )

    def _load_via_finetune(self):
        """Load using our own fine-tuned EnzyformerBART from finetune_enzyformer.py."""
        from cascade_planner.expand.finetune_enzyformer import (
            EnzyformerBART, tokenize, detokenize, VOCAB_SIZE, TOKEN2ID, ID2TOKEN,
            BOS, EOS, PAD,
        )

        model = EnzyformerBART(vocab_size=VOCAB_SIZE).to(self._device)
        sd = torch.load(str(self._ckpt_path), map_location=self._device, weights_only=False)

        # Handle key remapping (older checkpoints use fc/pos instead of token_fc/pos_emb)
        if "fc.weight" in sd and "token_fc.weight" not in sd:
            remap = {"fc.weight": "token_fc.weight", "fc.bias": "token_fc.bias", "pos": "pos_emb"}
            sd = {remap.get(k, k): v for k, v in sd.items()}

        model.load_state_dict(sd, strict=True)
        model.eval()
        self._model = model
        self._tokenizer = _FinetuneTokenizer(tokenize, detokenize)

    def _load_via_enzyformer_native(self):
        """Try loading using Enzyformer's own code."""
        # Enzyformer typically defines a model class that inherits from Chemformer's BARTModel
        # The exact import path depends on the repo structure
        try:
            from model import EnzyformerModel  # type: ignore
            model = EnzyformerModel.load_from_checkpoint(str(self._ckpt_path))
        except ImportError:
            # Try alternative import paths
            from molbart.models import BARTModel  # type: ignore
            model = BARTModel.load_from_checkpoint(str(self._ckpt_path))

        model.eval()
        model.to(self._device)
        self._model = model
        self._tokenizer = getattr(model, "tokenizer", None) or self._build_tokenizer()

    def _load_via_chemformer(self):
        """Try loading via MolecularAI Chemformer's BARTModel."""
        from molbart.models import BARTModel  # type: ignore

        model = BARTModel.load_from_checkpoint(
            str(self._ckpt_path),
            map_location=self._device,
        )
        model.eval()
        model.to(self._device)
        self._model = model
        self._tokenizer = getattr(model, "tokenizer", None) or self._build_tokenizer()

    def _load_standalone(self):
        """Load checkpoint as raw state_dict and reconstruct minimal BART.

        This is the most compatible approach when Chemformer/Enzyformer code
        has version conflicts with our environment.
        """
        # Load checkpoint — handle PL checkpoint format
        ckpt = torch.load(str(self._ckpt_path), map_location=self._device, weights_only=False)

        # PL checkpoints store state_dict under 'state_dict' key
        if "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        else:
            state_dict = ckpt

        # Extract hyperparameters if available
        hparams = ckpt.get("hyper_parameters", ckpt.get("hparams", {}))

        # Build a HuggingFace BART model with matching config
        # Chemformer/Enzyformer uses a custom BART; we approximate with HF BART
        from transformers import BartForConditionalGeneration, BartConfig

        # Infer model dimensions from state_dict keys
        d_model = self._infer_dim(state_dict, "d_model", default=256)
        n_heads = hparams.get("n_heads", hparams.get("num_attention_heads", 8))
        n_layers = hparams.get("n_layers", hparams.get("num_decoder_layers", 6))
        vocab_size = self._infer_vocab_size(state_dict, default=530)

        config = BartConfig(
            vocab_size=vocab_size,
            d_model=d_model,
            encoder_layers=n_layers,
            decoder_layers=n_layers,
            encoder_attention_heads=n_heads,
            decoder_attention_heads=n_heads,
            encoder_ffn_dim=d_model * 4,
            decoder_ffn_dim=d_model * 4,
            max_position_embeddings=512,
        )

        model = BartForConditionalGeneration(config)

        # Try to load state_dict (may have key mismatches — load what we can)
        try:
            # Remap Chemformer keys to HF BART keys
            remapped = self._remap_state_dict(state_dict, model)
            model.load_state_dict(remapped, strict=False)
            logger.info("Loaded state_dict with partial key matching.")
        except Exception as e:
            logger.warning("State dict remapping failed: %s. Model will use random weights.", e)
            # TODO: Implement full key remapping for Chemformer -> HF BART
            # For now, this path means the model won't produce meaningful results
            # until proper key mapping is implemented.

        model.eval()
        model.to(self._device)
        self._model = model
        self._tokenizer = self._build_tokenizer()

    def _infer_dim(self, state_dict: dict, name: str, default: int) -> int:
        """Infer model dimension from state_dict tensor shapes."""
        for key, tensor in state_dict.items():
            # Match both 'embed' (HF) and 'emb' (Chemformer) keys
            if ("embed" in key or key == "emb.weight") and tensor.dim() == 2:
                return tensor.shape[1]
        # Fallback: check encoder self_attn out_proj
        for key, tensor in state_dict.items():
            if "self_attn.out_proj.weight" in key and tensor.dim() == 2:
                return tensor.shape[0]
        return default

    def _infer_vocab_size(self, state_dict: dict, default: int) -> int:
        """Infer vocab size from embedding layer."""
        for key, tensor in state_dict.items():
            if "embed_tokens" in key and tensor.dim() == 2:
                return tensor.shape[0]
            if "shared" in key and "embed" in key and tensor.dim() == 2:
                return tensor.shape[0]
            # Chemformer uses 'emb.weight'
            if key == "emb.weight" and tensor.dim() == 2:
                return tensor.shape[0]
        return default

    def _remap_state_dict(self, state_dict: dict, model) -> dict:
        """Attempt to remap Chemformer state_dict keys to HF BART format.

        TODO: This needs refinement based on actual Enzyformer checkpoint structure.
        The key mapping between Chemformer's custom BART and HuggingFace BART
        differs in prefix naming conventions.
        """
        remapped = {}
        model_keys = set(model.state_dict().keys())

        for key, value in state_dict.items():
            # Strip common PL prefixes
            new_key = key
            for prefix in ["model.", "bart.", "encoder_decoder."]:
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix):]
                    break

            if new_key in model_keys:
                remapped[new_key] = value
            else:
                # Try with 'model.' prefix (HF BART convention)
                hf_key = f"model.{new_key}"
                if hf_key in model_keys:
                    remapped[hf_key] = value

        return remapped

    def _build_tokenizer(self):
        """Build a SMILES tokenizer compatible with Chemformer/Enzyformer.

        Chemformer uses a regex-based SMILES tokenizer. We replicate the
        tokenization pattern here for standalone operation.
        """
        # Always use our regex tokenizer — pysmilesutils requires vocabulary
        # initialization from training data which we don't have available.
        return _RegexSmilesTokenizer()

    def predict(
        self,
        product_smiles: str,
        ec_token: str = "",
        top_k: int = 10,
        beam_size: int = 10,
    ) -> list[dict]:
        """Predict reactants for a given product SMILES.

        Args:
            product_smiles: Product SMILES string.
            ec_token: EC number (e.g. "3.1.1.1") to condition the prediction.
                     If empty, runs unconditional retrosynthesis.
            top_k: Number of top results to return.
            beam_size: Beam search width (>= top_k for best results).

        Returns:
            List of dicts with keys:
                - main_reactant: str (canonical SMILES of primary reactant)
                - aux_reactants: list[str] (other reactants)
                - rxn_smiles: str (reactants>>product)
                - score: float (beam score, higher = more confident)
                - ec: str (EC number used for conditioning)
                - source: "enzyformer"
        """
        try:
            self._load_model()
        except RuntimeError as e:
            logger.error("Enzyformer load failed: %s", e)
            return []

        # Validate input
        mol = Chem.MolFromSmiles(product_smiles)
        if mol is None:
            logger.debug("Invalid product SMILES: %s", product_smiles)
            return []

        canon_product = Chem.MolToSmiles(mol)

        # Build input: EC-conditioned R-SMILES format
        # Enzyformer format: "[EC:x.x.x.x] product_smiles" or just "product_smiles"
        if ec_token:
            input_seq = f"[EC:{ec_token}] {canon_product}"
        else:
            input_seq = canon_product

        # Generate predictions via beam search
        beam_size = max(beam_size, top_k)
        try:
            raw_outputs = self._generate_beam(input_seq, beam_size=beam_size)
        except Exception as e:
            logger.warning("Enzyformer generation failed for %s: %s", product_smiles, e)
            return []

        # Parse and deduplicate results
        results = []
        seen = set()
        for score, output_smi in raw_outputs:
            # Clean up output SMILES
            output_smi = output_smi.strip()
            if not output_smi:
                continue

            # Validate output
            reactant_parts = output_smi.split(".")
            valid_parts = []
            for part in reactant_parts:
                part = part.strip()
                if not part:
                    continue
                canon = _canonicalize(part)
                if canon:
                    valid_parts.append(canon)

            if not valid_parts:
                continue

            # Deduplicate by canonical reactant set
            key = frozenset(valid_parts)
            if key in seen:
                continue
            seen.add(key)

            main_reactant = valid_parts[0]
            aux_reactants = valid_parts[1:]
            rxn_smiles = ".".join(valid_parts) + ">>" + canon_product

            results.append({
                "main_reactant": main_reactant,
                "aux_reactants": aux_reactants,
                "rxn_smiles": rxn_smiles,
                "score": float(score),
                "ec": ec_token,
                "source": "enzyformer",
            })

            if len(results) >= top_k:
                break

        return results

    def _generate_beam(self, input_seq: str, beam_size: int = 10) -> list[tuple[float, str]]:
        """Run beam search generation on the loaded model.

        Returns list of (score, output_smiles) sorted by descending score.
        """
        # --- Path A: Our fine-tuned EnzyformerBART with beam_search method ---
        if hasattr(self._model, "beam_search"):
            return self._generate_finetune(input_seq, beam_size)

        # --- Path B: Native Chemformer/Enzyformer model with .predict() ---
        if hasattr(self._model, "predict") or hasattr(self._model, "sample_molecules"):
            return self._generate_native(input_seq, beam_size)

        # --- Path C: HuggingFace BartForConditionalGeneration ---
        if hasattr(self._model, "generate"):
            return self._generate_hf(input_seq, beam_size)

        logger.error("Model has no recognized generation interface.")
        return []

    def _generate_finetune(self, input_seq: str, beam_size: int) -> list[tuple[float, str]]:
        """Generate using our fine-tuned EnzyformerBART beam_search."""
        tokenizer = self._tokenizer
        model = self._model

        src_ids = torch.tensor(
            tokenizer.encode(input_seq), dtype=torch.long, device=self._device
        ).unsqueeze(1)  # (seq_len, 1) for batch_first=False

        results = model.beam_search(src_ids, num_beams=beam_size, max_len=200)
        return [(score, tokenizer.decode(seq)) for score, seq in results]

    def _generate_native(self, input_seq: str, beam_size: int) -> list[tuple[float, str]]:
        """Generate using Chemformer/Enzyformer native API."""
        model = self._model

        # Chemformer typically uses model.predict() or model.sample_molecules()
        if hasattr(model, "predict"):
            # predict(smiles_list, num_beams=N) -> list of predictions
            try:
                preds = model.predict([input_seq], num_beams=beam_size)
                # preds format varies: list[list[str]] or list[dict]
                if isinstance(preds, list) and preds:
                    if isinstance(preds[0], list):
                        # list of beam outputs for each input
                        outputs = preds[0]
                        return [(1.0 / (i + 1), s) for i, s in enumerate(outputs)]
                    elif isinstance(preds[0], dict):
                        return [(p.get("score", 1.0 / (i + 1)), p.get("smiles", ""))
                                for i, p in enumerate(preds)]
                    elif isinstance(preds[0], str):
                        return [(1.0 / (i + 1), s) for i, s in enumerate(preds)]
            except Exception as e:
                logger.debug("model.predict() failed: %s", e)

        if hasattr(model, "sample_molecules"):
            try:
                samples = model.sample_molecules([input_seq], num_beams=beam_size)
                if isinstance(samples, list):
                    return [(1.0 / (i + 1), s) for i, s in enumerate(samples)]
            except Exception as e:
                logger.debug("model.sample_molecules() failed: %s", e)

        # Try forward pass with beam search manually
        raise RuntimeError("Native model API not compatible")

    def _generate_hf(self, input_seq: str, beam_size: int) -> list[tuple[float, str]]:
        """Generate using HuggingFace BART generate() method."""
        tokenizer = self._tokenizer
        model = self._model

        # Tokenize input
        input_ids = tokenizer.encode(input_seq)
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=self._device)

        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_tensor,
                max_length=256,
                num_beams=beam_size,
                num_return_sequences=beam_size,
                early_stopping=True,
                return_dict_in_generate=True,
                output_scores=True,
            )

        # Decode outputs
        sequences = outputs.sequences
        # Compute sequence scores
        if hasattr(outputs, "sequences_scores") and outputs.sequences_scores is not None:
            scores = outputs.sequences_scores.cpu().tolist()
        else:
            scores = [1.0 / (i + 1) for i in range(len(sequences))]

        results = []
        for i, seq in enumerate(sequences):
            decoded = tokenizer.decode(seq.cpu().tolist())
            results.append((scores[i], decoded))

        # Sort by score descending
        results.sort(key=lambda x: x[0], reverse=True)
        return results


class _FinetuneTokenizer:
    """Tokenizer adapter for our fine-tuned EnzyformerBART."""

    def __init__(self, encode_fn, decode_fn):
        self._encode = encode_fn
        self._decode = decode_fn

    def encode(self, smi: str) -> list[int]:
        return self._encode(smi)

    def decode(self, ids: list[int]) -> str:
        return self._decode(ids)


class _RegexSmilesTokenizer:
    """Minimal regex-based SMILES tokenizer matching Chemformer's convention.

    Chemformer tokenizes SMILES character-by-character with special handling
    for multi-char tokens like Br, Cl, [atoms], and ring numbers.
    """

    import re
    _PATTERN = re.compile(
        r"(\[[^\]]+\]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>|<|\*|\$|\%[0-9]{2}|[0-9])"
    )

    # Special tokens
    PAD = "<pad>"
    BOS = "<s>"
    EOS = "</s>"
    UNK = "<unk>"

    def __init__(self):
        # Build vocab from common SMILES tokens
        special = [self.PAD, self.BOS, self.EOS, self.UNK]
        # Common SMILES characters
        atoms = ["B", "C", "N", "O", "P", "S", "F", "I", "Br", "Cl",
                 "b", "c", "n", "o", "p", "s"]
        bonds = ["=", "#", "-", "+", "\\", "/", ":", "~", ".", "@", "@@"]
        brackets = ["(", ")", "[", "]", "<", ">"]
        digits = [str(i) for i in range(10)]
        misc = ["*", "$", "?", " "]
        # EC tokens
        ec_tokens = [f"[EC:{i}]" for i in range(8)]  # EC class 1-7

        all_tokens = special + atoms + bonds + brackets + digits + misc + ec_tokens
        # Add bracket atom patterns
        for a in atoms:
            all_tokens.append(f"[{a}]")
            all_tokens.append(f"[{a}H]")
            all_tokens.append(f"[{a}H2]")
            all_tokens.append(f"[{a}-]")
            all_tokens.append(f"[{a}+]")

        # Deduplicate preserving order
        seen = set()
        self._vocab = []
        for t in all_tokens:
            if t not in seen:
                self._vocab.append(t)
                seen.add(t)

        self._token2id = {t: i for i, t in enumerate(self._vocab)}
        self._id2token = {i: t for i, t in enumerate(self._vocab)}
        self._unk_id = self._token2id[self.UNK]
        self._bos_id = self._token2id[self.BOS]
        self._eos_id = self._token2id[self.EOS]
        self._pad_id = self._token2id[self.PAD]

    def encode(self, smiles: str) -> list[int]:
        """Tokenize and encode SMILES to token IDs."""
        tokens = [self.BOS] + self._tokenize(smiles) + [self.EOS]
        return [self._token2id.get(t, self._unk_id) for t in tokens]

    def decode(self, ids: list[int]) -> str:
        """Decode token IDs back to SMILES string."""
        tokens = []
        for i in ids:
            t = self._id2token.get(i, "")
            if t in (self.PAD, self.BOS, self.EOS, self.UNK):
                continue
            tokens.append(t)
        return "".join(tokens)

    def _tokenize(self, smiles: str) -> list[str]:
        """Split SMILES into tokens."""
        # Handle EC prefix
        if smiles.startswith("[EC:"):
            # Extract EC token
            end = smiles.index("]") + 1
            ec_tok = smiles[:end]
            rest = smiles[end:].strip()
            return [ec_tok] + self._PATTERN.findall(rest)
        return self._PATTERN.findall(smiles)


class _PySmilesTokenizer:
    """Tokenizer using pysmilesutils (if available)."""

    def __init__(self):
        from pysmilesutils.tokenize import SMILESTokenizer  # type: ignore
        self._tok = SMILESTokenizer()

    def encode(self, smiles: str) -> list[int]:
        return self._tok.encode(smiles)

    def decode(self, ids: list[int]) -> str:
        return self._tok.decode(ids)
