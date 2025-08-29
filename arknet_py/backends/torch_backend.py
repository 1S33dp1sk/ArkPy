# torch_backend.py — Torch backend for Arknet (training + HF text-gen helpers)
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterator, Optional

# ---- optional heavy deps ---------------------------------------------------
try:
    import torch  # type: ignore
except Exception:  # pragma: no cover
    torch = None  # type: ignore

try:
    from transformers import (  # type: ignore
        AutoModelForCausalLM,
        AutoTokenizer,
        TextIteratorStreamer,
        GenerationConfig,
    )
except Exception:  # pragma: no cover
    AutoModelForCausalLM = None  # type: ignore
    AutoTokenizer = None  # type: ignore
    TextIteratorStreamer = None  # type: ignore
    GenerationConfig = None  # type: ignore

try:
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover
    np = None  # type: ignore


# ---- trainer-facing backend ------------------------------------------------

class TorchBackend:
    """
    Minimal training backend wrapper expected by training/trainer.py.

    Methods:
      - is_supported(model) -> bool
      - weight_commit_hex() -> str
      - export_safetensors(path) -> None
      - train_step(batch, lr, opt) -> dict (loss=...)
      - create_optimizer(opt_cfg) -> torch.optim.Optimizer|None
    """

    def __init__(self, model: Any):
        _ensure_torch()
        if not self.is_supported(model):
            raise TypeError("TorchBackend: model must be a torch.nn.Module")
        self.model = model

    @staticmethod
    def is_supported(model: Any) -> bool:
        return (torch is not None) and hasattr(torch, "nn") and isinstance(model, torch.nn.Module)  # type: ignore[attr-defined]

    # ----- commits / export -------------------------------------------------

    def _state_dict_numpy(self) -> Dict[str, "np.ndarray"]:
        """
        Convert model.state_dict() → {name: np.ndarray} with:
          - CPU, contiguous
          - little-endian
          - safe BF16 handling if NumPy lacks it (upcast to float32)
        """
        if np is None:
            raise RuntimeError("numpy is required to export weights deterministically")

        sd = self.model.state_dict()
        out: Dict[str, "np.ndarray"] = {}
        for k, t in sd.items():
            a = t.detach().to("cpu").contiguous()
            # dtype normalization
            if str(a.dtype) == "torch.bfloat16":
                # If NumPy supports bfloat16, use it; else upcast to fp32.
                try:
                    _ = np.dtype("bfloat16")  # may raise on old NumPy
                    arr = a.numpy().astype(np.dtype("bfloat16"), copy=False)
                except Exception:
                    arr = a.numpy().astype(np.float32, copy=False)
            else:
                arr = a.numpy()

            # Ensure C-contiguous, little-endian
            if not arr.flags.c_contiguous:
                arr = np.ascontiguousarray(arr)
            if arr.dtype.byteorder == ">" or (arr.dtype.byteorder == "=" and not np.little_endian):
                arr = arr.byteswap().newbyteorder("<")
            out[str(k)] = arr
        return out

    def weight_commit_hex(self) -> str:
        # Lazy import to avoid cycles at module import time
        from ..training.exporter import tensors_to_safetensors_bytes
        from ..utils.hashing import sha256_hex

        tensors = self._state_dict_numpy()
        # exporter sorts names & builds canonical bytes
        data = tensors_to_safetensors_bytes(tensors)
        return sha256_hex(data)

    def export_safetensors(self, out_path: str) -> None:
        from ..training.exporter import export_safetensors
        tensors = self._state_dict_numpy()
        export_safetensors(out_path, tensors)

    # ----- training step / optimizer ---------------------------------------

    def create_optimizer(self, opt_cfg: Dict[str, Any]) -> Optional["torch.optim.Optimizer"]:
        try:
            name = str(opt_cfg.get("name", "adamw")).lower()
            lr = float(opt_cfg.get("lr", 1e-3))
            wd = float(opt_cfg.get("weight_decay", 0.0))
        except Exception:
            name, lr, wd = "adamw", 1e-3, 0.0

        if lr == 0.0:
            # "no-op" optimizer: training loop will treat as no updates
            return None

        try:
            if name in ("adamw", "adamw_torch"):
                return torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=wd)  # type: ignore[attr-defined]
            if name in ("sgd",):
                mom = float(opt_cfg.get("momentum", 0.0))
                nesterov = bool(opt_cfg.get("nesterov", False))
                return torch.optim.SGD(self.model.parameters(), lr=lr, momentum=mom, nesterov=nesterov, weight_decay=wd)  # type: ignore[attr-defined]
        except Exception:
            return None
        return None

    def train_step(self, batch: Any, lr: float, opt: Optional["torch.optim.Optimizer"] = None) -> Dict[str, Any]:
        """
        Reference-friendly single step.
        If the model exposes a custom `train_step(batch, lr, opt)`, delegate to it.
        Otherwise, do nothing and return {"loss": 0.0}.
        """
        if hasattr(self.model, "train_step") and callable(getattr(self.model, "train_step")):
            try:
                out = self.model.train_step(batch, lr, opt)  # type: ignore[misc]
                if isinstance(out, dict) and "loss" in out:
                    return {"loss": float(out["loss"])}
            except Exception:
                pass
        return {"loss": 0.0}


# ---- generation helpers (optional; used by artifact runner) ----------------

@dataclass
class TorchTextGen:
    model: Any
    tokenizer: Any
    device: str
    eos_token_id: Optional[int]
    pad_token_id: Optional[int]


def _ensure_torch() -> None:
    if torch is None:
        raise RuntimeError(
            "torch is not available. Install PyTorch (CUDA build if needed)."
        )


def _ensure_hf() -> None:
    if AutoModelForCausalLM is None or AutoTokenizer is None or GenerationConfig is None:
        raise RuntimeError(
            "transformers not available. Install: pip install transformers"
        )


def _pick_device(manifest: Dict[str, Any]) -> str:
    want = str(manifest.get("device", "") or "").lower()
    if want in ("cuda", "gpu") and hasattr(torch, "cuda") and torch.cuda.is_available():  # type: ignore[attr-defined]
        return "cuda"
    if want in ("cpu",):
        return "cpu"
    return "cuda" if (hasattr(torch, "cuda") and torch.cuda.is_available()) else "cpu"  # type: ignore[attr-defined]


def _pick_dtype(manifest: Dict[str, Any]):
    d = str(manifest.get("dtype", "auto")).lower()
    if d in ("auto", ""):
        return None
    if d in ("float16", "fp16"):
        return torch.float16
    if d in ("bfloat16", "bf16"):
        return torch.bfloat16
    if d in ("float32", "fp32"):
        return torch.float32
    return None


def _load_manifest(artifact_dir: str) -> Dict[str, Any]:
    path = os.path.join(artifact_dir, "manifest.json")
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}
    except Exception as e:
        raise RuntimeError(f"manifest read failed: {e}") from e


def _seed_all(seed: Optional[int]) -> None:
    if seed is None:
        return
    try:
        import random
        random.seed(seed)
    except Exception:
        pass
    try:
        if np is not None:
            np.random.seed(seed)  # type: ignore
    except Exception:
        pass
    try:
        torch.manual_seed(seed)          # type: ignore[attr-defined]
        if hasattr(torch, "cuda"):
            torch.cuda.manual_seed_all(seed)  # type: ignore[attr-defined]
        # tighten determinism as much as feasible without crashing
        try:
            torch.use_deterministic_algorithms(True)  # type: ignore[attr-defined]
        except Exception:
            pass
        try:
            torch.backends.cudnn.deterministic = True  # type: ignore[attr-defined]
            torch.backends.cudnn.benchmark = False     # type: ignore[attr-defined]
        except Exception:
            pass
    except Exception:
        pass


def _truncate_at_stop(text: str, stops: Optional[list[str]]) -> str:
    if not stops:
        return text
    cut = len(text)
    for s in stops:
        if not s:
            continue
        j = text.find(s)
        if j != -1 and j < cut:
            cut = j
    return text[:cut]


def _map_params(params: Dict[str, Any]) -> Dict[str, Any]:
    p = dict(params or {})
    if "max_tokens" in p and p.get("max_tokens") is not None:
        p["num_predict"] = int(p["max_tokens"])
    num_new = int(p.get("num_predict", 256))
    return dict(
        max_new_tokens=num_new,
        do_sample=float(p.get("temperature", 0)) > 0.0,
        temperature=float(p.get("temperature", 0.7)),
        top_p=float(p.get("top_p", 0.95)),
        top_k=int(p.get("top_k", 40)),
        repetition_penalty=float(p.get("repeat_penalty", 1.0)),
    )


def load_model(artifact_dir: str) -> TorchTextGen:
    _ensure_torch()
    _ensure_hf()

    manifest = _load_manifest(artifact_dir)
    src = manifest.get("hf_model") or artifact_dir

    device = _pick_device(manifest)
    dtype = _pick_dtype(manifest)
    trust = bool(manifest.get("trust_remote_code", False))

    tokenizer = AutoTokenizer.from_pretrained(src, use_fast=True, trust_remote_code=trust)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        src,
        torch_dtype=dtype,
        trust_remote_code=trust,
        device_map="auto" if device == "cuda" else None,
    )
    if device == "cpu":
        model = model.to("cpu")

    return TorchTextGen(
        model=model,
        tokenizer=tokenizer,
        device=device,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )


def generate(m: TorchTextGen, prompt: str, params: Dict[str, Any]) -> str:
    _ensure_torch()
    _ensure_hf()
    _seed_all(params.get("seed"))

    gen_kwargs = _map_params(params)
    toks = m.tokenizer(prompt, return_tensors="pt")
    input_ids = toks.input_ids.to(m.model.device)  # type: ignore[attr-defined]

    gc = GenerationConfig(
        max_new_tokens=gen_kwargs["max_new_tokens"],
        do_sample=gen_kwargs["do_sample"],
        temperature=gen_kwargs["temperature"],
        top_p=gen_kwargs["top_p"],
        top_k=gen_kwargs["top_k"],
        repetition_penalty=gen_kwargs["repetition_penalty"],
        eos_token_id=m.eos_token_id,
        pad_token_id=m.pad_token_id,
    )

    with torch.no_grad():  # type: ignore[attr-defined]
        out_ids = m.model.generate(input_ids=input_ids, generation_config=gc)

    gen_ids = out_ids[0, input_ids.shape[-1]:]
    text = m.tokenizer.decode(gen_ids, skip_special_tokens=True)
    return _truncate_at_stop(text, params.get("stop"))


def stream_generate(m: TorchTextGen, prompt: str, params: Dict[str, Any]) -> Iterator[str]:
    _ensure_torch()
    _ensure_hf()
    _seed_all(params.get("seed"))

    gen_kwargs = _map_params(params)
    toks = m.tokenizer(prompt, return_tensors="pt")
    input_ids = toks.input_ids.to(m.model.device)  # type: ignore[attr-defined]

    streamer = TextIteratorStreamer(
        m.tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
    )

    gc = GenerationConfig(
        max_new_tokens=gen_kwargs["max_new_tokens"],
        do_sample=gen_kwargs["do_sample"],
        temperature=gen_kwargs["temperature"],
        top_p=gen_kwargs["top_p"],
        top_k=gen_kwargs["top_k"],
        repetition_penalty=gen_kwargs["repetition_penalty"],
        eos_token_id=m.eos_token_id,
        pad_token_id=m.pad_token_id,
    )

    with torch.no_grad():  # type: ignore[attr-defined]
        m.model.generate(
            input_ids=input_ids,
            generation_config=gc,
            streamer=streamer,
        )

    buf = ""
    stops = params.get("stop")
    for piece in streamer:
        if not piece:
            continue
        buf += piece
        if stops:
            trimmed = _truncate_at_stop(buf, stops)
            if len(trimmed) < len(buf):
                if trimmed:
                    yield trimmed
                return
        yield piece
