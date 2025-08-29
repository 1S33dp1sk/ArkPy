# torch_backend.py — reference wrapper for HF/torch causal LMs (deterministic-friendly)
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterator, Optional

# Lazy imports to avoid pulling heavy deps unless used
try:
    import torch  # type: ignore
    from transformers import (  # type: ignore
        AutoModelForCausalLM,
        AutoTokenizer,
        TextIteratorStreamer,
        GenerationConfig,
    )
except Exception:  # pragma: no cover
    torch = None  # type: ignore


@dataclass
class TorchTextGen:
    model: Any
    tokenizer: Any
    device: str
    eos_token_id: Optional[int]
    pad_token_id: Optional[int]


# -------------------------- helpers ----------------------------------------


def _raise_if_missing() -> None:
    if torch is None:
        raise RuntimeError(
            "torch/transformers not available. "
            "Install: pip install torch transformers --extra-index-url https://download.pytorch.org/whl/cu121"
        )


def _pick_device(manifest: Dict[str, Any]) -> str:
    want = str(manifest.get("device", "") or "").lower()
    if want in ("cuda", "gpu") and torch.cuda.is_available():  # type: ignore[attr-defined]
        return "cuda"
    if want in ("cpu",):
        return "cpu"
    # auto
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
        import random, numpy as np  # type: ignore
        random.seed(seed)
        np.random.seed(seed)
    except Exception:
        pass
    try:
        torch.manual_seed(seed)          # type: ignore[attr-defined]
        torch.cuda.manual_seed_all(seed) # type: ignore[attr-defined]
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
    """
    Normalize Ollama-style params to transformers.generate config.
    """
    p = dict(params or {})
    # synonyms
    if "max_tokens" in p and p.get("max_tokens") is not None:
        p["num_predict"] = int(p["max_tokens"])
    num_new = int(p.get("num_predict", 256))

    out = dict(
        max_new_tokens=num_new,
        do_sample=float(p.get("temperature", 0)) > 0.0,
        temperature=float(p.get("temperature", 0.7)),
        top_p=float(p.get("top_p", 0.95)),
        top_k=int(p.get("top_k", 40)),
        repetition_penalty=float(p.get("repeat_penalty", 1.0)),
    )
    # penalties (presence/frequency) aren’t native; ignoring by default
    return out


# -------------------------- loader -----------------------------------------


def load_model(artifact_dir: str) -> TorchTextGen:
    """
    Load a HF causal LM from either:
      - manifest["hf_model"] (model id or local path), or
      - artifact_dir (treat artifact as a local HF repo).
    Optional manifest keys:
      - device: "auto"|"cuda"|"cpu" (default auto)
      - dtype:  "auto"|"float16"|"bfloat16"|"float32" (default auto)
      - trust_remote_code: bool
    """
    _raise_if_missing()
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

    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id
    return TorchTextGen(model=model, tokenizer=tokenizer, device=device, eos_token_id=eos_id, pad_token_id=pad_id)


# -------------------------- generation -------------------------------------


def generate(m: TorchTextGen, prompt: str, params: Dict[str, Any]) -> str:
    """
    Non-streaming text generation. Applies seed (if provided), maps common
    params, and trims on stop strings client-side.
    """
    _raise_if_missing()
    _seed_all(params.get("seed"))

    gen_kwargs = _map_params(params)
    toks = m.tokenizer(prompt, return_tensors="pt")
    input_ids = toks.input_ids.to(m.model.device)  # type: ignore[attr-defined]

    # Prepare config
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
        out_ids = m.model.generate(
            input_ids=input_ids,
            generation_config=gc,
        )

    # Take only the completion tail
    gen_ids = out_ids[0, input_ids.shape[-1] :]
    text = m.tokenizer.decode(gen_ids, skip_special_tokens=True)
    text = _truncate_at_stop(text, params.get("stop"))
    return text


def stream_generate(m: TorchTextGen, prompt: str, params: Dict[str, Any]) -> Iterator[str]:
    """
    Streaming generation using TextIteratorStreamer. Yields fully-formed text
    chunks. Caller (runner) will wrap into GenChunk.
    """
    _raise_if_missing()
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

    # Run in the current thread (caller controls threading if desired)
    with torch.no_grad():  # type: ignore[attr-defined]
        gen_out = m.model.generate(
            input_ids=input_ids,
            generation_config=gc,
            streamer=streamer,
        )

    # The streamer yields strings incrementally
    buf = ""
    stops = params.get("stop")
    for piece in streamer:
        if not piece:
            continue
        buf += piece
        if stops:
            trimmed = _truncate_at_stop(buf, stops)
            if len(trimmed) < len(buf):
                # We hit a stop; yield the final trimmed delta and end.
                delta = trimmed  # yield whole trimmed buffer as one final chunk
                if delta:
                    yield delta
                return
        yield piece

    # normal end; if a stop lands exactly on EOS the loop above has already yielded everything
