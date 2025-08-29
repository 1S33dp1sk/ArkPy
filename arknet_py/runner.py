"""
arknet_py.runner — lightweight Ollama-style runner facade.

Artifact contract (directory):
- MUST contain:   manifest.json, model.py
- model.py SHOULD expose one of:
    * stream_generate(model, prompt:str, params:dict) -> Iterator[str|dict]
    * generate(model, prompt:str, params:dict) -> str|dict
    * load_model(artifact_dir) -> object with .infer(prompt:str, **params)->str   (fallback)
- Optionally for chat:
    * format_chat_prompt(messages:list[dict], params:dict|None) -> str
      (if not present, we synthesize a simple chat prompt from messages)

Backends may return dicts with richer info; we always surface .text (or "response")
when possible.

This module provides:
- ArknetModel (generate/chat/stream facade)
- GenChunk (streaming chunk)
- run_once(artifact_dir, prompt, **kwargs) convenience
"""

from __future__ import annotations

import importlib.util
import json
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Union


# ---- defaults (mirrors common Ollama knobs) -----------------------------------

_DEFAULT_PARAMS: Dict[str, Any] = {
    "temperature": 0.7,         # 0..~2
    "top_p": 0.95,              # nucleus sampling
    "top_k": 40,                # k-sampling
    "repeat_penalty": 1.1,      # >= 1.0
    "repeat_last_n": 64,        # tokens
    "presence_penalty": 0.0,    # >= 0
    "frequency_penalty": 0.0,   # >= 0
    "num_predict": 256,         # aka max_tokens
    "stop": [],                 # list[str]
    "seed": None,               # int|None (deterministic inference)
    # Arknet additions:
    "system": None,             # optional system prompt
    "template": None,           # override manifest/template default
}


@dataclass
class GenChunk:
    """
    Streaming chunk (Ollama-esque).
    - text: accumulated new text for this chunk
    - done: True on final chunk
    - stop_reason: "stop"/"length"/"eos"/"user"/None
    - tokens: optional running token counts (backend-defined)
    """
    text: str
    done: bool = False
    stop_reason: Optional[str] = None
    tokens: Optional[Dict[str, int]] = None


# ---- utilities ----------------------------------------------------------------


def _safe_load_json(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}
    except Exception as e:
        raise RuntimeError(f"manifest load failed: {e}") from e


def _normalize_params(user: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    p = dict(_DEFAULT_PARAMS)
    if user:
        p.update({k: v for k, v in user.items() if v is not None})

    # synonyms: max_tokens -> num_predict
    if "max_tokens" in p and p.get("max_tokens") is not None:
        p["num_predict"] = int(p["max_tokens"])
    p["num_predict"] = int(p["num_predict"])

    # stop normalization
    stop = p.get("stop")
    if stop is None:
        p["stop"] = []
    elif isinstance(stop, str):
        p["stop"] = [stop]
    elif isinstance(stop, (list, tuple)):
        p["stop"] = [str(s) for s in stop]
    else:
        p["stop"] = []

    # numeric coercions (tolerant)
    for k in ("temperature", "top_p"):
        try:
            p[k] = float(p[k])
        except Exception:
            pass
    for k in ("top_k", "repeat_last_n"):
        try:
            p[k] = int(p[k])
        except Exception:
            pass
    for k in ("repeat_penalty", "presence_penalty", "frequency_penalty"):
        try:
            p[k] = float(p[k])
        except Exception:
            pass

    # seed can be None or int
    if p.get("seed") is not None:
        try:
            p["seed"] = int(p["seed"])
        except Exception:
            p["seed"] = None

    return p


def _apply_seed_local(seed: Optional[int]) -> None:
    """
    Best-effort, inference-only determinism without touching global env.
    (The CLI also offers a stronger process-wide profile.)
    """
    if seed is None:
        return
    random.seed(seed)
    try:
        import numpy as np  # type: ignore
        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch  # type: ignore
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Avoid flipping CuDNN global flags here; leave that to the determinism profile.
    except Exception:
        pass


def _import_module(entry_py: str, module_name: str) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, entry_py)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import: {entry_py}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _synthesize_chat_prompt(messages: List[Dict[str, str]], system: Optional[str], template: Optional[str]) -> str:
    """
    Simple, deterministic chat template:
      - Optional system preface.
      - Alternating "User:" / "Assistant:" lines.
      - Ends with 'Assistant:' cue.
    If template is provided (a Python str.format template), variables:
      {system}, {conversation}
    """
    parts: List[str] = []
    if system:
        parts.append(f"System: {system}".strip())

    for m in messages:
        role = (m.get("role") or "user").lower()
        content = (m.get("content") or "").strip()
        if role == "system":
            if content:
                parts.insert(0, f"System: {content}")
        elif role == "assistant":
            parts.append(f"Assistant: {content}")
        else:
            parts.append(f"User: {content}")

    parts.append("Assistant:")
    conv = "\n".join(parts)

    if template:
        try:
            return template.format(system=(system or ""), conversation=conv)
        except Exception:
            return conv
    return conv


def _extract_system_from_messages(messages: List[Dict[str, str]]) -> Optional[str]:
    sys_msgs = [m.get("content", "") for m in messages if (m.get("role") or "").lower() == "system"]
    joined = "\n".join(s for s in sys_msgs if s)
    return joined if joined.strip() else None


def _kwargs_for_infer(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Pass-through subset for .infer(**kwargs) backends that don't accept a single params dict.
    Keep this lean; many lightweight demos simply ignore extras.
    """
    keys = (
        "temperature", "top_p", "top_k", "num_predict", "stop",
        "repeat_penalty", "repeat_last_n", "presence_penalty", "frequency_penalty",
    )
    return {k: params[k] for k in keys if k in params}


def _coerce_text(x: Any) -> str:
    if isinstance(x, str):
        return x
    if isinstance(x, dict):
        for k in ("text", "response", "output", "completion"):
            v = x.get(k)
            if isinstance(v, str):
                return v
        return json.dumps(x, ensure_ascii=False)
    return str(x)


# ---- main facade --------------------------------------------------------------


class ArknetModel:
    """
    Ollama-like runner for Arknet artifacts.
    - load manifest.json for defaults/capabilities
    - import model.py and prefer stream_generate/generate; fallback to load_model().infer()
    """
    def __init__(self, artifact_dir: str):
        self.artifact_dir = os.path.realpath(artifact_dir)
        self.manifest = _safe_load_json(os.path.join(self.artifact_dir, "manifest.json"))
        self.module = self._load_module()
        self.model = self._maybe_load_model()
        self.template = self._derive_template()

    # --- loading & metadata ------------------------------------------------

    def _load_module(self) -> Any:
        entry = os.path.join(self.artifact_dir, "model.py")
        if not os.path.exists(entry):
            raise FileNotFoundError("model.py not found in artifact directory")
        return _import_module(entry, "arknet_model_entry")

    def _maybe_load_model(self) -> Any:
        # Some backends are stateless module-level functions (no object to load).
        # If load_model is present, call it; otherwise return module itself.
        if hasattr(self.module, "load_model"):
            return self.module.load_model(self.artifact_dir)  # type: ignore[attr-defined]
        return self.module

    def _derive_template(self) -> Optional[str]:
        try:
            t = self.manifest.get("template")
        except Exception:
            t = None
        if isinstance(t, str) and t.strip():
            return t
        return None

    # --- public APIs -------------------------------------------------------

    def generate(
        self,
        prompt: str,
        params: Optional[Dict[str, Any]] = None,
        stream: bool = False
    ) -> Union[str, Iterator[GenChunk]]:
        """
        Single-turn text generation. If stream=True, yields GenChunk.
        """
        p = _normalize_params(params)
        if p.get("system"):
            sys_preface = f"System: {p['system']}\n" if p["system"] else ""
            prompt = f"{sys_preface}{prompt}"
        _apply_seed_local(p.get("seed"))

        if stream:
            return self._stream_call(prompt, p)
        else:
            return self._nonstream_call(prompt, p)

    def chat(
        self,
        messages: List[Dict[str, str]],
        params: Optional[Dict[str, Any]] = None,
        stream: bool = False
    ) -> Union[str, Iterator[GenChunk]]:
        """
        Chat-style API (messages = [{role, content}, ...]).
        Uses manifest.template or module.format_chat_prompt if available.
        """
        p = _normalize_params(params)
        _apply_seed_local(p.get("seed"))

        # If the backend provides its own prompt formatter, use it
        if hasattr(self.module, "format_chat_prompt"):
            try:
                prompt = self.module.format_chat_prompt(messages, p)  # type: ignore[attr-defined]
            except Exception as e:
                raise RuntimeError(f"format_chat_prompt failed: {e}") from e
        else:
            system = p.get("system") or _extract_system_from_messages(messages)
            prompt = _synthesize_chat_prompt(messages, system, p.get("template") or self.template)

        if stream:
            return self._stream_call(prompt, p)
        else:
            return self._nonstream_call(prompt, p)

    # --- backend calls -----------------------------------------------------

    def _nonstream_call(self, prompt: str, params: Dict[str, Any]) -> str:
        """
        Prefer module.generate(model, prompt, params) -> str | dict
        Fallback: model.infer(prompt, **params) -> str
        """
        if hasattr(self.module, "generate"):
            out = self.module.generate(self.model, prompt, params)  # type: ignore[attr-defined]
            return _coerce_text(out)

        if hasattr(self.model, "infer"):
            out = self.model.infer(prompt, **_kwargs_for_infer(params))
            return _coerce_text(out)

        raise RuntimeError("No generate() or .infer() available in model")

    def _stream_call(self, prompt: str, params: Dict[str, Any]) -> Iterator[GenChunk]:
        """
        Prefer module.stream_generate(model, prompt, params) -> iterator
        Fallback: emulate streaming by chunking non-stream output.
        """
        if hasattr(self.module, "stream_generate"):
            try:
                it = self.module.stream_generate(self.model, prompt, params)  # type: ignore[attr-defined]
            except Exception as e:
                raise RuntimeError(f"stream_generate() failed: {e}") from e
            for item in it:
                # item may be str or dict
                if isinstance(item, str):
                    yield GenChunk(text=item, done=False)
                elif isinstance(item, dict):
                    txt = item.get("text") or item.get("response") or ""
                    done = bool(item.get("done", False))
                    stop_reason = item.get("stop_reason")
                    tokens = item.get("tokens")
                    yield GenChunk(text=str(txt), done=done, stop_reason=stop_reason, tokens=tokens)
                else:
                    yield GenChunk(text=str(item), done=False)
            # Ensure a final done if the backend didn't send one
            yield GenChunk(text="", done=True, stop_reason="eos")
            return

        # Fallback: non-stream call split into coarse chunks
        full = self._nonstream_call(prompt, params)
        for i in range(0, len(full), 256):
            chunk = full[i : i + 256]
            yield GenChunk(text=chunk, done=False)
        yield GenChunk(text="", done=True, stop_reason="eos")

    # --- convenience -------------------------------------------------------

    def infer(self, prompt: str, **kwargs) -> str:
        """Back-compat shortcut (non-stream)."""
        return self.generate(prompt, params=kwargs, stream=False)  # type: ignore[return-value]


# ---- one-liner helper ---------------------------------------------------------

def run_once(artifact_dir: str, prompt: str, **kwargs) -> str:
    m = ArknetModel(artifact_dir)
    return m.infer(prompt, **kwargs)
