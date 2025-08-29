# dummy_backend.py — reference no-deps backend for tests & plumbing
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterator, Optional

# Runner contract support:
# - load_model(artifact_dir) -> model_obj
# - generate(model, prompt:str, params:dict) -> str|dict
# - stream_generate(model, prompt:str, params:dict) -> Iterator[str|dict]
# Optional:
# - format_chat_prompt(messages:list[dict], params:dict|None) -> str


@dataclass
class _DummyModel:
    artifact_dir: str


def load_model(artifact_dir: str) -> _DummyModel:
    """
    Minimal loader: returns a lightweight object to satisfy the runner's
    load_model() contract. We don't read the artifact; deterministic by design.
    """
    return _DummyModel(artifact_dir=artifact_dir)


def _param_int(params: Dict[str, Any], key: str, default: int) -> int:
    try:
        v = params.get(key, default)
        return int(v)
    except Exception:
        return default


def _truncate_at_stop(text: str, stops: Optional[list[str]]) -> str:
    if not stops:
        return text
    idx = len(text)
    for s in stops:
        if not s:
            continue
        j = text.find(s)
        if j != -1 and j < idx:
            idx = j
    return text[:idx]


def generate(model: _DummyModel, prompt: str, params: Dict[str, Any]) -> str:
    """
    Deterministic echo useful for CLI/tests. Mirrors the shapes of real backends:
    - honors num_predict/max_tokens
    - trims on first stop sequence if provided
    """
    n = _param_int(params, "num_predict", _param_int(params, "max_tokens", 128))
    out = f"GEN::{prompt}::n={n}"
    return _truncate_at_stop(out, params.get("stop"))


def stream_generate(model: _DummyModel, prompt: str, params: Dict[str, Any]) -> Iterator[str]:
    """
    Stream a single deterministic chunk (kept tiny to exercise the runner’s
    streaming path). A real backend would yield sub-token or token chunks.
    """
    n = _param_int(params, "num_predict", _param_int(params, "max_tokens", 128))
    text = f"STREAM::{prompt}::n={n}"
    # respect stops (trim before yielding)
    text = _truncate_at_stop(text, params.get("stop"))
    yield text


def format_chat_prompt(messages: list[dict], params: Dict[str, Any] | None = None) -> str:
    """
    Simple, deterministic prompt builder for chat; mirrors runner’s fallback.
    Allows artifacts to rely entirely on backend formatting if they want.
    """
    parts: list[str] = []
    system = None
    if params and params.get("system"):
        system = str(params["system"]).strip()
    # collect explicit system messages
    for m in messages:
        if (m.get("role") or "").lower() == "system":
            s = (m.get("content") or "").strip()
            if s:
                system = f"{system}\n{s}" if system else s
    if system:
        parts.append(f"System: {system}")
    for m in messages:
        role = (m.get("role") or "user").lower()
        content = (m.get("content") or "").strip()
        if role == "assistant":
            parts.append(f"Assistant: {content}")
        elif role != "system":
            parts.append(f"User: {content}")
    parts.append("Assistant:")
    return "\n".join(parts)
