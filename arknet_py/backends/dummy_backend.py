# arknet_py/backends/dummy_backend.py
# reference no-deps backend for tests & plumbing
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterator, Optional

# Public runner contract (used by runner / artifact entrypoints):
# - load_model(artifact_dir) -> model_obj
# - generate(model, prompt:str, params:dict) -> str|dict
# - stream_generate(model, prompt:str, params:dict) -> Iterator[str|dict]
# - format_chat_prompt(messages:list[dict], params:dict|None) -> str
#
# Training shim for trainer.py:
# - class DummyBackend with:
#     is_supported(model) -> True
#     weight_commit_hex() -> str
#     export_safetensors(path) -> None
#     train_step(batch, lr, opt) -> {"loss": 0.0}
#     create_optimizer(opt_cfg) -> None

# ----------------------- runner-side dummy model ----------------------------

@dataclass
class _DummyModel:
    artifact_dir: str


def load_model(artifact_dir: str) -> _DummyModel:
    """Minimal loader: returns a lightweight object; fully deterministic."""
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
    Deterministic echo useful for CLI/tests. Mirrors shapes of real backends:
    - honors num_predict/max_tokens
    - trims on first stop sequence if provided
    """
    n = _param_int(params, "num_predict", _param_int(params, "max_tokens", 128))
    out = f"GEN::{prompt}::n={n}"
    return _truncate_at_stop(out, params.get("stop"))


def stream_generate(model: _DummyModel, prompt: str, params: Dict[str, Any]) -> Iterator[str]:
    """
    Stream a single deterministic chunk (kept tiny to exercise streaming path).
    """
    n = _param_int(params, "num_predict", _param_int(params, "max_tokens", 128))
    text = f"STREAM::{prompt}::n={n}"
    text = _truncate_at_stop(text, params.get("stop"))
    yield text


def format_chat_prompt(messages: list[dict], params: Dict[str, Any] | None = None) -> str:
    """
    Simple, deterministic prompt builder for chat; mirrors runner’s fallback.
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


# ----------------------- trainer-side dummy backend -------------------------


class _DummyBackend:
    """
    Deterministic placeholder backend:
      - weight_commit_hex(): stable fingerprint not tied to memory addresses or temp paths.
      - export_safetensors(): tiny deterministic payload.
      - train_step(): no-op; returns fixed loss 0.0.
    """
    def __init__(self, model: Any):
        self.model = model

    @staticmethod
    def is_supported(model: Any) -> bool:
        return True

    def weight_commit_hex(self) -> str:
        # Stable across runs: avoid repr(self.model) or anything with paths / ids
        cls = self.model.__class__
        snap = {
            "class": getattr(cls, "__name__", "Unknown"),
            "module": getattr(cls, "__module__", "builtins"),
        }
        return sha256_domain_hex(
            b"ARK/DUMMY/WEIGHTS/v1\n",
            dumps_canon(snap).encode("utf-8"),
        )

    def export_safetensors(self, out_path: str) -> None:
        payload = dumps_canon({
            "dummy": True,
            "model_class": type(self.model).__name__,
        }).encode("utf-8")
        atomic_write(out_path, payload)

    def train_step(self, batch: Any, lr: float, opt: Any = None) -> Dict[str, Any]:
        return {"loss": 0.0}

    def create_optimizer(self, opt_cfg: Dict[str, Any]) -> Any:
        return None