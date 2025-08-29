# arknet_py/training/dataset.py
from __future__ import annotations
import io, os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

from ..utils.hashing import sha256_hex, domain_hash_hex
from ..utils.tar_canon import build_canonical_tar_bytes
from ..utils.json_canon import dumps_canon

_DATASET_TAR_DOMAIN   = b"ARK/DATASET/TAR/v1\n"
_DATASET_JSONL_DOMAIN = b"ARK/DATASET/JSONL/v1\n"
_PERM_DOMAIN          = b"ARK/DATA/PERM/v1\n"

# ---------- RNG + permutation ----------

class _XorShift64Star:
    __slots__ = ("_x",)
    def __init__(self, seed: int) -> None:
        self._x = seed & ((1 << 64) - 1) or 0x9E3779B97F4A7C15
    def rand_u64(self) -> int:
        x = self._x
        x ^= (x >> 12) & ((1 << 64) - 1)
        x ^= (x << 25) & ((1 << 64) - 1)
        x ^= (x >> 27) & ((1 << 64) - 1)
        self._x = x
        return (x * 0x2545F4914F6CDD1D) & ((1 << 64) - 1)
    def rand_below(self, n: int) -> int:
        if n <= 0: return 0
        limit = ((1 << 64) // n) * n
        while True:
            r = self.rand_u64()
            if r < limit:
                return int(r % n)

def fisher_yates_perm(n: int, seed64: int) -> List[int]:
    p = list(range(n))
    rng = _XorShift64Star(seed64)
    for i in range(n - 1, 0, -1):
        j = rng.rand_below(i + 1)
        p[i], p[j] = p[j], p[i]
    return p

def _seed_from_commit(commit_hex: str, extra_seed: int = 0) -> int:
    be = int(extra_seed).to_bytes(8, "big", signed=False)
    h = domain_hash_hex(bytes.fromhex(commit_hex) + be, _PERM_DOMAIN)
    return int.from_bytes(bytes.fromhex(h)[:8], "big", signed=False)

# ---------- dataset commit ----------

def _hash_jsonl_file(path: str) -> str:
    with open(path, "rb") as f:
        out = io.BytesIO()
        for raw in f:
            line = raw.decode("utf-8", errors="strict")
            if line.endswith("\r\n"): line = line[:-2]
            elif line.endswith("\n"): line = line[:-1]
            out.write(line.encode("utf-8"))
            out.write(b"\n")
        buf = out.getvalue()
    return domain_hash_hex(buf, _DATASET_JSONL_DOMAIN)

def compute_dataset_commit(path: str, fmt: str = "auto") -> str:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    if os.path.isdir(path):
        tar_bytes = build_canonical_tar_bytes(path)
        return domain_hash_hex(tar_bytes, _DATASET_TAR_DOMAIN)
    ext = os.path.splitext(path.lower())[1]
    if fmt == "jsonl" or (fmt == "auto" and ext == ".jsonl"):
        return _hash_jsonl_file(path)
    with open(path, "rb") as f:
        return sha256_hex(f.read())

# ---------- spec + runtime dataset ----------

@dataclass(frozen=True)
class DatasetSpec:
    kind: str           # "jsonl" for this backend
    path: str
    ratios: Optional[Dict[str, float]] = None  # e.g. {"train":1.0}
    split: str = "train"

class _JSONLDataset:
    def __init__(self, spec: DatasetSpec, seed: int):
        self.path = os.path.realpath(spec.path)
        self.kind = "jsonl"
        self.commit_hex = compute_dataset_commit(self.path, fmt="jsonl")
        # load lines deterministically
        with open(self.path, "r", encoding="utf-8") as f:
            self._lines = [ln.rstrip("\r\n") for ln in f]
        self._n = len(self._lines)
        # build global permutation
        perm = fisher_yates_perm(self._n, _seed_from_commit(self.commit_hex, seed))
        # split (default 100% train)
        ratios = spec.ratios or {"train": 1.0}
        total = float(sum(ratios.values()))
        if not (0.99 <= total <= 1.01):  # keep it simple
            raise ValueError("split ratios must sum to ~1.0")
        names = list(ratios.keys())
        fracs = [float(ratios[k]) / total for k in names]
        counts = [int(x * self._n) for x in fracs]
        used = sum(counts); counts[-1] += self._n - used
        slices = {}
        cur = 0
        for name, cnt in zip(names, counts):
            slices[name] = (cur, cur + cnt); cur += cnt
        a, b = slices.get(spec.split, (0, self._n))
        self._order = perm[a:b]
        self._cursor = 0

    def next_batch(self, batch_size: int) -> Tuple[List[Any], Dict[str, Any]]:
        if self._cursor >= len(self._order):
            # deterministic empty tail
            return [], {"indices": []}
        end = min(self._cursor + int(batch_size), len(self._order))
        idxs = self._order[self._cursor:end]
        self._cursor = end
        batch = [self._lines[i] for i in idxs]  # content; backend may ignore
        meta = {"indices": list(idxs)}          # <-- ONLY indices; no paths!
        return batch, meta

    def batch_from_meta(self, meta: Dict[str, Any]) -> List[Any]:
        idxs = [int(i) for i in meta.get("indices", [])]
        return [self._lines[i] for i in idxs]

def open_dataset(spec: DatasetSpec, seed: int = 0):
    kind = (spec.kind or "jsonl").lower()
    if kind != "jsonl":
        raise NotImplementedError(f"dataset kind not supported in MVP: {kind}")
    return _JSONLDataset(spec, seed)
