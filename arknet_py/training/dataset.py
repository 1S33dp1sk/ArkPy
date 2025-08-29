# training/dataset.py — dataset commit hashing + deterministic split/permutation
from __future__ import annotations

import io
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

from ..utils.hashing import sha256_hex, sha256_domain_hex
from ..utils.tar_canon import canonical_tar_bytes
from ..utils.json_canon import dumps_canonical


# Domains (keep aligned to C constants naming style)
_DATASET_TAR_DOMAIN   = b"ARK/DATASET/TAR/v1\n"
_DATASET_JSONL_DOMAIN = b"ARK/DATASET/JSONL/v1\n"
_PERM_DOMAIN          = b"ARK/DATA/PERM/v1\n"


# ---------- Deterministic PRNG (xorshift64*) & Fisher–Yates -----------------

class _XorShift64Star:
    __slots__ = ("_x",)

    def __init__(self, seed: int) -> None:
        # guard against zero state
        self._x = seed & ((1 << 64) - 1) or 0x9E3779B97F4A7C15

    def rand_u64(self) -> int:
        x = self._x
        x ^= (x >> 12) & ((1 << 64) - 1)
        x ^= (x << 25) & ((1 << 64) - 1)
        x ^= (x >> 27) & ((1 << 64) - 1)
        self._x = x
        return (x * 0x2545F4914F6CDD1D) & ((1 << 64) - 1)

    def rand_below(self, n: int) -> int:
        """Uniform in [0, n) via rejection sampling to avoid modulo bias."""
        if n <= 0:
            return 0
        limit = ((1 << 64) // n) * n
        while True:
            r = self.rand_u64()
            if r < limit:
                return int(r % n)


def fisher_yates_perm(n: int, seed64: int) -> List[int]:
    """Deterministic permutation 0..n-1 using xorshift64* RNG."""
    p = list(range(n))
    rng = _XorShift64Star(seed64)
    # in-place shuffle
    for i in range(n - 1, 0, -1):
        j = rng.rand_below(i + 1)
        p[i], p[j] = p[j], p[i]
    return p


def _seed_from_commit(commit_hex: str, extra_seed: int = 0) -> int:
    """
    Derive a 64-bit seed from (domain || commit || be64(extra_seed)).
    """
    be = extra_seed.to_bytes(8, "big", signed=False)
    h = sha256_domain_hex(_PERM_DOMAIN, bytes.fromhex(commit_hex) + be)
    # first 8 bytes of the digest as big-endian u64
    return int.from_bytes(bytes.fromhex(h)[:8], "big", signed=False)


# ---------- Dataset commit (content hash) -----------------------------------

def _hash_jsonl_file(path: str) -> str:
    """
    Canonical JSONL hash:
      - UTF-8 decode per line, strip trailing '\r\n', re-join with '\n'.
      - We *do not* parse JSON (robust to whitespace), only normalize newlines.
      - Domain-separated hash to distinguish from tar hash.
    """
    with open(path, "rb") as f:
        # Read in chunks to avoid huge memory for large datasets
        out = io.BytesIO()
        for raw in f:
            # normalize \r?\n
            line = raw.decode("utf-8", errors="strict")
            if line.endswith("\r\n"):
                line = line[:-2]
            elif line.endswith("\n"):
                line = line[:-1]
            out.write(line.encode("utf-8"))
            out.write(b"\n")
        buf = out.getvalue()
    return sha256_domain_hex(_DATASET_JSONL_DOMAIN, buf)


def compute_dataset_commit(path: str, fmt: str = "auto") -> str:
    """
    Returns a hex-64 commit for a dataset path.
      - If path is a directory -> canonical tar hash (domain separated).
      - If path is a file and format is "jsonl" (or .jsonl extension) -> jsonl hash.
      - Otherwise -> raw file bytes sha256 (no domain tag).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    if os.path.isdir(path):
        tar_bytes = canonical_tar_bytes(path)
        return sha256_domain_hex(_DATASET_TAR_DOMAIN, tar_bytes)

    # file
    lower = path.lower()
    ext = os.path.splitext(lower)[1]
    use_jsonl = (fmt == "jsonl") or (fmt == "auto" and ext == ".jsonl")
    if use_jsonl:
        return _hash_jsonl_file(path)

    # fallback: raw bytes sha256
    with open(path, "rb") as f:
        return sha256_hex(f.read())


# ---------- Splits & permutations ------------------------------------------

@dataclass(frozen=True)
class SplitPlan:
    """
    Deterministic split plan using a single global permutation.
    Slices are represented as (start, end) on the perm array (half-open).
    """
    commit: str
    n_items: int
    order: Tuple[int, ...]            # the perm itself as an immutable tuple
    slices: Dict[str, Tuple[int, int]]  # e.g., {"train": (0, N1), "val": (N1, N2)}

    def indices_for(self, split: str) -> List[int]:
        a, b = self.slices[split]
        return list(self.order[a:b])


def make_split_plan(
    commit_hex: str,
    n_items: int,
    ratios: Optional[Dict[str, float]],
    seed: int = 0,
    default_train_val: Tuple[float, float] = (0.98, 0.02),
) -> SplitPlan:
    """
    Create a deterministic split plan for a dataset of size n_items.
    - commit_hex: dataset commit (hex-64)
    - ratios: map of split->float that must sum ~1.0 (if None, use 98/2 train/val)
    - seed: lets callers produce different permutations if desired
    """
    if n_items < 0:
        raise ValueError("n_items must be >= 0")
    if n_items == 0:
        # trivial plan
        return SplitPlan(commit_hex, 0, tuple(), {"train": (0, 0)})

    if not ratios:
        ratios = {"train": default_train_val[0], "val": default_train_val[1]}
    total = float(sum(ratios.values()))
    if not (0.99 <= total <= 1.01):
        raise ValueError(f"split ratios must sum to ~1.0 (got {total})")

    # deterministic permutation
    seed64 = _seed_from_commit(commit_hex, seed)
    perm = fisher_yates_perm(n_items, seed64)

    # integer bucket sizes via floor + remainder to the last split
    names = list(ratios.keys())
    fracs = [float(ratios[k]) for k in names]
    # normalize
    fracs = [x / total for x in fracs]

    counts = [int(x * n_items) for x in fracs]
    used = sum(counts)
    # hand the remainder to the last split
    counts[-1] += n_items - used

    # build slices
    slices: Dict[str, Tuple[int, int]] = {}
    cur = 0
    for name, cnt in zip(names, counts):
        slices[name] = (cur, cur + cnt)
        cur += cnt

    return SplitPlan(commit_hex, n_items, tuple(perm), slices)


# ---------- Interleave across multiple datasets ----------------------------

@dataclass(frozen=True)
class InterleavePlan:
    """
    Deterministic interleave of multiple datasets according to weights.
    Yields a list of (ds_index, item_index) pairs in a stable order.
    """
    pairs: Tuple[Tuple[int, int], ...]  # ((ds_i, idx_in_perm), ...)


def make_interleave_plan(
    commits: List[str],
    sizes: List[int],
    weights: Optional[List[float]] = None,
    seed: int = 0,
    total_examples: Optional[int] = None,
) -> InterleavePlan:
    """
    Round-robin-with-weights over per-dataset permutations.
    - commits/sizes must align 1:1
    - weights default to uniform if not provided
    - seed lets you derive a global mixing permutation deterministically
    - total_examples truncates the plan (else full pass over the largest dataset)
    """
    if len(commits) != len(sizes):
        raise ValueError("commits and sizes must have the same length")
    m = len(sizes)
    if m == 0:
        return InterleavePlan(tuple())

    if not weights:
        weights = [1.0] * m
    if len(weights) != m:
        raise ValueError("weights length must match datasets")

    # build per-dataset perms
    perms: List[List[int]] = []
    cursors = [0] * m
    for i in range(m):
        seed64 = _seed_from_commit(commits[i], seed)
        perms.append(fisher_yates_perm(sizes[i], seed64))

    # compute how many from each dataset per "cycle"
    wsum = sum(float(w) for w in weights)
    take = [max(1, int(round((float(w) / wsum) * 100))) for w in weights]  # scale to small ints
    cycle = sum(take)

    # default total_examples: full pass over the largest dataset in cycles
    if total_examples is None:
        total_examples = max(sizes) * cycle

    out: List[Tuple[int, int]] = []
    ds = 0
    while len(out) < total_examples:
        for i in range(m):
            k = take[i]
            for _ in range(k):
                if cursors[i] >= sizes[i]:
                    continue  # this dataset exhausted; still deterministic
                idx = perms[i][cursors[i]]
                cursors[i] += 1
                out.append((i, idx))
                if len(out) >= total_examples:
                    break
            if len(out) >= total_examples:
                break
        ds = (ds + 1) % m

    return InterleavePlan(tuple(out))
