# merkle.py — Canonical SHA-256 Merkle (leaf prefix & domain-separated)
from __future__ import annotations

import hashlib
from typing import Iterable, List, Sequence, Tuple

__all__ = [
    "ARK_MERKLE_DOMAIN",
    "merkle_root",
    "merkle_root_hex",
    "build_proof",
    "verify_proof",
    "levels_debug",
]

# Keep in sync with C constants (include/core/constants.h -> ARK_DOM_MERKLE)
ARK_MERKLE_DOMAIN = b"ARK/MERKLE/v1"

# Domain-separated prefixes to avoid ambiguity / second-preimage shenanigans
_LEAF_TAG = b"\x00"
_NODE_TAG = b"\x01"


def _h(data: bytes) -> bytes:
    return hashlib.sha256(data).digest()


def _leaf_hash(b: bytes) -> bytes:
    # H( DOM || 0x00 || leaf )
    return _h(ARK_MERKLE_DOMAIN + _LEAF_TAG + b)


def _node_hash(l: bytes, r: bytes) -> bytes:
    # H( DOM || 0x01 || left || right )
    return _h(ARK_MERKLE_DOMAIN + _NODE_TAG + l + r)


def merkle_root(leaves: Sequence[bytes]) -> bytes:
    """
    Canonical Merkle:
      - leaves hashed as H(DOM||0x00||leaf)
      - internal nodes H(DOM||0x01||L||R)
      - odd promotion: duplicate the last hash at the level (Bitcoin-style)
      - empty set: SHA256(DOM||0x00||"")  (root of a single empty leaf)
    Returns 32-byte digest.
    """
    if not leaves:
        return _leaf_hash(b"")

    lvl = [_leaf_hash(x) for x in leaves]
    while len(lvl) > 1:
        nxt: List[bytes] = []
        n = len(lvl)
        i = 0
        while i < n:
            a = lvl[i]
            b = lvl[i + 1] if (i + 1) < n else a  # duplicate last if odd
            nxt.append(_node_hash(a, b))
            i += 2
        lvl = nxt
    return lvl[0]


def merkle_root_hex(leaves: Sequence[bytes]) -> str:
    return merkle_root(leaves).hex()


# ----- Proofs --------------------------------------------------------------

# Proof element: (sibling_hash, is_left)
# is_left=True  means: sibling is the LEFT child, so parent = H(sib, cur)
# is_left=False means: sibling is the RIGHT child, so parent = H(cur, sib)
Proof = List[Tuple[bytes, bool]]


def build_proof(leaves: Sequence[bytes], index: int) -> Proof:
    """
    Build Merkle audit path for leaf at 'index'.
    Raises IndexError if index is out of range.
    """
    if index < 0 or index >= len(leaves):
        raise IndexError("proof index out of range")
    if not leaves:
        # single empty leaf case: no path
        return []

    # Start from leaf hashes
    lvl = [_leaf_hash(x) for x in leaves]
    idx = index
    proof: Proof = []

    while len(lvl) > 1:
        n = len(lvl)
        is_last_odd = (n % 2 == 1) and (idx == n - 1)
        if is_last_odd:
            # sibling is itself (duplication case)
            sib = lvl[idx]
            # In duplication, we treat sibling as the RIGHT (cur, sib)
            proof.append((sib, False))
            idx = idx // 2
        else:
            if idx % 2 == 0:
                # even index: sibling on the right
                sib = lvl[idx + 1]
                proof.append((sib, False))  # sib is RIGHT
            else:
                # odd index: sibling on the left
                sib = lvl[idx - 1]
                proof.append((sib, True))   # sib is LEFT
            idx = idx // 2

        # build next level
        nxt: List[bytes] = []
        i = 0
        while i < n:
            a = lvl[i]
            b = lvl[i + 1] if (i + 1) < n else a
            nxt.append(_node_hash(a, b))
            i += 2
        lvl = nxt

    return proof


def verify_proof(leaf: bytes, proof: Proof, root: bytes) -> bool:
    """
    Verify audit path for 'leaf' against 'root'.
    """
    cur = _leaf_hash(leaf)
    for sib, is_left in proof:
        if is_left:
            cur = _node_hash(sib, cur)
        else:
            cur = _node_hash(cur, sib)
    return cur == root


# ----- Debug helpers -------------------------------------------------------

def levels_debug(leaves: Sequence[bytes]) -> List[List[str]]:
    """
    Return list of levels as hex strings (for testing/inspection).
    Level[0] = leaf hashes, last level = root (single element).
    """
    if not leaves:
        return [[_leaf_hash(b"").hex()]]

    out: List[List[str]] = []
    lvl = [_leaf_hash(x) for x in leaves]
    out.append([h.hex() for h in lvl])
    while len(lvl) > 1:
        nxt: List[bytes] = []
        n = len(lvl)
        i = 0
        while i < n:
            a = lvl[i]
            b = lvl[i + 1] if (i + 1) < n else a
            nxt.append(_node_hash(a, b))
            i += 2
        lvl = nxt
        out.append([h.hex() for h in lvl])
    return out
