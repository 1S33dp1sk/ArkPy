# training/transcript.py — training transcript leaves + Merkle builder/verify
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

from ..utils.json_canon import dumps_canonical
from ..utils.hashing import sha256_hex, sha256_domain, sha256_domain_hex
from ..utils.merkle import merkle_root as _merkle_root_bytes
from ..utils.iohelpers import write_text_atomic

# Domains: keep aligned with on-chain/C constants naming style
_TRANSCRIPT_LEAF_DOMAIN  = b"ARK/TRAIN/LEAF/v1\n"
_TRANSCRIPT_MERKLE_DOMAIN = b"ARK/TRAIN/MERKLE/v1\n"


# ------------------------------ leaf model ---------------------------------

@dataclass(frozen=True)
class TranscriptLeaf:
    """
    A single canonical leaf in the training transcript Merkle tree.
    `payload` must be JSON-serializable; hashing uses canonical JSON bytes
    domain-separated with _TRANSCRIPT_LEAF_DOMAIN.
    """
    type: str
    step: int                   # global step/epoch counter as agreed by the recipe
    payload: Dict[str, Any]

    def preimage(self) -> bytes:
        """
        Canonical preimage bytes for hashing the leaf.
        Structure: canonical JSON of {"type", "step", "payload"} (stable key order),
        then domain-separated SHA-256.
        """
        obj = {"type": self.type, "step": int(self.step), "payload": self.payload}
        return dumps_canonical(obj)

    def leaf_hash(self) -> bytes:
        return sha256_domain(_TRANSCRIPT_LEAF_DOMAIN, self.preimage())

    def hex(self) -> str:
        return sha256_domain_hex(_TRANSCRIPT_LEAF_DOMAIN, self.preimage())


# Convenience constructors for common leaf kinds -----------------------------

def leaf_batch(
    step: int,
    dataset_commit: str,
    batch_indices_commit: Optional[str],
    loss_scalar: Optional[float],
    extra: Optional[Dict[str, Any]] = None,
) -> TranscriptLeaf:
    """
    Batch processing record:
      - dataset_commit: hex-64 of dataset
      - batch_indices_commit: hex-64 commit of indices (or None)
      - loss_scalar: optional training loss
    """
    payload: Dict[str, Any] = {
        "dataset_commit": dataset_commit,
        "batch_indices_commit": batch_indices_commit,
        "loss": loss_scalar,
    }
    if extra:
        payload["extra"] = extra
    return TranscriptLeaf("batch", step, payload)


def leaf_optimizer(step: int, optimizer_digest_hex: str, extra: Optional[Dict[str, Any]] = None) -> TranscriptLeaf:
    """
    Optimizer state snapshot digest (e.g., sha256 over a canonical dump).
    """
    payload: Dict[str, Any] = {"optimizer_commit": optimizer_digest_hex}
    if extra:
        payload["extra"] = extra
    return TranscriptLeaf("optimizer", step, payload)


def leaf_export(step: int, weights_commit_hex: str, spec_hash_hex: str, extra: Optional[Dict[str, Any]] = None) -> TranscriptLeaf:
    """
    Final (or periodic) exported weights checkpoint reference.
    - weights_commit_hex: sha256 over .safetensors file (with export domain)
    - spec_hash_hex: canonical training spec hash
    """
    payload: Dict[str, Any] = {"weights_commit": weights_commit_hex, "spec_hash": spec_hash_hex}
    if extra:
        payload["extra"] = extra
    return TranscriptLeaf("export", step, payload)


# ------------------------------ builder ------------------------------------

class TranscriptBuilder:
    """
    Incrementally build a transcript (ordered list of leaves), compute Merkle root,
    and optionally persist a JSONL trace (one canonical JSON object per line).
    """
    def __init__(self) -> None:
        self._leaves: List[TranscriptLeaf] = []

    def add(self, leaf: TranscriptLeaf) -> None:
        self._leaves.append(leaf)

    def extend(self, leaves: Iterable[TranscriptLeaf]) -> None:
        self._leaves.extend(list(leaves))

    @property
    def leaves(self) -> List[TranscriptLeaf]:
        return list(self._leaves)

    def merkle_root_hex(self) -> str:
        if not self._leaves:
            # Merkle of empty list — define as hash(domain || "EMPTY")
            return sha256_domain_hex(_TRANSCRIPT_MERKLE_DOMAIN, b"EMPTY")
        leaf_hashes = [lf.leaf_hash() for lf in self._leaves]
        root = _merkle_root_bytes(leaf_hashes, domain=_TRANSCRIPT_MERKLE_DOMAIN)
        return root.hex()

    def to_jsonl(self) -> str:
        """
        Deterministic JSONL: each line is canonical JSON of the leaf object
        (the same bytes used as leaf preimages).
        """
        lines = [dumps_canonical({"type": lf.type, "step": lf.step, "payload": lf.payload}).decode("utf-8")
                 for lf in self._leaves]
        return "\n".join(lines) + ("\n" if lines else "")

    def write_jsonl(self, path: str) -> None:
        write_text_atomic(path, self.to_jsonl())


# ------------------------------ verification --------------------------------

def verify_transcript_merkle(
    leaves: Iterable[TranscriptLeaf],
    expected_root_hex: str,
) -> bool:
    tb = TranscriptBuilder()
    tb.extend(leaves)
    return tb.merkle_root_hex().lower() == expected_root_hex.lower()
