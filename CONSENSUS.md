# Arknet Proof-of-Training: Consensus without Bit-Exact Determinism

## Abstract

Training on modern GPUs is **not** bit-deterministic across devices, drivers, or parallelism settings. Small numeric drift (kernel ordering, TF32, fused ops, atomics) means two honest trainers can produce slightly different floating-point weights—even with the same seed and data. Requiring strict bit-equality therefore kills performance and portability.

**Arknet Proof-of-Training (PoT)** is a practical consensus protocol that:

1. **Quantizes** key observables (weights & losses) into **deterministic buckets** so small, benign drift collapses to the same canonical value.
2. Records every step in a **Merkle-hashed transcript** (dataset commit, batch index, quantized loss, pre/post weight bucket).
3. Periodically writes **anchors**: canonical, portable weight snapshots.
4. Lets verifiers perform **windowed audits**: replay a short segment from the nearest anchor to a sampled step and check that the **bucketed** post-state and the **quantized** loss match the transcript.

This yields strong evidence that “the model was actually trained on this data with this recipe,” while staying fast, GPU-friendly, and chain-verifiable.

---

## 1. Problem: GPUs are fast—and non-deterministic

**Sources of drift**

* Reduction order & race conditions in parallel kernels
* Mixed precision (FP16/BF16/TF32) and hardware-specific rounding
* Library heuristics and vendor kernel selection
* Data parallelism sharding order, non-associativity of FP ops

Even with fixed seeds and “determinism modes,” small numeric differences accumulate across thousands of steps.

**What we need**

* A way for *honest but non-bit-identical* trainers to **agree** on “same enough” outcomes.
* Cryptographic commitments that let anyone **audit** training without re-doing the full run.

---

## 2. Design overview

We replace bit-equality with **equivalence under quantization** + cryptographic commitments.

### 2.1 Bucketed weight consensus

Define a deterministic **bucket function** `B(·)` mapping a full weight state → a short hash:

* **fp16\_rne** mode: cast each tensor to IEEE FP16 with **round-to-nearest-even**, concatenate *(name || 0x00 || bytes)* in sorted name order, hash with SHA-256.
* **grid** mode: quantize FP32 weights to a uniform grid with step `ε` (e.g., 1e-6), then hash as above.

Small numeric drift collapses to the same bucket. A different model (or materially different weights) lands in a different bucket.

### 2.2 Quantized losses

Per-step scalar loss is rounded to a fixed number of decimals (default: **1 dp**). Honest small variance disappears; big discrepancies do not.

### 2.3 Merkle transcript

For each training step, we write a **leaf** with:

* `index` (step), `dataset_commit`, `batch` metadata (split, indices, seed lineage),
* `pre_bucket`, `post_bucket`,
* `loss_q` (quantized loss),
* optional `tags` (e.g., `"noop"`).

Each leaf is canonical-JSON’d and domain-tagged, then hashed; the Merkle root commits to the **entire sequence**. The transcript header includes the **TrainingSpec hash**, **environment snapshot hash**, and **consensus parameters** (bucket mode, grid ε, loss decimals, anchor stride).

### 2.4 Anchors

Every *N* steps (configurable), write a canonical **anchor** file:

```
anchors/step_00001234.safetensors
```

and record its **bucket** as an `anchor` leaf. Anchors are compact checkpoints that let verifiers jump in and replay a short window.

### 2.5 Windowed probabilistic verification

A verifier:

1. Loads the **manifest** and **transcript**, checks hashes & domains.
2. Uniformly samples `K` step indices (public seed) from `train_step` leaves.
3. For each sampled step `s`, find nearest anchor `a ≤ s`. Replay steps `a..s` (bounded by `max_window`) using the same dataset plan, optimizer, and schedule.
4. Check:

   * recomputed **post\_bucket == transcript.post\_bucket**,
   * recomputed **loss\_q == transcript.loss\_q**.
5. Accept if all sampled windows pass.

This provides **evidence of work** without full recomputation.

---

## 3. Protocol details

### 3.1 Domains & hashing

* All hashes are **SHA-256** with explicit **domain separation** (ASCII tag + `\n`).
* Examples:
  `ARK/TRAIN/STEP/v1`, `ARK/TRAIN/TRANSCRIPT/v1`, `ARK/TRAIN/SPEC/v1`, `ARK/DATASET/TAR/v1`, `ARK/ENV/SNAPSHOT/v1`.

### 3.2 Canonicalization

* **Artifacts:** deterministic USTAR tar (sorted POSIX arcnames, zeroed metadata) → hash → `commit.json`.
* **JSON:** canonical dump (sorted keys, no NaN/Inf, UTF-8; no whitespace fluff).
* **Weights:** canonical safetensors (sorted tensor names, fixed endianness) for anchors; **bucket** function for consensus.

### 3.3 TrainingSpec & environment

* **Spec hash** commits to seed, optimizer, schedule, dataset(s), batch shape, and **consensus knobs**:

  * `bucket_mode` (`fp16_rne`|`grid`),
  * `grid_eps` (e.g., 1e-6),
  * `loss_decimals` (e.g., 1),
  * `anchor_stride` (e.g., 1000).
* **Env snapshot:** Python, CUDA/cuDNN/Torch versions, visible GPUs, determinism env vars → hash in header.

### 3.4 Dataset commitments

* Directories: canonical tar → domain-tagged hash.
* JSONL: newline-normalized bytes → domain-tagged hash.
* Splits/permutation: derived via deterministic PRNG seeded from dataset commit (+ user seed).

### 3.5 Leaves

**Step leaf**

```json
{
  "kind": "train_step",
  "index": 123,
  "dataset_commit": "<hex64>",
  "batch": {"split":"train","indices":[...],"seq":"..."},
  "pre_bucket": "<hex64>|null",
  "post_bucket": "<hex64>|null",
  "loss_q": 1.2,
  "tags": ["noop"]
}
```

**Anchor leaf**

```json
{
  "kind": "anchor",
  "index": 120,
  "file": "anchors/step_00000120.safetensors",
  "bucket": "<hex64>"
}
```

### 3.6 Merkle root

`root = Merkle( H( domain_step || canon_json(leaf_i) ) )` over the ordered leaf list. The transcript file includes `header`, `leaves[]`, `root`, and `domain`.

---

## 4. Security & correctness

### 4.1 What we guarantee

* **Consensus under honest drift:** Trainers using different GPUs/parallelism get the same **buckets** and **loss\_q** with overwhelmingly high probability.
* **Tamper evidence:** Any insertion, deletion, or edit in the transcript breaks the **Merkle root**.
* **Spec/environment binding:** Replaying with a different recipe/spec/env is detectable (spec/env hashes disagree).
* **Probabilistic assurance:** Windowed audits force the prover to maintain local step-to-step consistency near anchors.

### 4.2 Threat model & attacks

* **Bit-twiddling within a bucket:** Adversary could nudge values while staying in the same bucket. Mitigations:

  * tight bucket (`fp16_rne`) or small `grid_eps`,
  * random windowed replays from anchors,
  * optional **dual-bucket** verification (e.g., check both `fp16_rne` and a fine grid).
* **Fabricated transcript without training:** Must produce anchors consistent with many sampled windows; replay checks (loss & bucket) amplify cost. Increasing `K` and reducing anchor stride raises the bar.
* **Data substitution:** The **dataset commit** + deterministic split plan is part of each leaf; mismatch is detectable.

### 4.3 Liveness & performance

* No requirement for full determinism; trainers can keep **fast kernels and parallelism**.
* Anchors are periodic and small; transcripts are compact (JSONL option for streaming).
* Verifiers replay only **short windows** (seconds → minutes), not the whole run.

---

## 5. Parameters & tuning

* `bucket_mode`:

  * `fp16_rne` (default): robust, hardware-agnostic, tight buckets.
  * `grid`: choose `grid_eps` (e.g., `1e-6`) to balance tolerance vs. collision risk.
* `loss_decimals`: default **1**; raise to 2 for tighter checks if kernels are stable.
* `anchor_stride`: trade storage for auditability (e.g., 500–2,000 steps).
* Verifier knobs: `samples` (e.g., 64), `max_window` (e.g., 8–32).

---

## 6. Verifier algorithm (sketch)

```
Input: trained artifact dir T, spec S, K samples, window W
1. Load manifest, transcript (header, leaves[], root):
   - Check domain tags & recompute Merkle root.
   - Check header.spec_hash == hash(S).
2. Gather step leaves and anchor leaves; index anchors by step.
3. For i in 1..K:
   a. s ← sample step index (PRNG seeded by public seed)
   b. a ← nearest anchor ≤ s; if none and stride==0 → fail
   c. Load anchor weights; set model state to anchor
   d. For t in [a+1 .. s]:
        - deterministically reconstruct batch (dataset_commit, split plan)
        - compute LR from schedule
        - run one training step
   e. Compute post_bucket, loss_q at step s
   f. Check equality with transcript leaf s
4. Accept if all K pass.
```

---

## 7. Implementation map (high-level)

* **Consensus knobs:** `constants.py` (bucket mode, eps, loss decimals, anchor stride).
* **Export/buckets:** `training/exporter.py` (`weight_bucket_hex`, canonical safetensors IO).
* **Transcript:** `training/transcript.py` (leaf builders, Merkle, writer).
* **Backends:** `backends/torch_backend.py` (`weight_bucket_hex`, canonical export); dummy backend mirrors API.
* **Trainer:** `training/trainer.py`

  * replaces raw weight “commits” with **bucketed** pre/post,
  * quantizes loss,
  * writes **anchor** leaves & files,
  * emits transcript with Merkle root in manifest.
* **Verifier:** `training/audit.py` (windowed replay).
* **Spec/env/data commits:** `training/spec.py`, `training/container.py`, `training/dataset.py`.
* **Artifact commit:** `commit.py` (canonical tar + domain-tagged hash).

---

## 8. Why this works

Floating-point drift is inevitable on high-performance hardware. Instead of fighting physics, Arknet PoT **contains** drift within **tight, deterministic equivalence classes** and ties every step to data & recipe via **cryptographic commitments**. Anchors turn global training into many **local constraints** that are cheap to verify yet collectively hard to forge.

The result: **portable, performant training** with **verifiable provenance**—no CPU fallbacks, no single-vendor lock-in, no brittle bit-equality.

---

## 9. Limitations & future work

* Very long windows or missing anchors reduce verifier leverage. (Tune stride and `max_window`.)
* If kernels are extremely unstable, `loss_decimals=1` may be too tight; consider 2 dp or a tolerance band on loss.
* Optional **multi-anchor precision** (store both fp16 and bf16 anchors) further reduces collision surface.
* Extend transcripts with **gradient-norm** or **activation-checksum** side-channels for even stronger locality checks.

---

## 10. TL;DR

**Arknet Proof-of-Training** replaces bit-exact determinism with **bucketed consensus** on weights and **quantized losses**, secured by **Merkle transcripts** and **anchor-based windowed verification**. You keep your fast GPUs and parallelism; we keep verifiability.
