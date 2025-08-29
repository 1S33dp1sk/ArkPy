# arknet-py

Deterministic model commit tooling for Arknet.

## Concepts

- **Artifact directory**: a folder containing at least `manifest.json` and `model.py` (for inference).
- **Commit**: SHA-256 over a canonical USTAR tar of the artifact directory (sorted file order, fixed metadata). The
  hex digest is the **model commit** that appears on-chain and is reproducible in C.

## Quickstart

```bash
arknet-py commit ./my_model --write --json
arknet-py run ./my_model --prompt "hello"
arknet-py train ./my_model --spec job.json --out ./my_model_out
