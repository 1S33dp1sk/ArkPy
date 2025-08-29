# arknet_py/cli.py
from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, List, Optional

from .commit import (
    compute_commit,
    write_commit_files,
    load_commit_manifest,
    verify_commit,
)
from .runner import ArknetModel
from .determinism import apply_determinism_profile, determinism_report
from .training.trainer import train_once


# ------------------------------ utils -------------------------------------


def _load_json_from_path(path: str) -> Any:
    if path == "-":
        return json.load(sys.stdin)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _kv_list_to_dict(items: Optional[List[str]]) -> Dict[str, Any]:
    """--param k=v (repeatable) → dict with best-effort JSON coercion."""
    out: Dict[str, Any] = {}
    if not items:
        return out
    for it in items:
        if "=" not in it:
            out[it] = True
            continue
        k, v = it.split("=", 1)
        k = k.strip()
        v = v.strip()
        if not k:
            continue
        try:
            out[k] = json.loads(v)
        except Exception:
            try:
                if v.isdigit() or (v.startswith("-") and v[1:].isdigit()):
                    out[k] = int(v)
                else:
                    out[k] = float(v)
            except Exception:
                out[k] = v
    return out


def _merge_params(args) -> Dict[str, Any]:
    p = _kv_list_to_dict(getattr(args, "param", None))
    if getattr(args, "max_tokens", None) is not None:
        p["max_tokens"] = int(args.max_tokens)
    if getattr(args, "system", None):
        p["system"] = args.system
    if getattr(args, "seed", None) is not None:
        p["seed"] = int(args.seed)
    return p


def _print_stream(chunks) -> None:
    for ch in chunks:
        if ch.text:
            print(ch.text, end="", flush=True)
    if sys.stdout.isatty():
        print()  # nicety for TTYs


def _load_prompt(path: str) -> str:
    if path == "-":
        return sys.stdin.read()
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


# ------------------------------ subcommands --------------------------------


def _cmd_commit(args) -> int:
    if args.verify:
        ok = verify_commit(args.artifact, args.verify)
        print("OK" if ok else "MISMATCH")
        return 0 if ok else 2

    digest, manifest = compute_commit(args.artifact)
    if args.write:
        write_commit_files(args.artifact, out_dir=args.write_out or None)
    print(digest)
    if args.json:
        print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


def _cmd_commit_show(args) -> int:
    mf = load_commit_manifest(args.artifact)
    print(json.dumps(mf, indent=2, sort_keys=True))
    return 0


def _cmd_run(args) -> int:
    # process-wide determinism (env + libs)
    if args.seed is not None or args.allow_tf32:
        apply_determinism_profile(seed=args.seed, allow_tf32=bool(args.allow_tf32))

    params = _merge_params(args)
    runner = ArknetModel(args.artifact)

    try:
        if args.chat:
            msgs = _load_json_from_path(args.chat)
            if not isinstance(msgs, list):
                raise ValueError("--chat must be a JSON array of {role,content} messages")
            if args.stream:
                _print_stream(runner.chat(msgs, params=params, stream=True))
            else:
                out = runner.chat(msgs, params=params, stream=False)
                print(out)
        else:
            prompt = _load_prompt(args.prompt_file) if args.prompt_file else (args.prompt or "")
            if not prompt:
                raise ValueError("provide --prompt or --prompt-file or --chat")
            if args.stream:
                _print_stream(runner.generate(prompt, params=params, stream=True))
            else:
                out = runner.generate(prompt, params=params, stream=False)
                print(out)
        return 0
    except Exception as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


def _cmd_train(args) -> int:
    # determinism profile for training (stronger parity)
    spec: Dict[str, Any] = {}
    if args.spec:
        spec = _load_json_from_path(args.spec)
        if not isinstance(spec, dict):
            raise ValueError("--spec must be a JSON object")
    if args.seed is not None or args.allow_tf32:
        apply_determinism_profile(seed=args.seed, allow_tf32=bool(args.allow_tf32))
        if args.seed is not None:
            spec.setdefault("seed", int(args.seed))

    digest = train_once(args.artifact, spec, args.out)
    print(digest)
    return 0


def _cmd_env_report(_args) -> int:
    print(json.dumps(determinism_report(), indent=2, sort_keys=True))
    return 0


# ------------------------------ main ---------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(prog="arknet-py", description="Arknet model tooling")
    sub = p.add_subparsers(dest="cmd", required=True)

    # commit
    pc = sub.add_parser("commit", help="Compute/verify canonical model commit hash")
    pc.add_argument("artifact", help="Path to artifact directory")
    pc.add_argument("--write", action="store_true", help="Write commit.json (defaults to artifact dir)")
    pc.add_argument("--write-out", help="Directory to write commit.json to (implies --write)")
    pc.add_argument("--json", action="store_true", help="Also print manifest JSON containing the commit")
    pc.add_argument("--verify", metavar="HEX64", help="Verify the commit equals HEX64; exit 0/2")
    pc.set_defaults(func=_cmd_commit)

    # commit.show (pretty print manifest/commit)
    pcs = sub.add_parser("commit.show", help="Show commit/manifest JSON (prefers commit.json)")
    pcs.add_argument("artifact", help="Path to artifact directory")
    pcs.set_defaults(func=_cmd_commit_show)

    # run
    pr = sub.add_parser("run", help="Run inference (prompt or chat)")
    pr.add_argument("artifact", help="Path to artifact directory (or exported out_dir)")
    g_in = pr.add_mutually_exclusive_group(required=True)
    g_in.add_argument("--prompt", help="Prompt text (single-turn)")
    g_in.add_argument("--prompt-file", help="Read prompt from file or '-' for stdin")
    g_in.add_argument("--chat", help="JSON file (or '-') containing messages: [{role,content}, ...]")
    pr.add_argument("--system", help="Optional system prompt (also works for chat)")
    pr.add_argument("--max-tokens", type=int, default=None, help="Max tokens / completion length")
    pr.add_argument("--param", action="append", metavar="K=V", help="Extra generation param (repeatable)")
    pr.add_argument("--seed", type=int, help="Deterministic seed for RNGs")
    pr.add_argument("--allow-tf32", action="store_true", help="Allow TF32 (off by default for parity)")
    pr.add_argument("--stream", action="store_true", help="Stream tokens/chunks to stdout")
    pr.set_defaults(func=_cmd_run)

    # train
    pt = sub.add_parser("train", help="Run deterministic training via trainer.py")
    pt.add_argument("artifact", help="Path to artifact directory containing trainer.py")
    pt.add_argument("--spec", help="JSON file (or '-') with job_spec (seed, datasets, recipe, etc.)")
    pt.add_argument("--out", required=True, help="Output directory for the trained artifact")
    pt.add_argument("--seed", type=int, help="Deterministic seed for training workflow")
    pt.add_argument("--allow-tf32", action="store_true", help="Allow TF32 during training (off by default)")
    pt.set_defaults(func=_cmd_train)

    # env.report
    pe = sub.add_parser("env.report", help="Print determinism/environment report (JSON)")
    pe.set_defaults(func=_cmd_env_report)

    args = p.parse_args()

    # --write-out implies --write
    if getattr(args, "write_out", None):
        args.write = True

    rc = args.func(args)
    raise SystemExit(rc)


if __name__ == "__main__":
    main()
