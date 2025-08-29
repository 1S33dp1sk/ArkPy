#!/usr/bin/env bash
# bin/test_arknet_full.sh
# Thorough staged test for arknet_py (commit/run/chat/stream/determinism/train).

set -Eeuo pipefail

KEEP=0
QUIET=0
PY=${PYTHON:-python3}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -k) KEEP=1; shift;;
    -q) QUIET=1; shift;;
    -p) PY="$2"; shift 2;;
    *) echo "unknown arg: $1" >&2; exit 2;;
  esac
done

is_tty(){ [[ -t 1 ]] && [[ -n "${TERM:-}" ]]; }
if is_tty; then
  C_GRN=$'\e[32m'; C_YLW=$'\e[33m'; C_RED=$'\e[31m'; C_CYN=$'\e[36m'; C_RST=$'\e[0m'
else
  C_GRN=; C_YLW=; C_RED=; C_CYN=; C_RST=
fi
log(){ [[ $QUIET -eq 0 ]] && echo "${C_CYN}[*]${C_RST} $*"; }
ok(){ echo "${C_GRN}[ok]${C_RST} $*"; }
warn(){ echo "${C_YLW}[warn]${C_RST} $*" >&2; }
die(){ echo "${C_RED}[err]${C_RST} $*"; exit 1; }

# Ensure debug transcript for byte-for-byte diffing
export ARKNET_DEBUG_TRANSCRIPT=1

# -------- portable sha256 tool ----------
hash_cmd() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$@"
  elif command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$@"
  else
    die "need sha256sum or shasum"
  fi
}

# List files (portable, sorted, no -printf)
list_files() {
  ( cd "$1" && find . -type f -print | sed 's#^\./##' | LC_ALL=C sort )
}

# --- handy diagnostics if a trained-commit mismatch ------
diag_diff() {
  local A="$1" B="$2"

  echo "[*] per-file sha256 for A=$A"
  while IFS= read -r p; do
    hash_cmd "$A/$p" | awk -v f="$p" '{print $1"  "f}'
  done < <(list_files "$A")

  echo "[*] per-file sha256 for B=$B"
  while IFS= read -r p; do
    hash_cmd "$B/$p" | awk -v f="$p" '{print $1"  "f}'
  done < <(list_files "$B")

  echo "[*] file listing diff (A vs B):"
  diff -u <(list_files "$A") <(list_files "$B") || true

  echo "[*] transcript roots:"
  if command -v jq >/dev/null 2>&1; then
    [[ -f "$A/transcript.json" ]] && jq -r '.root' "$A/transcript.json" | sed 's/^/  A.root: /'
    [[ -f "$B/transcript.json" ]] && jq -r '.root' "$B/transcript.json" | sed 's/^/  B.root: /'
  else
    echo "  (jq not installed; skipping transcript root read)"
  fi

  # Prefer debug JSON if present
  local TA="$A/transcript.debug.json"
  local TB="$B/transcript.debug.json"
  [[ -f "$TA" ]] || TA="$A/transcript.json"
  [[ -f "$TB" ]] || TB="$B/transcript.json"

  if command -v transcript_diff.py >/dev/null 2>&1; then
    echo "[*] transcript_diff.py:"
    transcript_diff.py "$TA" "$TB" || true
  elif [[ -x "./scripts/transcript_diff.py" ]]; then
    echo "[*] scripts/transcript_diff.py:"
    ./scripts/transcript_diff.py "$TA" "$TB" || true
  else
    echo "[warn] transcript_diff.py not found; skipping canonical leaf diff"
  fi
}

req(){ command -v "$1" >/dev/null 2>&1 || die "missing command: $1"; }
req "$PY"
req jq

# Try importing arknet_py; if it fails, prepend repo root to PYTHONPATH
if ! "$PY" - <<'PY' >/dev/null; then
import importlib; importlib.import_module("arknet_py")
PY
  SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
  REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
  export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
  warn "arknet_py not on sys.path; adding repo root: ${REPO_ROOT}"
  "$PY" - <<'PY' || { echo 1>&2 "unable to import arknet_py"; exit 1; }
import importlib; importlib.import_module("arknet_py")
PY
fi
ok "arknet_py importable"

"$PY" - <<'PY'
import sys, arknet_py
print(f"Python: {sys.version.split()[0]}  exe={sys.executable}")
print(f"arknet_py: {arknet_py.__file__}")
PY

ARTIFACT="$(mktemp -d -t arknet-artifact.XXXXXX)"
OUT1="$(mktemp -d -t arknet-trained.XXXXXX)"
OUT2="$(mktemp -d -t arknet-trained.XXXXXX)"
OUT3="$(mktemp -d -t arknet-trained.XXXXXX)"
DATA="$(mktemp -t dataset.XXXXXX.jsonl)"

cleanup(){
  [[ $KEEP -eq 1 ]] && return
  for d in "$ARTIFACT" "$OUT1" "$OUT2" "$OUT3"; do [[ -d "${d:-}" ]] && rm -rf "$d"; done
  [[ -f "${DATA:-}" ]] && rm -f "$DATA"
}
trap cleanup EXIT

log "artifact dir: $ARTIFACT"
log "out dirs:     $OUT1 , $OUT2 , $OUT3"

# --- toy artifact ---
cat >"$ARTIFACT/manifest.json" <<'JSON'
{"name":"toy-echo","format_version":1,"template":null}
JSON

cat >"$ARTIFACT/model.py" <<'PYMOD'
def generate(_model, prompt: str, params: dict):
    n = int(params.get("max_tokens") or params.get("num_predict") or 0)
    return f"GEN::{prompt}::n={n}"
def stream_generate(_model, prompt: str, params: dict):
    n = int(params.get("max_tokens") or params.get("num_predict") or 0)
    yield f"STREAM::{prompt}::n={n}"
def format_chat_prompt(messages, params=None):
    sys = "".join(m.get("content","") for m in messages if (m.get("role") or "").lower()=="system")
    lines=[]
    if sys.strip(): lines.append(f"System: {sys.strip()}")
    for m in messages:
        r=(m.get("role") or "user").lower(); c=(m.get("content") or "").strip()
        if r=="system": continue
        lines.append(("Assistant" if r=="assistant" else "User")+": "+c)
    lines.append("Assistant:")
    return "\n".join(lines)
def load_model(_): return object()
PYMOD
ok "wrote toy artifact"

# --- commit / verify ---
DIGEST="$("$PY" -m arknet_py.cli commit "$ARTIFACT" | head -n1)"
[[ ${#DIGEST} -eq 64 ]] || die "bad digest: $DIGEST"
ok "commit computed: $DIGEST"

"$PY" -m arknet_py.cli commit "$ARTIFACT" --write >/dev/null
ok "commit.json written"

OUT_OK="$("$PY" -m arknet_py.cli commit "$ARTIFACT" --verify "$DIGEST")"
[[ "$OUT_OK" == "OK" ]] || die "verify mismatch: $OUT_OK"
ok "commit verify OK"

# --- run tests ---
OUT1_GEN="$("$PY" -m arknet_py.cli run "$ARTIFACT" --prompt "hello" --max-tokens 10)"
[[ "$OUT1_GEN" == "GEN::hello::n=10" ]] || die "run non-stream mismatch: $OUT1_GEN"
ok "run (prompt, non-stream)"

OUT2_STR="$("$PY" -m arknet_py.cli run "$ARTIFACT" --prompt "ping" --max-tokens 9 --stream)"
[[ "$OUT2_STR" == "STREAM::ping::n=9" ]] || die "run stream mismatch: $OUT2_STR"
ok "run (prompt, stream)"

CHAT_FILE="$(mktemp -t chat.XXXX.json)"
cat >"$CHAT_FILE" <<'JSON'
[{"role":"system","content":"be terse"},{"role":"user","content":"hi"}]
JSON
OUT_CHAT="$("$PY" -m arknet_py.cli run "$ARTIFACT" --chat "$CHAT_FILE" --max-tokens 5)"
[[ "$OUT_CHAT" == GEN::* ]] || die "chat mismatch: $OUT_CHAT"
rm -f "$CHAT_FILE"
ok "run (chat, non-stream)"

# --- dataset for training (tiny JSONL) ---
cat >"$DATA" <<'JSONL'
{"text":"a"}
{"text":"b"}
{"text":"c"}
{"text":"d"}
{"text":"e"}
JSONL

DATA_CANON="$("$PY" - <<PY "$DATA"
import os, sys; print(os.path.realpath(sys.argv[1]))
PY
)"

# training spec: 3 steps, batch_size=2, zero LR (weights stable), single split
SPEC="$(mktemp -t spec.XXXX.json)"
cat >"$SPEC" <<JSON
{
  "seed": 123,
  "steps": 3,
  "batch_size": 2,
  "allow_tf32": false,
  "dataset": { "kind": "jsonl", "path": "${DATA_CANON}", "ratios": {"train": 1.0}, "split": "train" },
  "optimizer": {"name": "sgd", "lr": 0.0},
  "schedule": {}
}
JSON

# --- train twice with same spec -> identical commits ---
"$PY" -m arknet_py.cli train "$ARTIFACT" --spec "$SPEC" --out "$OUT1" >/dev/null
C1="$(jq -r .commit "$OUT1/commit.json")"

"$PY" -m arknet_py.cli train "$ARTIFACT" --spec "$SPEC" --out "$OUT2" >/dev/null
C2="$(jq -r .commit "$OUT2/commit.json")"

if [[ "$C1" != "$C2" ]]; then
  warn "trained commits differ with same seed:"
  echo "  C1: $C1"
  echo "  C2: $C2"
  diag_diff "$OUT1" "$OUT2"
  die "trained commit mismatch"
fi
ok "train determinism (same seed) -> identical commit"

# check transcript roots match
R1="$(jq -r .root "$OUT1/transcript.json")"
R2="$(jq -r .root "$OUT2/transcript.json")"
if [[ "$R1" != "$R2" || ${#R1} -ne 64 ]]; then
  warn "transcript roots mismatch: $R1 vs $R2"
  diag_diff "$OUT1" "$OUT2"
  die "transcript roots mismatch"
fi
ok "transcript root stable"

# --- change seed -> commit should change ---
SPEC2="$(mktemp -t spec2.XXXX.json)"
jq '.seed=789' "$SPEC" > "$SPEC2"

"$PY" -m arknet_py.cli train "$ARTIFACT" --spec "$SPEC2" --out "$OUT3" >/dev/null
C3="$(jq -r .commit "$OUT3/commit.json")"

if [[ "$C3" == "$C1" ]]; then
  warn "commit should differ for different seed, but didn't"
  diag_diff "$OUT1" "$OUT3"
  if [[ -f "$OUT1/spec.public.json" && -f "$OUT3/spec.public.json" ]]; then
    echo "[*] spec.public.json sha256:"
    hash_cmd "$OUT1/spec.public.json" "$OUT3/spec.public.json" || true
  fi
  if [[ -f "$OUT1/spec.hash.txt" && -f "$OUT3/spec.hash.txt" ]]; then
    echo "[*] spec.hash.txt values:"
    echo "  OUT1: $(cat "$OUT1/spec.hash.txt" 2>/dev/null || true)"
    echo "  OUT3: $(cat "$OUT3/spec.hash.txt" 2>/dev/null || true)"
  fi
  die "different-seed commit mismatch"
fi
ok "train (different seed) -> different commit"

ok "ALL STAGES PASSED (with training)"
