#!/usr/bin/env bash
# scripts/test_arknet_serve.sh — e2e test for arknet-py serve (Windows/MSYS-safe)
set -Eeuo pipefail

# --- config ---
if command -v python3 >/dev/null 2>&1; then PY=python3; else PY=python; fi
export PYTHONUNBUFFERED=1
export PYTHONPATH="${PYTHONPATH:-$PWD}"

HOST=${ARK_SERVE_HOST:-127.0.0.1}
PORT=${ARK_SERVE_PORT:-0}
TOKEN=${ARK_SERVE_TOKEN:-}
TIMEOUT_S=${TIMEOUT_S:-20}
LOG="${LOG:-/tmp/arknet_serve_test.log}"

die(){ echo "ERR: $*" >&2; exit 2; }
log(){ echo "-- $*"; }

need_py(){
  "$PY" - <<'PY' || exit 1
import importlib
for m in ("arknet_py","arknet_py.cli","arknet_py.serve"): importlib.import_module(m)
print("IMPORT_OK")
PY
}

free_port(){
  "$PY" - <<'PY'
import socket
s=socket.socket(socket.AF_INET,socket.SOCK_STREAM); s.bind(('127.0.0.1',0))
print(s.getsockname()[1]); s.close()
PY
}

# -------- RPC helpers (no JSON-in-argv) --------------------------------------
rpc_call_simple(){  # method
  local method="$1"
  "$PY" - "$HOST" "$PORT" "$TOKEN" "$method" <<'PY'
import sys, json, socket, struct
def recv_exact(s,n):
    b=bytearray()
    while len(b)<n:
        c=s.recv(n-len(b))
        if not c: raise RuntimeError(f"short_read:{len(b)}/{n}")
        b+=c
    return bytes(b)
host,port,tok,method = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
req={"v":1,"id":method,"method":method,"params":{}}
if tok: req["t"]=tok
buf=json.dumps(req,separators=(",",":")).encode()
s=socket.socket(socket.AF_INET,socket.SOCK_STREAM); s.settimeout(5.0); s.connect((host,port))
s.sendall(struct.pack(">I",len(buf))+buf)
hdr=recv_exact(s,4); n=struct.unpack(">I",hdr)[0]; body=recv_exact(s,n); s.close()
sys.stdout.write(body.decode("utf-8","replace"))
PY
}

rpc_train_step(){  # S_b64 B_b64
  local S="$1" B="$2"
  "$PY" - "$HOST" "$PORT" "$TOKEN" "$S" "$B" <<'PY'
import sys, json, socket, struct
def recv_exact(s,n):
    b=bytearray()
    while len(b)<n:
        c=s.recv(n-len(b))
        if not c: raise RuntimeError(f"short_read:{len(b)}/{n}")
        b+=c
    return bytes(b)
host,port,tok,S,B = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5]
req={"v":1,"id":"train.step","method":"train.step","params":{"state":S,"batch":B}}
if tok: req["t"]=tok
buf=json.dumps(req,separators=(",",":")).encode()
s=socket.socket(socket.AF_INET,socket.SOCK_STREAM); s.settimeout(5.0); s.connect((host,port))
s.sendall(struct.pack(">I",len(buf))+buf)
hdr=recv_exact(s,4); n=struct.unpack(">I",hdr)[0]; body=recv_exact(s,n); s.close()
sys.stdout.write(body.decode("utf-8","replace"))
PY
}

port_accepts(){
  "$PY" - "$HOST" "$PORT" <<'PY' >/dev/null 2>&1
import sys,socket
s=socket.socket(); s.settimeout(0.5)
try: s.connect((sys.argv[1],int(sys.argv[2]))); ok=True
except Exception: ok=False
s.close(); sys.exit(0 if ok else 1)
PY
}

await_up(){
  local deadline=$((SECONDS + TIMEOUT_S))
  while (( SECONDS < deadline )); do
    if port_accepts; then
      r="$(rpc_call_simple ping || true)"
      if grep -q '"ok":true' <<<"$r"; then return 0; fi
    fi
    sleep 0.2
  done
  return 1
}

# --- preflight ---
log "checking Python modules..."
need_py || die "arknet_py import failed"
[[ "$PORT" = "0" ]] && PORT="$(free_port)"
[[ -n "$PORT" ]] || die "failed to pick free port"

ART="$(mktemp -d -t arkserve-art-XXXXXX)"
trap 'kill "$SERVE_PID" 2>/dev/null || true; rm -rf "$ART" "$LOG"' EXIT
cat >"$ART/manifest.json" <<JSON
{ "name": "arknet-serve-test", "format_version": 1 }
JSON
cat >"$ART/model.py" <<'PY'
def generate(model, prompt, params): return "ok"
PY

# --- start server ---
log "starting server on $HOST:$PORT (artifact: $ART) …"
if [[ -n "$TOKEN" ]]; then
  "$PY" -m arknet_py.cli serve --host "$HOST" --port "$PORT" --artifact "$ART" --token "$TOKEN" >"$LOG" 2>&1 &
else
  "$PY" -m arknet_py.cli serve --host "$HOST" --port "$PORT" --artifact "$ART" >"$LOG" 2>&1 &
fi
SERVE_PID=$!

# --- readiness ---
log "waiting for readiness…"
if ! await_up; then
  echo "==== serve log ====" >&2
  { command -v tail >/dev/null 2>&1 && tail -n +200 "$LOG" || sed -n '1,200p' "$LOG"; } >&2 || true
  die "server did not become ready"
fi

# --- tests ---
pass=0; fail=0
run(){ local name="$1"; if "$name"; then log "✓ $name"; pass=$((pass+1)); else log "✗ $name"; fail=$((fail+1)); fi; }

test_ping(){ r="$(rpc_call_simple ping)"; grep -q '"ok":true' <<<"$r" && grep -q '"pong":true' <<<"$r"; }
test_healthz(){ r="$(rpc_call_simple healthz)"; grep -q '"ok":true' <<<"$r" && grep -q '"version":1' <<<"$r"; }
test_train_step(){
  local S B r
  S="$("$PY" - <<'PY'
import base64, binascii; print(base64.b64encode(binascii.unhexlify("000102030405")).decode())
PY
)"; B="$("$PY" - <<'PY'
import base64, binascii; print(base64.b64encode(binascii.unhexlify("0a0b0c")).decode())
PY
)"
  r="$(rpc_train_step "$S" "$B")"
  grep -q '"ok":true' <<<"$r" && grep -q '"state":"' <<<"$r"
}

run test_ping
run test_healthz
run test_train_step

if (( fail == 0 )); then
  log "ALL TESTS PASSED ($pass)"
  exit 0
else
  log "$fail test(s) failed ($pass passed) — see $LOG"
  exit 1
fi
