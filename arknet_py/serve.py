# arknet_py/serve.py
from __future__ import annotations
import asyncio, base64, hashlib, importlib, os, signal, struct, sys, time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union
from .compblob import Decompressor, b64 as _b64
from .compblob import build_getz_prelude  # server-side pre-compress + header

# ---- JSON (fast if available) ----------------------------------------------
try:
    import orjson as _json  # type: ignore
    _loads = _json.loads
    def _dumps(x: Any) -> bytes: return _json.dumps(x)
except Exception:
    import json as _json
    _loads = _json.loads
    def _dumps(x: Any) -> bytes: return _json.dumps(x, separators=(",", ":"), ensure_ascii=False).encode()

# ---- protocol constants -----------------------------------------------------
PROTO_VERSION = 1
MAX_FRAME = 64 * 1024 * 1024
REQ_TIMEOUT_S = 60.0
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 9763
MAX_BLOB = 8 * 1024 * 1024 * 1024  # 8 GiB ceiling for safety
BLOB_CHUNK = 4 * 1024 * 1024       # 4 MiB streaming chunks

# ---- config ----------------------------------------------------------------
@dataclass(slots=True)
class ServeConfig:
    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    uds: Optional[str] = None
    rt_module: str = "arknet_py.rt"
    artifact: Optional[str] = None
    token: Optional[str] = None
    max_inflight: int = 64
    req_timeout_s: float = REQ_TIMEOUT_S
    allow_remote: bool = False

# ---- errors ----------------------------------------------------------------
class RpcError(Exception):
    __slots__ = ("code", "msg")
    def __init__(self, code: str, msg: str = ""):
        super().__init__(msg or code)
        self.code, self.msg = code, (msg or code)

# ---- helpers ----------------------------------------------------------------
def _b64dec(s: str) -> bytes:
    try:
        return base64.b64decode(s, validate=True)
    except Exception as e:
        raise RpcError("bad_base64", str(e))

def _now() -> float:
    return time.time()

# sentinel to signal "already sent response + streamed extra bytes"
class _Streamed: pass
STREAMED = _Streamed()

# ---- server -----------------------------------------------------------------
class RpcServer:
    def __init__(self, cfg: ServeConfig):
        self.cfg = cfg
        self.started_at = _now()
        self.sem = asyncio.Semaphore(cfg.max_inflight)
        self.rt = self._load_rt(cfg.rt_module, cfg.artifact)
        self._artifact_cache: Tuple[float, Optional[str], Optional[Dict[str, Any]]] = (0.0, None, None)

    def _load_rt(self, modname: str, artifact: Optional[str]):
        try:
            mod = importlib.import_module(modname)
        except ModuleNotFoundError:
            from . import rt as mod
        if not hasattr(mod, "train_step"):
            raise RuntimeError(f"{mod.__name__} must expose train_step(state:bytes,batch:bytes)->bytes")
        if hasattr(mod, "init"):
            try: mod.init(artifact)
            except Exception: pass
        return mod

    async def handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        try:
            while True:
                hdr = await reader.readexactly(4)
                n = struct.unpack(">I", hdr)[0]
                if n > MAX_FRAME:
                    raise RpcError("frame_too_large", str(n))
                body = await reader.readexactly(n)

                # token presence (cheap pre-check)
                if self.cfg.token and b"\"t\"" not in body:
                    await self._send(writer, {"v": PROTO_VERSION, "id": None, "ok": False,
                                              "error": {"code": "no_token", "message": "missing t"}})
                    continue
                try:
                    req = _loads(body)
                except Exception as e:
                    await self._send(writer, {"v": PROTO_VERSION, "id": None, "ok": False,
                                              "error": {"code": "bad_json", "message": str(e)}})
                    continue

                v = int(req.get("v", 0))
                rid = req.get("id")
                method = req.get("method")
                params = req.get("params") or {}
                tok = req.get("t")

                if v != PROTO_VERSION:
                    await self._send(writer, {"v": PROTO_VERSION, "id": rid, "ok": False,
                                              "error": {"code": "bad_version", "message": str(v)}})
                    continue
                if self.cfg.token and tok != self.cfg.token:
                    await self._send(writer, {"v": PROTO_VERSION, "id": rid, "ok": False,
                                              "error": {"code": "bad_token", "message": "unauthorized"}})
                    continue

                try:
                    async with self.sem:
                        # pass reader/writer so methods can stream
                        result = await asyncio.wait_for(self._dispatch(method, params, reader, writer),
                                                        timeout=self.cfg.req_timeout_s)
                    if result is STREAMED:
                        # method already sent response and streamed payload
                        continue
                    resp = {"v": PROTO_VERSION, "id": rid, "ok": True, "result": result}
                except asyncio.TimeoutError:
                    resp = {"v": PROTO_VERSION, "id": rid, "ok": False,
                            "error": {"code": "timeout", "message": "request timed out"}}
                except RpcError as re:
                    resp = {"v": PROTO_VERSION, "id": rid, "ok": False,
                            "error": {"code": re.code, "message": re.msg}}
                except Exception as e:
                    resp = {"v": PROTO_VERSION, "id": rid, "ok": False,
                            "error": {"code": "internal", "message": str(e)}}

                await self._send(writer, resp)
        except (asyncio.IncompleteReadError, ConnectionResetError):
            pass
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass

    async def _send(self, writer: asyncio.StreamWriter, obj: Dict[str, Any]) -> None:
        blob = _dumps(obj)
        writer.write(struct.pack(">I", len(blob)))
        writer.write(blob)
        await writer.drain()

    # ---- RPC dispatch ------------------------------------------------------
    async def _dispatch(
        self,
        method: str,
        p: Dict[str, Any],
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> Union[Dict[str, Any], _Streamed]:

        if method == "ping":
            return {"pong": True, "uptime_s": int(_now() - self.started_at)}

        if method == "healthz":
            from .determinism import determinism_report
            rep = determinism_report()
            art = await self._artifact_info_cached()
            return {
                "version": PROTO_VERSION,
                "env": rep,
                "artifact": art,
                "caps": {
                    "methods": ["ping", "healthz", "train.step", "blob.putz", "blob.getz"],
                    "max_frame": MAX_FRAME,
                    "max_blob": MAX_BLOB,
                },
            }

        if method == "train.step":
            if not isinstance(p, dict):
                raise RpcError("bad_params", "params must be object")
            try:
                state = _b64dec(p["state"])
                batch = _b64dec(p["batch"])
            except KeyError as ke:
                raise RpcError("bad_params", f"missing {ke.args[0]}")
            t0 = _now()
            out = self.rt.train_step(state, batch)
            if not isinstance(out, (bytes, bytearray)):
                raise RpcError("train_step_nonbytes")
            return {
                "state": base64.b64encode(out).decode("ascii"),
                "dur_ms": int((_now() - t0) * 1000),
            }

        if method == "blob.putz":
            return await self._recv_blobz(p, reader)

        if method == "blob.getz":
            await self._send_blob_getz(p, writer)
            return STREAMED

        raise RpcError("unknown_method", str(method))

    # ---- blob.putz (receive compressed → decompress → write file) ----------
    async def _recv_blobz(self, p: Dict[str, Any], reader: asyncio.StreamReader) -> Dict[str, Any]:
        if not isinstance(p, dict): raise RpcError("bad_params", "params must be object")
        codec = str(p.get("codec") or "zstd")
        level = int(p.get("level") or 3)
        raw_size = int(p["raw_size"])
        csize = int(p["csize"])
        if raw_size < 0 or csize < 0 or raw_size > MAX_BLOB or csize > MAX_BLOB:
            raise RpcError("bad_size", f"raw={raw_size} c={csize}")
        path = p.get("path")
        mkdirs = bool(p.get("mkdirs", False))
        want_sha = bool(p.get("sha256", False))

        if path:
            d = os.path.dirname(path)
            if d and mkdirs:
                os.makedirs(d, exist_ok=True)

        dec = Decompressor(codec)  # level unused at decode; retained for audit
        h = hashlib.sha256() if want_sha else None

        # precompute mid indexes
        mid_lo = (raw_size - 1) // 2 if raw_size > 0 else 0
        mid_hi = raw_size // 2 if raw_size > 1 else 0
        mid_lo_b = None
        mid_hi_b = None

        remaining = csize
        out_pos = 0
        chunk = BLOB_CHUNK
        f = None
        try:
            if path:
                f = open(path, "wb", buffering=0)
            while remaining:
                to_read = chunk if remaining > chunk else remaining
                buf = await reader.readexactly(to_read)
                remaining -= len(buf)
                decomp = dec.feed(buf)
                if decomp:
                    if f: f.write(decomp)
                    if h: h.update(decomp)
                    if raw_size:
                        if mid_lo_b is None and out_pos <= mid_lo < out_pos + len(decomp):
                            mid_lo_b = decomp[mid_lo - out_pos]
                        if raw_size > 1 and mid_hi_b is None and out_pos <= mid_hi < out_pos + len(decomp):
                            mid_hi_b = decomp[mid_hi - out_pos]
                    out_pos += len(decomp)
            tail = dec.flush()
            if tail:
                if f: f.write(tail)
                if h: h.update(tail)
                if raw_size:
                    if mid_lo_b is None and out_pos <= mid_lo < out_pos + len(tail):
                        mid_lo_b = tail[mid_lo - out_pos]
                    if raw_size > 1 and mid_hi_b is None and out_pos <= mid_hi < out_pos + len(tail):
                        mid_hi_b = tail[mid_hi - out_pos]
                out_pos += len(tail)
        finally:
            if f:
                try: f.close()
                except Exception: pass

        if out_pos != raw_size:
            raise RpcError("size_mismatch", f"{out_pos}!={raw_size}")

        if raw_size == 0:
            mid = b""
        elif raw_size == 1:
            mid = bytes([mid_lo_b if mid_lo_b is not None else 0])
        else:
            mid = bytes([
                mid_lo_b if mid_lo_b is not None else 0,
                mid_hi_b if mid_hi_b is not None else 0
            ])

        res = {
            "bytes": out_pos,
            "cbytes": csize,
            "ratio": (float(csize)/float(out_pos) if out_pos else 1.0),
            "mid2": _b64(mid),
            "saved": bool(path),
            "codec": codec,
            "level": level,
        }
        if h:
            res["sha256"] = h.hexdigest()
        return res

    # ---- blob.getz (prelude JSON frame → raw csize bytes) -------------------
    async def _send_blob_getz(self, p: Dict[str, Any], writer: asyncio.StreamWriter) -> None:
        if not isinstance(p, dict): raise RpcError("bad_params", "params must be object")
        path = p.get("path")
        if not path or not os.path.isfile(path):
            raise RpcError("not_found", "path missing or not a file")

        # Pre-compress to temp to know csize, include mid2 + sha256 for client verification
        pre = build_getz_prelude(
            path,
            codec="zstd",
            level=3,
            include_mid2=True,
            include_sha256=True,
        )
        header = pre.header
        csize = int(pre.comp_size)
        tmp = pre.tmp_path

        # send JSON prelude (framed)
        await self._send(writer, {"v": PROTO_VERSION, "id": "blob.getz", "ok": True, "result": header})

        # then stream exactly csize raw bytes
        try:
            with open(tmp, "rb", buffering=0) as f:
                # Use buffered chunks (cross-platform, loop-friendly)
                left = csize
                while left:
                    chunk = f.read(min(BLOB_CHUNK, left))
                    if not chunk:
                        raise RpcError("short_read_src", f"read<{left}")
                    writer.write(chunk)
                    left -= len(chunk)
                    if writer.transport and writer.transport.is_closing():
                        break
                    if left <= 0 or writer.transport.get_write_buffer_size() > (8 * BLOB_CHUNK):
                        await writer.drain()
                await writer.drain()
        finally:
            try: os.remove(tmp)
            except Exception: pass

    # ---- artifact cache -----------------------------------------------------
    async def _artifact_info_cached(self) -> Optional[Dict[str, Any]]:
        if not self.cfg.artifact:
            return None
        ts, dig, mf = self._artifact_cache
        if (_now() - ts) < 3.0 and dig and mf:
            return {"commit": dig, "manifest": mf}
        from .commit import compute_commit, load_commit_manifest
        dig, _ = compute_commit(self.cfg.artifact)
        mf = load_commit_manifest(self.cfg.artifact)
        self._artifact_cache = (_now(), dig, mf)
        return {"commit": dig, "manifest": mf}

# ---- listeners ---------------------------------------------------------------
async def _serve_tcp(srv: RpcServer, cfg: ServeConfig):
    if not cfg.allow_remote and cfg.host not in ("127.0.0.1", "::1", "localhost"):
        raise RuntimeError("refusing non-local bind without --allow-remote")
    server = await asyncio.start_server(srv.handle, cfg.host, cfg.port, reuse_address=True)
    return server

async def _serve_uds(srv: RpcServer, path: str):
    try: os.unlink(path)
    except FileNotFoundError: pass
    server = await asyncio.start_unix_server(srv.handle, path)
    try: os.chmod(path, 0o600)
    except Exception: pass
    return server

# ---- entrypoint (module callable) -------------------------------------------
async def run_server(cfg: ServeConfig) -> None:
    srv = RpcServer(cfg)
    server = await (_serve_uds(srv, cfg.uds) if cfg.uds else _serve_tcp(srv, cfg))

    # visible readiness for tests
    addrs = []
    if server.sockets:
        for s in server.sockets:
            try: addrs.append(s.getsockname())
            except Exception: pass
    print(f"[arkserve] ready {addrs}", file=sys.stderr, flush=True)

    loop = asyncio.get_running_loop()
    stop = asyncio.Event()
    def _grace(*_): stop.set()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try: loop.add_signal_handler(sig, _grace)
        except NotImplementedError: pass

    async with server:
        waiter = asyncio.create_task(server.serve_forever())
        stopper = asyncio.create_task(stop.wait())
        done, _ = await asyncio.wait({waiter, stopper}, return_when=asyncio.FIRST_COMPLETED)
        for t in (waiter, stopper):
            if not t.done(): t.cancel()
        server.close(); await server.wait_closed()

__all__ = ["ServeConfig", "run_server", "DEFAULT_HOST", "DEFAULT_PORT", "REQ_TIMEOUT_S"]
