# arknet_py/fastblob.py
from __future__ import annotations
import os, socket, struct, hashlib
from dataclasses import dataclass
from typing import Any, Dict, Optional, Literal
from .compblob import compress_to_temp, mid2_bytes, sha256_file, b64, Decompressor

try:
    import orjson as _json  # type: ignore
    _loads = _json.loads
    def _dumps(x: Any) -> bytes: return _json.dumps(x)
except Exception:
    import json as _json
    _loads = _json.loads
    def _dumps(x: Any) -> bytes: return _json.dumps(x, separators=(",", ":"), ensure_ascii=False).encode()

VerifyMode = Literal["none", "mid2", "sha256"]

@dataclass(slots=True)
class BlobClientConfig:
    host: str = "127.0.0.1"
    port: int = 9763
    token: Optional[str] = None
    timeout_s: float = 120.0
    chunk_sz: int = 1 << 20  # 1 MiB

class BlobClient:
    def __init__(self, cfg: BlobClientConfig):
        self.cfg = cfg
        self._s: Optional[socket.socket] = None

    def __enter__(self) -> "BlobClient":
        self.connect(); return self

    def __exit__(self, et, ev, tb) -> None:
        self.close()

    # ---- socket lifecycle ----
    def connect(self) -> None:
        if self._s: return
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        s.settimeout(self.cfg.timeout_s)
        s.connect((self.cfg.host, self.cfg.port))
        self._s = s

    def close(self) -> None:
        s, self._s = self._s, None
        if s:
            try: s.shutdown(socket.SHUT_RDWR)
            except Exception: pass
            try: s.close()
            except Exception: pass

    # ---- framing ----
    def _send_json(self, obj: Dict[str, Any]) -> None:
        assert self._s is not None
        body = _dumps(obj)
        self._s.sendall(struct.pack(">I", len(body)))
        self._s.sendall(body)

    def _recv_exact(self, n: int) -> bytes:
        assert self._s is not None
        buf = bytearray()
        view = memoryview(buf)
        while len(buf) < n:
            chunk = self._s.recv(n - len(buf))
            if not chunk: raise RuntimeError(f"short_read:{len(buf)}/{n}")
            buf += chunk
            view = memoryview(buf)
        return bytes(view)

    def _recv_json(self) -> Dict[str, Any]:
        n = struct.unpack(">I", self._recv_exact(4))[0]
        body = self._recv_exact(n)
        return _loads(body)

    # ---------- compressed PUT (blob.putz) ----------
    def put_file_z(
        self,
        local_path: str,
        remote_path: Optional[str] = None,
        *,
        codec: str = "zstd",
        level: int = 3,
        mkdirs: bool = True,
        verify: VerifyMode = "mid2",
    ) -> Dict[str, Any]:
        cr = compress_to_temp(local_path, codec=codec, level=level)
        # compute verification on the raw source once (O(1) I/O for mid2, O(n) for sha256)
        raw_mid2 = b64(mid2_bytes(local_path, cr.raw_size)) if verify in ("mid2", "sha256") else None
        raw_sha  = sha256_file(local_path) if verify == "sha256" else None

        req = {
            "v": 1, "id": "blob.putz", "method": "blob.putz",
            "params": {
                "codec": cr.codec, "level": cr.level,
                "raw_size": cr.raw_size, "csize": cr.comp_size,
                "path": remote_path, "mkdirs": mkdirs,
                "sha256": (verify == "sha256"),
            },
        }
        if self.cfg.token: req["t"] = self.cfg.token
        self._send_json(req)

        # stream compressed payload (use sendfile when available for large files)
        assert self._s is not None
        with open(cr.tmp_path, "rb", buffering=0) as f:
            sendfile = getattr(self._s, "sendfile", None)
            if callable(sendfile):
                offset, remaining = 0, cr.comp_size
                # Some platforms return None (fallback); keep robust loop.
                while remaining:
                    sent = sendfile(f, offset, remaining)
                    if not sent:
                        break
                    offset += sent
                    remaining -= sent
                if remaining:
                    f.seek(offset)
                    for chunk in iter(lambda: f.read(self.cfg.chunk_sz), b""):
                        self._s.sendall(chunk)
            else:
                for chunk in iter(lambda: f.read(self.cfg.chunk_sz), b""):
                    self._s.sendall(chunk)

        # response + verification
        resp = self._recv_json()
        if not resp.get("ok", False):
            err = resp.get("error") or {}
            raise RuntimeError(f"rpc_error:{err.get('code')}:{err.get('message')}")
        out = resp["result"]

        if verify == "mid2" and raw_mid2 and out.get("mid2") != raw_mid2:
            raise RuntimeError("mid2_mismatch")
        if verify == "sha256" and raw_sha and out.get("sha256") != raw_sha:
            raise RuntimeError("sha256_mismatch")

        try: os.remove(cr.tmp_path)
        except Exception: pass
        return out

    # ---------- compressed GET (blob.getz) ----------
    def get_file_z(
        self,
        remote_path: str,
        local_path: str,
        *,
        mkdirs: bool = True,
        fsync: bool = True,
        verify: VerifyMode = "mid2",
    ) -> Dict[str, Any]:
        req = {
            "v": 1, "id": "blob.getz", "method": "blob.getz",
            "params": { "path": remote_path },
        }
        if self.cfg.token: req["t"] = self.cfg.token
        self._send_json(req)

        # prelude (JSON frame) with codec/raw_size/csize and optional mid2/sha256
        resp = self._recv_json()
        if not resp.get("ok", False):
            err = resp.get("error") or {}
            raise RuntimeError(f"rpc_error:{err.get('code')}:{err.get('message')}")
        hdr = resp["result"]
        codec = hdr["codec"]
        raw_size = int(hdr["raw_size"])
        csize = int(hdr["csize"])
        enc_mid2 = hdr.get("mid2")
        enc_sha  = hdr.get("sha256")

        # ensure dirs
        if mkdirs:
            d = os.path.dirname(os.path.abspath(local_path))
            if d and not os.path.isdir(d):
                os.makedirs(d, exist_ok=True)

        # stream csize bytes → decompress → write local_path
        dec = Decompressor(codec)  # zstd or gzip
        h = hashlib.sha256() if verify == "sha256" or (enc_sha and verify != "none") else None
        wrote = 0

        assert self._s is not None
        with open(local_path, "wb", buffering=0) as fout:
            left = csize
            while left:
                to = min(self.cfg.chunk_sz, left)
                chunk = self._s.recv(to)
                if not chunk:
                    raise RuntimeError("short_read_payload")
                left -= len(chunk)
                out = dec.feed(chunk)
                if out:
                    if h: h.update(out)
                    fout.write(out)
                    wrote += len(out)
            # flush any tail from decompressor
            tail = dec.flush()
            if tail:
                if h: h.update(tail)
                fout.write(tail)
                wrote += len(tail)
            if fsync:
                try: os.fsync(fout.fileno())
                except Exception: pass

        # size check + verify
        if raw_size and wrote != raw_size:
            raise RuntimeError(f"size_mismatch:{wrote}!={raw_size}")

        if verify == "mid2":
            want_mid2 = enc_mid2  # server prelude supplied expected mid2
            have_mid2 = b64(mid2_bytes(local_path, wrote))
            if want_mid2 and have_mid2 != want_mid2:
                raise RuntimeError("mid2_mismatch")
        elif verify == "sha256":
            have_sha = (h.hexdigest() if h else sha256_file(local_path))
            want_sha = enc_sha
            if want_sha and have_sha != want_sha:
                raise RuntimeError("sha256_mismatch")

        return {
            "path": local_path,
            "bytes": wrote,
            "codec": codec,
            "cbytes": csize,
            "mid2": b64(mid2_bytes(local_path, wrote)) if wrote else "",
            "sha256": (h.hexdigest() if h else None),
        }
