# arknet_py/compblob.py
from __future__ import annotations
import base64, hashlib, os, tempfile, zlib
from dataclasses import dataclass
from typing import Generator, Iterable, Literal, Optional, Tuple

Codec = Literal["zstd", "gzip"]

try:
    import zstandard as zstd  # type: ignore
    _HAS_ZSTD = True
except Exception:
    _HAS_ZSTD = False

# ---------- tiny primitives ----------

def file_size(path: str) -> int:
    return int(os.stat(path).st_size)

def mid2_bytes(path: str, size: Optional[int] = None) -> bytes:
    n = file_size(path) if size is None else int(size)
    if n == 0:
        return b""
    if n == 1:
        with open(path, "rb", buffering=0) as f:
            f.seek(0); return f.read(1)
    lo = (n - 1) // 2
    hi = n // 2
    with open(path, "rb", buffering=0) as f:
        f.seek(lo); b0 = f.read(1) or b"\x00"
        f.seek(hi); b1 = f.read(1) or b"\x00"
    return b0 + b1

def sha256_file(path: str, chunk_sz: int = 1<<20) -> str:
    h = hashlib.sha256()
    with open(path, "rb", buffering=0) as f:
        for chunk in iter(lambda: f.read(chunk_sz), b""):
            h.update(chunk)
    return h.hexdigest()

def b64(x: bytes) -> str:
    return base64.b64encode(x).decode("ascii")

# ---------- compression (to temp or streaming) ----------

@dataclass(slots=True)
class CompressResult:
    tmp_path: str
    raw_size: int
    comp_size: int
    codec: Codec
    level: int
    mid2_b64: Optional[str] = None
    sha256_hex: Optional[str] = None

def compress_to_temp(
    path: str,
    *,
    codec: Codec = "zstd",
    level: int = 3,
    chunk_sz: int = 1<<20,
    tmpdir: Optional[str] = None,
    want_mid2: bool = False,
    want_sha256: bool = False,
) -> CompressResult:
    raw_size = file_size(path)
    os.makedirs(tmpdir or tempfile.gettempdir(), exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix="arkblob-", suffix=f".{codec}", dir=tmpdir)
    os.close(fd)
    mid2_b64 = b64(mid2_bytes(path, raw_size)) if want_mid2 else None
    sha_hex = sha256_file(path) if want_sha256 else None
    try:
        if codec == "zstd":
            if not _HAS_ZSTD:
                raise RuntimeError("zstd codec requested but python-zstandard not installed")
            cctx = zstd.ZstdCompressor(level=level, write_content_size=False)
            with open(path, "rb", buffering=0) as fin, open(tmp_path, "wb", buffering=0) as fout:
                with cctx.stream_writer(fout) as zw:
                    for chunk in iter(lambda: fin.read(chunk_sz), b""):
                        zw.write(chunk)
        elif codec == "gzip":
            import gzip
            with open(path, "rb", buffering=0) as fin, gzip.GzipFile(tmp_path, "wb", compresslevel=level, mtime=0) as fout:
                for chunk in iter(lambda: fin.read(chunk_sz), b""):
                    fout.write(chunk)
        else:
            raise ValueError("unsupported codec")
    except Exception:
        try: os.remove(tmp_path)
        except Exception: pass
        raise
    comp_size = file_size(tmp_path)
    return CompressResult(
        tmp_path=tmp_path,
        raw_size=raw_size,
        comp_size=comp_size,
        codec=codec,
        level=level,
        mid2_b64=mid2_b64,
        sha256_hex=sha_hex,
    )

# streaming compressors (for future chunked framing; not required by fixed-length prelude)

class _ZstdStreamWriter:
    __slots__ = ("_cctx", "_buf", "_level")
    def __init__(self, level: int = 3):
        if not _HAS_ZSTD:
            raise RuntimeError("zstd codec requested but python-zstandard not installed")
        self._cctx = zstd.ZstdCompressor(level=level, write_content_size=False)
        self._buf = bytearray()
        self._level = level
    def feed(self, data: bytes) -> bytes:
        return self._cctx.compress(data)
    def flush(self) -> bytes:
        # zstandard one-shot compressor already returns final frames per feed; nothing to flush.
        return b""

class _GzipStreamWriter:
    __slots__ = ("_obj",)
    def __init__(self, level: int = 6):
        self._obj = zlib.compressobj(level=level, wbits=16 + zlib.MAX_WBITS)
    def feed(self, data: bytes) -> bytes:
        return self._obj.compress(data)
    def flush(self) -> bytes:
        return self._obj.flush(zlib.Z_FINISH)

def streaming_compressor(codec: Codec, level: int = 3):
    if codec == "zstd":
        return _ZstdStreamWriter(level=level)
    if codec == "gzip":
        return _GzipStreamWriter(level=level)
    raise ValueError("unsupported codec")

def iter_file(path: str, chunk_sz: int = 1<<20) -> Iterable[bytes]:
    with open(path, "rb", buffering=0) as f:
        for chunk in iter(lambda: f.read(chunk_sz), b""):
            yield chunk

# ---------- streaming decompression (server/client shared) ----------

class Decompressor:
    __slots__ = ("codec", "_obj")
    def __init__(self, codec: Codec):
        self.codec = codec
        if codec == "zstd":
            if not _HAS_ZSTD:
                raise RuntimeError("zstd codec requested but python-zstandard not installed")
            self._obj = zstd.ZstdDecompressor().decompressobj()
        elif codec == "gzip":
            self._obj = zlib.decompressobj(wbits=16 + zlib.MAX_WBITS)
        else:
            raise ValueError("unsupported codec")

    def feed(self, data: bytes) -> bytes:
        return self._obj.decompress(data)

    def flush(self) -> bytes:
        try:
            return self._obj.flush()
        except Exception:
            return b""

# ---------- server helpers for blob.getz ----------

@dataclass(slots=True)
class GetzPrelude:
    header: dict
    tmp_path: str
    comp_size: int

def build_getz_prelude(
    path: str,
    *,
    codec: Codec = "zstd",
    level: int = 3,
    chunk_sz: int = 1<<20,
    tmpdir: Optional[str] = None,
    include_mid2: bool = True,
    include_sha256: bool = False,
) -> GetzPrelude:
    """
    Prepare a JSON prelude for server-side blob.getz with fixed-size payload:
      - compresses source file to a temporary file to know csize
      - returns header {codec, level, raw_size, csize, [mid2], [sha256]}
      - tmp_path points at the compressed payload to stream after the prelude
    """
    cr = compress_to_temp(
        path,
        codec=codec,
        level=level,
        chunk_sz=chunk_sz,
        tmpdir=tmpdir,
        want_mid2=include_mid2,
        want_sha256=include_sha256,
    )
    header: dict = {
        "codec": cr.codec,
        "level": cr.level,
        "raw_size": cr.raw_size,
        "csize": cr.comp_size,
    }
    if include_mid2 and cr.mid2_b64 is not None:
        header["mid2"] = cr.mid2_b64
    if include_sha256 and cr.sha256_hex is not None:
        header["sha256"] = cr.sha256_hex
    return GetzPrelude(header=header, tmp_path=cr.tmp_path, comp_size=cr.comp_size)
