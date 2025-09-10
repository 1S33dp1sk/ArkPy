#!/usr/bin/env python3
from __future__ import annotations
import argparse, ast, builtins, importlib, importlib.util, json, os, sys
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set, Tuple

# ---------- config ----------
DEFAULT_PACKAGE = "arknet_py"
DEFAULT_PROBES = [
    "import arknet_py.determinism as d; d.apply_determinism_profile(seed=0, allow_tf32=False, strict_ops=True)",  # numpy/torch/tf if present
    "import arknet_py.commit as c; getattr(c,'compute_commit',lambda *a,**k:None)",                                # orjson if used
    # add more if needed:
    # "import arknet_py.runner as r; getattr(r,'ArknetModel',None)",
    # "import arknet_py.serve as s; getattr(s,'ServeConfig',None)",
]
PYPI_HINTS: Dict[str, str] = {
    "PIL": "Pillow", "yaml": "PyYAML", "sklearn": "scikit-learn",
    "cv2": "opencv-python", "orjson":"orjson", "ujson":"ujson",
    "zstandard":"zstandard", "zstd":"zstandard",
    "numpy":"numpy", "scipy":"scipy", "pandas":"pandas",
    "torch":"torch", "tensorflow":"tensorflow", "jax":"jax", "jaxlib":"jaxlib",
    "requests":"requests", "tqdm":"tqdm", "cryptography":"cryptography", "protobuf":"protobuf",
}

try:
    STDLIB: Set[str] = set(sys.stdlib_module_names)  # py311+
except Exception:
    STDLIB = set()

@dataclass
class Report:
    required: Set[str]
    optional: Set[str]
    runtime: Set[str]
    third_party: Set[str]

def _walk_py(root: str, package_dir: str) -> Iterable[str]:
    for base, dirs, files in os.walk(os.path.join(root, package_dir)):
        dirs[:] = [d for d in dirs if d not in (".git", ".venv", "venv", "__pycache__", "build", "dist")]
        for f in files:
            if f.endswith(".py"):
                yield os.path.join(base, f)

def _mod_top(name: str) -> str:
    return name.split(".", 1)[0]

def _is_stdlib(mod: str) -> bool:
    if mod in STDLIB: return True
    try:
        spec = importlib.util.find_spec(mod)
    except Exception:
        return False
    if spec is None or spec.origin is None:
        return False
    origin = str(spec.origin).lower()
    if origin == "built-in": return True
    if "site-packages" in origin or "dist-packages" in origin: return False
    return origin.startswith(os.path.dirname(os.__file__).lower())

def _scan_static(root: str, package_dir: str) -> Tuple[Set[str], Set[str]]:
    required, optional = set(), set()
    for path in _walk_py(root, package_dir):
        with open(path, "rb") as fh:
            try:
                tree = ast.parse(fh.read(), filename=path)
            except Exception:
                continue
        # raw imports
        for n in ast.walk(tree):
            if isinstance(n, ast.Import):
                for a in n.names:
                    required.add(_mod_top(a.name))
            elif isinstance(n, ast.ImportFrom) and (n.level or 0) == 0 and n.module:
                required.add(_mod_top(n.module))
        # imports in try/except ImportError → optional
        for n in ast.walk(tree):
            if isinstance(n, ast.Try) and n.handlers:
                catches_import_error = any(
                    (h.type and isinstance(h.type, (ast.Name, ast.Attribute)) and
                     (getattr(h.type, "id", None) in ("ImportError","ModuleNotFoundError") or
                      getattr(h.type, "attr", None) in ("ImportError","ModuleNotFoundError")))
                    for h in n.handlers
                )
                if catches_import_error:
                    for stmt in n.body:
                        for m in ast.walk(stmt):
                            if isinstance(m, ast.Import):
                                for a in m.names: optional.add(_mod_top(a.name))
                            elif isinstance(m, ast.ImportFrom) and (m.level or 0) == 0 and m.module:
                                optional.add(_mod_top(m.module))
    # optional ⊆ required → keep classification but exclude locals/stdlib later
    return required, optional

def _trace_runtime(probes: List[str], pkg_root: str) -> Set[str]:
    seen: Set[str] = set()
    real_import = builtins.__import__
    def hook(name, globals=None, locals=None, fromlist=(), level=0):
        top = _mod_top(name)
        seen.add(top)
        return real_import(name, globals, locals, fromlist, level)
    builtins.__import__ = hook
    try:
        for code in probes:
            try:
                exec(compile(code, "<probe>", "exec"), {})
            except SystemExit:
                pass
            except Exception:
                # ignore probe errors; we only care about imports seen
                pass
    finally:
        builtins.__import__ = real_import
    return seen

def _pypi_name(mod: str) -> str:
    return PYPI_HINTS.get(mod, mod.replace("_","-"))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".", help="repo root")
    ap.add_argument("--package", default=DEFAULT_PACKAGE)
    ap.add_argument("--out", default="build/deps")
    ap.add_argument("--probe", action="append", default=None,
                    help="python snippet to execute to trigger lazy imports")
    args = ap.parse_args()

    root = os.path.abspath(args.root)
    pkg = args.package
    os.makedirs(args.out, exist_ok=True)

    req_raw, opt_raw = _scan_static(root, pkg)
    rt_seen = _trace_runtime(args.probe or DEFAULT_PROBES, pkg)

    # classify
    locals_ = {pkg}
    raw = (req_raw | rt_seen) - locals_ - {"__future__"}
    required = set(m for m in raw if not _is_stdlib(m))
    optional = set(m for m in (opt_raw | (rt_seen - req_raw)) if m in required)

    third_party = required
    report = Report(required=required, optional=optional, runtime=rt_seen, third_party=third_party)

    # emit JSON report
    rep = {
        "required_modules": sorted(report.required),
        "optional_modules": sorted(report.optional),
        "runtime_seen": sorted(report.runtime),
        "third_party_modules": sorted(report.third_party),
        "pypi_map": {m: _pypi_name(m) for m in sorted(report.third_party)},
    }
    with open(os.path.join(args.out, "discovered.json"), "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=2, sort_keys=True)

    # emit suggested pyproject extras (minimal)
    groups: Dict[str, List[str]] = {
        "speed": [], "numpy": [], "torch": [], "tf": [], "jax": [], "misc": []
    }
    for m in sorted(report.third_party):
        p = _pypi_name(m)
        if m in {"orjson","ujson","zstandard"}: groups["speed"].append(p)
        elif m == "numpy": groups["numpy"].append(p)
        elif m == "torch": groups["torch"].append(p)
        elif m == "tensorflow": groups["tf"].append(p)
        elif m in {"jax","jaxlib"}: groups["jax"].append(p)
        else: groups["misc"].append(p)

    # collapse empties and duplicates
    for k in list(groups.keys()):
        pkgs = sorted(dict.fromkeys(groups[k]))
        if not pkgs: del groups[k]
        else: groups[k] = pkgs

    # always include a 'full' union
    all_pkgs = sorted({p for lst in groups.values() for p in lst})
    out_toml = ["[project.optional-dependencies]"]
    for k, lst in groups.items():
        out_toml.append(f'{k} = [{", ".join(repr(x) for x in lst)}]')
    out_toml.append(f'full = [{", ".join(repr(x) for x in all_pkgs)}]')
    with open(os.path.join(args.out, "pyproject.deps.suggested.toml"), "w", encoding="utf-8") as f:
        f.write("\n".join(out_toml) + "\n")

    # emit requirements-style candidates
    with open(os.path.join(args.out, "requirements.discovered.in"), "w", encoding="utf-8") as f:
        for p in all_pkgs:
            f.write(p + "\n")

    print(f"[discover_deps] required={len(required)} optional≈{len(optional)} → {args.out}")

if __name__ == "__main__":
    main()
