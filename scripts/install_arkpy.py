#!/usr/bin/env python3
# scripts/install_arkpy.py
# Cross-platform installer for ArkPy: builds a wheel (default) or installs editable.
# Usage examples:
#   python scripts/install_arkpy.py                   # build wheel → install user-local if not in venv
#   python scripts/install_arkpy.py --editable        # pip install -e .
#   python scripts/install_arkpy.py --system          # system-wide (needs admin/sudo)
#   python scripts/install_arkpy.py --constraints toolchains/constraints.txt
#   python scripts/install_arkpy.py --upgrade

from __future__ import annotations
import argparse
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "build" / "py" / "dist"
PYPROJECT = ROOT / "pyproject.toml"

def run(cmd: list[str], cwd: Path | None = None) -> None:
    print(f"+ {' '.join(cmd)}")
    subprocess.run(cmd, cwd=cwd or ROOT, check=True)

def in_venv() -> bool:
    return (hasattr(sys, "real_prefix") or (getattr(sys, "base_prefix", sys.prefix) != sys.prefix) or bool(os.environ.get("VIRTUAL_ENV")))

def default_user_install(system: bool) -> bool:
    if system or in_venv():
        return False
    # Prefer --user when not in venv to avoid admin/sudo; works on Linux/macOS/Windows.
    return True

def ensure_build_module() -> None:
    try:
        import build  # noqa: F401
    except Exception:
        run([sys.executable, "-m", "pip", "install", "--upgrade", "build"])

def parse_project_name() -> str | None:
    try:
        import tomllib  # Python 3.11+
    except Exception:
        try:
            import tomli as tomllib  # type: ignore
        except Exception:
            return None
    try:
        data = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
        name = (
            data.get("project", {}).get("name")
            or data.get("tool", {}).get("poetry", {}).get("name")
        )
        return name
    except Exception:
        return None

def check_requires_python() -> None:
    try:
        import tomllib  # py311+
    except Exception:
        try:
            import tomli as tomllib  # type: ignore
        except Exception:
            return
    try:
        data = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
        req = (
            data.get("project", {}).get("requires-python")
            or data.get("tool", {}).get("poetry", {}).get("dependencies", {}).get("python")
        )
        if not req:
            return
        try:
            from packaging.specifiers import SpecifierSet
            from packaging.version import Version
        except Exception:
            return
        spec = SpecifierSet(req)
        cur = Version(".".join(map(str, sys.version_info[:3])))
        if cur not in spec:
            print(f"ERROR: Current Python {cur} does not satisfy requires-python '{req}'.", file=sys.stderr)
            sys.exit(3)
    except Exception:
        return

def build_wheel(outdir: Path) -> Path:
    ensure_build_module()
    outdir.mkdir(parents=True, exist_ok=True)
    for p in outdir.glob("*.whl"):
        p.unlink()
    run([sys.executable, "-m", "build", "--wheel", "--outdir", str(outdir)])
    wheels = sorted(outdir.glob("*.whl"))
    if not wheels:
        print("ERROR: Wheel build produced no artifacts.", file=sys.stderr)
        sys.exit(4)
    return wheels[-1]

def pip_install(args: list[str], constraints: Path | None, user: bool, system: bool, upgrade: bool, prefix: Path | None) -> None:
    cmd = [sys.executable, "-m", "pip", "install"]
    if upgrade:
        cmd.append("--upgrade")
    if constraints:
        cmd += ["-c", str(constraints)]
    if prefix:
        cmd += ["--prefix", str(prefix)]
    elif user and not system:
        cmd.append("--user")
    cmd += args
    run([sys.executable, "-m", "pip", "--version"])
    run(cmd)

def main() -> None:
    ap = argparse.ArgumentParser(description="Install ArkPy (wheel by default or editable).")
    ap.add_argument("--editable", action="store_true", help="pip install -e .")
    ap.add_argument("--system", action="store_true", help="force system-wide install (no --user).")
    ap.add_argument("--user", action="store_true", help="force user install.")
    ap.add_argument("--prefix", type=Path, help="pip --prefix target.")
    ap.add_argument("--constraints", type=Path, help="pip constraints file.")
    ap.add_argument("--upgrade", action="store_true", help="pip --upgrade.")
    args = ap.parse_args()

    os_name = platform.system().lower()
    print(f"== ArkPy installer ==")
    print(f"root: {ROOT}")
    print(f"os:   {os_name}, python: {sys.version.split()[0]}, venv: {in_venv()}")

    if not PYPROJECT.exists():
        print(f"ERROR: pyproject.toml not found at {PYPROJECT}", file=sys.stderr)
        sys.exit(2)

    check_requires_python()

    constraints = args.constraints
    if not constraints:
        default_cons = ROOT / "toolchains" / "constraints.txt"
        if default_cons.exists():
            constraints = default_cons

    # Decide user/system flags
    user_flag = args.user or default_user_install(args.system)
    system_flag = args.system

    if args.editable:
        print("Installing in editable mode...")
        pip_install(["-e", str(ROOT)], constraints, user_flag, system_flag, args.upgrade, args.prefix)
    else:
        print("Building wheel and installing...")
        wheel = build_wheel(DIST)
        print(f"wheel: {wheel.name}")
        pip_install([str(wheel)], constraints, user_flag, system_flag, True if args.upgrade else False, args.prefix)

    print("OK: ArkPy installed.")

if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as e:
        print(f"FAILED: {e}", file=sys.stderr)
        sys.exit(e.returncode if isinstance(e.returncode, int) else 1)
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        sys.exit(130)
