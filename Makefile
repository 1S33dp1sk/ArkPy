# Cross-platform Python package builder with venv (MSYS2, Linux, macOS)
# Targets: build | wheel | sdist | sync | venv | check | lock | clean | doctor

SHELL := bash
.SHELLFLAGS := -euo pipefail -c
.ONESHELL:
.DEFAULT_GOAL := build

ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
VENV ?= $(ROOT)/.venv-build
OUTDIR ?= $(ROOT)/wheelhouse
REQ ?= $(ROOT)/requirements.txt
CONSTRAINTS ?= $(ROOT)/toolchains/constraints.txt

# Bootstrap Python (python3 -> python -> py -3)
PYBOOT := $(shell \
  (command -v python3 >/dev/null 2>&1 && echo python3) || \
  (command -v python  >/dev/null 2>&1 && echo python)  || \
  (command -v py      >/dev/null 2>&1 && echo py -3) )

# cygpath for MSYS path→Windows path (used for --outdir / constraints)
HAS_CYGPATH := $(shell command -v cygpath >/dev/null 2>&1 && echo 1 || echo 0)
define P2H
$(if $(filter 1,$(HAS_CYGPATH)),$(shell cygpath -w '$(1)'),$(1))
endef

OUTDIR_HOST := $(call P2H,$(OUTDIR))
CONSTRAINTS_HOST := $(call P2H,$(CONSTRAINTS))

export PYTHONUTF8 := 1
# export SOURCE_DATE_EPOCH ?= 1730937600  # enable for deterministic timestamps

.PHONY: help
help:
	@echo "make build     # clean + venv + deps + sdist+wheel → $(OUTDIR)"
	@echo "make wheel     # wheel → $(OUTDIR)"
	@echo "make sdist     # sdist → $(OUTDIR)"
	@echo "make sync      # install build deps + project reqs into venv"
	@echo "make venv      # create/repair virtualenv: $(VENV)"
	@echo "make check     # twine check artifacts"
	@echo "make lock      # snapshot venv to .venv-lock/requirements.txt"
	@echo "make clean     # remove build artifacts"
	@echo "make doctor    # print environment info"

.PHONY: doctor
doctor:
	@echo "[env] SHELL=$$SHELL"
	@echo "[env] OS=$(OS)"
	@echo "[env] PYBOOT=$(PYBOOT)"
	@if [ -x "$(VENV)/bin/python" ]; then "$(VENV)/bin/python" -V; fi
	@if [ -x "$(VENV)/Scripts/python.exe" ]; then "$(VENV)/Scripts/python.exe" -V; fi

.PHONY: clean
clean:
	@rm -rf "$(ROOT)/build" "$(ROOT)/dist" "$(ROOT)/arknet_py.egg-info" "$(ROOT)"/*.egg-info "$(OUTDIR)"

.PHONY: venv
venv:
	@mkdir -p "$(OUTDIR)"
	@if [ ! -x "$(VENV)/bin/python" ] && [ ! -x "$(VENV)/Scripts/python.exe" ]; then \
	  echo "[venv] creating at $(VENV) using $(PYBOOT)"; \
	  $(PYBOOT) -m venv "$(VENV)" || { echo "[venv] failed"; exit 1; }; \
	fi
	@VE_PY="$(VENV)/bin/python"; [ -x "$$VE_PY" ] || VE_PY="$(VENV)/Scripts/python.exe"; \
	echo "[venv] python=$$($$VE_PY -V)"; \
	"$$VE_PY" -m pip -q install --upgrade pip setuptools wheel

.PHONY: sync
sync: venv
	@VE_PY="$(VENV)/bin/python"; [ -x "$$VE_PY" ] || VE_PY="$(VENV)/Scripts/python.exe"; \
	echo "[sync] installing build toolchain"; \
	"$$VE_PY" -m pip -q install --upgrade build; \
	if [ -f "$(REQ)" ]; then \
	  if [ -f "$(CONSTRAINTS)" ]; then \
	    echo "[sync] constraints: $(CONSTRAINTS)"; \
	    PIP_CONSTRAINT="$(CONSTRAINTS_HOST)" "$$VE_PY" -m pip -q install -r "$(REQ)"; \
	  else \
	    "$$VE_PY" -m pip -q install -r "$(REQ)"; \
	  fi; \
	fi

.PHONY: _build_all
_build_all:
	@VE_PY="$(VENV)/bin/python"; [ -x "$$VE_PY" ] || VE_PY="$(VENV)/Scripts/python.exe"; \
	mkdir -p "$(OUTDIR)"; \
	echo "[build] sdist+wheel → $(OUTDIR)"; \
	"$$VE_PY" -m build --no-isolation --outdir "$(OUTDIR_HOST)" --sdist --wheel; \
	ls -lh "$(OUTDIR)" || true

.PHONY: build
build: clean tidy sync _build_all

.PHONY: wheel
wheel:  tidy sync
	@VE_PY="$(VENV)/bin/python"; [ -x "$$VE_PY" ] || VE_PY="$(VENV)/Scripts/python.exe"; \
	mkdir -p "$(OUTDIR)"; \
	echo "[build] wheel → $(OUTDIR)"; \
	"$$VE_PY" -m build --no-isolation --outdir "$(OUTDIR_HOST)" --wheel; \
	ls -lh "$(OUTDIR)" || true

.PHONY: sdist
sdist: tidy sync
	@VE_PY="$(VENV)/bin/python"; [ -x "$$VE_PY" ] || VE_PY="$(VENV)/Scripts/python.exe"; \
	mkdir -p "$(OUTDIR)"; \
	echo "[build] sdist → $(OUTDIR)"; \
	"$$VE_PY" -m build --no-isolation --outdir "$(OUTDIR_HOST)" --sdist; \
	ls -lh "$(OUTDIR)" || true

.PHONY: check
check:
	@VE_PY="$(VENV)/bin/python"; [ -x "$$VE_PY" ] || VE_PY="$(VENV)/Scripts/python.exe"; \
	"$$VE_PY" -m pip -q install --upgrade twine; \
	find "$(OUTDIR)" -maxdepth 1 -type f \( -name '*.whl' -o -name '*.tar.gz' \) -print0 | xargs -0 -r "$$VE_PY" -m twine check

.PHONY: lock
lock:
	@mkdir -p "$(ROOT)/.venv-lock"; \
	VE_PY="$(VENV)/bin/python"; [ -x "$$VE_PY" ] || VE_PY="$(VENV)/Scripts/python.exe"; \
	"$$VE_PY" -m pip freeze > "$(ROOT)/.venv-lock/requirements.txt"; \
	echo "[lock] wrote .venv-lock/requirements.txt"

.PHONY: tidy
tidy:
	@find "$(ROOT)/arknet_py" -type d -name "__pycache__" -prune -exec rm -rf {} +
