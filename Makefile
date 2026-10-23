# Run UNCAWE experiments on the remote GPU server (default) or on this machine (WHERE=local).
#
# Typical workflow:
#   make init                 # once: create REMOTE_DIR and copy env.json, results, ... from OLD_DIR
#   make setup                # push the code and install the environment with uv
#   make run-main-classic     # start an experiment in a detached tmux session (ATTACH=1 to attach to it)
#   make status / make logs EXP=main-classic / make attach EXP=main-classic
#   make explog EXP=main      # experiment log <OUT_DIR>/<experiment>.log (FOLLOW=1 to keep reading)
#   make pull                 # copy tables and plots back to ./output
#
# Local runs: `make fetch-data` once (pretrain files and datasets from REMOTE), then
#   make run-binform WHERE=local JAX=cpu
# and `make push-results` / `make pull-results` to merge result files between the two machines.
#
# Every variable can be overridden on the command line, e.g. `make run-predist GPU=2 ARGS="--datasets letter"`.

WHERE      ?= remote
REMOTE     ?= bcuda
REMOTE_DIR ?= /home/volpi/uncawe-devel
OLD_DIR    ?= /home/volpi/uncawe
REMOTE_UV  ?= /home/volpi/miniconda3/bin/uv
# Interpreter of the remote environment: the stored results were produced with this Python (3.12.11); bcuda's
# /usr/bin/python3 (3.14) has no development headers, so dependencies without 3.14 wheels (quadprog) fail to build.
REMOTE_PY  ?= /home/volpi/miniconda3/bin/python3
GPU        ?= 0
JAX        ?= cpu
ARGS       ?=
LOCAL_OUT  ?= ./output
PULL_DIRS  ?= tables plots
LOG_LINES  ?= 50
FOLLOW     ?=
RESULTS    ?= main binform predist

# Files and directories of OLD_DIR worth keeping in REMOTE_DIR (never overwritten if already present).
OLD_FILES  ?= env.json sorted_sets.json output

# Code pushed to REMOTE_DIR. Directories are mirrored (--delete), so stale modules do not survive.
PUSH_DIRS  := src scripts tests
PUSH_FILES := pyproject.toml uv.lock README.md

# Where `fetch-data` puts pretrain files and datasets (the defaults of cap.env on this machine).
LOCAL_CAP_DATA ?= $(HOME)/cap_data
LOCAL_DATA_ROOT ?= $(HOME)

# Experiments: run-<name> runs `uv run <command> $(ARGS)`.
# Table targets carry no mode flag: pass it in ARGS (--tables, --pdf or --all), e.g. `make run-tables-main ARGS=--tables`.
EXP_pretrain-classic := python -m uncawe.pretrain.classic
EXP_main-classic     := python -m uncawe.main --classic
EXP_binform          := python -m uncawe.binform
EXP_predist          := python -m uncawe.predist
EXP_tables-main      := python -m uncawe.tables.main --classic
EXP_tables-rank      := python -m uncawe.tables.rank --classic
EXP_tables-binform   := python -m uncawe.tables.binform
EXP_tables-estimand-gap := python -m uncawe.tables.estimand_gap
EXP_tables-displacement := python -m uncawe.tables.displacement
EXP_plot-avc         := python -m uncawe.plots.amplitude_vs_coverage --domain classic
EXP_plot-aew         := python -m uncawe.plots.ae_vs_winkler --domain classic
EXP_plot-rank        := python -m uncawe.plots.rank_heatmap --domain classic
EXP_plot-binform     := python -m uncawe.plots.binform_simplex
EXP_plot-displacement := python -m uncawe.plots.displacement
EXP_test             := python -m unittest discover -s tests
EXP_bench-bayes-jobs := python scripts/bench/bayes_n_jobs.py
EXP_bench-xla-stored := python scripts/bench/xla_vs_stored.py

EXPERIMENTS := $(patsubst EXP_%,%,$(filter EXP_%,$(.VARIABLES)))

RSYNC := rsync -az --exclude __pycache__ --exclude '*.egg-info' --exclude .venv
SSH   := ssh $(REMOTE)

# Where commands run: ON runs a quoted shell command, DIR is the project folder, UV the uv binary.
ifeq ($(WHERE),local)
  ON       := bash -c
  ON_TTY   := bash -c
  DIR      := $(CURDIR)
  UV       := uv
  RUN_DEPS :=
else ifeq ($(WHERE),remote)
  ON       := $(SSH)
  ON_TTY   := ssh -t $(REMOTE)
  DIR      := $(REMOTE_DIR)
  UV       := UV_PYTHON=$(REMOTE_PY) $(REMOTE_UV)
  RUN_DEPS := push
else
  $(error WHERE must be 'local' or 'remote', got '$(WHERE)')
endif

# Header of every run log: code revision (computed here, so it matches what `push` just sent) and settings.
REV_COMMIT   := $(shell git rev-parse HEAD)
REV_DESCRIBE := $(shell git describe --always --dirty)
JAX_ENV      := $(if $(JAX),JAX_PLATFORMS=$(JAX))

.PHONY: help list init push setup run-% status logs explog attach stop pull fetch-data pull-results push-results

help:
	@sed -n '2,15p' $(firstword $(MAKEFILE_LIST)) | sed 's/^# \{0,1\}//'
	@echo "Experiments (make run-<name>):"
	@$(foreach e,$(sort $(EXPERIMENTS)),printf '  %-18s %s\n' '$(e)' '$(EXP_$(e))';)

list:
	@echo $(sort $(EXPERIMENTS))

## One-time initialisation: copy useful files from the old folder (remote-to-remote, no overwrite).
init:
	$(SSH) 'set -e; mkdir -p $(REMOTE_DIR); cd $(OLD_DIR); \
	  rsync -a --ignore-existing --exclude .Trash-1000 $(OLD_FILES) $(REMOTE_DIR)/; \
	  ls -la $(REMOTE_DIR)'

## Push the working tree (including uncommitted changes).
push:
	$(SSH) 'mkdir -p $(REMOTE_DIR)'
	$(RSYNC) --delete $(PUSH_DIRS) $(REMOTE):$(REMOTE_DIR)/
	$(RSYNC) $(PUSH_FILES) $(REMOTE):$(REMOTE_DIR)/

## Install the environment pinned by uv.lock (pushing first when remote).
setup: $(RUN_DEPS)
	$(ON) 'cd $(DIR) && $(UV) sync --locked'

## Start an experiment in a detached tmux session named uncawe-<name>; output goes to <OUT_DIR>/runs/.
## ATTACH=1 attaches the terminal to the session once it has started (detach with C-b d; the run goes on).
run-%: $(RUN_DEPS)
	@test -n '$(EXP_$*)' || { echo "Unknown experiment '$*'. Known: $(sort $(EXPERIMENTS))"; exit 1; }
	$(ON) 'cd $(DIR) && mkdir -p output/runs && \
	  if tmux has-session -t uncawe-$* 2>/dev/null; then echo "uncawe-$* is already running"; exit 1; fi; \
	  log=output/runs/$*_$$(date +%Y%m%d-%H%M%S).out; ln -sfn $$(basename $$log) output/runs/$*.latest; \
	  printf "%s\n" "commit: $(REV_COMMIT)" "describe: $(REV_DESCRIBE)" "host: $$(hostname)" \
	    "cmd: $(EXP_$*) $(ARGS)" "gpu: $(GPU)" "jax: $(or $(JAX),default)" "start: $$(date -Iseconds)" > $$log; \
	  tmux new-session -d -s uncawe-$* "set -o pipefail; \
	    CUDA_VISIBLE_DEVICES=$(GPU) $(JAX_ENV) $(UV) run --locked $(EXP_$*) $(ARGS) 2>&1 | tee -a $$log; \
	    echo \"exit: \$$? end: \$$(date -Iseconds)\" >> $$log" && echo "started uncawe-$* ($(WHERE)), log: $(DIR)/$$log"'
	$(if $(ATTACH),$(ON_TTY) 'tmux attach -t uncawe-$* 2>/dev/null || echo "uncawe-$* has already ended: see make logs EXP=$*"')

status:
	@$(ON) 'tmux ls 2>/dev/null | grep "^uncawe-" || echo "no uncawe sessions"; \
	  nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv'

logs:
	@test -n '$(EXP)' || { echo "usage: make logs EXP=<name> [WHERE=local]"; exit 1; }
	@$(ON) 'tail -n 50 $(DIR)/output/runs/$(EXP).latest'

## Tail the experiment log <OUT_DIR>/<EXP>.log written by the code (EXP is the experiment: main, binform, predist,
## pretrain.classic), with OUT_DIR resolved as the code does (env.json entry of the host, else the cap.env default).
explog:
	@test -n '$(EXP)' || { echo "usage: make explog EXP=<main|binform|predist|pretrain.classic> [FOLLOW=1] [LOG_LINES=n] [WHERE=local]"; exit 1; }
	@$(ON) 'cd $(DIR) && out=$$($(UV) run --locked python -c "import os, uncawe.env as e; print(os.path.dirname(e.root_dir))" 2>/dev/null | tail -n 1) && \
	  echo "==> $$out/$(EXP).log" && tail -n $(LOG_LINES) $(if $(FOLLOW),-f) "$$out/$(EXP).log"'

attach:
	@test -n '$(EXP)' || { echo "usage: make attach EXP=<name> [WHERE=local]"; exit 1; }
	$(ON_TTY) 'tmux attach -t uncawe-$(EXP)'

stop:
	@test -n '$(EXP)' || { echo "usage: make stop EXP=<name> [WHERE=local]"; exit 1; }
	$(ON) 'tmux kill-session -t uncawe-$(EXP)'

## Copy generated tables and plots back (not the raw parquet results).
pull:
	mkdir -p $(LOCAL_OUT)/uncawe
	$(foreach d,$(PULL_DIRS),rsync -az $(REMOTE):$(REMOTE_DIR)/output/uncawe/$(d) $(LOCAL_OUT)/uncawe/;)

## Copy pretrain files and datasets from REMOTE so that local runs find them (no overwrite).
fetch-data:
	mkdir -p $(LOCAL_CAP_DATA) $(LOCAL_DATA_ROOT)
	rsync -az --ignore-existing $(REMOTE):/home/volpi/cap_data/pretrain $(LOCAL_CAP_DATA)/
	rsync -az --ignore-existing $(REMOTE):/home/volpi/quapy_data $(REMOTE):/home/volpi/scikit_learn_data $(LOCAL_DATA_ROOT)/

## Merge result files of the experiments in RESULTS. Result files are written once per cell, so
## --ignore-existing never replaces a result: a cell computed on both machines keeps the receiver's copy.
pull-results:
	mkdir -p $(LOCAL_OUT)/uncawe
	$(foreach r,$(RESULTS),rsync -az --ignore-existing $(REMOTE):$(REMOTE_DIR)/output/uncawe/$(r) $(LOCAL_OUT)/uncawe/;)

push-results:
	$(foreach r,$(RESULTS),test ! -d $(LOCAL_OUT)/uncawe/$(r) || rsync -az --ignore-existing $(LOCAL_OUT)/uncawe/$(r) $(REMOTE):$(REMOTE_DIR)/output/uncawe/;)
