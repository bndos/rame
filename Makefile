FEATURES ?= --all-features
BENCH_EXTRAS ?= rame paddle-hpi-gpu-cu118
TOML_FILES := $(shell git ls-files --cached --others --exclude-standard '*.toml')
YAML_FILES := $(shell git ls-files --cached --others --exclude-standard '*.yaml' '*.yml')

.PHONY: help
help:
	@printf '%s\n' \
		'Targets:' \
		'  make fmt                         Format Rust, TOML, YAML, and benchmark Python files' \
		'  make check                       Run format checks, Clippy, and tests' \
		'  make feature-check               Check supported Rust feature combinations' \
		'  make test                        Run tests with FEATURES' \
		'  make hooks                       Install Lefthook and sync Git hooks' \
		'  make bench-env                   Sync benchmark environment with BENCH_EXTRAS' \
		'  make bench-run-config CONFIG=... Run benchmark config'

.PHONY: fmt
fmt:
	cargo fmt
	taplo format $(TOML_FILES)
	yamlfmt $(YAML_FILES)
	uvx ruff check --fix .
	uvx ruff format .

.PHONY: check
check:
	cargo fmt --check
	taplo format --check $(TOML_FILES)
	yamlfmt -lint $(YAML_FILES)
	uvx ruff format --check .
	uvx ruff check .
	cargo clippy $(FEATURES) --lib
	cargo clippy -p rame-python --all-targets
	cargo test $(FEATURES)
	cargo test -p rame-python

.PHONY: ci
ci: check feature-check

.PHONY: feature-check
feature-check:
	cargo check --no-default-features
	cargo check --features metrics
	cargo check --workspace --all-features
	cargo check -p rame-python --features cuda
	cargo check -p rame-python --features tensorrt

.PHONY: test
test:
	cargo test $(FEATURES)

.PHONY: hooks
hooks:
	@if ! command -v lefthook >/dev/null 2>&1; then \
		if command -v go >/dev/null 2>&1; then \
			go install github.com/evilmartians/lefthook/v2@latest; \
		elif command -v brew >/dev/null 2>&1; then \
			brew install lefthook; \
		else \
			printf '%s\n' 'warning: install lefthook manually; neither go nor brew is available'; \
			exit 0; \
		fi; \
	fi
	lefthook install

.PHONY: bench-env
bench-env:
	uv sync $(foreach extra,$(BENCH_EXTRAS),--extra $(extra))

.PHONY: bench-run-config
bench-run-config:
	@test -n "$(CONFIG)" || (printf '%s\n' 'usage: make bench-run-config CONFIG=configs/name.yaml'; exit 2)
	uv run rame-bench run-config $(abspath $(CONFIG))
