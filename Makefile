FEATURES ?= --all-features

.PHONY: help
help:
	@printf '%s\n' \
		'Targets:' \
		'  make fmt       Format Rust, TOML, YAML, and benchmark Python files' \
		'  make check     Run format checks, Clippy, and tests' \
		'  make test      Run tests with FEATURES' \
		'  make hooks     Install Lefthook and sync Git hooks' \
		'  make bench-env Sync benchmark environment; set EXTRA=paddle-cu126 as needed'

.PHONY: fmt
fmt:
	cargo fmt
	taplo format Cargo.toml benchmarks/pyproject.toml benchmarks/rame-bench/Cargo.toml python/Cargo.toml python/pyproject.toml
	yamlfmt .github/workflows
	cd benchmarks && uv run ruff check --fix --select I .
	cd benchmarks && uv run ruff format .
	cd python && uv run ruff check --fix --select I .
	cd python && uv run ruff format .

.PHONY: check
check:
	cargo fmt --check
	taplo format --check Cargo.toml benchmarks/pyproject.toml benchmarks/rame-bench/Cargo.toml python/Cargo.toml python/pyproject.toml
	yamlfmt -lint .github/workflows
	cd benchmarks && uv run ruff format --check .
	cd benchmarks && uv run ruff check .
	cd python && uv run ruff format --check .
	cd python && uv run ruff check .
	cargo clippy $(FEATURES) --lib
	cargo clippy -p rame-bench --all-targets
	cargo clippy -p rame-python --all-targets
	cargo test $(FEATURES)
	cargo test -p rame-bench
	cargo test -p rame-python

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
	cd benchmarks && uv sync $(if $(EXTRA),--extra $(EXTRA),)
