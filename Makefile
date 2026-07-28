.PHONY: help
help:
	@printf '%s\n' \
		'Targets:' \
		'  make fmt       Format Rust, TOML, and benchmark Python files' \
		'  make check     Run format checks, Clippy, and tests' \
		'  make test      Run tests with all features' \
		'  make bench-env Sync benchmark environment; set EXTRA=paddle-cu126 as needed'

.PHONY: fmt
fmt:
	cargo fmt
	taplo format Cargo.toml benchmarks/pyproject.toml
	cd benchmarks && uv run ruff format .

.PHONY: check
check:
	cargo fmt --check
	taplo format --check Cargo.toml benchmarks/pyproject.toml
	cd benchmarks && uv run ruff format --check .
	cd benchmarks && uv run ruff check .
	cargo clippy --all-features --lib
	cargo test --all-features

.PHONY: test
test:
	cargo test --all-features

.PHONY: bench-env
bench-env:
	cd benchmarks && uv sync $(if $(EXTRA),--extra $(EXTRA),)
