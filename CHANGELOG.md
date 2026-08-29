# Changelog

All notable changes to tiny-agent will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [0.3.0] - 2026-08-29

### Added
- Comprehensive pytest test suite in `tests/` directory
- Improved `pyproject.toml` with proper classifiers, URLs, and tool configs
- `tests/test_agent.py`: unit tests for Message, ToolRegistry, Tool decorator, Memory, Observable, Plan, BudgetGovernor, IdempotencyGuard, HTTPAdapter
- `tests/test_adapters.py`: adapter initialization tests
- `tests/conftest.py`: pytest fixtures and configuration
- Quick reference section in README.md

### Changed
- pyproject.toml now declares all optional dependencies groups (dev, all)
- Added pytest, coverage, ruff, and mypy tool configurations

## [0.2.0] - 2026-08-18

### Added
- Streaming ReAct loop via `stream=True` parameter
- Checkpoint/suspend integration
- Budget Governor with USD enforcement and warning callbacks
- Idempotency Guard using SHA-1 dedup keys
- Better MCP tool schemas (JSON Schema auto-generation)
- 5 real-world examples (bounty hunter, code reviewer, data pipeline, etc.)

## [0.1.0] - 2026-08-10

### Added
- Initial release
- Core ReAct loop
- Tool registry with decorator-based registration
- OpenAI, Anthropic, Ollama adapters
- HTTP and User adapters
- In-memory memory backend
- Multi-agent delegation
