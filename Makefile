MODULE = src
ARGS ?=
.PHONY: install run debug clean lint lint-strict

install:
	uv sync
run:
	uv run python -m $(MODULE) $(ARGS)
debug:
	uv run python -m pdb -m $(MODULE) $(ARGS)
clean:
	rm -rf __pycache__ .mypy_cache .pytest_cache
	find . -type d -name "__pycache__" -exec rm -r {} +
	find . -type f -name "*.pyc" -delete
lint:
	uv run flake8 src/
	uv run mypy src/ --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs
lint-strict:
	uv run flake8 src/
	uv run mypy --strict src/