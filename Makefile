.PHONY: install lint test smoke train eval demo clean

install:          ## sync deps (add extras: uv sync --extra dev --extra track)
	uv sync --extra dev

lint:
	uv run ruff check src tests scripts
	uv run ruff format --check src tests scripts

test:             ## hermetic unit tests only (CI-safe, no network)
	uv run pytest -m "not smoke"

smoke:            ## end-to-end walking skeleton on a tiny model (needs network + torch)
	uv run pytest -m smoke --run-smoke

train:            ## Phase-1 training entrypoint (configure via .env / config.py)
	uv run python -m lexfr_embed.train

eval:             ## evaluate a model on BSARD + held-out set
	uv run python -m lexfr_embed.evaluate

demo:             ## launch the FastAPI /search demo (requires .[serve])
	uv run uvicorn lexfr_embed.serve:app --reload --port 8000

clean:
	rm -rf .pytest_cache .ruff_cache **/__pycache__
