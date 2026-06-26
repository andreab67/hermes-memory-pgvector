.PHONY: build test test-local docker-up docker-dev down clean

build:
	pip install -e ".[test,mcp]"

test:
	docker compose -f docker/compose.yml --profile test up --build --abort-on-container-exit --exit-code-from test

test-local:
	pytest tests/ -v

docker-up:
	docker compose -f docker/compose.yml --profile mcp up -d --build

docker-dev:
	docker compose -f docker/compose.yml --profile dev up -d pg

down:
	docker compose -f docker/compose.yml down --profile all

clean:
	rm -rf .pytest_cache .ruff_cache hexus.egg-info build dist
	find . -type d -name "__pycache__" -exec rm -rf {} +
