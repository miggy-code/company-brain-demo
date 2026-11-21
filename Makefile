BUSINESS ?= casa-prado

.PHONY: install up check permissions demo

install:
	uv sync

up:
	docker compose up -d --wait

# What CI runs: lint, types, tests and the permission suite, all on the fakes.
check: up
	uv run ruff check
	uv run ruff format --check
	uv run mypy src
	uv run pytest
	uv run brain business create $(BUSINESS)
	uv run brain index $(BUSINESS) --models fake
	uv run brain eval $(BUSINESS) --suite permissions

# The real models (BGE-M3 + reranker, ~4.5 GB on first run) and one mock-model question.
# For a live answer, add --model live --keys-file <your provider keys file>.
demo: up
	uv run brain business create $(BUSINESS)
	uv run brain index $(BUSINESS) --models bge
	uv run brain ask $(BUSINESS) --as lucia.quispe@casaprado.example --model mock \
		"¿En qué estado está la separación del departamento 502 en Mirador del Parque?"
