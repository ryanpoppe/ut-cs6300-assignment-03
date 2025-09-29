setup:
	pipx install poetry

install:
	poetry install
	playwright install

run:
	poetry run python run.py

test:
	poetry run pytest
