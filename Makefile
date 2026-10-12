.PHONY: install train test lint run docker

install:
	pip install -r requirements-dev.txt

train:
	python train.py

test:
	pytest -q

lint:
	ruff check .

run:
	uvicorn app.main:app --reload

docker:
	docker compose up --build
