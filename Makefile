.PHONY: install migrate seed run test lint clean

install:
	pip install -e .
	pip install pytest pytest-asyncio httpx

migrate:
	cd /opt/gami && alembic upgrade head

seed:
	cd /opt/gami && python -m storage.sql.seed

run:
	cd /opt/gami && uvicorn api.main:app --host 0.0.0.0 --port 9000 --reload

workers:
	cd /opt/gami && celery -A workers.celery_app worker --loglevel=info --concurrency=4

test:
	cd /opt/gami && pytest tests/ -v

lint:
	cd /opt/gami && python -m py_compile api/main.py

clean:
	find /opt/gami -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
