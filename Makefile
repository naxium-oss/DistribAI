# DistribAI contributor shortcuts. Mirrors .github/workflows/ci.yml.
.PHONY: help lint test test-unit test-security test-integration typecheck coverage proto ci

help:
	@echo "Targets: lint test test-unit test-security test-integration typecheck coverage proto ci"

lint:
	ruff check .
	ruff format --check .

typecheck:
	mypy services_python/ worker/src/ --ignore-missing-imports --no-strict-optional --follow-imports=silent

test-unit:
	pytest tests/unit -v

test-security:
	pytest tests/security -v

test-integration:
	pytest tests/integration -v --timeout=900

test: test-unit test-security

coverage:
	pytest tests/unit tests/security --cov=services_python --cov=worker --cov-report=term-missing --cov-report=xml

proto:
	python -m grpc_tools.protoc --python_out=worker/src/distribai_proto --grpc_python_out=worker/src/distribai_proto -I proto proto/distribai.proto
	python -c "from pathlib import Path; p=Path('worker/src/distribai_proto/distribai_pb2_grpc.py'); p.write_text(p.read_text().replace('import distribai_pb2 as distribai__pb2', 'from . import distribai_pb2 as distribai__pb2'))"

ci: lint typecheck test coverage
