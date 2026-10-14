UV ?= uv
PYTHON ?= 3.12

.PHONY: uv-venv uv-sync uv-sync-full uv-sync-integration test test-full test-integration coverage coverage-full coverage-integration

uv-venv:
	$(UV) venv --python $(PYTHON) --allow-existing

uv-sync: uv-venv
	./scripts/setup_uv.sh core

uv-sync-full: uv-venv
	./scripts/setup_uv.sh full

uv-sync-integration: uv-venv
	./scripts/setup_uv.sh integration

test: uv-sync
	./scripts/test_uv.sh core

test-full: uv-sync-full
	./scripts/test_uv.sh full

test-integration: uv-sync-integration
	./scripts/test_uv.sh integration

coverage: uv-sync
	./scripts/coverage_uv.sh core

coverage-full: uv-sync-full
	./scripts/coverage_uv.sh full

coverage-integration: uv-sync-integration
	./scripts/coverage_uv.sh integration
