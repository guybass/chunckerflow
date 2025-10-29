.PHONY: help install install-dev format lint test test-cov clean docker-build docker-run ci

help:  ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install package
	pip install -e .

install-dev:  ## Install package with development dependencies
	pip install -e ".[dev]"
	pre-commit install

install-all:  ## Install package with all optional dependencies
	pip install -e ".[all]"

format:  ## Format code with black and isort
	@echo "Running black..."
	black chunk_flow tests scripts examples
	@echo "Running isort..."
	isort chunk_flow tests scripts examples

lint:  ## Lint code with ruff and mypy
	@echo "Running ruff..."
	ruff check chunk_flow tests scripts examples
	@echo "Running mypy..."
	mypy chunk_flow

test:  ## Run tests
	pytest tests/ -v

test-cov:  ## Run tests with coverage
	pytest tests/ -v --cov=chunk_flow --cov-report=html --cov-report=term-missing

test-integration:  ## Run integration tests
	pytest tests/ -v -m integration

test-e2e:  ## Run end-to-end tests
	pytest tests/ -v -m e2e

test-benchmark:  ## Run benchmark tests
	pytest tests/ -v -m benchmark

test-all:  ## Run all tests including slow ones
	pytest tests/ -v --run-slow

clean:  ## Clean build artifacts and cache
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf .ruff_cache
	rm -rf htmlcov/
	rm -rf .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

docker-build:  ## Build Docker image
	docker build -t chunk-flow:latest .

docker-build-dev:  ## Build development Docker image
	docker build -f Dockerfile.dev -t chunk-flow:dev .

docker-run:  ## Run Docker container
	docker run -p 8000:8000 chunk-flow:latest

docker-compose-up:  ## Start all services with docker-compose
	docker-compose up -d

docker-compose-down:  ## Stop all services
	docker-compose down

ci:  ## Run full CI pipeline locally
	@echo "=== Running CI pipeline ==="
	@echo "1. Formatting check..."
	black --check chunk_flow tests scripts examples
	isort --check-only chunk_flow tests scripts examples
	@echo "2. Linting..."
	ruff check chunk_flow tests scripts examples
	mypy chunk_flow
	@echo "3. Testing..."
	pytest tests/ -v --cov=chunk_flow --cov-report=term-missing
	@echo "=== CI pipeline complete ==="

pre-commit:  ## Run pre-commit hooks on all files
	pre-commit run --all-files

docs-build:  ## Build documentation
	cd docs && make html

docs-serve:  ## Serve documentation locally
	cd docs && python -m http.server 8080

release-patch:  ## Bump patch version and create tag
	@echo "Bumping patch version..."
	# Add version bumping logic here

release-minor:  ## Bump minor version and create tag
	@echo "Bumping minor version..."
	# Add version bumping logic here

release-major:  ## Bump major version and create tag
	@echo "Bumping major version..."
	# Add version bumping logic here

# Development shortcuts
dev-server:  ## Run development server
	uvicorn chunk_flow.api.app:app --reload --host 0.0.0.0 --port 8000

dev-dashboard:  ## Run Streamlit dashboard
	streamlit run chunk_flow/api/dashboard.py

# Quick checks before commit
check: format lint test  ## Run format, lint, and test
