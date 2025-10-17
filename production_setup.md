# Production-Grade Python Project: Best Practices Guide

## Project Structure

Keep a clean separation: `chunk_flow/` for source code, `tests/` for tests, `docs/` for documentation, `scripts/` for automation, `config/` for configurations. Never mix data files, notebooks, or logs in the repo - use `.gitignore` aggressively. Structure your package with clear boundaries: `api/`, `core/`, `chunking/`, `embeddings/`, `evaluation/`, `results/`, `utils/`.

## Git Practices

**Never commit:** Jupyter notebooks (especially with outputs/data), data files (CSV, Parquet, JSON), model weights (use DVC or external storage), logs, `.env` files, IDE configs, `__pycache__` or virtual environments.

**Always commit:** Source code, tests, documentation, requirements files, Dockerfiles, CI/CD configs, example scripts (not notebooks - write Python scripts instead).

Use `.gitattributes` to enforce LF line endings for Python files and mark binary files explicitly. Keep commits atomic and meaningful. Use conventional commits (feat:, fix:, docs:, refactor:, test:).

## Logging Over Print

Replace ALL `print()` statements with structured logging using `structlog` or Python's `logging`. Configure JSON logging for production (parseable by log aggregators) and pretty console output for development. Always include context: request IDs, user IDs, document IDs, timestamps. Use log levels correctly: DEBUG for verbose details, INFO for normal operations, WARNING for recoverable issues, ERROR for failures.

Example pattern: `logger.info("chunk_completed", doc_id=doc_id, num_chunks=5, time_ms=123)` gives you queryable, structured logs.

## Configuration Management

Use `pydantic-settings` to load configuration from environment variables, YAML files, or both. Never hardcode secrets - use `.env` files locally (gitignored) and environment variables in production. Support multiple environments: `config/default.yaml`, `config/production.yaml`, `config/test.yaml`. Validate all configuration at startup.

## Code Quality Tools

Setup `pyproject.toml` with Black (formatting), Ruff (fast linting), isort (import sorting), and mypy (type checking). Configure pre-commit hooks to run these automatically before every commit. Add a hook to block print statements and notebooks with outputs. Run pytest with coverage reporting (`--cov`), aiming for 80%+ coverage.

## Testing Strategy

Three test levels: **Unit tests** for individual functions (fast, mocked dependencies), **Integration tests** for component interactions (use real dependencies but test data), **E2E tests** for full workflows. Use `pytest` with `pytest-asyncio` for async code. Mock external API calls in unit tests, use test configs in integration tests. Run tests in CI on every PR.

## Docker Setup

Create a **multi-stage Dockerfile**: builder stage installs dependencies, runtime stage copies only what's needed. Use slim base images (`python:3.11-slim`), run as non-root user, add health checks. Separate `Dockerfile` for production (optimized, small) and `Dockerfile.dev` (with dev tools, hot reload).

Write `docker-compose.yml` for local development with all services (API, Redis, databases). Use named volumes for persistence, environment variables for configuration. Never bake secrets into images.

## CI/CD Pipeline

GitHub Actions workflow: **lint job** (Black, Ruff, mypy), **test job** (pytest on multiple Python versions with coverage), **docker job** (build image, test container), **security job** (Bandit, dependency scanning). Run on every PR, block merge if any job fails.

Release workflow: tag triggers automatic PyPI publish, Docker image build and push to registry, changelog generation. Use semantic versioning strictly.

## Package Management

Use `pyproject.toml` (PEP 621) as single source of truth. Define dependencies with version constraints, optional dependencies for providers (`[openai]`, `[huggingface]`, `[dev]`). Keep `requirements.txt` and `requirements-dev.txt` in sync using `pip-compile` or similar. Support Python 3.9+ for compatibility.

## API Best Practices

FastAPI with Pydantic models for validation, structured logging for every request (request ID, user, endpoint, latency), proper error handling with custom exceptions, health/readiness endpoints for orchestration, OpenAPI docs auto-generated. Use middleware for logging, CORS, rate limiting. Deploy with Gunicorn/Uvicorn with multiple workers.

## Monitoring & Observability

Instrument code with metrics (Prometheus format), structured logs (JSON in production), health checks, distributed tracing (OpenTelemetry). Export DataFrame results to Parquet (for analytics), expose metrics endpoint, log all errors with full context. Use container orchestration readiness/liveness probes.

## Documentation

Write docstrings for all public APIs (Google/NumPy style), generate API docs with Sphinx/MkDocs, include example scripts (`.py` not `.ipynb`), maintain CHANGELOG.md with all changes, write CONTRIBUTING.md for contributors. Keep README.md concise with quick start only.

## Security

Scan dependencies regularly (Dependabot, Safety), never commit secrets (use secret managers), validate all inputs with Pydantic, sanitize error messages (no stack traces to users), run containers as non-root, use security headers in API responses, keep dependencies updated.

## Key Commands

```bash
# Development setup
make install-dev          # Install with dev dependencies
pre-commit install        # Setup git hooks

# Code quality
make format              # Black + isort
make lint                # Ruff + mypy
make test                # Run all tests with coverage

# Docker
make docker-build        # Build production image
make docker-run          # Run container
make docker-test         # Test in container

# CI/CD
make ci                  # Run full CI locally
make release             # Tag and trigger release
```

## Performance Optimization

### Numba JIT Compilation
Use `@numba.jit(nopython=True)` for hot loops and numerical computations. Compiles Python to machine code for 10-100x speedups. Perfect for distance calculations, similarity matrices, boundary detection algorithms. Avoid in I/O-bound code - only helps CPU-bound operations. Use `@numba.jit(parallel=True)` for automatic parallelization of loops.

### Vectorization
Replace Python loops with NumPy/Pandas vectorized operations. Compute embeddings in batches (100-1000 at once) instead of one-by-one. Use `np.einsum` for complex tensor operations, `scipy.spatial.distance.cdist` for pairwise distances. This gives 50-500x speedups over naive loops.

### Async/Await for I/O
All API calls, file operations, and database queries must be async. Use `httpx.AsyncClient` for HTTP (connection pooling), `aiofiles` for file I/O, `asyncpg` for PostgreSQL. Process multiple documents concurrently with `asyncio.gather()` and semaphores for rate limiting. Never block the event loop.

### Caching Strategies
Cache embeddings aggressively - they're expensive and deterministic. Use Redis for distributed cache, `functools.lru_cache` for in-process. Cache strategy outputs for identical configs. Implement cache invalidation by content hash. Consider persistent cache to disk with diskcache or LMDB.

### Batch Processing
Batch API calls: embed 100 chunks in one request vs 100 separate requests. Use `asyncio.Queue` with producer-consumer pattern for memory-bounded batching. Configure batch sizes based on API limits and memory constraints. Monitor queue depths.

### Memory Optimization
Stream large files instead of loading into memory. Use generators for lazy evaluation. Delete large objects explicitly and call `gc.collect()` if needed. Use `__slots__` for classes with many instances. Consider memory-mapped files for huge datasets (`np.memmap`). Profile memory with `memory_profiler`.

### Profiling Tools
**CPU profiling:** `cProfile`, `py-spy` (no code changes), `line_profiler` for line-by-line. **Memory profiling:** `memory_profiler`, `tracemalloc`. **Async profiling:** `aioprofile`. Profile in production-like conditions, not dev environment. Focus on the 20% of code that takes 80% of time.

### Database & Storage
Use connection pooling for databases. Batch inserts (1000s of rows at once). Index columns used in WHERE clauses. Use Parquet for columnar data (10x faster than CSV). Consider DuckDB for analytics on Parquet files - faster than Pandas for large datasets. Use async database drivers.

### Model Optimization
Use ONNX Runtime for 2-5x faster inference than PyTorch/TF. Quantize models (int8) for 4x speedup and 75% memory reduction. Use smaller models when quality/speed tradeoff acceptable. Consider model distillation. Cache model outputs.

### Key Metrics to Track
- Throughput (documents/second)
- P50/P95/P99 latency
- Memory usage per document
- Cache hit rate
- API cost per 1K documents
- CPU/GPU utilization

Profile first, optimize what matters, measure improvements.

## Anti-Patterns to Avoid

❌ Committing notebooks with outputs  
❌ Using `print()` instead of logger  
❌ Hardcoding configuration values  
❌ Committing data or model files  
❌ Running as root in Docker  
❌ No tests or <50% coverage  
❌ Not pinning dependency versions  
❌ Mixing source and test code  
❌ Large monolithic functions  
❌ No type hints  
❌ Premature optimization (profile first!)  
❌ Blocking I/O in async code  
❌ Loading entire files into memory  
❌ No caching for expensive operations  

Follow these practices and your codebase will be maintainable, deployable, and production-ready from day one.