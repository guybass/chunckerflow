# Contributing to ChunkFlow

Thank you for your interest in contributing to ChunkFlow! This document provides guidelines and instructions for contributing.

## Code of Conduct

Be respectful, inclusive, and collaborative. We're building this for the community, and we welcome contributors of all backgrounds and skill levels.

## How to Contribute

### Reporting Bugs

- Use GitHub Issues to report bugs
- Include clear description, steps to reproduce, expected vs actual behavior
- Provide Python version, OS, and relevant dependencies
- Include code snippets or minimal reproducible examples

### Suggesting Features

- Open a GitHub Issue with the "enhancement" label
- Describe the use case and why it's valuable
- Provide examples of how it would be used
- Discuss implementation approach if you have ideas

### Pull Requests

1. **Fork the repository** and create a branch from `main`
2. **Install development dependencies**: `make install-dev`
3. **Make your changes** following our coding standards
4. **Add tests** for new functionality
5. **Run the test suite**: `make test`
6. **Format and lint**: `make format && make lint`
7. **Commit with clear messages** (see Commit Guidelines below)
8. **Push to your fork** and submit a Pull Request

## Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR-USERNAME/chunk-flow.git
cd chunk-flow

# Install with development dependencies
make install-dev

# Set up pre-commit hooks
pre-commit install

# Run tests
make test

# Format code
make format

# Lint code
make lint
```

## Coding Standards

### Python Style

- Follow PEP 8 (enforced by Black and Ruff)
- Use type hints for all functions
- Write docstrings for all public APIs (Google style)
- Maximum line length: 100 characters
- Use f-strings for formatting

### Code Quality

- **NO print statements** - use structured logging (structlog/logging)
- **Type hints required** for all function signatures
- **Docstrings required** for all public functions/classes
- **Tests required** for all new functionality
- **Aim for 80%+ test coverage**

### Example Code

```python
import structlog
from typing import List, Optional

logger = structlog.get_logger()


async def chunk_text(
    text: str,
    chunk_size: int = 512,
    overlap: int = 0,
) -> List[str]:
    """
    Chunk text into segments.

    Args:
        text: Input text to chunk
        chunk_size: Maximum size of each chunk in characters
        overlap: Number of characters to overlap between chunks

    Returns:
        List of text chunks

    Raises:
        ValueError: If chunk_size <= 0 or overlap < 0

    Example:
        >>> chunks = await chunk_text("Hello world", chunk_size=5)
        >>> len(chunks)
        3
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    logger.info("chunking_text", text_length=len(text), chunk_size=chunk_size)

    # Implementation...
    return chunks
```

## Commit Guidelines

We use Conventional Commits for clear history:

- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `style:` Code style changes (formatting, no logic change)
- `refactor:` Code refactoring
- `test:` Adding or updating tests
- `chore:` Maintenance tasks (dependencies, build)
- `perf:` Performance improvements

Examples:
```
feat: add semantic chunking strategy
fix: handle empty documents in recursive chunker
docs: update embedding provider guide
test: add integration tests for OpenAI provider
```

## Testing Guidelines

### Test Structure

```
tests/
├── unit/           # Fast, isolated tests with mocked dependencies
├── integration/    # Tests with real dependencies (APIs, databases)
├── e2e/            # Full workflow tests
└── benchmarks/     # Performance benchmarks
```

### Writing Tests

```python
import pytest
from chunk_flow.chunking import RecursiveCharacterChunker


class TestRecursiveCharacterChunker:
    """Test suite for RecursiveCharacterChunker."""

    def test_chunks_simple_text(self) -> None:
        """Test chunking of simple text."""
        chunker = RecursiveCharacterChunker(config={"chunk_size": 10})
        text = "Hello world! This is a test."

        result = await chunker.chunk(text)

        assert len(result.chunks) > 0
        assert all(len(chunk) <= 10 for chunk in result.chunks)

    @pytest.mark.integration
    async def test_with_real_api(self) -> None:
        """Integration test with real API."""
        # Test with actual API calls
        pass
```

### Test Markers

- `@pytest.mark.slow` - Slow tests (skip in CI)
- `@pytest.mark.integration` - Integration tests
- `@pytest.mark.e2e` - End-to-end tests
- `@pytest.mark.benchmark` - Benchmark tests

## Documentation

- Update docstrings for any changed functions
- Add examples to documentation for new features
- Update README.md if adding major features
- Create tutorial notebooks for complex features

## Pull Request Process

1. **Update CHANGELOG.md** with your changes
2. **Ensure all tests pass** (`make test`)
3. **Ensure code is formatted** (`make format`)
4. **Ensure linting passes** (`make lint`)
5. **Update documentation** if needed
6. **Request review** from maintainers
7. **Address feedback** and update PR

### PR Checklist

- [ ] Tests added/updated
- [ ] Documentation updated
- [ ] CHANGELOG.md updated
- [ ] Code formatted (Black + isort)
- [ ] Linting passes (Ruff + mypy)
- [ ] All tests pass
- [ ] No merge conflicts
- [ ] Descriptive commit messages

## Adding New Components

### New Chunking Strategy

1. Create file in `chunk_flow/chunking/strategies/`
2. Inherit from `ChunkingStrategy`
3. Implement abstract methods
4. Add tests in `tests/unit/chunking/`
5. Register in strategy registry
6. Add documentation

### New Embedding Provider

1. Create file in `chunk_flow/embeddings/providers/`
2. Inherit from `EmbeddingProvider`
3. Implement abstract methods
4. Add tests (with mocked API calls)
5. Add to factory
6. Update documentation

### New Evaluation Metric

1. Create file in `chunk_flow/evaluation/metrics/`
2. Inherit from `EvaluationMetric`
3. Implement `compute()` method
4. Add tests with known ground truth
5. Register in metric registry
6. Add documentation

## Questions?

- Open a GitHub Discussion
- Tag maintainers in issues
- Join our community chat (link TBD)

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to ChunkFlow! Together we're building the framework the RAG community needs.
