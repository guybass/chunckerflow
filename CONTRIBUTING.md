# Contributing to ChunkFlow

Thank you for your interest in ChunkFlow!

## Project Status

ChunkFlow is currently a **solo project** developed and maintained by a single author. While the codebase is open-source and available for use under the MIT License, **contributions are not currently being accepted**.

## How You Can Help

While we're not accepting code contributions at this time, you can still help:

- **Report Bugs**: Open GitHub Issues with detailed bug reports
- **Suggest Features**: Share your ideas via GitHub Issues
- **Spread the Word**: Star the repo, share on social media
- **Use ChunkFlow**: Build amazing things and share your experiences

## Reporting Bugs

When reporting bugs, please include:

- Clear description of the issue
- Steps to reproduce the behavior
- Expected vs actual behavior
- Python version, OS, and relevant dependencies
- Code snippets or minimal reproducible examples
- Full error traceback

**Example Bug Report:**

```
Title: RecursiveChunker fails on empty input

Description:
The RecursiveCharacterChunker raises an unexpected IndexError
when given an empty string instead of returning an empty list.

Steps to Reproduce:
1. Create RecursiveCharacterChunker with default config
2. Call chunk("") with empty string
3. See error

Expected: Returns ChunkResult with empty chunks list
Actual: Raises IndexError

Environment:
- Python 3.11.2
- chunk-flow 0.1.0
- OS: Windows 11

Error Traceback:
IndexError: string index out of range
  at chunk_flow/chunking/strategies/recursive.py:45
```

## Suggesting Features

When suggesting features, please include:

- **Use case**: Describe the problem you're trying to solve
- **Proposed solution**: How should this feature work?
- **Examples**: Show how you'd use this feature
- **Research**: Link to relevant papers or implementations
- **Alternatives**: What alternatives did you consider?

**Example Feature Request:**

```
Title: Add support for PDF chunking

Use Case:
Users want to chunk PDF documents directly without manual extraction.

Proposed Solution:
New PDFChunker strategy that extracts text and respects PDF structure
(pages, sections, etc.)

Example Usage:
chunker = StrategyRegistry.create("pdf", {"respect_pages": True})
result = await chunker.chunk_file("document.pdf")

Related Research:
- PyPDF2 library for PDF extraction
- Similar to MarkdownChunker but for PDF structure

Alternatives:
- Users manually extract with PyPDF2 then use existing chunkers
```

## Code Standards (For Reference)

If you're forking or learning from the codebase, here are the standards used:

### Python Style
- Follow PEP 8 (enforced by Black and Ruff)
- Type hints for all functions
- Google-style docstrings for all public APIs
- Maximum line length: 100 characters
- Use f-strings for formatting

### Code Quality
- NO print statements - use structured logging (structlog)
- Type hints required for all function signatures
- Docstrings required for all public functions/classes
- Comprehensive test coverage
- Async-first design

### Example

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

## Questions?

- **GitHub Issues**: For bugs and feature requests
- **GitHub Discussions**: For general questions and discussions
- **Documentation**: Check the `/docs` folder for guides

## License

By using ChunkFlow, you agree to the terms of the MIT License. See [LICENSE](LICENSE) for details.

---

**Built with passion for the neglected field of text chunking** 🚀
