"""Document-based chunking for structured content (Markdown, HTML, Code)."""

import re
import time
from typing import Any, Dict, List, Optional, Tuple

from chunk_flow.core.base import ChunkingStrategy
from chunk_flow.core.exceptions import ChunkingError, ConfigurationError
from chunk_flow.core.models import ChunkMetadata, ChunkResult
from chunk_flow.utils.logging import get_logger

logger = get_logger(__name__)


class MarkdownChunker(ChunkingStrategy):
    """
    Markdown-aware chunking strategy.

    Splits text based on Markdown header hierarchy, preserving logical structure.
    Creates chunks at header boundaries while respecting document organization.

    Best for: Markdown documentation, README files, technical docs.
    """

    VERSION = "1.0.0"
    NAME = "markdown"

    def get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            "headers_to_split": ["#", "##", "###", "####"],
            "max_chunk_size": 2000,
            "include_header_in_chunk": True,
        }

    def _validate_config(self) -> None:
        """Validate configuration."""
        if "headers_to_split" not in self.config:
            raise ConfigurationError("Missing required config: headers_to_split")

    async def chunk(self, text: str, doc_id: Optional[str] = None) -> ChunkResult:
        """
        Chunk Markdown text by headers.

        Args:
            text: Markdown text to chunk
            doc_id: Optional document identifier

        Returns:
            ChunkResult with chunks and metadata
        """
        self.validate_input(text)
        start_time = time.time()

        try:
            headers_to_split = self.config["headers_to_split"]
            max_chunk_size = self.config.get("max_chunk_size", 2000)
            include_header = self.config.get("include_header_in_chunk", True)

            # Parse headers and content
            sections = await self._parse_markdown_sections(text, headers_to_split)

            # Create chunks from sections
            chunks: List[str] = []
            metadata: List[ChunkMetadata] = []

            for i, (header, content, start_idx) in enumerate(sections):
                if include_header and header:
                    chunk_text = f"{header}\n{content}"
                else:
                    chunk_text = content

                # Split large sections if needed
                if len(chunk_text) > max_chunk_size:
                    sub_chunks = self._split_large_section(chunk_text, max_chunk_size)
                    for j, sub_chunk in enumerate(sub_chunks):
                        chunks.append(sub_chunk)
                        metadata.append(
                            ChunkMetadata(
                                chunk_id=f"{doc_id or 'doc'}_{self.NAME}_{i}_{j}",
                                start_idx=start_idx,
                                end_idx=start_idx + len(sub_chunk),
                                token_count=len(sub_chunk.split()),
                                char_count=len(sub_chunk),
                                version=self.VERSION,
                                strategy_name=self.NAME,
                                custom_fields={"header": header, "section_index": i},
                            )
                        )
                else:
                    if chunk_text.strip():
                        chunks.append(chunk_text)
                        metadata.append(
                            ChunkMetadata(
                                chunk_id=f"{doc_id or 'doc'}_{self.NAME}_{i}",
                                start_idx=start_idx,
                                end_idx=start_idx + len(chunk_text),
                                token_count=len(chunk_text.split()),
                                char_count=len(chunk_text),
                                version=self.VERSION,
                                strategy_name=self.NAME,
                                custom_fields={"header": header, "section_index": i},
                            )
                        )

            processing_time = (time.time() - start_time) * 1000

            logger.info(
                "markdown_chunking_completed",
                num_chunks=len(chunks),
                num_sections=len(sections),
                processing_time_ms=processing_time,
            )

            return ChunkResult(
                chunks=chunks,
                metadata=metadata,
                processing_time_ms=processing_time,
                strategy_version=self.VERSION,
                config=self.config,
                doc_id=doc_id,
            )

        except Exception as e:
            logger.error("markdown_chunking_failed", error=str(e), exc_info=True)
            raise ChunkingError(f"Markdown chunking failed: {str(e)}") from e

    async def _parse_markdown_sections(
        self, text: str, headers: List[str]
    ) -> List[Tuple[str, str, int]]:
        """
        Parse Markdown into sections based on headers.

        Returns:
            List of (header, content, start_position) tuples
        """
        sections: List[Tuple[str, str, int]] = []

        # Create regex pattern for headers
        header_pattern = "|".join(re.escape(h) + r"\s" for h in headers)
        pattern = f"^({header_pattern})"

        lines = text.split("\n")
        current_header = ""
        current_content: List[str] = []
        current_start = 0

        for i, line in enumerate(lines):
            if re.match(pattern, line):
                # Save previous section
                if current_content or current_header:
                    sections.append(
                        (current_header, "\n".join(current_content), current_start)
                    )

                # Start new section
                current_header = line
                current_content = []
                current_start = sum(len(l) + 1 for l in lines[:i])  # +1 for newline
            else:
                current_content.append(line)

        # Add final section
        if current_content or current_header:
            sections.append((current_header, "\n".join(current_content), current_start))

        return sections

    def _split_large_section(self, text: str, max_size: int) -> List[str]:
        """Split large section into smaller chunks."""
        chunks: List[str] = []
        paragraphs = text.split("\n\n")

        current_chunk: List[str] = []
        current_size = 0

        for para in paragraphs:
            para_size = len(para)

            if current_size + para_size <= max_size:
                current_chunk.append(para)
                current_size += para_size
            else:
                if current_chunk:
                    chunks.append("\n\n".join(current_chunk))

                if para_size > max_size:
                    # Split very large paragraph
                    for i in range(0, len(para), max_size):
                        chunks.append(para[i : i + max_size])
                    current_chunk = []
                    current_size = 0
                else:
                    current_chunk = [para]
                    current_size = para_size

        if current_chunk:
            chunks.append("\n\n".join(current_chunk))

        return chunks


class HTMLChunker(ChunkingStrategy):
    """
    HTML-aware chunking strategy.

    Splits HTML based on tags, preserving document structure.

    Best for: Web pages, HTML documentation.
    """

    VERSION = "1.0.0"
    NAME = "html"

    def get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            "tags_to_split": ["h1", "h2", "h3", "h4", "p", "div", "section", "article"],
            "max_chunk_size": 2000,
            "strip_tags": True,
        }

    def _validate_config(self) -> None:
        """Validate configuration."""
        if "tags_to_split" not in self.config:
            raise ConfigurationError("Missing required config: tags_to_split")

    async def chunk(self, text: str, doc_id: Optional[str] = None) -> ChunkResult:
        """
        Chunk HTML text by tags.

        Args:
            text: HTML text to chunk
            doc_id: Optional document identifier

        Returns:
            ChunkResult with chunks and metadata
        """
        self.validate_input(text)
        start_time = time.time()

        try:
            # Simple HTML parsing (for production, consider using BeautifulSoup)
            tags = self.config["tags_to_split"]
            strip_tags = self.config.get("strip_tags", True)

            chunks: List[str] = []
            metadata: List[ChunkMetadata] = []

            # Simple tag-based splitting
            for tag in tags:
                pattern = f"<{tag}[^>]*>(.*?)</{tag}>"
                matches = re.finditer(pattern, text, re.DOTALL | re.IGNORECASE)

                for i, match in enumerate(matches):
                    content = match.group(1).strip()

                    if strip_tags:
                        # Remove nested tags
                        content = re.sub(r"<[^>]+>", "", content)

                    if content:
                        chunks.append(content)
                        metadata.append(
                            ChunkMetadata(
                                chunk_id=f"{doc_id or 'doc'}_{self.NAME}_{tag}_{i}",
                                start_idx=match.start(),
                                end_idx=match.end(),
                                token_count=len(content.split()),
                                char_count=len(content),
                                version=self.VERSION,
                                strategy_name=self.NAME,
                                custom_fields={"tag": tag},
                            )
                        )

            # If no chunks found, fall back to simple splitting
            if not chunks:
                chunks = [text]
                metadata = [
                    ChunkMetadata(
                        chunk_id=f"{doc_id or 'doc'}_{self.NAME}_0",
                        start_idx=0,
                        end_idx=len(text),
                        token_count=len(text.split()),
                        char_count=len(text),
                        version=self.VERSION,
                        strategy_name=self.NAME,
                    )
                ]

            processing_time = (time.time() - start_time) * 1000

            return ChunkResult(
                chunks=chunks,
                metadata=metadata,
                processing_time_ms=processing_time,
                strategy_version=self.VERSION,
                config=self.config,
                doc_id=doc_id,
            )

        except Exception as e:
            logger.error("html_chunking_failed", error=str(e), exc_info=True)
            raise ChunkingError(f"HTML chunking failed: {str(e)}") from e
