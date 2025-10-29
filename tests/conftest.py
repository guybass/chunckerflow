"""Pytest configuration and fixtures."""

import pytest


@pytest.fixture
def sample_text():
    """Sample text for testing."""
    return """
    # Introduction to Machine Learning
    
    Machine learning is a subset of artificial intelligence that enables
    computers to learn from data without explicit programming.
    
    ## Types of Machine Learning
    
    There are three main types: supervised learning, unsupervised learning,
    and reinforcement learning. Each has different use cases and approaches.
    
    ## Applications
    
    Machine learning powers recommendation systems, fraud detection, image
    recognition, and natural language processing applications.
    """


@pytest.fixture
def sample_chunks():
    """Sample chunks for testing."""
    return [
        "Introduction to Machine Learning",
        "Machine learning is a subset of artificial intelligence.",
        "Types: supervised, unsupervised, reinforcement learning.",
        "Applications include recommendations and fraud detection.",
    ]


@pytest.fixture
def sample_embeddings():
    """Sample embeddings for testing."""
    return [
        [0.1, 0.2, 0.3, 0.4],
        [0.2, 0.3, 0.4, 0.5],
        [0.15, 0.25, 0.35, 0.45],
        [0.3, 0.4, 0.5, 0.6],
    ]


@pytest.fixture
def sample_query_embedding():
    """Sample query embedding for testing."""
    return [0.2, 0.3, 0.4, 0.5]


@pytest.fixture
def sample_ground_truth(sample_query_embedding):
    """Sample ground truth for testing."""
    return {
        "query_embedding": sample_query_embedding,
        "relevant_indices": [0, 1],
    }
