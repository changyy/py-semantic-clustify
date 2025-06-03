"""
Pytest configuration and shared fixtures for semantic-clustify tests.
"""

import pytest
import numpy as np
import tempfile
import json
import glob
import os
from pathlib import Path
from typing import List, Dict, Any

from semantic_clustify import SemanticClusterer


@pytest.fixture(autouse=True)
def cleanup_temp_files():
    """Automatically clean up temporary output files generated during CLI tests."""
    # Store initial set of files
    initial_files = set(glob.glob("*.jsonl"))

    yield  # Run the test

    # Clean up any new .jsonl files that match CLI output patterns
    current_files = set(glob.glob("*.jsonl"))
    new_files = current_files - initial_files

    cli_output_patterns = [
        "_kmeans_grouped_",
        "_dbscan_grouped_",
        "_hierarchical_grouped_",
        "_gmm_grouped_",
        "_kmeans_labeled_",
        "_dbscan_labeled_",
        "_hierarchical_labeled_",
        "_gmm_labeled_",
        "tmp",
        "clustered_output_",
    ]

    for file_path in new_files:
        # Check if it matches any CLI output pattern
        if any(pattern in file_path for pattern in cli_output_patterns):
            try:
                os.unlink(file_path)
                print(f"Cleaned up temporary file: {file_path}")
            except OSError:
                pass  # File might already be removed


@pytest.fixture
def sample_vectors_2d():
    """Generate 2D sample vectors with clear clusters for testing."""
    np.random.seed(42)

    # Create three distinct clusters
    cluster1 = np.random.normal([0, 0], 0.2, (15, 2))
    cluster2 = np.random.normal([3, 3], 0.2, (15, 2))
    cluster3 = np.random.normal([0, 3], 0.2, (10, 2))

    return np.vstack([cluster1, cluster2, cluster3])


@pytest.fixture
def sample_vectors_high_dim():
    """Generate high-dimensional sample vectors for testing."""
    np.random.seed(42)

    # Create two clusters in 50-dimensional space
    cluster1 = np.random.normal(0, 1, (20, 50))
    cluster2 = np.random.normal(2, 1, (20, 50))

    return np.vstack([cluster1, cluster2])


@pytest.fixture
def sample_documents():
    """Sample text documents with embeddings."""
    return [
        {
            "text": "Machine learning algorithms for data analysis",
            "embedding": [0.1, 0.2, 0.3, 0.4, 0.5],
            "category": "technology",
        },
        {
            "text": "Deep neural networks and artificial intelligence",
            "embedding": [0.15, 0.25, 0.35, 0.45, 0.55],
            "category": "technology",
        },
        {
            "text": "Statistical methods for scientific research",
            "embedding": [0.2, 0.3, 0.4, 0.5, 0.6],
            "category": "science",
        },
        {
            "text": "Python programming for beginners tutorial",
            "embedding": [0.05, 0.15, 0.25, 0.35, 0.45],
            "category": "programming",
        },
        {
            "text": "Quantum computing and quantum algorithms",
            "embedding": [0.8, 0.7, 0.6, 0.5, 0.4],
            "category": "science",
        },
        {
            "text": "Web development with modern frameworks",
            "embedding": [0.08, 0.18, 0.28, 0.38, 0.48],
            "category": "programming",
        },
    ]


@pytest.fixture
def temp_jsonl_file(sample_documents):
    """Create a temporary JSONL file with sample documents."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        for doc in sample_documents:
            json.dump(doc, f)
            f.write("\n")
        temp_path = f.name

    yield temp_path

    # Cleanup
    Path(temp_path).unlink(missing_ok=True)


@pytest.fixture
def clusterer_factory():
    """Factory function to create clusterers with different configurations."""

    def _create_clusterer(method: str, **kwargs):
        return SemanticClusterer(method=method, **kwargs)

    return _create_clusterer


@pytest.fixture(params=["kmeans", "dbscan", "hierarchical", "gmm"])
def all_algorithms(request):
    """Parametrized fixture to test all clustering algorithms."""
    return request.param


@pytest.fixture
def cluster_quality_threshold():
    """Minimum quality thresholds for clustering results."""
    return {
        "silhouette_score": 0.0,  # Minimum acceptable silhouette score
        "min_clusters": 1,  # Minimum number of clusters
        "max_clusters": 10,  # Maximum reasonable number of clusters
    }
