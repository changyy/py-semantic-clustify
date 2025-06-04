"""
Basic smoke tests for semantic-clustify.
"""

import pytest
from semantic_clustify import SemanticClusterer


@pytest.mark.smoke
def test_import():
    """Test that we can import the main classes."""
    assert SemanticClusterer is not None


@pytest.mark.smoke 
def test_clusterer_creation():
    """Test basic clusterer creation."""
    clusterer = SemanticClusterer(method="kmeans")
    assert clusterer is not None
    assert clusterer.method == "kmeans"


@pytest.mark.quick
def test_supported_methods():
    """Test that all supported methods are available."""
    expected_methods = ["kmeans", "dbscan", "hierarchical", "gmm"]
    
    for method in expected_methods:
        clusterer = SemanticClusterer(method=method)
        assert clusterer.method == method


@pytest.mark.core
def test_basic_clustering():
    """Test basic clustering functionality."""
    # Sample data with vectors
    data = [
        {"title": "Doc 1", "embedding": [0.1, 0.2, 0.3]},
        {"title": "Doc 2", "embedding": [0.15, 0.25, 0.35]},
        {"title": "Doc 3", "embedding": [0.8, 0.1, 0.2]},
        {"title": "Doc 4", "embedding": [0.85, 0.15, 0.25]},
    ]
    
    clusterer = SemanticClusterer(method="kmeans", n_clusters=2)
    clusters = clusterer.fit_predict(data, vector_field="embedding")
    
    assert isinstance(clusters, list)
    assert len(clusters) <= 2  # Should have at most 2 clusters
    assert sum(len(cluster) for cluster in clusters) == len(data)


if __name__ == "__main__":
    pytest.main([__file__])
