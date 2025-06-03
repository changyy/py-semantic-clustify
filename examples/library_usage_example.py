#!/usr/bin/env python3
"""
Library Usage Example for semantic-clustify

This script demonstrates how to use semantic-clustify as a Python library
rather than as a command-line tool.
"""

import json
import numpy as np
from semantic_clustify import SemanticClusterer

def main():
    """Demonstrate library usage with different input/output formats."""
    
    print("🔬 semantic-clustify Library Usage Example")
    print("=" * 50)
    
    # Example 1: List[Dict] input -> List[List[Dict]] output
    print("\n📊 Example 1: List[Dict] -> List[List[Dict]]")
    print("-" * 40)
    
    # Sample data as List[Dict]
    documents = [
        {"text": "Machine learning algorithms", "embedding": [0.1, 0.2, 0.3, 0.4, 0.5]},
        {"text": "Deep neural networks", "embedding": [0.15, 0.25, 0.35, 0.45, 0.55]},
        {"text": "Python programming", "embedding": [0.8, 0.7, 0.6, 0.5, 0.4]},
        {"text": "Web development", "embedding": [0.75, 0.65, 0.55, 0.45, 0.35]},
        {"text": "Data visualization", "embedding": [0.2, 0.3, 0.4, 0.5, 0.6]},
        {"text": "Statistical analysis", "embedding": [0.25, 0.35, 0.45, 0.55, 0.65]},
    ]
    
    # Initialize clusterer
    clusterer = SemanticClusterer(
        method="kmeans",
        n_clusters=2,
        random_state=42
    )
    
    # Perform clustering
    clusters = clusterer.fit_predict(documents, vector_field="embedding")
    
    print(f"Input: {len(documents)} documents")
    print(f"Output: {len(clusters)} clusters")
    for i, cluster in enumerate(clusters):
        print(f"  Cluster {i}: {len(cluster)} documents")
        for doc in cluster:
            print(f"    - {doc['text']}")
    
    # Example 2: JSONL string input -> List[List[Dict]] output
    print("\n📊 Example 2: JSONL string -> List[List[Dict]]")
    print("-" * 40)
    
    # JSONL format data
    jsonl_data = '\n'.join([json.dumps(doc) for doc in documents])
    print(f"JSONL input length: {len(jsonl_data.splitlines())} lines")
    
    # Parse JSONL and cluster
    parsed_docs = [json.loads(line) for line in jsonl_data.strip().split('\n')]
    clusters = clusterer.fit_predict(parsed_docs, vector_field="embedding")
    
    print(f"Output: {len(clusters)} clusters (same as before)")
    
    # Example 3: Different clustering algorithms
    print("\n📊 Example 3: Different Algorithms")
    print("-" * 40)
    
    algorithms = ["kmeans", "hierarchical", "gmm"]
    
    for method in algorithms:
        clusterer = SemanticClusterer(
            method=method,
            n_clusters=2,
            random_state=42
        )
        clusters = clusterer.fit_predict(documents, vector_field="embedding")
        print(f"{method.upper()}: {len(clusters)} clusters")
    
    # Example 4: Auto-detect optimal clusters
    print("\n📊 Example 4: Auto-detect Clusters")
    print("-" * 40)
    
    clusterer = SemanticClusterer(
        method="kmeans",
        n_clusters="auto",
        max_clusters=4,
        random_state=42
    )
    
    clusters = clusterer.fit_predict(documents, vector_field="embedding")
    quality = clusterer.get_quality_metrics()
    
    print(f"Auto-detected clusters: {len(clusters)}")
    print(f"Silhouette score: {quality.get('silhouette_score', 'N/A'):.3f}")
    print(f"Calinski-Harabasz score: {quality.get('calinski_harabasz_score', 'N/A'):.3f}")
    
    # Example 5: Working with different vector fields
    print("\n📊 Example 5: Different Vector Fields")
    print("-" * 40)
    
    docs_with_multiple_vectors = [
        {
            "text": "Machine learning",
            "text_embedding": [0.1, 0.2, 0.3],
            "title_embedding": [0.4, 0.5, 0.6]
        },
        {
            "text": "Deep learning",
            "text_embedding": [0.15, 0.25, 0.35],
            "title_embedding": [0.45, 0.55, 0.65]
        },
        {
            "text": "Programming",
            "text_embedding": [0.8, 0.7, 0.6],
            "title_embedding": [0.9, 0.8, 0.7]
        },
        {
            "text": "Development",
            "text_embedding": [0.75, 0.65, 0.55],
            "title_embedding": [0.85, 0.75, 0.65]
        }
    ]
    
    # Cluster using text embeddings
    clusterer_text = SemanticClusterer(method="kmeans", n_clusters=2)
    clusters_text = clusterer_text.fit_predict(docs_with_multiple_vectors, vector_field="text_embedding")
    
    # Cluster using title embeddings
    clusterer_title = SemanticClusterer(method="kmeans", n_clusters=2)
    clusters_title = clusterer_title.fit_predict(docs_with_multiple_vectors, vector_field="title_embedding")
    
    print(f"Text embedding clustering: {len(clusters_text)} clusters")
    print(f"Title embedding clustering: {len(clusters_title)} clusters")
    
    print("\n✅ Library usage examples completed!")
    print("\n💡 Key Features Demonstrated:")
    print("   ✓ List[Dict] input/output format")
    print("   ✓ JSONL input/output format")
    print("   ✓ Multiple clustering algorithms")
    print("   ✓ Automatic cluster optimization")
    print("   ✓ Quality metrics")
    print("   ✓ Flexible vector field selection")

if __name__ == "__main__":
    main()
