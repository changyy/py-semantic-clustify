#!/usr/bin/env python3
"""
Hybrid DBSCAN + K-Means Clustering Algorithm Complete Demo

This demo shows how to use the hybrid clustering algorithm to process news data,
particularly suitable for Taiwan's 24-hour news clustering scenarios.
"""

import numpy as np
import json
import sys
from datetime import datetime
from pathlib import Path

# Add project root directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from semantic_clustify.core import SemanticClusterer
from semantic_clustify.algorithms import HybridDBSCANKMeansClusterer


def create_news_like_vectors(n_samples: int = 120) -> tuple:
    """
    Create news-like vector data, simulating Taiwan's 24-hour news scenario
    
    Args:
        n_samples: Number of news articles to generate
        
    Returns:
        tuple: (vector array, true labels, article list)
    """
    np.random.seed(42)
    
    # Simulate major events (will have many related reports)
    major_events = [
        {"center": np.array([1.0, 0.0, 0.0, 0.0, 0.0]), "count": 25, "name": "Major Political Event"},
        {"center": np.array([0.0, 1.0, 0.0, 0.0, 0.0]), "count": 20, "name": "Economic Policy News"},
        {"center": np.array([0.0, 0.0, 1.0, 0.0, 0.0]), "count": 15, "name": "Social Events"},
    ]
    
    # Simulate minor events (only a few reports)
    minor_events = [
        {"center": np.array([0.0, 0.0, 0.0, 1.0, 0.0]), "count": 8, "name": "Sports News"},
        {"center": np.array([0.0, 0.0, 0.0, 0.0, 1.0]), "count": 7, "name": "Technology News"},
        {"center": np.array([0.5, 0.5, 0.0, 0.0, 0.0]), "count": 6, "name": "Entertainment News"},
        {"center": np.array([0.0, 0.0, 0.5, 0.5, 0.0]), "count": 5, "name": "International News"},
        {"center": np.array([0.3, 0.0, 0.0, 0.0, 0.7]), "count": 4, "name": "Health News"},
    ]
    
    vectors = []
    labels = []
    articles = []
    
    current_label = 0
    
    # Generate major event data
    for event in major_events:
        for i in range(event["count"]):
            noise = np.random.normal(0, 0.1, 5)
            vector = event["center"] + noise
            vectors.append(vector)
            labels.append(current_label)
            articles.append({
                "title": f"{event['name']} - Report {i+1}",
                "content": f"This is the {i+1}th report about {event['name']}, containing detailed background information and latest developments.",
                "source": f"Media{np.random.randint(1, 10)}",
                "embedding": vector.tolist()
            })
        current_label += 1
    
    # Generate minor event data
    for event in minor_events:
        for i in range(event["count"]):
            noise = np.random.normal(0, 0.15, 5)
            vector = event["center"] + noise
            vectors.append(vector)
            labels.append(current_label)
            articles.append({
                "title": f"{event['name']} - Report {i+1}",
                "content": f"This is the {i+1}th report about {event['name']}, providing relevant information and analysis.",
                "source": f"Media{np.random.randint(1, 10)}",
                "embedding": vector.tolist()
            })
        current_label += 1
    
    # Add some noise data (independent small news)
    remaining = n_samples - len(vectors)
    for i in range(remaining):
        vector = np.random.normal(0, 0.3, 5)
        vectors.append(vector)
        labels.append(-1)  # Noise label
        articles.append({
            "title": f"Independent News {i+1}",
            "content": f"This is the {i+1}th independent small news, covering various different topics.",
            "source": f"Media{np.random.randint(1, 10)}",
            "embedding": vector.tolist()
        })
    
    return np.array(vectors), np.array(labels), articles


def test_hybrid_algorithm():
    """Test the core functionality of the hybrid algorithm"""
    print("🧪 Testing Hybrid DBSCAN + K-Means Algorithm")
    print("=" * 60)
    
    # Create test data
    vectors, true_labels, articles = create_news_like_vectors(120)
    print(f"📊 Generated {len(articles)} simulated news articles")
    
    # Use hybrid algorithm
    clusterer = SemanticClusterer(
        method="hybrid-dbscan-kmeans",
        target_clusters=30,
        major_event_threshold=10,
        dbscan_eps=0.2,
        min_cluster_size=3
    )
    
    print("\n🔄 Executing hybrid clustering...")
    clusters = clusterer.fit_predict(articles, vector_field="embedding")
    
    print(f"✅ Clustering completed!")
    print(f"   - Final cluster count: {len(clusters)}")
    
    # Display stage information
    if hasattr(clusterer.algorithm, 'get_stage_info'):
        stage_info = clusterer.algorithm.get_stage_info()
        print(f"\n📋 Detailed stage information:")
        print(f"   - DBSCAN discovered clusters: {stage_info.get('dbscan', {}).get('n_clusters', 'N/A')}")
        print(f"   - DBSCAN noise points: {stage_info.get('dbscan', {}).get('n_noise', 'N/A')}")
        print(f"   - Major events count: {stage_info.get('analysis', {}).get('major_clusters', 'N/A')}")
        print(f"   - Documents to reorganize: {stage_info.get('analysis', {}).get('minor_documents', 'N/A')}")
        print(f"   - Target cluster count: {stage_info.get('kmeans', {}).get('target_clusters', 'N/A')}")
    
    # Display cluster size distribution
    print(f"\n📈 Cluster size distribution:")
    cluster_sizes = [len(cluster) for cluster in clusters]
    cluster_sizes.sort(reverse=True)
    for i, size in enumerate(cluster_sizes[:10]):  # Only show top 10
        print(f"   Cluster {i+1}: {size} articles")
    
    # Display quality metrics
    metrics = clusterer.get_quality_metrics()
    print(f"\n📊 Clustering quality metrics:")
    print(f"   - Silhouette Score: {metrics.get('silhouette_score', 'N/A'):.4f}")
    print(f"   - Calinski-Harabasz Score: {metrics.get('calinski_harabasz_score', 'N/A'):.2f}")
    print(f"   - Davies-Bouldin Score: {metrics.get('davies_bouldin_score', 'N/A'):.4f}")
    
    # Display some cluster examples
    print(f"\n📰 Cluster examples (top 3 largest clusters):")
    for i, cluster in enumerate(clusters[:3]):
        print(f"\n   Cluster {i+1} ({len(cluster)} articles):")
        for j, article in enumerate(cluster[:3]):  # Show first 3 articles per cluster
            print(f"     - {article['title']}")
        if len(cluster) > 3:
            print(f"     ... and {len(cluster) - 3} more articles")
    
    return clusters


def create_test_data_file():
    """Create test JSONL file"""
    print("\n💾 Creating test data file...")
    vectors, true_labels, articles = create_news_like_vectors(120)
    
    # Create test file in current directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"hybrid_demo_news_{timestamp}.jsonl"
    
    with open(filename, 'w', encoding='utf-8') as f:
        for article in articles:
            json.dump(article, f, ensure_ascii=False)
            f.write('\n')
    
    print(f"✅ Test data saved to {filename} ({len(articles)} articles)")
    return filename


def main():
    """Main function"""
    print("🚀 Hybrid Clustering Algorithm Complete Demo")
    print("=" * 60)
    
    # Execute core functionality test
    clusters = test_hybrid_algorithm()
    
    # Create test data file
    test_file = create_test_data_file()
    
    print("\n" + "=" * 60)
    print("✅ Demo completed! You can test the CLI with the following command:")
    print(f"\nsemantic-clustify cluster --algorithm hybrid-dbscan-kmeans \\")
    print(f"  --target-clusters 30 --embedding-field embedding \\")
    print(f"  --major-event-threshold 8 --quality-metrics \\")
    print(f"  --verbose {test_file}")
    
    print(f"\n💡 Tip: Test file has been saved as {test_file}")
    print("We recommend manually deleting the test file after use to keep the directory clean.")


if __name__ == "__main__":
    main()
