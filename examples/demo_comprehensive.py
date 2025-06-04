#!/usr/bin/env python3
"""
Comprehensive Demo Script for semantic-clustify

This script demonstrates all major features of the semantic-clustify package,
including different algorithms, parameter optimization, and quality metrics.
"""

import json
import numpy as np
from pathlib import Path
import tempfile

# Import semantic-clustify components
from semantic_clustify import SemanticClusterer
from semantic_clustify.utils import save_jsonl


def create_sample_data():
    """Create sample text data with realistic embeddings."""
    # Technology cluster
    tech_docs = [
        "Machine learning algorithms for data analysis",
        "Deep neural networks and artificial intelligence",
        "Data visualization techniques and tools",
        "Natural language processing applications",
        "Blockchain technology and cryptocurrency",
    ]

    # Science cluster
    science_docs = [
        "Quantum computing and quantum algorithms",
        "Statistical methods for scientific research",
        "Mathematical analysis and calculus",
        "Physics experiments and laboratory techniques",
        "Chemistry molecular structure analysis",
    ]

    # Programming cluster
    programming_docs = [
        "Python programming for beginners tutorial",
        "Web development with modern frameworks",
        "Database design and optimization",
        "Software engineering best practices",
        "API development and microservices",
    ]

    all_docs = tech_docs + science_docs + programming_docs

    # Generate realistic embeddings (normally you'd use a real embedding model)
    np.random.seed(42)
    data = []

    for i, doc in enumerate(all_docs):
        if i < len(tech_docs):
            # Tech cluster embeddings (centered around [0.2, 0.3, 0.4])
            base = np.array([0.2, 0.3, 0.4, 0.5, 0.6])
            embedding = base + np.random.normal(0, 0.1, 5)
            category = "technology"
        elif i < len(tech_docs) + len(science_docs):
            # Science cluster embeddings (centered around [0.7, 0.6, 0.5])
            base = np.array([0.7, 0.6, 0.5, 0.4, 0.3])
            embedding = base + np.random.normal(0, 0.1, 5)
            category = "science"
        else:
            # Programming cluster embeddings (centered around [0.1, 0.2, 0.8])
            base = np.array([0.1, 0.2, 0.8, 0.7, 0.6])
            embedding = base + np.random.normal(0, 0.1, 5)
            category = "programming"

        data.append(
            {
                "text": doc,
                "embedding": embedding.tolist(),
                "true_category": category,
                "doc_id": i,
            }
        )

    return data


def demo_library_usage():
    """Demonstrate using semantic-clustify as a Python library."""
    print("\n" + "=" * 60)
    print("📚 LIBRARY USAGE DEMONSTRATION")
    print("=" * 60)

    # Create sample data
    data = create_sample_data()
    print(f"📊 Created {len(data)} sample documents")

    # Initialize clusterer
    clusterer = SemanticClusterer()

    # Test different algorithms
    algorithms = [
        ("KMeans", {"method": "kmeans", "n_clusters": 3}),
        ("DBSCAN", {"method": "dbscan", "eps": 0.3}),
        ("Hierarchical", {"method": "hierarchical", "n_clusters": 3}),
        ("GMM", {"method": "gmm", "n_clusters": 3}),
    ]

    results = {}

    for algo_name, params in algorithms:
        print(f"\n🔬 Testing {algo_name} clustering...")

        # Create clusterer with specific algorithm
        clusterer = SemanticClusterer(**params)
        clusters = clusterer.fit_predict(data=data, vector_field="embedding")

        # Get metrics
        metrics = clusterer.get_quality_metrics()

        result = {"clusters": clusters, "metrics": metrics}

        results[algo_name] = result

        print(f"   ✅ Clusters found: {len(result['clusters'])}")
        print(f"   📈 Silhouette score: {result['metrics']['silhouette_score']:.3f}")

        # Show cluster composition
        for i, cluster in enumerate(result["clusters"]):
            categories = [doc.get("true_category", "unknown") for doc in cluster]
            category_counts = {}
            for cat in categories:
                category_counts[cat] = category_counts.get(cat, 0) + 1
            print(f"   📂 Cluster {i}: {len(cluster)} docs - {category_counts}")

    return results


def demo_auto_optimization():
    """Demonstrate automatic cluster number optimization."""
    print("\n" + "=" * 60)
    print("🎯 AUTO-OPTIMIZATION DEMONSTRATION")
    print("=" * 60)

    data = create_sample_data()
    clusterer = SemanticClusterer()

    print("🔍 Testing automatic cluster number detection...")

    clusterer = SemanticClusterer(method="kmeans", n_clusters="auto", max_clusters=8)

    clusters = clusterer.fit_predict(data=data, vector_field="embedding")

    metrics = clusterer.get_quality_metrics()

    result = {"clusters": clusters, "metrics": metrics}

    optimal_k = len(result["clusters"])
    print(f"   ✅ Optimal cluster number detected: {optimal_k}")
    print(f"   📈 Silhouette score: {result['metrics']['silhouette_score']:.3f}")

    # Show cluster quality scores for different k values
    print("\n📊 Cluster quality comparison:")
    for k in range(2, 6):
        test_clusterer = SemanticClusterer(method="kmeans", n_clusters=k)
        test_clusters = test_clusterer.fit_predict(data=data, vector_field="embedding")
        test_metrics = test_clusterer.get_quality_metrics()
        score = test_metrics["silhouette_score"]
        marker = " ⭐" if k == optimal_k else ""
        print(f"   k={k}: silhouette={score:.3f}{marker}")


def demo_quality_metrics():
    """Demonstrate comprehensive quality metrics."""
    print("\n" + "=" * 60)
    print("📈 QUALITY METRICS DEMONSTRATION")
    print("=" * 60)

    data = create_sample_data()
    clusterer = SemanticClusterer(method="kmeans", n_clusters=3)

    clusters = clusterer.fit_predict(data=data, vector_field="embedding")

    metrics = clusterer.get_quality_metrics()

    result = {"clusters": clusters, "metrics": metrics}

    metrics = result["metrics"]

    print("🎯 Clustering Quality Metrics:")
    print(f"   📊 Silhouette Score: {metrics['silhouette_score']:.3f}")
    print(f"      (Range: -1 to 1, higher is better)")

    print(f"   📊 Calinski-Harabasz Score: {metrics['calinski_harabasz_score']:.2f}")
    print(f"      (Higher values indicate better defined clusters)")

    print(f"   📊 Davies-Bouldin Score: {metrics['davies_bouldin_score']:.3f}")
    print(f"      (Lower values indicate better clustering)")

    print(f"\n📏 Cluster Statistics:")
    print(f"   🔢 Number of clusters: {metrics['n_clusters']}")
    print(f"   🔢 Number of noise points: {metrics['n_noise']}")

    cluster_sizes = [len(cluster) for cluster in result["clusters"]]
    print(
        f"   📦 Cluster sizes: min={min(cluster_sizes)}, max={max(cluster_sizes)}, avg={np.mean(cluster_sizes):.1f}"
    )


def demo_file_operations():
    """Demonstrate file I/O operations."""
    print("\n" + "=" * 60)
    print("💾 FILE I/O DEMONSTRATION")
    print("=" * 60)

    data = create_sample_data()

    # Create temporary files for demonstration
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)

        # Save data to JSONL
        input_file = temp_dir / "sample_data.jsonl"
        save_jsonl(data, str(input_file))
        print(f"📝 Saved sample data to: {input_file}")

        # Run clustering and save results
        from semantic_clustify.utils import load_jsonl

        # Load data back
        loaded_data = load_jsonl(str(input_file))

        # Create clusterer and run clustering
        clusterer = SemanticClusterer(method="kmeans", n_clusters=3)
        clusters = clusterer.fit_predict(data=loaded_data, vector_field="embedding")

        # Save results
        from semantic_clustify.utils import save_grouped_jsonl

        output_file = temp_dir / "clusters.jsonl"
        save_grouped_jsonl(clusters, str(output_file))

        metrics = clusterer.get_quality_metrics()

        result = {"clusters": clusters, "metrics": metrics}

        print(f"✅ Clustering completed and saved to: {temp_dir / 'clusters.jsonl'}")
        print(
            f"📊 Results: {len(result['clusters'])} clusters, {sum(len(c) for c in result['clusters'])} documents"
        )

        # Show file contents preview
        with open(temp_dir / "clusters.jsonl", "r") as f:
            content = f.read()
            lines = content.split("\n")[:5]  # First 5 lines
            print(f"\n📄 Output file preview (first 5 lines):")
            for i, line in enumerate(lines):
                if line.strip():
                    preview = line[:80] + "..." if len(line) > 80 else line
                    print(f"   {i+1}: {preview}")


def main():
    """Run all demonstrations."""
    print("🚀 SEMANTIC-CLUSTIFY COMPREHENSIVE DEMO")
    print("=" * 60)
    print("This demo showcases all major features of semantic-clustify:")
    print("• Library usage with different algorithms")
    print("• Automatic parameter optimization")
    print("• Comprehensive quality metrics")
    print("• File I/O operations")
    print()
    print("The demo uses artificially generated data that simulates")
    print("three distinct topic clusters: Technology, Science, and Programming")

    try:
        # Run all demonstrations
        demo_library_usage()
        demo_auto_optimization()
        demo_quality_metrics()
        demo_file_operations()

        print("\n" + "=" * 60)
        print("✅ ALL DEMONSTRATIONS COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print()
        print("🎯 Key Takeaways:")
        print("• semantic-clustify successfully identified meaningful clusters")
        print("• Different algorithms have different strengths for different data")
        print("• Auto-optimization can find optimal cluster numbers")
        print("• Comprehensive metrics help evaluate clustering quality")
        print("• Easy file I/O for production workflows")
        print()
        print("📚 Next Steps:")
        print("• Try with your own vector embeddings")
        print("• Experiment with different algorithms and parameters")
        print("• Use the CLI for quick experimentation")
        print("• Integrate into your text analysis pipeline")

    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        raise


if __name__ == "__main__":
    main()
