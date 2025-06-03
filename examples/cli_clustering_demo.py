"""
Interactive demo for semantic-clustify clustering capabilities.

This script demonstrates the main features of semantic-clustify with sample data
and shows how to use different clustering algorithms.
"""

import json
import numpy as np
from pathlib import Path
from semantic_clustify import SemanticClusterer


def generate_sample_data():
    """Generate sample documents with embeddings for demonstration."""
    
    # Sample documents about different topics
    documents = [
        {
            "title": "Introduction to Machine Learning",
            "content": "Machine learning is a subset of artificial intelligence that focuses on algorithms.",
            "category": "AI/ML"
        },
        {
            "title": "Deep Neural Networks",
            "content": "Deep learning uses neural networks with multiple layers to learn complex patterns.",
            "category": "AI/ML"
        },
        {
            "title": "Natural Language Processing",
            "content": "NLP combines computational linguistics with machine learning and deep learning.",
            "category": "AI/ML"
        },
        {
            "title": "Python Programming Basics",
            "content": "Python is a high-level programming language known for its simplicity.",
            "category": "Programming"
        },
        {
            "title": "Web Development with Flask",
            "content": "Flask is a lightweight web framework for Python applications.",
            "category": "Programming"
        },
        {
            "title": "JavaScript for Beginners",
            "content": "JavaScript is the programming language of the web, used for interactive websites.",
            "category": "Programming"
        },
        {
            "title": "Data Visualization Techniques",
            "content": "Data visualization helps in understanding patterns and insights from data.",
            "category": "Data Science"
        },
        {
            "title": "Statistical Analysis Methods",
            "content": "Statistics provides methods for collecting, analyzing and interpreting data.",
            "category": "Data Science"
        },
        {
            "title": "Big Data Processing",
            "content": "Big data technologies help process and analyze large volumes of data.",
            "category": "Data Science"
        },
        {
            "title": "Database Design Principles",
            "content": "Good database design ensures data integrity and efficient querying.",
            "category": "Database"
        },
        {
            "title": "SQL Query Optimization",
            "content": "SQL optimization techniques improve database query performance.",
            "category": "Database"
        },
        {
            "title": "NoSQL Database Systems",
            "content": "NoSQL databases provide flexible schemas for modern applications.",
            "category": "Database"
        },
    ]
    
    # Generate embeddings that cluster by category
    # This simulates real embeddings where similar content has similar vectors
    np.random.seed(42)
    
    category_centers = {
        "AI/ML": [0.1, 0.9, 0.2, 0.8, 0.1],
        "Programming": [0.8, 0.1, 0.9, 0.2, 0.1], 
        "Data Science": [0.2, 0.1, 0.1, 0.9, 0.8],
        "Database": [0.9, 0.2, 0.1, 0.1, 0.9],
    }
    
    for doc in documents:
        category = doc["category"]
        center = category_centers[category]
        # Add some noise to make it realistic
        noise = np.random.normal(0, 0.1, len(center))
        embedding = np.array(center) + noise
        doc["embedding"] = embedding.tolist()
    
    return documents


def save_sample_data(data, filename="sample_data.jsonl"):
    """Save sample data to JSONL file."""
    with open(filename, 'w') as f:
        for item in data:
            json.dump(item, f, ensure_ascii=False)
            f.write('\n')
    
    print(f"✅ Sample data saved to {filename}")
    return filename


def demo_clustering_methods(data):
    """Demonstrate different clustering methods."""
    print("\n🚀 Semantic Clustering Demo")
    print("=" * 50)
    
    methods = ["kmeans", "dbscan", "hierarchical", "gmm"]
    
    for method in methods:
        print(f"\n📊 Testing {method.upper()} clustering:")
        print("-" * 30)
        
        try:
            clusterer = SemanticClusterer(
                method=method,
                n_clusters="auto" if method != "dbscan" else None,
                min_cluster_size=2,
                random_state=42
            )
            
            clusters = clusterer.fit_predict(data, vector_field="embedding")
            metrics = clusterer.get_quality_metrics()
            
            print(f"   Clusters found: {len(clusters)}")
            print(f"   Total documents: {sum(len(cluster) for cluster in clusters)}")
            
            if metrics.get("silhouette_score"):
                print(f"   Silhouette score: {metrics['silhouette_score']:.3f}")
            
            # Show cluster contents
            for i, cluster in enumerate(clusters):
                print(f"\n   Cluster {i} ({len(cluster)} documents):")
                categories = set()
                for doc in cluster:
                    categories.add(doc.get("category", "Unknown"))
                    title = doc["title"][:40] + "..." if len(doc["title"]) > 40 else doc["title"]
                    print(f"     • {title}")
                
                print(f"     Categories: {', '.join(categories)}")
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")


def demo_parameter_comparison(data):
    """Demonstrate the effect of different parameters."""
    print(f"\n🔧 Parameter Comparison Demo")
    print("=" * 50)
    
    print(f"\n📈 KMeans with different cluster numbers:")
    for k in [2, 3, 4, 5]:
        try:
            clusterer = SemanticClusterer(method="kmeans", n_clusters=k, random_state=42)
            clusters = clusterer.fit_predict(data, vector_field="embedding")
            metrics = clusterer.get_quality_metrics()
            
            sil_score = metrics.get("silhouette_score", 0)
            print(f"   k={k}: {len(clusters)} clusters, silhouette={sil_score:.3f}")
            
        except Exception as e:
            print(f"   k={k}: Failed - {e}")
    
    print(f"\n🔍 DBSCAN with different eps values:")
    for eps in [0.1, 0.3, 0.5, 0.8]:
        try:
            clusterer = SemanticClusterer(method="dbscan", eps=eps, min_samples=2)
            clusters = clusterer.fit_predict(data, vector_field="embedding")
            metrics = clusterer.get_quality_metrics()
            
            n_noise = metrics.get("n_noise", 0)
            print(f"   eps={eps}: {len(clusters)} clusters, {n_noise} noise points")
            
        except Exception as e:
            print(f"   eps={eps}: Failed - {e}")


def demo_cli_usage(filename):
    """Show CLI usage examples."""
    print(f"\n💻 CLI Usage Examples")
    print("=" * 50)
    
    examples = [
        f"# Basic KMeans clustering",
        f"semantic-clustify --input {filename} --embedding-field embedding --method kmeans",
        f"",
        f"# Auto-detect optimal clusters",
        f"semantic-clustify --input {filename} --embedding-field embedding --method kmeans --n-clusters auto",
        f"",
        f"# DBSCAN clustering",
        f"semantic-clustify --input {filename} --embedding-field embedding --method dbscan",
        f"",
        f"# Show quality metrics",
        f"semantic-clustify --input {filename} --embedding-field embedding --method kmeans --quality-metrics",
        f"",
        f"# Different output format",
        f"semantic-clustify --input {filename} --embedding-field embedding --method kmeans --output-format labeled",
        f"",
        f"# Using stdin",
        f"cat {filename} | semantic-clustify --embedding-field embedding --method kmeans",
    ]
    
    for example in examples:
        print(example)


def main():
    """Main demo function."""
    print("🎯 semantic-clustify Interactive Demo")
    print("=====================================")
    
    # Generate and save sample data
    print("\n📝 Generating sample documents...")
    data = generate_sample_data()
    filename = save_sample_data(data)
    
    print(f"\n📊 Sample data overview:")
    print(f"   Total documents: {len(data)}")
    
    categories = {}
    for doc in data:
        cat = doc.get("category", "Unknown")
        categories[cat] = categories.get(cat, 0) + 1
    
    print(f"   Categories:")
    for cat, count in categories.items():
        print(f"     • {cat}: {count} documents")
    
    # Run clustering demos
    demo_clustering_methods(data)
    demo_parameter_comparison(data)
    demo_cli_usage(filename)
    
    print(f"\n✨ Demo completed!")
    print(f"   Sample data: {filename}")
    print(f"   Try the CLI commands above to experiment further!")


if __name__ == "__main__":
    main()
