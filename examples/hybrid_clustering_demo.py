#!/usr/bin/env python3
"""
Hybrid DBSCAN + K-Means Clustering Algorithm Demo

This example shows how to use the new HybridDBSCANKMeansClusterer for news article clustering.
Especially suitable for clustering Taiwan's 24-hour news events.
"""

import json
import numpy as np
from pathlib import Path
import logging
from typing import List, Dict, Any

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_sample_news_data() -> List[Dict[str, Any]]:
    """Create simulated news data"""
    # Simulate news vectors (in real use, these would be actual embedding vectors)
    np.random.seed(42)
    
    news_data = []
    
    # Major event 1: Election related (15 articles)
    election_base = np.array([1.0, 0.8, 0.2, 0.1, 0.3])
    for i in range(15):
        vector = election_base + np.random.normal(0, 0.1, 5)
        news_data.append({
            "id": f"election_{i}",
            "title": f"Election related news {i+1}",
            "content": f"News content about election {i+1}",
            "source": f"Media{i%5+1}",
            "embedding": vector.tolist(),
            "category": "election"
        })
    
    # Major event 2: Economic policy (12 articles)
    economy_base = np.array([0.2, 1.0, 0.8, 0.1, 0.2])
    for i in range(12):
        vector = economy_base + np.random.normal(0, 0.1, 5)
        news_data.append({
            "id": f"economy_{i}",
            "title": f"Economic policy news {i+1}",
            "content": f"News content about economic policy {i+1}",
            "source": f"FinancialMedia{i%3+1}",
            "embedding": vector.tolist(),
            "category": "economy"
        })
    
    # Major event 3: COVID-19 related (18 articles)
    covid_base = np.array([0.1, 0.2, 1.0, 0.7, 0.1])
    for i in range(18):
        vector = covid_base + np.random.normal(0, 0.1, 5)
        news_data.append({
            "id": f"covid_{i}",
            "title": f"Pandemic related news {i+1}",
            "content": f"News content about pandemic {i+1}",
            "source": f"HealthMedia{i%4+1}",
            "embedding": vector.tolist(),
            "category": "health"
        })
    
    # Minor events and miscellaneous news (30 articles)
    small_events = [
        ("sports", np.array([0.8, 0.1, 0.1, 1.0, 0.2]), 6),
        ("entertainment", np.array([0.3, 0.2, 0.1, 0.2, 1.0]), 5),
        ("technology", np.array([0.5, 0.6, 0.3, 0.4, 0.7]), 7),
        ("local", np.array([0.2, 0.3, 0.4, 0.3, 0.6]), 8),
        ("international", np.array([0.6, 0.7, 0.5, 0.6, 0.4]), 4)
    ]
    
    for category, base_vector, count in small_events:
        for i in range(count):
            vector = base_vector + np.random.normal(0, 0.15, 5)
            news_data.append({
                "id": f"{category}_{i}",
                "title": f"{category.title()} news {i+1}",
                "content": f"News content about {category} {i+1}",
                "source": f"{category.title()}Media{i%2+1}",
                "embedding": vector.tolist(),
                "category": category
            })
    
    logger.info(f"Created {len(news_data)} simulated news articles")
    return news_data

def demonstrate_hybrid_clustering():
    """Demonstrate hybrid clustering algorithm"""
    
    # Import clusterer (for actual use)
    try:
        from semantic_clustify import SemanticClusterer
        logger.info("Successfully imported SemanticClusterer")
    except ImportError:
        logger.error("Cannot import SemanticClusterer, please ensure package is properly installed")
        return
    
    # Create simulated data
    news_data = create_sample_news_data()
    
    # Prepare clusterer configurations
    clusterer_configs = [
        {
            "name": "Default hybrid clustering",
            "method": "hybrid-dbscan-kmeans",
            "target_clusters": 30,
            "major_event_threshold": 10,
            "kmeans_strategy": "remaining_slots"
        },
        {
            "name": "Strict major event detection",
            "method": "hybrid-dbscan-kmeans", 
            "target_clusters": 25,
            "major_event_threshold": 15,
            "kmeans_strategy": "remaining_slots"
        },
        {
            "name": "Flexible clustering strategy",
            "method": "hybrid-dbscan-kmeans",
            "target_clusters": 35,
            "major_event_threshold": 8,
            "kmeans_strategy": "adaptive"
        }
    ]
    
    # Test different configurations
    for config in clusterer_configs:
        print(f"\n{'='*60}")
        print(f"Testing configuration: {config['name']}")
        print(f"{'='*60}")
        
        try:
            # Create clusterer
            clusterer = SemanticClusterer(
                method=config["method"],
                target_clusters=config["target_clusters"],
                major_event_threshold=config["major_event_threshold"],
                kmeans_strategy=config["kmeans_strategy"],
                min_cluster_size=2,
                random_state=42
            )
            
            # Perform clustering
            logger.info(f"Starting {config['name']} clustering...")
            clusters = clusterer.fit_predict(news_data, vector_field="embedding")
            
            # Analyze results
            analyze_clustering_results(clusters, config["name"])
            
            # If it's HybridDBSCANKMeansClusterer, show stage information
            if hasattr(clusterer.algorithm, 'get_stage_info'):
                stage_info = clusterer.algorithm.get_stage_info()
                print(f"\nDetailed stage information:")
                print(f"  DBSCAN stage: {stage_info.get('dbscan', {})}")
                print(f"  Analysis stage: {stage_info.get('analysis', {})}")
                print(f"  K-Means stage: {stage_info.get('kmeans', {})}")
            
        except Exception as e:
            logger.error(f"Error occurred during clustering: {e}")
            continue

def analyze_clustering_results(clusters: List[List[Dict]], config_name: str):
    """Analyze clustering results"""
    
    print(f"\n📊 {config_name} - Clustering result analysis:")
    print(f"   Total clusters: {len(clusters)}")
    
    # Cluster size distribution
    cluster_sizes = [len(cluster) for cluster in clusters]
    print(f"   Cluster size distribution: min={min(cluster_sizes)}, max={max(cluster_sizes)}, avg={np.mean(cluster_sizes):.1f}")
    
    # Analyze major event identification
    major_clusters = [i for i, cluster in enumerate(clusters) if len(cluster) >= 10]
    print(f"   Major event clusters (≥10 articles): {len(major_clusters)}")
    
    # Analyze clustering purity by original category
    category_analysis = {}
    for cluster_idx, cluster in enumerate(clusters):
        categories = {}
        for doc in cluster:
            cat = doc.get('category', 'unknown')
            categories[cat] = categories.get(cat, 0) + 1
        
        # Find main category
        if categories:
            main_category = max(categories.items(), key=lambda x: x[1])
            purity = main_category[1] / len(cluster)
            category_analysis[cluster_idx] = {
                'size': len(cluster),
                'main_category': main_category[0],
                'purity': purity,
                'distribution': categories
            }
    
    # Show high-purity large clusters
    high_quality_clusters = [
        (idx, info) for idx, info in category_analysis.items() 
        if info['size'] >= 5 and info['purity'] >= 0.8
    ]
    
    print(f"   High-quality clusters (≥5 articles and purity≥80%): {len(high_quality_clusters)}")
    for idx, info in high_quality_clusters[:5]:  # Show only top 5
        print(f"     Cluster {idx}: {info['size']} articles, main category={info['main_category']}, purity={info['purity']:.1%}")

def create_sample_output_file():
    """Create sample output file"""
    
    # Create sample data
    news_data = create_sample_news_data()
    
    # Save as JSONL format
    output_file = Path("taiwan_news_24h_sample.jsonl")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for news in news_data:
            f.write(json.dumps(news, ensure_ascii=False) + '\n')
    
    print(f"✅ Sample data saved to: {output_file}")
    print(f"   Contains {len(news_data)} news articles")
    print(f"\nUsage:")
    print(f"semantic-clustify \\")
    print(f"  --input {output_file} \\")
    print(f"  --embedding-field embedding \\")
    print(f"  --method hybrid-dbscan-kmeans \\")
    print(f"  --target-clusters 30 \\")
    print(f"  --major-event-threshold 10 \\")
    print(f"  --quality-metrics \\")
    print(f"  --verbose \\")
    print(f"  --output hybrid_clustering_results.jsonl")

if __name__ == "__main__":
    print("🚀 Hybrid DBSCAN + K-Means Clustering Algorithm Demo")
    print("="*60)
    
    # Create sample file
    create_sample_output_file()
    
    print(f"\n{'='*60}")
    print("Starting clustering demonstration...")
    
    # Run demonstration
    demonstrate_hybrid_clustering()
    
    print(f"\n{'='*60}")
    print("✅ Demo completed!")
