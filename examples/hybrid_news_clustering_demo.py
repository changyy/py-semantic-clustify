#!/usr/bin/env python3
"""
Hybrid DBSCAN + K-Means News Clustering Demo

This demo shows how to use hybrid algorithm for news article clustering scenario:
- Stage 1: Use DBSCAN to naturally discover major news events
- Stage 2: Use K-Means to reorganize minor events to target cluster count

Use case: 1000+ news articles in 24 hours, producing 30 topic summaries
"""

import numpy as np
import json
from typing import List, Dict, Any
from semantic_clustify import SemanticClusterer


def create_news_simulation_data(n_articles: int = 1200) -> List[Dict[str, Any]]:
    """
    Create simulated news data
    
    Simulate Taiwan's 24-hour news situation:
    - Several major events (each with many related reports)
    - Many minor events (only a few reports each)
    - Some noise articles
    """
    np.random.seed(42)
    articles = []
    
    # Define center vectors for major events (dimension=768, simulating sentence-transformer output)
    major_events = {
        "Legislative Yuan bill discussion": np.random.normal([0.8, 0.1, 0.1, 0.0, 0.0] + [0.0] * 763, 0.05, 768),
        "TSMC earnings report": np.random.normal([0.1, 0.8, 0.1, 0.0, 0.0] + [0.0] * 763, 0.05, 768),
        "Typhoon warning issued": np.random.normal([0.1, 0.1, 0.8, 0.0, 0.0] + [0.0] * 763, 0.05, 768),
        "Presidential Office important statement": np.random.normal([0.1, 0.1, 0.1, 0.8, 0.0] + [0.0] * 763, 0.05, 768),
        "International trade negotiations": np.random.normal([0.1, 0.1, 0.1, 0.1, 0.8] + [0.0] * 763, 0.05, 768),
    }
    
    # Major event articles (80-120 articles per event)
    article_id = 0
    for event_name, center_vector in major_events.items():
        n_articles_for_event = np.random.randint(80, 121)
        for i in range(n_articles_for_event):
            # Different media report the same event from different angles
            noise = np.random.normal(0, 0.02, 768)
            article_vector = center_vector + noise
            
            # Normalize vector
            article_vector = article_vector / np.linalg.norm(article_vector)
            
            articles.append({
                "id": f"news_{article_id:04d}",
                "title": f"{event_name} - Related report {i+1}",
                "content": f"This is detailed news content about {event_name}...",
                "source": f"Media{np.random.randint(1, 11)}",
                "timestamp": f"2024-06-04 {np.random.randint(0, 24):02d}:{np.random.randint(0, 60):02d}",
                "embedding": article_vector.tolist(),
                "event_type": "major",
                "true_topic": event_name
            })
            article_id += 1
    
    # Minor event articles (2-8 articles per event)
    n_minor_events = 50
    for event_idx in range(n_minor_events):
        # Randomly generate center vector for minor events
        center_vector = np.random.normal(0, 0.3, 768)
        center_vector = center_vector / np.linalg.norm(center_vector)
        
        n_articles_for_event = np.random.randint(2, 9)
        event_name = f"Minor event {event_idx+1}"
        
        for i in range(n_articles_for_event):
            noise = np.random.normal(0, 0.05, 768)
            article_vector = center_vector + noise
            article_vector = article_vector / np.linalg.norm(article_vector)
            
            articles.append({
                "id": f"news_{article_id:04d}",
                "title": f"{event_name} - Report {i+1}",
                "content": f"This is news content about {event_name}...",
                "source": f"Media{np.random.randint(1, 11)}",
                "timestamp": f"2024-06-04 {np.random.randint(0, 24):02d}:{np.random.randint(0, 60):02d}",
                "embedding": article_vector.tolist(),
                "event_type": "minor", 
                "true_topic": event_name
            })
            article_id += 1
    
    # Noise articles (completely random)
    n_noise = n_articles - len(articles)
    for i in range(n_noise):
        random_vector = np.random.normal(0, 0.1, 768)
        random_vector = random_vector / np.linalg.norm(random_vector)
        
        articles.append({
            "id": f"news_{article_id:04d}",
            "title": f"Random news {i+1}",
            "content": "This is random news content...",
            "source": f"Media{np.random.randint(1, 11)}",
            "timestamp": f"2024-06-04 {np.random.randint(0, 24):02d}:{np.random.randint(0, 60):02d}",
            "embedding": random_vector.tolist(),
            "event_type": "noise",
            "true_topic": "Noise"
        })
        article_id += 1
    
    # Randomly shuffle order
    np.random.shuffle(articles)
    
    print(f"📰 Created {len(articles)} simulated news articles")
    print(f"   - Major events: {len([a for a in articles if a['event_type'] == 'major'])} articles")
    print(f"   - Minor events: {len([a for a in articles if a['event_type'] == 'minor'])} articles")
    print(f"   - Noise articles: {len([a for a in articles if a['event_type'] == 'noise'])} articles")
    
    return articles


def analyze_clustering_results(clusters: List[List[Dict[str, Any]]], original_data: List[Dict[str, Any]]):
    """Analyze clustering result quality"""
    print("\n📊 Clustering Result Analysis")
    print("=" * 50)
    
    # Basic statistics
    total_clustered = sum(len(cluster) for cluster in clusters)
    print(f"Number of clusters: {len(clusters)}")
    print(f"Total clustered articles: {total_clustered}")
    print(f"Original articles: {len(original_data)}")
    print(f"Unclustered articles: {len(original_data) - total_clustered}")
    
    # Cluster size distribution
    cluster_sizes = [len(cluster) for cluster in clusters]
    print(f"\nCluster size distribution:")
    print(f"   Largest cluster: {max(cluster_sizes) if cluster_sizes else 0} articles")
    print(f"   Smallest cluster: {min(cluster_sizes) if cluster_sizes else 0} articles")
    print(f"   Average size: {np.mean(cluster_sizes):.1f} articles")
    print(f"   Median size: {np.median(cluster_sizes):.1f} articles")
    
    # Analyze topic purity of each cluster
    print(f"\n🎯 Topic distribution of top 10 largest clusters:")
    for i, cluster in enumerate(sorted(clusters, key=len, reverse=True)[:10]):
        topic_counts = {}
        for article in cluster:
            topic = article.get('true_topic', 'Unknown')
            topic_counts[topic] = topic_counts.get(topic, 0) + 1
        
        # Find main topic
        main_topic = max(topic_counts.items(), key=lambda x: x[1])
        purity = main_topic[1] / len(cluster)
        
        print(f"   Cluster {i+1:2d} ({len(cluster):3d} articles): {main_topic[0]} ({purity:.1%} purity)")
        
        # Show topic distribution
        if len(topic_counts) > 1:
            other_topics = sorted([f"{topic}({count})" for topic, count in topic_counts.items() 
                                 if topic != main_topic[0]], key=lambda x: int(x.split('(')[1].split(')')[0]), reverse=True)[:3]
            if other_topics:
                print(f"             Others: {', '.join(other_topics)}")


def main():
    """Main demo program"""
    print("🚀 Hybrid DBSCAN + K-Means News Clustering Demo")
    print("=" * 60)
    
    # 1. Create simulated news data
    print("\n📝 Step 1: Create simulated news data")
    news_articles = create_news_simulation_data(1200)
    
    # 2. Configure hybrid clusterer
    print("\n⚙️  Step 2: Configure hybrid clusterer")
    clusterer = SemanticClusterer(
        method="hybrid-dbscan-kmeans",
        target_clusters=30,              # Target: produce 30 topics
        major_event_threshold=15,        # 15+ articles considered major event
        dbscan_eps=0.15,                # DBSCAN neighborhood radius
        dbscan_min_samples=3,           # DBSCAN minimum samples
        min_cluster_size=3,             # Minimum cluster size
        kmeans_strategy="remaining_slots",  # K-Means strategy
        random_state=42
    )
    
    print(f"   - Target cluster count: 30")
    print(f"   - Major event threshold: 15 articles")
    print(f"   - DBSCAN eps: 0.15")
    print(f"   - Minimum cluster size: 3")
    
    # 3. Perform clustering
    print("\n🔄 Step 3: Perform hybrid clustering")
    clusters = clusterer.fit_predict(news_articles, vector_field="embedding")
    
    # 4. Get stage information
    print("\n📈 Step 4: Analyze stage results")
    if hasattr(clusterer.algorithm, 'get_stage_info'):
        stage_info = clusterer.algorithm.get_stage_info()
        
        print(f"Stage 1 - DBSCAN discovery:")
        print(f"   Clusters found: {stage_info.get('dbscan', {}).get('n_clusters', 'N/A')}")
        print(f"   Noise points: {stage_info.get('dbscan', {}).get('n_noise', 'N/A')}")
        
        print(f"\nStage 2 - Event analysis:")
        print(f"   Major events: {stage_info.get('analysis', {}).get('major_clusters', 'N/A')}")
        print(f"   Documents to reorganize: {stage_info.get('analysis', {}).get('minor_documents', 'N/A')}")
        
        print(f"\nStage 3 - K-Means reorganization:")
        print(f"   Target cluster count: {stage_info.get('kmeans', {}).get('target_clusters', 'N/A')}")
        print(f"   Final cluster count: {stage_info.get('kmeans', {}).get('final_clusters', 'N/A')}")
    
    # 5. Show quality metrics
    print("\n📊 Step 5: Clustering quality evaluation")
    try:
        metrics = clusterer.get_quality_metrics()
        print(f"   Silhouette Score: {metrics.get('silhouette_score', 'N/A'):.3f}")
        print(f"   Calinski-Harabasz Score: {metrics.get('calinski_harabasz_score', 'N/A'):.1f}")
        print(f"   Davies-Bouldin Score: {metrics.get('davies_bouldin_score', 'N/A'):.3f}")
    except Exception as e:
        print(f"   Quality metrics calculation failed: {e}")
    
    # 6. Analyze clustering results
    analyze_clustering_results(clusters, news_articles)
    
    # 7. Show real application scenario
    print("\n📝 Step 6: Generate news summary examples")
    print("=" * 50)
    
    # Show top 5 largest clusters as news summaries
    top_clusters = sorted(clusters, key=len, reverse=True)[:5]
    
    for i, cluster in enumerate(top_clusters):
        print(f"\n📰 News Topic {i+1} ({len(cluster)} reports)")
        
        # Find main topic
        topic_counts = {}
        sources = set()
        for article in cluster:
            topic = article.get('true_topic', 'Unknown')
            topic_counts[topic] = topic_counts.get(topic, 0) + 1
            sources.add(article.get('source', 'Unknown'))
        
        main_topic = max(topic_counts.items(), key=lambda x: x[1])[0]
        
        print(f"   Main event: {main_topic}")
        print(f"   Reporting media: {len(sources)} outlets ({', '.join(sorted(sources)[:5])})")
        print(f"   Time span: Within 24 hours")
        print(f"   Suggested summary: This is an important news event about {main_topic},")
        print(f"                    with {len(cluster)} related reports, worth attention.")
    
    print(f"\n🎉 Hybrid clustering demo completed!")
    print(f"Successfully clustered {len(news_articles)} news articles into {len(clusters)} topics")


if __name__ == "__main__":
    main()
