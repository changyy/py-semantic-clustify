"""
Comprehensive clustering workflow guide for semantic-clustify.

This example demonstrates a complete workflow from data preparation 
to clustering analysis and result interpretation.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Any
from semantic_clustify import SemanticClusterer


def create_realistic_dataset():
    """Create a more realistic dataset with different text domains."""
    
    # Technology articles
    tech_articles = [
        {
            "title": "Artificial Intelligence Revolution",
            "content": "AI is transforming industries through machine learning and automation.",
            "domain": "technology",
            "url": "https://example.com/ai-revolution"
        },
        {
            "title": "Quantum Computing Breakthrough", 
            "content": "Quantum computers promise exponential speedup for certain computational problems.",
            "domain": "technology",
            "url": "https://example.com/quantum-computing"
        },
        {
            "title": "5G Networks and IoT",
            "content": "Fifth-generation wireless technology enables massive IoT deployments.",
            "domain": "technology", 
            "url": "https://example.com/5g-iot"
        },
        {
            "title": "Blockchain Applications",
            "content": "Distributed ledger technology extends beyond cryptocurrency to supply chains.",
            "domain": "technology",
            "url": "https://example.com/blockchain-apps"
        },
    ]
    
    # Health articles
    health_articles = [
        {
            "title": "Precision Medicine Advances",
            "content": "Personalized treatments based on genetic profiles improve patient outcomes.",
            "domain": "health",
            "url": "https://example.com/precision-medicine"
        },
        {
            "title": "Mental Health Awareness",
            "content": "Understanding depression and anxiety disorders through modern psychology.",
            "domain": "health",
            "url": "https://example.com/mental-health"
        },
        {
            "title": "Nutrition and Longevity",
            "content": "Dietary patterns influence aging processes and life expectancy.",
            "domain": "health",
            "url": "https://example.com/nutrition-longevity"
        },
        {
            "title": "Vaccine Development Process",
            "content": "Modern vaccine research accelerates through advanced biotechnology methods.",
            "domain": "health",
            "url": "https://example.com/vaccine-development"
        },
    ]
    
    # Finance articles  
    finance_articles = [
        {
            "title": "Cryptocurrency Market Analysis",
            "content": "Digital currencies exhibit high volatility and regulatory uncertainty.",
            "domain": "finance",
            "url": "https://example.com/crypto-analysis"
        },
        {
            "title": "ESG Investing Trends",
            "content": "Environmental, social, and governance factors drive investment decisions.",
            "domain": "finance",
            "url": "https://example.com/esg-investing"
        },
        {
            "title": "Central Bank Digital Currencies",
            "content": "CBDCs represent government-backed digital money for modern economies.",
            "domain": "finance",
            "url": "https://example.com/cbdc"
        },
        {
            "title": "Algorithmic Trading Systems",
            "content": "Automated trading strategies use mathematical models and market data.",
            "domain": "finance",
            "url": "https://example.com/algo-trading"
        },
    ]
    
    # Climate articles
    climate_articles = [
        {
            "title": "Renewable Energy Transition",
            "content": "Solar and wind power installations accelerate global decarbonization efforts.",
            "domain": "climate",
            "url": "https://example.com/renewable-energy"
        },
        {
            "title": "Carbon Capture Technologies",
            "content": "Direct air capture and storage systems remove CO2 from atmosphere.",
            "domain": "climate", 
            "url": "https://example.com/carbon-capture"
        },
        {
            "title": "Climate Change Adaptation",
            "content": "Communities develop resilience strategies for rising sea levels and extreme weather.",
            "domain": "climate",
            "url": "https://example.com/climate-adaptation"
        },
        {
            "title": "Sustainable Agriculture Practices",
            "content": "Regenerative farming methods restore soil health and biodiversity.",
            "domain": "climate",
            "url": "https://example.com/sustainable-agriculture"
        },
    ]
    
    all_articles = tech_articles + health_articles + finance_articles + climate_articles
    
    # Generate embeddings that reflect semantic similarity
    np.random.seed(42)
    
    domain_embeddings = {
        "technology": [0.9, 0.1, 0.1, 0.1, 0.2, 0.8, 0.1, 0.1],
        "health": [0.1, 0.9, 0.1, 0.1, 0.1, 0.1, 0.8, 0.2], 
        "finance": [0.1, 0.1, 0.9, 0.1, 0.8, 0.1, 0.1, 0.1],
        "climate": [0.1, 0.1, 0.1, 0.9, 0.1, 0.2, 0.1, 0.8],
    }
    
    for article in all_articles:
        domain = article["domain"]
        base_embedding = domain_embeddings[domain]
        
        # Add noise and variation
        noise = np.random.normal(0, 0.05, len(base_embedding))
        embedding = np.array(base_embedding) + noise
        
        # Normalize to unit vector (common in embeddings)
        embedding = embedding / np.linalg.norm(embedding)
        
        article["embedding"] = embedding.tolist()
        article["embedding_dim"] = len(embedding)
    
    return all_articles


def analyze_clustering_quality(clusters: List[List[Dict]], true_domains: List[str]) -> Dict:
    """Analyze clustering quality against true domain labels."""
    
    # Create mapping from document to cluster
    doc_to_cluster = {}
    cluster_to_domains = {}
    
    for cluster_id, cluster in enumerate(clusters):
        cluster_to_domains[cluster_id] = {}
        for doc in cluster:
            doc_to_cluster[doc["url"]] = cluster_id
            domain = doc["domain"]
            cluster_to_domains[cluster_id][domain] = cluster_to_domains[cluster_id].get(domain, 0) + 1
    
    # Calculate purity for each cluster
    cluster_purities = []
    for cluster_id, domain_counts in cluster_to_domains.items():
        total_docs = sum(domain_counts.values())
        max_domain_count = max(domain_counts.values()) if domain_counts else 0
        purity = max_domain_count / total_docs if total_docs > 0 else 0
        cluster_purities.append(purity)
    
    # Overall metrics
    total_docs = sum(len(cluster) for cluster in clusters)
    weighted_purity = sum(len(clusters[i]) * purity for i, purity in enumerate(cluster_purities)) / total_docs
    
    return {
        "cluster_purities": cluster_purities,
        "weighted_purity": weighted_purity,
        "cluster_to_domains": cluster_to_domains,
        "n_clusters": len(clusters),
        "total_documents": total_docs
    }


def comprehensive_clustering_analysis(data: List[Dict]) -> Dict:
    """Perform comprehensive clustering analysis with multiple methods."""
    
    results = {}
    methods = ["kmeans", "dbscan", "hierarchical", "gmm"]
    
    print(f"\n🔬 Comprehensive Clustering Analysis")
    print("=" * 50)
    
    for method in methods:
        print(f"\n📊 Analyzing {method.upper()}...")
        
        try:
            # Try different parameter settings
            if method == "kmeans":
                # Test different k values
                best_score = -1
                best_result = None
                
                for k in range(2, 8):
                    clusterer = SemanticClusterer(method=method, n_clusters=k, random_state=42)
                    clusters = clusterer.fit_predict(data, vector_field="embedding")
                    metrics = clusterer.get_quality_metrics()
                    
                    score = metrics.get("silhouette_score", -1)
                    if score > best_score:
                        best_score = score
                        best_result = {
                            "clusters": clusters,
                            "metrics": metrics,
                            "parameters": {"n_clusters": k}
                        }
                
                results[method] = best_result
                print(f"   Best k: {best_result['parameters']['n_clusters']}")
                print(f"   Silhouette score: {best_score:.3f}")
                
            elif method == "dbscan":
                # Test different eps values
                best_score = -1
                best_result = None
                
                for eps in [0.1, 0.2, 0.3, 0.4, 0.5]:
                    clusterer = SemanticClusterer(method=method, eps=eps, min_samples=2)
                    clusters = clusterer.fit_predict(data, vector_field="embedding")
                    metrics = clusterer.get_quality_metrics()
                    
                    if len(clusters) > 1:  # Only consider results with multiple clusters
                        score = metrics.get("silhouette_score", -1)
                        if score > best_score:
                            best_score = score
                            best_result = {
                                "clusters": clusters,
                                "metrics": metrics,
                                "parameters": {"eps": eps}
                            }
                
                if best_result:
                    results[method] = best_result
                    print(f"   Best eps: {best_result['parameters']['eps']}")
                    print(f"   Silhouette score: {best_score:.3f}")
                else:
                    print(f"   No valid clustering found")
                    
            else:
                # Use auto parameters for hierarchical and GMM
                clusterer = SemanticClusterer(method=method, n_clusters="auto", random_state=42)
                clusters = clusterer.fit_predict(data, vector_field="embedding")
                metrics = clusterer.get_quality_metrics()
                
                results[method] = {
                    "clusters": clusters,
                    "metrics": metrics,
                    "parameters": clusterer.get_algorithm_params()
                }
                
                score = metrics.get("silhouette_score", 0)
                print(f"   Clusters: {len(clusters)}")
                print(f"   Silhouette score: {score:.3f}")
                
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            results[method] = None
    
    return results


def generate_clustering_report(data: List[Dict], results: Dict) -> str:
    """Generate a comprehensive clustering report."""
    
    report = []
    report.append("# Semantic Clustering Analysis Report")
    report.append("=" * 50)
    report.append("")
    
    # Dataset overview
    report.append("## Dataset Overview")
    report.append(f"- Total documents: {len(data)}")
    
    domains = {}
    for doc in data:
        domain = doc.get("domain", "Unknown")
        domains[domain] = domains.get(domain, 0) + 1
    
    report.append("- Domain distribution:")
    for domain, count in domains.items():
        report.append(f"  - {domain}: {count} documents")
    report.append("")
    
    # Method comparison
    report.append("## Method Comparison")
    report.append("")
    
    for method, result in results.items():
        if result is None:
            continue
            
        clusters = result["clusters"]
        metrics = result["metrics"]
        
        report.append(f"### {method.upper()}")
        report.append(f"- Clusters found: {len(clusters)}")
        report.append(f"- Silhouette score: {metrics.get('silhouette_score', 'N/A')}")
        report.append(f"- Parameters: {result['parameters']}")
        
        # Analyze domain purity
        true_domains = [doc["domain"] for doc in data]
        quality_analysis = analyze_clustering_quality(clusters, true_domains)
        report.append(f"- Domain purity: {quality_analysis['weighted_purity']:.3f}")
        
        # Show cluster contents
        report.append("- Cluster breakdown:")
        for i, cluster in enumerate(clusters):
            domains_in_cluster = {}
            for doc in cluster:
                domain = doc["domain"]
                domains_in_cluster[domain] = domains_in_cluster.get(domain, 0) + 1
            
            domain_str = ", ".join(f"{d}:{c}" for d, c in domains_in_cluster.items())
            report.append(f"  - Cluster {i}: {len(cluster)} docs ({domain_str})")
        
        report.append("")
    
    # Recommendations
    report.append("## Recommendations")
    
    # Find best method by silhouette score
    best_method = None
    best_score = -1
    
    for method, result in results.items():
        if result and result["metrics"].get("silhouette_score"):
            score = result["metrics"]["silhouette_score"]
            if score > best_score:
                best_score = score
                best_method = method
    
    if best_method:
        report.append(f"- Best performing method: **{best_method.upper()}** (silhouette score: {best_score:.3f})")
        
        best_result = results[best_method]
        quality_analysis = analyze_clustering_quality(best_result["clusters"], [doc["domain"] for doc in data])
        
        if quality_analysis["weighted_purity"] > 0.8:
            report.append("- High domain purity indicates good semantic clustering")
        elif quality_analysis["weighted_purity"] > 0.6:
            report.append("- Moderate domain purity, consider parameter tuning")
        else:
            report.append("- Low domain purity, may need different approach or features")
    
    report.append("")
    report.append("## Usage Examples")
    report.append("")
    report.append("```bash")
    if best_method:
        params = results[best_method]["parameters"]
        if best_method == "kmeans":
            report.append(f"semantic-clustify --input data.jsonl --vector-field embedding --method {best_method} --n-clusters {params.get('n_clusters', 'auto')}")
        elif best_method == "dbscan":
            report.append(f"semantic-clustify --input data.jsonl --vector-field embedding --method {best_method} --eps {params.get('eps', 0.5)}")
        else:
            report.append(f"semantic-clustify --input data.jsonl --vector-field embedding --method {best_method}")
    else:
        report.append("semantic-clustify --input data.jsonl --vector-field embedding --method kmeans --n-clusters auto")
    report.append("```")
    
    return "\n".join(report)


def main():
    """Main workflow demonstration."""
    print("🎯 Comprehensive Clustering Workflow Guide")
    print("=" * 50)
    
    # Step 1: Create realistic dataset
    print("\n📝 Step 1: Creating realistic dataset...")
    data = create_realistic_dataset()
    
    # Save dataset
    output_dir = Path("clustering_analysis")
    output_dir.mkdir(exist_ok=True)
    
    dataset_file = output_dir / "sample_dataset.jsonl"
    with open(dataset_file, 'w') as f:
        for item in data:
            json.dump(item, f, ensure_ascii=False)
            f.write('\n')
    
    print(f"✅ Dataset saved: {dataset_file}")
    print(f"   Documents: {len(data)}")
    print(f"   Domains: {len(set(doc['domain'] for doc in data))}")
    
    # Step 2: Comprehensive analysis
    print(f"\n🔬 Step 2: Running comprehensive analysis...")
    results = comprehensive_clustering_analysis(data)
    
    # Step 3: Generate report
    print(f"\n📊 Step 3: Generating analysis report...")
    report = generate_clustering_report(data, results)
    
    # Save report
    report_file = output_dir / "clustering_report.md"
    with open(report_file, 'w') as f:
        f.write(report)
    
    print(f"✅ Report saved: {report_file}")
    
    # Step 4: Save best clustering results
    best_method = None
    best_score = -1
    
    for method, result in results.items():
        if result and result["metrics"].get("silhouette_score"):
            score = result["metrics"]["silhouette_score"]
            if score > best_score:
                best_score = score
                best_method = method
    
    if best_method:
        print(f"\n🏆 Step 4: Saving best results ({best_method})...")
        best_clusters = results[best_method]["clusters"]
        
        # Save in both formats
        grouped_file = output_dir / f"best_clusters_grouped.jsonl"
        with open(grouped_file, 'w') as f:
            json.dump(best_clusters, f, ensure_ascii=False, indent=2)
        
        labeled_file = output_dir / f"best_clusters_labeled.jsonl"  
        with open(labeled_file, 'w') as f:
            for cluster_id, cluster in enumerate(best_clusters):
                for doc in cluster:
                    doc_copy = doc.copy()
                    doc_copy["cluster_id"] = cluster_id
                    json.dump(doc_copy, f, ensure_ascii=False)
                    f.write('\n')
        
        print(f"✅ Best clustering results saved:")
        print(f"   Grouped format: {grouped_file}")
        print(f"   Labeled format: {labeled_file}")
    
    # Step 5: Display summary
    print(f"\n📋 Workflow Summary")
    print("=" * 30)
    print(f"✅ Dataset created: {len(data)} documents")
    print(f"✅ Methods tested: {len([r for r in results.values() if r is not None])}")
    print(f"✅ Best method: {best_method or 'None'}")
    print(f"✅ Output directory: {output_dir}")
    
    print(f"\n🚀 Next Steps:")
    print(f"1. Review the analysis report: {report_file}")
    print(f"2. Try the CLI with recommended parameters")
    print(f"3. Experiment with your own data")
    print(f"4. Adjust parameters based on your domain")


if __name__ == "__main__":
    main()
