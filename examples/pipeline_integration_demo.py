#!/usr/bin/env python3
"""
Pipeline Integration Demo Script
Demonstrates how to use different output formats in real data pipelines
"""

import json
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Any


def create_large_dataset() -> str:
    """Create large test dataset"""
    documents = []
    
    # Technology documents
    tech_docs = [
        "Artificial intelligence and machine learning algorithms",
        "Deep neural networks and backpropagation",
        "Computer vision and image recognition",
        "Natural language processing techniques",
        "Reinforcement learning in robotics",
        "Quantum computing fundamentals",
        "Blockchain technology applications",
        "Cloud computing architecture",
        "Data structures and algorithms",
        "Software engineering best practices"
    ]
    
    # Business documents
    business_docs = [
        "Financial markets and investment strategies",
        "Marketing automation and customer segmentation",
        "Supply chain management optimization",
        "Digital transformation initiatives",
        "Business intelligence and analytics",
        "Project management methodologies",
        "Human resources and talent acquisition",
        "Strategic planning and execution",
        "Risk management frameworks",
        "Corporate governance principles"
    ]
    
    # Science documents
    science_docs = [
        "Climate change and environmental science",
        "Biotechnology and genetic engineering",
        "Physics research and quantum mechanics",
        "Chemistry and molecular biology",
        "Medical research and drug discovery",
        "Renewable energy technologies",
        "Space exploration and astronomy",
        "Mathematical modeling and statistics",
        "Geological surveys and earth science",
        "Environmental conservation strategies"
    ]
    
    # Generate documents with similar vectors for each category
    import random
    import numpy as np
    
    def generate_embedding(base_vector: List[float], noise_level: float = 0.1) -> List[float]:
        """Generate similar vector with noise"""
        return [val + random.uniform(-noise_level, noise_level) for val in base_vector]
    
    # Technology category base vector
    tech_base = [0.8, 0.7, 0.6, 0.2, 0.1]
    for i, doc in enumerate(tech_docs):
        documents.append({
            "title": doc,
            "category": "Technology",
            "document_id": f"tech_{i:03d}",
            "embedding": generate_embedding(tech_base)
        })
    
    # Business category base vector
    business_base = [0.2, 0.3, 0.8, 0.7, 0.6]
    for i, doc in enumerate(business_docs):
        documents.append({
            "title": doc,
            "category": "Business",
            "document_id": f"biz_{i:03d}",
            "embedding": generate_embedding(business_base)
        })
    
    # Science category base vector
    science_base = [0.1, 0.8, 0.2, 0.9, 0.7]
    for i, doc in enumerate(science_docs):
        documents.append({
            "title": doc,
            "category": "Science",
            "document_id": f"sci_{i:03d}",
            "embedding": generate_embedding(science_base)
        })
    
    # Save to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        for doc in documents:
            json.dump(doc, f, ensure_ascii=False)
            f.write('\n')
        return f.name


def pipeline_step_1_basic_clustering(input_file: str) -> str:
    """Pipeline Step 1: Basic clustering"""
    print("🔄 Pipeline Step 1: Executing basic clustering...")
    
    output_file = "pipeline_step1_clusters.jsonl"
    cmd = [
        "semantic-clustify",
        "--input", input_file,
        "--embedding-field", "embedding",
        "--method", "kmeans",
        "--n-clusters", "auto",
        "--output-format", "enriched-labeled",
        "--output", output_file
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"❌ Step 1 failed: {result.stderr}")
        return None
    
    print(f"✅ Step 1 completed: {output_file}")
    return output_file


def pipeline_step_2_filter_large_clusters(input_file: str) -> str:
    """Pipeline Step 2: Filter large clusters"""
    print("🔍 Pipeline Step 2: Filtering large clusters (cluster_size >= 8)...")
    
    large_cluster_docs = []
    cluster_stats = {}
    
    with open(input_file, 'r') as f:
        for line in f:
            if line.strip():
                doc = json.loads(line)
                cluster_size = doc.get("cluster_size", 0)
                cluster_id = doc.get("cluster_id", -1)
                
                # Collect cluster statistics
                if cluster_id not in cluster_stats:
                    cluster_stats[cluster_id] = {
                        "size": cluster_size,
                        "documents": 0,
                        "categories": set()
                    }
                
                cluster_stats[cluster_id]["documents"] += 1
                cluster_stats[cluster_id]["categories"].add(doc.get("category", "Unknown"))
                
                # Filter large clusters
                if cluster_size >= 8:
                    large_cluster_docs.append(doc)
    
    # Save filtered results
    output_file = "pipeline_step2_large_clusters.jsonl"
    with open(output_file, 'w') as f:
        for doc in large_cluster_docs:
            json.dump(doc, f, ensure_ascii=False)
            f.write('\n')
    
    print(f"✅ Step 2 completed: Filtered {len(large_cluster_docs)} large cluster documents from {len(cluster_stats)} clusters")
    print(f"   Cluster statistics:")
    for cluster_id, stats in cluster_stats.items():
        categories_str = ", ".join(stats["categories"])
        print(f"     Cluster {cluster_id}: {stats['documents']} documents, categories: {categories_str}")
    
    return output_file


def pipeline_step_3_refined_clustering(input_file: str) -> str:
    """Pipeline Step 3: Refine large clusters"""
    print("🔬 Pipeline Step 3: Performing refined clustering on large clusters...")
    
    output_file = "pipeline_step3_refined_clusters.jsonl"
    cmd = [
        "semantic-clustify",
        "--input", input_file,
        "--embedding-field", "embedding",
        "--method", "hierarchical",
        "--n-clusters", "auto",
        "--output-format", "streaming-grouped",
        "--output", output_file
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"❌ Step 3 failed: {result.stderr}")
        return None
    
    print(f"✅ Step 3 completed: {output_file}")
    return output_file


def analyze_pipeline_results(streaming_file: str):
    """Analyze pipeline results"""
    print("📊 Analyzing pipeline results...")
    
    metadata = None
    clusters = []
    summary = None
    
    with open(streaming_file, 'r') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                data_type = data.get("type")
                
                if data_type == "clustering_metadata":
                    metadata = data
                elif data_type == "cluster":
                    clusters.append(data)
                elif data_type == "clustering_summary":
                    summary = data
    
    print(f"\n📈 Final results:")
    print(f"   Clustering method: {metadata.get('method', 'N/A')}")
    print(f"   Number of clusters: {len(clusters)}")
    print(f"   Total documents: {summary.get('total_documents', 'N/A')}")
    print(f"   Silhouette score: {summary.get('silhouette_score', 0):.3f}" if summary.get('silhouette_score') else "   Silhouette score: N/A")
    
    print(f"\n📋 Cluster details:")
    for cluster in clusters:
        cluster_id = cluster.get("cluster_id", -1)
        size = cluster.get("size", 0)
        density = cluster.get("density", 0)
        
        # Analyze category distribution in clusters
        categories = {}
        documents = cluster.get("documents", [])
        for doc in documents:
            cat = doc.get("category", "Unknown")
            categories[cat] = categories.get(cat, 0) + 1
        
        category_str = ", ".join([f"{cat}({count})" for cat, count in categories.items()])
        print(f"   Cluster {cluster_id}: {size} documents, density {density:.3f}, category distribution: {category_str}")


def demonstrate_streaming_processing(streaming_file: str):
    """Demonstrate streaming processing capabilities"""
    print("\n🚰 Demonstrating streaming processing capabilities...")
    
    # Simulate streaming processing: read and process line by line
    print("   🔄 Simulating real-time processing of streaming-grouped format...")
    
    processed_clusters = 0
    processed_documents = 0
    
    with open(streaming_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            if line.strip():
                data = json.loads(line)
                data_type = data.get("type")
                
                if data_type == "clustering_metadata":
                    print(f"   📋 Line {line_num}: Received metadata - method: {data.get('method')}")
                elif data_type == "cluster":
                    cluster_id = data.get("cluster_id", -1)
                    size = data.get("size", 0)
                    processed_clusters += 1
                    processed_documents += size
                    print(f"   📦 Line {line_num}: Processed cluster {cluster_id} - {size} documents")
                elif data_type == "clustering_summary":
                    print(f"   📊 Line {line_num}: Received summary - total: {data.get('total_documents')} documents")
    
    print(f"   ✅ Streaming processing completed: Processed {processed_clusters} clusters, {processed_documents} documents")


def main():
    """Main function - demonstrate complete pipeline"""
    print("🎯 Data Pipeline Integration Demo")
    print("=" * 50)
    
    try:
        # Create test data
        print("📝 Creating large test dataset...")
        input_file = create_large_dataset()
        print(f"✅ Created test data with 30 documents: {input_file}")
        
        # Pipeline Step 1: Basic clustering
        step1_output = pipeline_step_1_basic_clustering(input_file)
        if not step1_output:
            return
        
        # Pipeline Step 2: Filter large clusters
        step2_output = pipeline_step_2_filter_large_clusters(step1_output)
        if not step2_output:
            return
        
        # Pipeline Step 3: Refine clustering
        step3_output = pipeline_step_3_refined_clustering(step2_output)
        if not step3_output:
            return
        
        # Analyze final results
        analyze_pipeline_results(step3_output)
        
        # Demonstrate streaming processing
        demonstrate_streaming_processing(step3_output)
        
        print(f"\n🎯 Pipeline integration advantages:")
        print("• 🔄 enriched-labeled format supports statistics-based filtering")
        print("• 🚰 streaming-grouped format supports structured streaming processing")
        print("• 📊 Rich metadata facilitates pipeline monitoring and debugging")
        print("• 💾 High memory efficiency, suitable for large-scale data processing")
        print("• 🔗 Easy integration with other tools and frameworks")
        
    except KeyboardInterrupt:
        print("\n⚠️ Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Error occurred during demo: {e}")
    finally:
        # Clean up temporary files
        for filename in [input_file, "pipeline_step1_clusters.jsonl", 
                        "pipeline_step2_large_clusters.jsonl", 
                        "pipeline_step3_refined_clusters.jsonl"]:
            if filename and Path(filename).exists():
                Path(filename).unlink(missing_ok=True)
        
        print(f"\n✨ Pipeline demo completed!")


if __name__ == "__main__":
    main()
