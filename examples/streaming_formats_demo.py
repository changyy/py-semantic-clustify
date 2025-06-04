#!/usr/bin/env python3
"""
Streaming Format Demo Script
Demonstrates the characteristics of different output formats and the advantages of streaming mechanisms
"""

import json
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Any


def create_demo_data() -> str:
    """Create demo data"""
    demo_data = []
    
    # Machine learning related documents
    ml_docs = [
        {"title": "Machine Learning Basics", "content": "Introduction to machine learning concepts", "category": "ML", "embedding": [0.1, 0.2, 0.3, 0.4, 0.5]},
        {"title": "Deep Learning Introduction", "content": "Neural network fundamentals", "category": "ML", "embedding": [0.12, 0.22, 0.32, 0.42, 0.52]},
        {"title": "Supervised Learning Methods", "content": "Classification and regression algorithms", "category": "ML", "embedding": [0.08, 0.18, 0.28, 0.38, 0.48]},
    ]
    
    # Data science related documents
    ds_docs = [
        {"title": "Data Analysis Techniques", "content": "Statistical analysis methods", "category": "DS", "embedding": [0.6, 0.1, 0.2, 0.3, 0.4]},
        {"title": "Data Visualization", "content": "Charts and dashboards", "category": "DS", "embedding": [0.62, 0.12, 0.22, 0.32, 0.42]},
        {"title": "Big Data Processing", "content": "Distributed computing", "category": "DS", "embedding": [0.58, 0.08, 0.18, 0.28, 0.38]},
    ]
    
    # Programming related documents  
    prog_docs = [
        {"title": "Python Programming", "content": "Python syntax fundamentals", "category": "Programming", "embedding": [0.9, 0.1, 0.05, 0.02, 0.03]},
        {"title": "JavaScript Development", "content": "Frontend development technologies", "category": "Programming", "embedding": [0.92, 0.12, 0.07, 0.04, 0.05]},
    ]
    
    demo_data.extend(ml_docs)
    demo_data.extend(ds_docs) 
    demo_data.extend(prog_docs)
    
    # Save to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        for doc in demo_data:
            json.dump(doc, f, ensure_ascii=False)
            f.write('\n')
        return f.name


def run_clustering(input_file: str, output_format: str) -> str:
    """Execute clustering and return output file path"""
    output_file = f"demo_output_{output_format.replace('-', '_')}.jsonl"
    
    cmd = [
        "semantic-clustify",
        "--input", input_file,
        "--embedding-field", "embedding", 
        "--method", "kmeans",
        "--n-clusters", "3",
        "--output-format", output_format,
        "--output", output_file
    ]
    
    print(f"🔄 Executing command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"❌ Error: {result.stderr}")
        return None
        
    print(f"✅ Successfully generated: {output_file}")
    return output_file


def analyze_format(file_path: str, format_name: str) -> Dict[str, Any]:
    """Analyze output format characteristics"""
    if not Path(file_path).exists():
        return {"error": "File does not exist"}
        
    file_size = Path(file_path).stat().st_size
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    analysis = {
        "format": format_name,
        "file_size": file_size,
        "line_count": len(content.strip().split('\n')),
        "streaming_friendly": format_name in ["labeled", "enriched-labeled", "streaming-grouped"],
        "memory_efficient": format_name != "grouped"
    }
    
    # Format-specific analysis
    if format_name == "grouped":
        try:
            data = json.loads(content)
            analysis["clusters"] = len(data)
            analysis["structure"] = "JSON array of arrays"
        except:
            analysis["parse_error"] = True
    
    elif format_name in ["labeled", "enriched-labeled"]:
        lines = [line for line in content.strip().split('\n') if line.strip()]
        analysis["document_lines"] = len(lines)
        analysis["structure"] = "JSONL documents with cluster_id"
        
        if format_name == "enriched-labeled":
            # Check if additional metadata exists
            try:
                first_doc = json.loads(lines[0])
                analysis["has_cluster_stats"] = "cluster_size" in first_doc
            except:
                pass
    
    elif format_name == "streaming-grouped":
        lines = [line for line in content.strip().split('\n') if line.strip()]
        analysis["total_lines"] = len(lines)
        analysis["structure"] = "JSONL with metadata, clusters, and summary"
        
        # Analyze different line types
        line_types = {}
        for line in lines:
            try:
                data = json.loads(line)
                line_type = data.get("type", "unknown")
                line_types[line_type] = line_types.get(line_type, 0) + 1
            except:
                pass
        analysis["line_types"] = line_types
    
    return analysis


def demonstrate_streaming_processing(file_path: str, format_name: str):
    """Demonstrate streaming processing capabilities"""
    print(f"\n🔄 Demonstrating {format_name} format streaming processing:")
    
    if format_name == "grouped":
        print("   ❌ Must be fully loaded into memory for processing")
        with open(file_path, 'r') as f:
            data = json.load(f)
            print(f"   📊 Loaded {len(data)} groups into memory")
    
    elif format_name == "labeled":
        print("   ✅ Can process line by line, memory efficient")
        line_count = 0
        cluster_counts = {}
        
        with open(file_path, 'r') as f:
            for line in f:
                if line.strip():
                    doc = json.loads(line)
                    cluster_id = doc.get("cluster_id", -1)
                    cluster_counts[cluster_id] = cluster_counts.get(cluster_id, 0) + 1
                    line_count += 1
        
        print(f"   📊 Stream processed {line_count} lines, found {len(cluster_counts)} groups")
    
    elif format_name == "enriched-labeled":
        print("   ✅ Can process line by line, each line contains complete context")
        large_clusters = []
        
        with open(file_path, 'r') as f:
            for line in f:
                if line.strip():
                    doc = json.loads(line)
                    if doc.get("cluster_size", 0) >= 3:  # Only process large groups
                        large_clusters.append(doc)
        
        print(f"   📊 Stream filtered {len(large_clusters)} documents belonging to large groups")
    
    elif format_name == "streaming-grouped":
        print("   ✅ Structured streaming processing, suitable for pipeline integration")
        metadata = None
        cluster_count = 0
        
        with open(file_path, 'r') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    if data.get("type") == "clustering_metadata":
                        metadata = data
                    elif data.get("type") == "cluster":
                        cluster_count += 1
        
        print(f"   📊 Metadata: {metadata.get('method')} method, {cluster_count} groups")


def main():
    """Main function"""
    print("🎯 Semantic Clustering Streaming Format Demo")
    print("=" * 50)
    
    # Create demo data
    print("\n📝 Creating demo data...")
    input_file = create_demo_data()
    print(f"✅ Created test data with 8 documents: {input_file}")
    
    # Test all formats
    formats = ["grouped", "labeled", "enriched-labeled", "streaming-grouped"]
    results = {}
    
    for format_name in formats:
        print(f"\n🧪 Testing {format_name} format:")
        output_file = run_clustering(input_file, format_name)
        
        if output_file:
            analysis = analyze_format(output_file, format_name)
            results[format_name] = {
                "file": output_file,
                "analysis": analysis
            }
            
            # Display analysis results
            print(f"   📊 File size: {analysis['file_size']} bytes")
            print(f"   📄 Line count: {analysis['line_count']}")
            print(f"   🚰 Streaming friendly: {'✅' if analysis['streaming_friendly'] else '❌'}")
            print(f"   💾 Memory efficient: {'✅' if analysis['memory_efficient'] else '❌'}")
            
            # Demonstrate streaming processing
            demonstrate_streaming_processing(output_file, format_name)
    
    # Summary comparison
    print(f"\n📊 Format comparison summary:")
    print("=" * 50)
    
    print(f"{'Format':<20} {'File Size':<10} {'Streaming':<8} {'Use Case'}")
    print("-" * 60)
    
    scenarios = {
        "grouped": "Small-scale experimental analysis",
        "labeled": "Basic pipeline processing", 
        "enriched-labeled": "Context-rich pipeline",
        "streaming-grouped": "Structured large-scale pipeline"
    }
    
    for format_name in formats:
        if format_name in results:
            analysis = results[format_name]["analysis"]
            size = analysis["file_size"]
            streaming = "✅" if analysis["streaming_friendly"] else "❌"
            scenario = scenarios.get(format_name, "Unknown")
            print(f"{format_name:<20} {size:<10} {streaming:<8} {scenario}")
    
    print(f"\n🎯 Recommended usage strategies:")
    print("• 🧪 Experimental stage: Use grouped format for analyzing cluster structure")
    print("• 🔄 Production pipeline: Use enriched-labeled, balancing efficiency and information completeness")
    print("• 🚀 Large-scale processing: Use streaming-grouped for optimal pipeline integration")
    print("• 🔗 Simple streaming: Use labeled for maximum memory efficiency")
    
    # Clean up temporary files
    Path(input_file).unlink(missing_ok=True)
    for result in results.values():
        Path(result["file"]).unlink(missing_ok=True)
    
    print(f"\n✨ Demo completed!")


if __name__ == "__main__":
    main()
