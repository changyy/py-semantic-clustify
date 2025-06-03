#!/usr/bin/env python3
"""
管道整合演示腳本
展示如何在實際資料管道中使用不同的輸出格式
"""

import json
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Any


def create_large_dataset() -> str:
    """創建大型測試數據集"""
    documents = []
    
    # 科技類文檔
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
    
    # 商業類文檔
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
    
    # 科學類文檔
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
    
    # 為每個類別生成帶有相似向量的文檔
    import random
    import numpy as np
    
    def generate_embedding(base_vector: List[float], noise_level: float = 0.1) -> List[float]:
        """生成具有噪聲的相似向量"""
        return [val + random.uniform(-noise_level, noise_level) for val in base_vector]
    
    # 科技類別基向量
    tech_base = [0.8, 0.7, 0.6, 0.2, 0.1]
    for i, doc in enumerate(tech_docs):
        documents.append({
            "title": doc,
            "category": "Technology",
            "document_id": f"tech_{i:03d}",
            "embedding": generate_embedding(tech_base)
        })
    
    # 商業類別基向量
    business_base = [0.2, 0.3, 0.8, 0.7, 0.6]
    for i, doc in enumerate(business_docs):
        documents.append({
            "title": doc,
            "category": "Business",
            "document_id": f"biz_{i:03d}",
            "embedding": generate_embedding(business_base)
        })
    
    # 科學類別基向量
    science_base = [0.1, 0.8, 0.2, 0.9, 0.7]
    for i, doc in enumerate(science_docs):
        documents.append({
            "title": doc,
            "category": "Science",
            "document_id": f"sci_{i:03d}",
            "embedding": generate_embedding(science_base)
        })
    
    # 保存到臨時文件
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        for doc in documents:
            json.dump(doc, f, ensure_ascii=False)
            f.write('\n')
        return f.name


def pipeline_step_1_basic_clustering(input_file: str) -> str:
    """管道步驟 1: 基礎聚類"""
    print("🔄 管道步驟 1: 執行基礎聚類...")
    
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
        print(f"❌ 步驟 1 失敗: {result.stderr}")
        return None
    
    print(f"✅ 步驟 1 完成: {output_file}")
    return output_file


def pipeline_step_2_filter_large_clusters(input_file: str) -> str:
    """管道步驟 2: 篩選大型聚類"""
    print("🔍 管道步驟 2: 篩選大型聚類（cluster_size >= 8）...")
    
    large_cluster_docs = []
    cluster_stats = {}
    
    with open(input_file, 'r') as f:
        for line in f:
            if line.strip():
                doc = json.loads(line)
                cluster_size = doc.get("cluster_size", 0)
                cluster_id = doc.get("cluster_id", -1)
                
                # 統計聚類信息
                if cluster_id not in cluster_stats:
                    cluster_stats[cluster_id] = {
                        "size": cluster_size,
                        "documents": 0,
                        "categories": set()
                    }
                
                cluster_stats[cluster_id]["documents"] += 1
                cluster_stats[cluster_id]["categories"].add(doc.get("category", "Unknown"))
                
                # 篩選大型聚類
                if cluster_size >= 8:
                    large_cluster_docs.append(doc)
    
    # 保存篩選結果
    output_file = "pipeline_step2_large_clusters.jsonl"
    with open(output_file, 'w') as f:
        for doc in large_cluster_docs:
            json.dump(doc, f, ensure_ascii=False)
            f.write('\n')
    
    print(f"✅ 步驟 2 完成: 從 {len(cluster_stats)} 個聚類中篩選出 {len(large_cluster_docs)} 個大型聚類文檔")
    print(f"   聚類統計:")
    for cluster_id, stats in cluster_stats.items():
        categories_str = ", ".join(stats["categories"])
        print(f"     聚類 {cluster_id}: {stats['documents']} 個文檔, 類別: {categories_str}")
    
    return output_file


def pipeline_step_3_refined_clustering(input_file: str) -> str:
    """管道步驟 3: 對大型聚類進行細化"""
    print("🔬 管道步驟 3: 對大型聚類進行細化聚類...")
    
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
        print(f"❌ 步驟 3 失敗: {result.stderr}")
        return None
    
    print(f"✅ 步驟 3 完成: {output_file}")
    return output_file


def analyze_pipeline_results(streaming_file: str):
    """分析管道結果"""
    print("📊 分析管道結果...")
    
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
    
    print(f"\n📈 最終結果:")
    print(f"   聚類方法: {metadata.get('method', 'N/A')}")
    print(f"   聚類數量: {len(clusters)}")
    print(f"   總文檔數: {summary.get('total_documents', 'N/A')}")
    print(f"   輪廓係數: {summary.get('silhouette_score', 0):.3f}" if summary.get('silhouette_score') else "   輪廓係數: N/A")
    
    print(f"\n📋 各聚類詳情:")
    for cluster in clusters:
        cluster_id = cluster.get("cluster_id", -1)
        size = cluster.get("size", 0)
        density = cluster.get("density", 0)
        
        # 分析聚類中的類別分布
        categories = {}
        documents = cluster.get("documents", [])
        for doc in documents:
            cat = doc.get("category", "Unknown")
            categories[cat] = categories.get(cat, 0) + 1
        
        category_str = ", ".join([f"{cat}({count})" for cat, count in categories.items()])
        print(f"   聚類 {cluster_id}: {size} 個文檔, 密度 {density:.3f}, 類別分布: {category_str}")


def demonstrate_streaming_processing(streaming_file: str):
    """演示串流處理能力"""
    print("\n🚰 演示串流處理能力...")
    
    # 模擬串流處理：逐行讀取並處理
    print("   🔄 模擬實時處理 streaming-grouped 格式...")
    
    processed_clusters = 0
    processed_documents = 0
    
    with open(streaming_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            if line.strip():
                data = json.loads(line)
                data_type = data.get("type")
                
                if data_type == "clustering_metadata":
                    print(f"   📋 行 {line_num}: 接收到元數據 - 方法: {data.get('method')}")
                elif data_type == "cluster":
                    cluster_id = data.get("cluster_id", -1)
                    size = data.get("size", 0)
                    processed_clusters += 1
                    processed_documents += size
                    print(f"   📦 行 {line_num}: 處理聚類 {cluster_id} - {size} 個文檔")
                elif data_type == "clustering_summary":
                    print(f"   📊 行 {line_num}: 接收到摘要 - 總計: {data.get('total_documents')} 個文檔")
    
    print(f"   ✅ 串流處理完成: 處理了 {processed_clusters} 個聚類，{processed_documents} 個文檔")


def main():
    """主函數 - 演示完整管道"""
    print("🎯 資料管道整合演示")
    print("=" * 50)
    
    try:
        # 創建測試數據
        print("📝 創建大型測試數據集...")
        input_file = create_large_dataset()
        print(f"✅ 創建了包含 30 個文檔的測試數據: {input_file}")
        
        # 管道步驟 1: 基礎聚類
        step1_output = pipeline_step_1_basic_clustering(input_file)
        if not step1_output:
            return
        
        # 管道步驟 2: 篩選大型聚類
        step2_output = pipeline_step_2_filter_large_clusters(step1_output)
        if not step2_output:
            return
        
        # 管道步驟 3: 細化聚類
        step3_output = pipeline_step_3_refined_clustering(step2_output)
        if not step3_output:
            return
        
        # 分析最終結果
        analyze_pipeline_results(step3_output)
        
        # 演示串流處理
        demonstrate_streaming_processing(step3_output)
        
        print(f"\n🎯 管道整合優勢:")
        print("• 🔄 enriched-labeled 格式支援基於統計的篩選")
        print("• 🚰 streaming-grouped 格式支援結構化串流處理")
        print("• 📊 豐富的元數據便於管道監控和調試")
        print("• 💾 記憶體效率高，適合大規模數據處理")
        print("• 🔗 易於與其他工具和框架整合")
        
    except KeyboardInterrupt:
        print("\n⚠️ 演示被用戶中斷")
    except Exception as e:
        print(f"\n❌ 演示過程中發生錯誤: {e}")
    finally:
        # 清理臨時文件
        for filename in [input_file, "pipeline_step1_clusters.jsonl", 
                        "pipeline_step2_large_clusters.jsonl", 
                        "pipeline_step3_refined_clusters.jsonl"]:
            if filename and Path(filename).exists():
                Path(filename).unlink(missing_ok=True)
        
        print(f"\n✨ 管道演示完成！")


if __name__ == "__main__":
    main()
