#!/usr/bin/env python3
"""
串接格式演示腳本
展示不同輸出格式的特點和串接機制的優勢
"""

import json
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Any


def create_demo_data() -> str:
    """創建演示數據"""
    demo_data = []
    
    # 機器學習相關文檔
    ml_docs = [
        {"title": "機器學習基礎", "content": "介紹機器學習概念", "category": "ML", "embedding": [0.1, 0.2, 0.3, 0.4, 0.5]},
        {"title": "深度學習入門", "content": "神經網絡基礎", "category": "ML", "embedding": [0.12, 0.22, 0.32, 0.42, 0.52]},
        {"title": "監督學習方法", "content": "分類和回歸算法", "category": "ML", "embedding": [0.08, 0.18, 0.28, 0.38, 0.48]},
    ]
    
    # 數據科學相關文檔
    ds_docs = [
        {"title": "數據分析技術", "content": "統計分析方法", "category": "DS", "embedding": [0.6, 0.1, 0.2, 0.3, 0.4]},
        {"title": "數據可視化", "content": "圖表和儀表板", "category": "DS", "embedding": [0.62, 0.12, 0.22, 0.32, 0.42]},
        {"title": "大數據處理", "content": "分布式計算", "category": "DS", "embedding": [0.58, 0.08, 0.18, 0.28, 0.38]},
    ]
    
    # 程式設計相關文檔  
    prog_docs = [
        {"title": "Python 程式設計", "content": "Python 語法基礎", "category": "Programming", "embedding": [0.9, 0.1, 0.05, 0.02, 0.03]},
        {"title": "JavaScript 開發", "content": "前端開發技術", "category": "Programming", "embedding": [0.92, 0.12, 0.07, 0.04, 0.05]},
    ]
    
    demo_data.extend(ml_docs)
    demo_data.extend(ds_docs) 
    demo_data.extend(prog_docs)
    
    # 保存到臨時文件
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        for doc in demo_data:
            json.dump(doc, f, ensure_ascii=False)
            f.write('\n')
        return f.name


def run_clustering(input_file: str, output_format: str) -> str:
    """執行聚類並返回輸出文件路径"""
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
    
    print(f"🔄 執行命令: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"❌ 錯誤: {result.stderr}")
        return None
        
    print(f"✅ 成功生成: {output_file}")
    return output_file


def analyze_format(file_path: str, format_name: str) -> Dict[str, Any]:
    """分析輸出格式特性"""
    if not Path(file_path).exists():
        return {"error": "文件不存在"}
        
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
    
    # 格式特定分析
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
            # 檢查是否有附加元數據
            try:
                first_doc = json.loads(lines[0])
                analysis["has_cluster_stats"] = "cluster_size" in first_doc
            except:
                pass
    
    elif format_name == "streaming-grouped":
        lines = [line for line in content.strip().split('\n') if line.strip()]
        analysis["total_lines"] = len(lines)
        analysis["structure"] = "JSONL with metadata, clusters, and summary"
        
        # 分析各種行類型
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
    """演示串流處理能力"""
    print(f"\n🔄 演示 {format_name} 格式的串流處理:")
    
    if format_name == "grouped":
        print("   ❌ 需要全部加載到記憶體中才能處理")
        with open(file_path, 'r') as f:
            data = json.load(f)
            print(f"   📊 加載了 {len(data)} 個群組到記憶體")
    
    elif format_name == "labeled":
        print("   ✅ 可以逐行處理，記憶體效率高")
        line_count = 0
        cluster_counts = {}
        
        with open(file_path, 'r') as f:
            for line in f:
                if line.strip():
                    doc = json.loads(line)
                    cluster_id = doc.get("cluster_id", -1)
                    cluster_counts[cluster_id] = cluster_counts.get(cluster_id, 0) + 1
                    line_count += 1
        
        print(f"   📊 串流處理了 {line_count} 行，發現 {len(cluster_counts)} 個群組")
    
    elif format_name == "enriched-labeled":
        print("   ✅ 可以逐行處理，且每行包含完整上下文")
        large_clusters = []
        
        with open(file_path, 'r') as f:
            for line in f:
                if line.strip():
                    doc = json.loads(line)
                    if doc.get("cluster_size", 0) >= 3:  # 只處理大群組
                        large_clusters.append(doc)
        
        print(f"   📊 串流篩選出 {len(large_clusters)} 個屬於大群組的文檔")
    
    elif format_name == "streaming-grouped":
        print("   ✅ 結構化串流處理，適合管道集成")
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
        
        print(f"   📊 元數據: {metadata.get('method')} 方法，{cluster_count} 個群組")


def main():
    """主函數"""
    print("🎯 語義聚類串接格式演示")
    print("=" * 50)
    
    # 創建演示數據
    print("\n📝 創建演示數據...")
    input_file = create_demo_data()
    print(f"✅ 創建了包含 8 個文檔的測試數據: {input_file}")
    
    # 測試所有格式
    formats = ["grouped", "labeled", "enriched-labeled", "streaming-grouped"]
    results = {}
    
    for format_name in formats:
        print(f"\n🧪 測試 {format_name} 格式:")
        output_file = run_clustering(input_file, format_name)
        
        if output_file:
            analysis = analyze_format(output_file, format_name)
            results[format_name] = {
                "file": output_file,
                "analysis": analysis
            }
            
            # 顯示分析結果
            print(f"   📊 文件大小: {analysis['file_size']} bytes")
            print(f"   📄 行數: {analysis['line_count']}")
            print(f"   🚰 串流友善: {'✅' if analysis['streaming_friendly'] else '❌'}")
            print(f"   💾 記憶體效率: {'✅' if analysis['memory_efficient'] else '❌'}")
            
            # 演示串流處理
            demonstrate_streaming_processing(output_file, format_name)
    
    # 總結比較
    print(f"\n📊 格式比較總結:")
    print("=" * 50)
    
    print(f"{'格式':<20} {'文件大小':<10} {'串流友善':<8} {'適用場景'}")
    print("-" * 60)
    
    scenarios = {
        "grouped": "小規模實驗分析",
        "labeled": "基礎管道處理", 
        "enriched-labeled": "上下文豐富的管道",
        "streaming-grouped": "結構化大規模管道"
    }
    
    for format_name in formats:
        if format_name in results:
            analysis = results[format_name]["analysis"]
            size = analysis["file_size"]
            streaming = "✅" if analysis["streaming_friendly"] else "❌"
            scenario = scenarios.get(format_name, "未知")
            print(f"{format_name:<20} {size:<10} {streaming:<8} {scenario}")
    
    print(f"\n🎯 建議使用策略:")
    print("• 🧪 實驗階段: 使用 grouped 格式，便於分析群組結構")
    print("• 🔄 生產管道: 使用 enriched-labeled，平衡效率和信息完整性")
    print("• 🚀 大規模處理: 使用 streaming-grouped，最佳化管道集成")
    print("• 🔗 簡單串接: 使用 labeled，最大記憶體效率")
    
    # 清理臨時文件
    Path(input_file).unlink(missing_ok=True)
    for result in results.values():
        Path(result["file"]).unlink(missing_ok=True)
    
    print(f"\n✨ 演示完成！")


if __name__ == "__main__":
    main()
