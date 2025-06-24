#!/usr/bin/env python3
"""
高级人物推荐系统测试脚本
用于验证各种检索策略的功能和性能
"""

import os
import sys
import time
from dotenv import load_dotenv

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

load_dotenv()

# 导入检索函数
from character_recommendation_app import (
    vector_query, bm25_sparse_search, hybrid_search, 
    semantic_chunk_search, multi_field_search, ensemble_search,
    client, collection_name
)

def test_search_strategy(strategy_name, search_func, query, top_k=5):
    """测试单个检索策略"""
    print(f"\n=== 测试 {strategy_name} ===")
    print(f"查询: {query}")
    
    try:
        start_time = time.time()
        results = search_func(query, top_k)
        end_time = time.time()
        
        print(f"执行时间: {end_time - start_time:.2f}秒")
        print(f"结果数量: {len(results) if results else 0}")
        
        if results:
            print("前3个结果:")
            for i, result in enumerate(results[:3], 1):
                print(f"{i}. {result['character_name']} (相似度: {result['similarity_score']:.3f})")
                if result.get('confidence'):
                    print(f"   置信度: {result['confidence']:.2f}")
        else:
            print("未找到结果")
            
        return results, end_time - start_time
        
    except Exception as e:
        print(f"测试失败: {e}")
        return None, 0

def compare_all_strategies(query, top_k=5):
    """比较所有检索策略"""
    print(f"\n{'='*60}")
    print(f"全面策略对比测试")
    print(f"查询: {query}")
    print(f"返回数量: {top_k}")
    print(f"{'='*60}")
    
    strategies = [
        ("稠密向量检索", lambda q, k: vector_query(client, collection_name, q, k)),
        ("BM25稀疏向量检索", bm25_sparse_search),
        ("混合检索", hybrid_search),
        ("语义分块检索", semantic_chunk_search),
        ("多字段检索", multi_field_search),
        ("集成检索", ensemble_search)
    ]
    
    results_summary = []
    
    for strategy_name, search_func in strategies:
        results, execution_time = test_search_strategy(strategy_name, search_func, query, top_k)
        
        if results:
            avg_score = sum(r['similarity_score'] for r in results) / len(results)
            max_score = max(r['similarity_score'] for r in results)
            min_score = min(r['similarity_score'] for r in results)
            
            results_summary.append({
                'strategy': strategy_name,
                'execution_time': execution_time,
                'result_count': len(results),
                'avg_score': avg_score,
                'max_score': max_score,
                'min_score': min_score
            })
        else:
            results_summary.append({
                'strategy': strategy_name,
                'execution_time': execution_time,
                'result_count': 0,
                'avg_score': 0,
                'max_score': 0,
                'min_score': 0
            })
    
    # 打印汇总报告
    print(f"\n{'='*60}")
    print(f"策略对比汇总报告")
    print(f"{'='*60}")
    print(f"{'策略名称':<15} {'耗时(秒)':<10} {'结果数':<8} {'平均分':<8} {'最高分':<8} {'最低分':<8}")
    print(f"{'-'*60}")
    
    for summary in results_summary:
        print(f"{summary['strategy']:<15} {summary['execution_time']:<10.2f} "
              f"{summary['result_count']:<8} {summary['avg_score']:<8.3f} "
              f"{summary['max_score']:<8.3f} {summary['min_score']:<8.3f}")
    
    # 找出最佳策略
    best_strategy = max(results_summary, key=lambda x: x['avg_score'])
    fastest_strategy = min(results_summary, key=lambda x: x['execution_time'])
    most_results_strategy = max(results_summary, key=lambda x: x['result_count'])
    
    print(f"\n🏆 最佳策略 (平均分最高): {best_strategy['strategy']} ({best_strategy['avg_score']:.3f})")
    print(f"⚡ 最快策略: {fastest_strategy['strategy']} ({fastest_strategy['execution_time']:.2f}秒)")
    print(f"📊 结果最多: {most_results_strategy['strategy']} ({most_results_strategy['result_count']}个)")

def test_query_variations():
    """测试不同类型的查询"""
    test_queries = [
        "勇敢的英雄角色",
        "幽默风趣的记者",
        "复杂的反派角色", 
        "智慧型侦探",
        "年轻缉毒警察",
        "登山队队长",
        "太空探险家",
        "古代将军",
        "现代医生",
        "科幻机器人"
    ]
    
    print(f"\n{'='*60}")
    print(f"查询变体测试")
    print(f"{'='*60}")
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n--- 测试查询 {i}: {query} ---")
        
        # 只测试集成检索以节省时间
        results, execution_time = test_search_strategy(
            "集成检索", ensemble_search, query, top_k=3
        )
        
        if results:
            print(f"最佳匹配: {results[0]['character_name']} (相似度: {results[0]['similarity_score']:.3f})")
        else:
            print("无匹配结果")

def test_performance_scaling():
    """测试性能扩展性"""
    print(f"\n{'='*60}")
    print(f"性能扩展性测试")
    print(f"{'='*60}")
    
    query = "勇敢的英雄角色"
    top_k_values = [1, 3, 5, 10, 15, 20]
    
    print(f"查询: {query}")
    print(f"{'返回数量':<10} {'耗时(秒)':<10} {'结果数':<8}")
    print(f"{'-'*30}")
    
    for top_k in top_k_values:
        start_time = time.time()
        results = ensemble_search(query, top_k)
        end_time = time.time()
        
        print(f"{top_k:<10} {end_time - start_time:<10.2f} {len(results) if results else 0:<8}")

def main():
    """主测试函数"""
    print("🚀 高级人物推荐系统测试")
    print("="*60)
    
    # 检查Milvus连接
    try:
        if not client.has_collection(collection_name):
            print(f"❌ 错误: 集合 '{collection_name}' 不存在")
            return
        print(f"✅ Milvus连接正常，集合 '{collection_name}' 可用")
    except Exception as e:
        print(f"❌ Milvus连接失败: {e}")
        return
    
    # 测试查询
    test_query = "勇敢坚韧的英雄角色"
    
    # 1. 全面策略对比
    compare_all_strategies(test_query, top_k=5)
    
    # 2. 查询变体测试
    test_query_variations()
    
    # 3. 性能扩展性测试
    test_performance_scaling()
    
    print(f"\n{'='*60}")
    print(f"🎉 测试完成!")
    print(f"{'='*60}")

if __name__ == "__main__":
    main() 