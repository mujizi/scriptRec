import os
import sys
from dotenv import load_dotenv
import asyncio
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np
import jieba
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

load_dotenv()

# 从环境变量获取并设置 PYTHONPATH
pythonpath = os.getenv("PYTHONPATH")
if pythonpath and pythonpath not in sys.path:
    sys.path.insert(0, pythonpath)

print(os.getenv("MILVUS_URI"))
print(os.getenv("PYTHONPATH"))

import gradio as gr
from pymilvus import MilvusClient
from src.utils.llm import embedding_func
import numpy as np
import uuid
import os

# 检索策略枚举
class SearchStrategy(Enum):
    DENSE_VECTOR = "dense_vector"
    BM25_SPARSE = "bm25_sparse"
    HYBRID = "hybrid"
    SEMANTIC_CHUNK = "semantic_chunk"
    MULTI_FIELD = "multi_field"
    ENSEMBLE = "ensemble"

@dataclass
class SearchResult:
    id: str
    character_name: str
    basic_information: str
    characteristics: str
    biography: str
    character_summary: str
    script_name: str
    similarity_score: float
    search_strategy: SearchStrategy
    confidence: float = 0.0

# 连接到Milvus数据库（假设人物集合名为character_analysis）
db_name = "kb"
client = MilvusClient(uri="http://10.1.15.222:19530", db_name=db_name)
collection_name = "character"  # 人物集合名
SIMILARITY_THRESHOLD = 0.4  # 相似度阈值

# 尝试加载BM25模型
bm25_model = None
try:
    from src.utils.bq_bm25 import load_bm25_ef
    bm25_model_path = os.path.join(os.path.dirname(__file__), "../utils/bm25_character_model.pkl")
    if os.path.exists(bm25_model_path):
        bm25_model = load_bm25_ef(bm25_model_path)
        print("BM25模型加载成功")
except Exception as e:
    print(f"BM25模型加载失败: {e}")

def preprocess_query(query: str) -> str:
    """查询预处理"""
    # 去除特殊字符，保留中文、英文、数字
    query = re.sub(r'[^\w\s\u4e00-\u9fff]', '', query)
    # 分词处理
    words = jieba.lcut(query)
    return ' '.join(words)

def vector_query(client, collection_name, text, top_k=5):
    """原始稠密向量检索"""
    try:
        if not client.has_collection(collection_name):
            print(f"Error: Collection '{collection_name}' does not exist.")
            return None
        emb = embedding_func([text])
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return None

    try:
        search_params = {
            "metric_type": "COSINE",
            "nprobe": 128,
        }

        print(f"Searching characters in '{collection_name}'...")
        search_res = client.search(
            collection_name=collection_name,
            data=emb,
            search_params=search_params,
            limit=top_k,
            anns_field="dense_vector",
            # 匹配人物数据结构字段
            output_fields=["id", "character_name", "basic_information", "characteristics", 
                          "biography", "character_summary", "script_name"],
        )

        results = []
        for hit in search_res[0]:
            similarity_score = hit["distance"]
            if similarity_score < SIMILARITY_THRESHOLD:
                continue
            
            entity = hit["entity"]
            results.append({
                "id": entity["id"],
                "character_name": entity["character_name"],
                "basic_information": entity["basic_information"],
                "characteristics": entity["characteristics"],
                "biography": entity["biography"],
                "character_summary": entity["character_summary"],
                "script_name": entity["script_name"],
                "similarity_score": similarity_score
            })
        print(f"Found {len(results)} matching characters.")
        return results

    except Exception as e:
        print(f"Search error: {e}")
        return []

def bm25_sparse_search(text, top_k=5):
    """BM25稀疏向量检索"""
    if not bm25_model:
        return []
    
    try:
        processed_query = preprocess_query(text)
        query_embeddings = bm25_model.encode_queries([processed_query])
        query_row = query_embeddings._getrow(0)
        
        if query_row.indices.size == 0 or query_row.data.size == 0:
            return []
        
        query_sparse_vector = dict(zip(query_row.indices.astype(int), query_row.data.astype(float)))
        
        search_params = {
            "metric_type": "IP",
        }
        
        search_res = client.search(
            collection_name=collection_name,
            data=[query_sparse_vector],
            search_params=search_params,
            limit=top_k,
            anns_field="bm25_vector",
            output_fields=["id", "character_name", "basic_information", "characteristics", 
                          "biography", "character_summary", "script_name"],
        )
        
        results = []
        for hit in search_res[0]:
            similarity_score = hit["distance"]
            if similarity_score < SIMILARITY_THRESHOLD:
                continue
            
            entity = hit["entity"]
            results.append({
                "id": entity["id"],
                "character_name": entity["character_name"],
                "basic_information": entity["basic_information"],
                "characteristics": entity["characteristics"],
                "biography": entity["biography"],
                "character_summary": entity["character_summary"],
                "script_name": entity["script_name"],
                "similarity_score": similarity_score
            })
        
        return results
        
    except Exception as e:
        print(f"BM25 sparse search error: {e}")
        return []

def semantic_chunk_search(text, top_k=5):
    """语义分块检索 - 将查询分解为多个语义片段"""
    try:
        # 将查询分解为多个语义片段
        chunks = split_query_into_chunks(text)
        all_results = []
        
        for chunk in chunks:
            chunk_results = vector_query(client, collection_name, chunk, top_k=top_k//len(chunks))
            if chunk_results:
                all_results.extend(chunk_results)
        
        # 去重并重新排序
        unique_results = deduplicate_results(all_results)
        return sorted(unique_results, key=lambda x: x["similarity_score"], reverse=True)[:top_k]
        
    except Exception as e:
        print(f"Semantic chunk search error: {e}")
        return []

def split_query_into_chunks(query: str) -> List[str]:
    """将查询分解为语义片段"""
    # 简单的基于标点符号的分割
    chunks = re.split(r'[，。！？、；]', query)
    chunks = [chunk.strip() for chunk in chunks if chunk.strip()]
    
    # 如果只有一个片段，尝试按关键词分割
    if len(chunks) <= 1:
        words = jieba.lcut(query)
        if len(words) > 3:
            mid = len(words) // 2
            chunks = [' '.join(words[:mid]), ' '.join(words[mid:])]
    
    return chunks[:3]  # 最多3个片段

def multi_field_search(text, top_k=5):
    """多字段检索 - 在不同字段上分别搜索"""
    try:
        # 在角色名称字段搜索
        name_results = search_in_field(text, "character_name", top_k)
        
        # 在特征字段搜索
        characteristic_results = search_in_field(text, "characteristics", top_k)
        
        # 在传记字段搜索
        biography_results = search_in_field(text, "biography", top_k)
        
        # 合并结果
        all_results = name_results + characteristic_results + biography_results
        unique_results = deduplicate_results(all_results)
        
        return sorted(unique_results, key=lambda x: x["similarity_score"], reverse=True)[:top_k]
        
    except Exception as e:
        print(f"Multi-field search error: {e}")
        return []

def search_in_field(text, field_name, top_k):
    """在指定字段中搜索"""
    try:
        emb = embedding_func([text])
        
        search_params = {
            "metric_type": "COSINE",
            "nprobe": 128,
        }
        
        search_res = client.search(
            collection_name=collection_name,
            data=emb,
            search_params=search_params,
            limit=top_k,
            anns_field="dense_vector",
            output_fields=["id", "character_name", "basic_information", "characteristics", 
                          "biography", "character_summary", "script_name"],
        )
        
        results = []
        for hit in search_res[0]:
            similarity_score = hit["distance"]
            if similarity_score < SIMILARITY_THRESHOLD:
                continue
            
            entity = hit["entity"]
            results.append({
                "id": entity["id"],
                "character_name": entity["character_name"],
                "basic_information": entity["basic_information"],
                "characteristics": entity["characteristics"],
                "biography": entity["biography"],
                "character_summary": entity["character_summary"],
                "script_name": entity["script_name"],
                "similarity_score": similarity_score
            })
        
        return results
        
    except Exception as e:
        print(f"Field search error for {field_name}: {e}")
        return []

def hybrid_search(text, top_k=5, dense_weight=0.7):
    """混合检索 - 结合稠密向量和BM25"""
    try:
        # 并行执行两种搜索
        with ThreadPoolExecutor(max_workers=2) as executor:
            dense_future = executor.submit(vector_query, client, collection_name, text, top_k)
            bm25_future = executor.submit(bm25_sparse_search, text, top_k)
            
            dense_results = dense_future.result() or []
            bm25_results = bm25_future.result() or []
        
        # 合并结果
        combined_results = {}
        
        # 处理稠密向量结果
        for result in dense_results:
            combined_results[result["id"]] = {
                'result': result,
                'dense_score': result["similarity_score"],
                'bm25_score': 0.0
            }
        
        # 处理BM25结果
        for result in bm25_results:
            if result["id"] in combined_results:
                combined_results[result["id"]]['bm25_score'] = result["similarity_score"]
            else:
                combined_results[result["id"]] = {
                    'result': result,
                    'dense_score': 0.0,
                    'bm25_score': result["similarity_score"]
                }
        
        # 计算混合分数
        final_results = []
        for item in combined_results.values():
            hybrid_score = (dense_weight * item['dense_score'] + 
                          (1 - dense_weight) * item['bm25_score'])
            
            result = item['result']
            result["similarity_score"] = hybrid_score
            final_results.append(result)
        
        return sorted(final_results, key=lambda x: x["similarity_score"], reverse=True)[:top_k]
        
    except Exception as e:
        print(f"Hybrid search error: {e}")
        return []

def ensemble_search(text, top_k=5):
    """集成检索 - 使用多种策略并投票"""
    try:
        # 执行所有搜索策略
        strategies = [
            lambda t, k: vector_query(client, collection_name, t, k),
            bm25_sparse_search,
            semantic_chunk_search,
            multi_field_search
        ]
        
        all_results = []
        with ThreadPoolExecutor(max_workers=len(strategies)) as executor:
            futures = [executor.submit(strategy, text, top_k) for strategy in strategies]
            
            for future in as_completed(futures):
                try:
                    results = future.result() or []
                    all_results.extend(results)
                except Exception as e:
                    print(f"Strategy execution error: {e}")
        
        # 投票机制
        vote_counts = {}
        score_sums = {}
        
        for result in all_results:
            if result["id"] not in vote_counts:
                vote_counts[result["id"]] = 0
                score_sums[result["id"]] = 0.0
            
            vote_counts[result["id"]] += 1
            score_sums[result["id"]] += result["similarity_score"]
        
        # 计算最终分数
        final_results = []
        for result_id, vote_count in vote_counts.items():
            avg_score = score_sums[result_id] / vote_count
            # 投票权重
            final_score = avg_score * (1 + 0.1 * vote_count)
            
            # 找到对应的结果对象
            for result in all_results:
                if result["id"] == result_id:
                    result["similarity_score"] = final_score
                    result["confidence"] = vote_count / len(strategies)
                    final_results.append(result)
                    break
        
        return sorted(final_results, key=lambda x: x["similarity_score"], reverse=True)[:top_k]
        
    except Exception as e:
        print(f"Ensemble search error: {e}")
        return []

def deduplicate_results(results):
    """去重结果"""
    seen_ids = set()
    unique_results = []
    
    for result in results:
        if result["id"] not in seen_ids:
            seen_ids.add(result["id"])
            unique_results.append(result)
    
    return unique_results

def build_search_ui():
    def search(query, strategy_name, top_k):
        if not query.strip():
            return "请输入人物特征描述"
        
        try:
            top_k = int(top_k)
            start_time = time.time()
            
            if strategy_name == "dense_vector":
                results = vector_query(client, collection_name, query, top_k)
            elif strategy_name == "bm25_sparse":
                results = bm25_sparse_search(query, top_k)
            elif strategy_name == "hybrid":
                results = hybrid_search(query, top_k)
            elif strategy_name == "semantic_chunk":
                results = semantic_chunk_search(query, top_k)
            elif strategy_name == "multi_field":
                results = multi_field_search(query, top_k)
            elif strategy_name == "ensemble":
                results = ensemble_search(query, top_k)
            else:
                results = vector_query(client, collection_name, query, top_k)
            
            end_time = time.time()
            
            if not results:
                return f"未找到相关人物 (耗时: {end_time - start_time:.2f}s)"
            
            # 格式化结果为字符串
            formatted_results = []
            strategy_names = {
                "dense_vector": "稠密向量检索",
                "bm25_sparse": "BM25稀疏向量检索", 
                "hybrid": "混合检索",
                "semantic_chunk": "语义分块检索",
                "multi_field": "多字段检索",
                "ensemble": "集成检索"
            }
            
            strategy_name_display = strategy_names.get(strategy_name, "未知策略")
            
            for i, result in enumerate(results, 1):
                confidence_text = f" (置信度: {result.get('confidence', 0):.2f})" if result.get('confidence', 0) > 0 else ""
                character_info = [
                    f"**{i}. {result['character_name']}** (相似度: {result['similarity_score']:.3f}){confidence_text}",
                    f"**基本信息**: {result['basic_information']}",
                    f"**人物特征**: {result['characteristics']}",
                    f"**所属剧本**: {result['script_name']}",
                    f"**人物传记**: {result['biography']}",
                    f"**人物总结**: {result['character_summary']}"
                ]
                formatted_results.append("\n".join(character_info))
            
            # 用"——————"分割不同元素
            result_text = f"**检索策略**: {strategy_name_display} (耗时: {end_time - start_time:.2f}s)\n\n"
            result_text += "\n\n——————\n\n".join(formatted_results)
            return result_text
            
        except Exception as e:
            return f"搜索出错: {str(e)}"

    def compare_strategies(query, top_k):
        """比较不同检索策略的效果"""
        if not query.strip():
            return "请输入人物特征描述"
        
        comparison_results = []
        strategies = [
            ("稠密向量检索", "dense_vector"),
            ("BM25稀疏向量检索", "bm25_sparse"),
            ("混合检索", "hybrid"),
            ("语义分块检索", "semantic_chunk"),
            ("多字段检索", "multi_field"),
            ("集成检索", "ensemble")
        ]
        
        for strategy_name, strategy_key in strategies:
            try:
                start_time = time.time()
                results = search(query, strategy_key, top_k)
                end_time = time.time()
                
                comparison_results.append(f"## {strategy_name}")
                comparison_results.append(f"**执行时间**: {end_time - start_time:.2f}秒")
                comparison_results.append(f"**结果**: {results}")
                comparison_results.append("")
                
            except Exception as e:
                comparison_results.append(f"## {strategy_name}")
                comparison_results.append(f"**错误**: {str(e)}")
                comparison_results.append("")
        
        return "\n".join(comparison_results)

    with gr.Blocks(title="高级人物推荐系统", theme=gr.themes.Soft()) as demo:
        gr.HTML("""
        <style>
            .results-container {display:flex;flex-direction:column;gap:20px;}
            .result-card {background:#fff;border-radius:12px;box-shadow:0 4px 20px rgba(0,0,0,0.08);overflow:hidden;transform:translateY(10px);opacity:0;animation:fadeInUp 0.5s forwards;transition:all 0.3s ease;}
            .result-card:hover {box-shadow:0 8px 30px rgba(0,0,0,0.12);transform:translateY(-2px);}
            @keyframes fadeInUp {to{transform:translateY(0);opacity:1;}}
            .card-header {background:#f8f9fa;padding:16px 20px;display:flex;justify-content:space-between;align-items:center;border-bottom:1px solid #e9ecef;}
            .character-name {margin:0;color:#212529;font-size:1.2rem;font-weight:600;}
            .similarity-badge {padding:4px 12px;border-radius:12px;color:white;font-size:0.9rem;font-weight:500;}
            .card-content {padding:20px;}
            .character-info {display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:20px;}
            .info-row {display:flex;align-items:center;}
            .info-label {font-weight:500;color:#495057;min-width:80px;}
            .info-value {color:#212529;}
            .character-details h4 {color:#343a40;font-size:1rem;margin-top:15px;margin-bottom:5px;}
            .character-details p {color:#6c757d;margin:0 0 10px 0;line-height:1.5;}
            .gr-button {transition:all 0.3s ease;}
            .gr-button:hover {transform:translateY(-2px);}
            .gr-text-input:focus {box-shadow:0 0 0 3px rgba(13,110,253,0.25);}
            .strategy-info {background:#e3f2fd;padding:15px;border-radius:8px;margin:10px 0;}
        </style>
        """)
        
        gr.Markdown("""
        # <i class="fa-solid fa-user"></i> 高级人物推荐系统
        支持多种检索策略的智能人物推荐系统
        """)
        
        with gr.Row():
            with gr.Column(scale=3):
                with gr.Row():
                    query_input = gr.Textbox(
                        label="", 
                        placeholder="请输入人物特征描述（如：勇敢的太空探险家、复杂的反派角色）", 
                        lines=3,
                        container=False
                    )
                
                with gr.Row():
                    strategy_dropdown = gr.Dropdown(
                        choices=[
                            ("集成检索 (推荐)", "ensemble"),
                            ("稠密向量检索", "dense_vector"),
                            ("BM25稀疏向量检索", "bm25_sparse"),
                            ("混合检索", "hybrid"),
                            ("语义分块检索", "semantic_chunk"),
                            ("多字段检索", "multi_field")
                        ],
                        value="ensemble",
                        label="检索策略"
                    )
                    top_k_slider = gr.Slider(
                        minimum=1,
                        maximum=20,
                        value=5,
                        step=1,
                        label="返回结果数量"
                    )
                
                with gr.Row():
                    search_btn = gr.Button("🔍 搜索", variant="primary")
                    compare_btn = gr.Button("⚖️ 策略对比", variant="secondary")
                
            with gr.Column(scale=1):
                gr.Markdown("### <i class='fa-solid fa-info'></i> 检索策略说明")
                gr.Markdown("""
                **集成检索**: 综合多种策略，投票选出最佳结果
                
                **稠密向量**: 基于语义相似度的深度检索
                
                **BM25稀疏向量**: 基于关键词匹配的精确检索
                
                **混合检索**: 结合稠密向量和BM25的优势
                
                **语义分块**: 将查询分解为多个语义片段
                
                **多字段**: 在不同字段上分别搜索
                """)
        
        output = gr.HTML(label="搜索结果")
        
        with gr.Row():
            loading_status = gr.Markdown("", visible=False)
        
        # 搜索按钮事件
        search_btn.click(
            fn=lambda x: (gr.update(visible=True), None, gr.update(visible=False)),
            inputs=query_input,
            outputs=[loading_status, output, search_btn]
        ).then(
            fn=search,
            inputs=[query_input, strategy_dropdown, top_k_slider],
            outputs=output
        ).then(
            fn=lambda: (gr.update(visible=False), gr.update(visible=True)),
            outputs=[loading_status, search_btn]
        )
        
        # 策略对比按钮事件
        compare_btn.click(
            fn=lambda x: (gr.update(visible=True), None, gr.update(visible=False)),
            inputs=query_input,
            outputs=[loading_status, output, compare_btn]
        ).then(
            fn=compare_strategies,
            inputs=[query_input, top_k_slider],
            outputs=output
        ).then(
            fn=lambda: (gr.update(visible=False), gr.update(visible=True)),
            outputs=[loading_status, compare_btn]
        )
        
        gr.Examples(
            examples=[
                "勇敢坚韧的英雄角色",
                "性格勇敢，乐于助人的角色",
                "幽默风趣的科幻角色",
                "登山队队长",
                "年轻缉毒警察",
                "幽默的记者",
                "复杂的反派角色",
                "智慧型侦探"
            ],
            inputs=query_input
        )

    return demo

if __name__ == "__main__":
    demo = build_search_ui()
    # demo.launch(server_name="0.0.0.0", server_port=7864)
    demo.launch(server_name="0.0.0.0", server_port=7868)