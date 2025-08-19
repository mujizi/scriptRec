#!/usr/bin/env python3
"""
人物数据BM25模型训练脚本
用于为人物推荐系统创建BM25稀疏向量模型
"""

import os
import sys
import pickle
from dotenv import load_dotenv
from pymilvus import MilvusClient
from bq_bm25 import create_bm25_embeddings, save_bm25_model
import jieba
import re

load_dotenv()

def preprocess_character_text(text):
    """预处理人物文本"""
    if not text:
        return ""
    
    # 去除特殊字符，保留中文、英文、数字
    text = re.sub(r'[^\w\s\u4e00-\u9fff]', '', text)
    # 分词处理
    words = jieba.lcut(text)
    return ' '.join(words)

def extract_character_texts_from_milvus():
    """从Milvus中提取人物文本数据"""
    try:
        # 连接到Milvus
        db_name = "kb"
        client = MilvusClient(uri="http://117.36.50.198:40056", db_name=db_name)
        collection_name = "character"
        
        if not client.has_collection(collection_name):
            print(f"Error: Collection '{collection_name}' does not exist.")
            return []
        
        # 查询所有人物数据
        print("正在从Milvus中提取人物数据...")
        results = client.query(
            collection_name=collection_name,
            filter="",
            output_fields=["character_name", "basic_information", "characteristics", 
                          "biography", "character_summary", "script_name"]
        )
        
        character_texts = []
        for item in results:
            # 组合所有文本字段
            combined_text = f"{item.get('character_name', '')} {item.get('basic_information', '')} {item.get('characteristics', '')} {item.get('biography', '')} {item.get('character_summary', '')} {item.get('script_name', '')}"
            
            # 预处理文本
            processed_text = preprocess_character_text(combined_text)
            if processed_text.strip():
                character_texts.append(processed_text)
        
        print(f"成功提取 {len(character_texts)} 个人物文本")
        return character_texts
        
    except Exception as e:
        print(f"从Milvus提取数据失败: {e}")
        return []

def create_character_bm25_model():
    """创建人物BM25模型"""
    try:
        # 提取人物文本
        character_texts = extract_character_texts_from_milvus()
        
        if not character_texts:
            print("没有找到人物文本数据")
            return None
        
        print(f"开始训练BM25模型，数据量: {len(character_texts)}")
        
        # 创建BM25模型
        bm25_model, embeddings = create_bm25_embeddings(character_texts)
        
        # 保存模型
        model_path = os.path.join(os.path.dirname(__file__), "bm25_character_model.pkl")
        save_bm25_model(bm25_model, model_path)
        
        print(f"BM25模型训练完成，已保存到: {model_path}")
        print(f"模型维度: {embeddings.shape[1]}")
        
        return bm25_model
        
    except Exception as e:
        print(f"创建BM25模型失败: {e}")
        return None

def test_bm25_model(bm25_model, test_queries):
    """测试BM25模型"""
    if not bm25_model:
        print("BM25模型未加载")
        return
    
    print("\n=== BM25模型测试 ===")
    
    for query in test_queries:
        try:
            processed_query = preprocess_character_text(query)
            query_embeddings = bm25_model.encode_queries([processed_query])
            query_row = query_embeddings._getrow(0)
            
            print(f"\n查询: {query}")
            print(f"处理后: {processed_query}")
            print(f"向量维度: {query_row.indices.size}")
            print(f"非零元素: {len(query_row.data)}")
            
        except Exception as e:
            print(f"查询 '{query}' 测试失败: {e}")

if __name__ == "__main__":
    print("=== 人物BM25模型训练脚本 ===")
    
    # 创建BM25模型
    bm25_model = create_character_bm25_model()
    
    if bm25_model:
        # 测试模型
        test_queries = [
            "勇敢的英雄角色",
            "幽默风趣的记者",
            "复杂的反派角色",
            "智慧型侦探",
            "年轻缉毒警察"
        ]
        
        test_bm25_model(bm25_model, test_queries)
        
        print("\n=== 训练完成 ===")
        print("BM25模型已准备就绪，可以在人物推荐系统中使用")
    else:
        print("BM25模型训练失败") 