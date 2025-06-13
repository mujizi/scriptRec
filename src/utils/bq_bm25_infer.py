from pymilvus import MilvusClient, CollectionSchema, FieldSchema, DataType, connections, Collection
import spacy
import re
from pymilvus.model.sparse.bm25.tokenizers import build_default_analyzer
from pymilvus.model.sparse import BM25EmbeddingFunction
import pickle
import time
import os
from sklearn.decomposition import PCA
# from config import MILVUS_HOST, MILVUS_PORT, VECTOR_DIM
import json
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict


def load_bm25_ef(filepath):
    with open(filepath, "rb") as f:
        bm25_ef = pickle.load(f)
    return bm25_ef


filepath = "/home/vto/WorkSpace/ry_project/milvus_test/organize/bq_bm25_save.pkl"
bm25_model = load_bm25_ef(filepath)


def connect_milvus(url, db_name, collection_name):
    client = MilvusClient(uri=url, db_name=db_name)
    collection_info = client.get_collection_stats(collection_name)
   
    print("Collection Info:", collection_info)

    # 列出集合中的索引并打印出来
    indexes = client.list_indexes(collection_name)
    print(f"Indexes: {indexes}")  # 打印 indexes 以检查其结构

    if len(indexes) == 0:
        raise ValueError(f"No indexes found for collection: {collection_name}")

    # 假设 indexes 返回的是一个索引名称的列表，直接使用第一个索引名称
    index_name = indexes[0]  # 如果是字符串列表，这一行不会报错

    # 获取索引信息
    index_info = client.describe_index(collection_name, index_name)
    print(f"Index Info: {index_info}")
    return client


url = "http://10.1.15.222:19530"
db_name = "test_database"
collection_name = "bq_1"
client = connect_milvus(url, db_name, collection_name)



def infer(prompt, client, bm25_model):
    query_embedding = bm25_model.encode_queries([prompt])
    sparse_vector_dict = dict(zip(query_embedding.indices, query_embedding.data))
    search_params = {
        "metric_type": "IP",
        # "params": {"drop_ratio_search": 0.2}, # the ratio of small vector values to be dropped during search.
    }

    search_res = client.search(
        collection_name=collection_name,
        data=[sparse_vector_dict],
        limit=5,
        anns_field="vector",
        metric_type="IP",
        output_fields=["text", "book"],
        search_params=search_params)
    return [[i["entity"]["text"], i["entity"]["book"]] for i in search_res[0]]


app = FastAPI()

# 定义请求体模型
class InferRequest(BaseModel):
    prompt: str

# 定义响应模型
class InferResponse(BaseModel):
    results: List[Dict[str, str]]


@app.post("/infer", response_model=InferResponse)
async def infer_endpoint(request: InferRequest):
    try:
        # 调用 infer 函数
        results = infer(request.prompt, client, bm25_model)
        # 返回结果
        return InferResponse(results=[{"text": res[0], "book": res[1]} for res in results])
    except Exception as e:
        # 处理异常
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    # prompt = "剧本的三幕结构是什么"
    # res = infer(prompt, client, bm25_model)
    # print(res)
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7300)
