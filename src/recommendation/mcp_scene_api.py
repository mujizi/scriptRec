import os
import sys
from dotenv import load_dotenv

load_dotenv()

# 从环境变量获取并设置 PYTHONPATH
pythonpath = os.getenv("PYTHONPATH")
if pythonpath and pythonpath not in sys.path:
    sys.path.insert(0, pythonpath)

# print(os.getenv("MILVUS_URI"))
# print(os.getenv("PYTHONPATH"))

import gradio as gr
from pymilvus import MilvusClient
from src.utils.llm import embedding_func
import numpy as np
import uuid
import os




# 连接到Milvus数据库（假设场景集合名为scene）
db_name = "kb"
client = MilvusClient(uri="http://10.1.15.222:19530", db_name=db_name)
collection_name = "scene"  # 场景集合名
SIMILARITY_THRESHOLD = 0.3  # 相似度阈值

from scene_recommendation_app import vector_query

from mcp.server.fastmcp import FastMCP


mcp = FastMCP(host="0.0.0.0", port=7013)


@mcp.tool()
def script_scene_recommendation(text):
    """
    这是一个可以被大模型调用的推荐场景工具，它的功能是：当用户输入关于一些剧本场景的描述，推荐最相关的的剧本场景。

    Args:
        text: 用户输入的剧本场景描述
    """
    
    results = vector_query(client, collection_name, text, top_k=5)
    formatted_results = []
    for result in results:
        # 排除id和similarity_score，其他字段用key加换行拼接
        scene_info = []
        for key, value in result.items():
            if key not in ["id", "similarity_score"]:
                scene_info.append(f"{key}:\n{value}")
        
        # 将单个角色的信息拼接
        formatted_results.append("\n".join(scene_info))
    
    # 用"——————"分割不同元素
    return "\n——————\n".join(formatted_results)


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
    print(script_scene_recommendation("晚上，辛西娅在家中焦急打电话，担心兰登的情绪和家中的困境。兰登独自坐在车里，显得疲惫沮丧。"))