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




# 连接到Milvus数据库（假设剧本集合名为script）
db_name = "kb"
client = MilvusClient(uri="http://10.1.15.222:19530", db_name=db_name)
collection_name = "script4"  # 人物集合名
SIMILARITY_THRESHOLD = 0.4  # 相似度阈值

from script_recommendation_app import semantic_search

from mcp.server.fastmcp import FastMCP


mcp = FastMCP(host="0.0.0.0", port=7014)


@mcp.tool()
def script_summary_recommendation(text):
    """
    这是一个可以被大模型调用的推荐剧本工具，它的功能是：当用户输入关于一些剧本的描述，推荐最相关的的剧本。

    Args:
        text: 用户输入的剧本描述
    """
    
    results = semantic_search(client, collection_name, text, top_k=5)
    formatted_results = []
    for result in results:
        # 排除id和similarity_score，其他字段用key加换行拼接
        script_info = []
        for key, value in result.items():
            if key not in ["id", "similarity_score"]:
                script_info.append(f"{key}:\n{value}")
        
        # 将单个角色的信息拼接
        formatted_results.append("\n".join(script_info))
    
    # 用"——————"分割不同元素
    return formatted_results
    # return "\n——————\n".join(formatted_results)


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
    # print(script_character_recommendation("律师"))