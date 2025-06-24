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
def script_recommendation(text):
    """
    剧本推荐工具：根据用户的需求描述，智能推荐最符合要求的剧本作品。
    
    当用户想要寻找特定类型、主题、风格或情节的剧本时，请调用此工具。
    支持的推荐场景包括但不限于：
    - 按主题推荐：如"爱与和平"、"悬疑推理"、"青春校园"等
    - 按风格推荐：如"浪漫喜剧"、"科幻惊悚"、"历史剧"等  
    - 按情节推荐：如"复仇故事"、"成长励志"、"家庭伦理"等
    - 按情感基调推荐：如"温馨治愈"、"紧张刺激"、"感人催泪"等

    Args:
        text: 用户对想要的剧本的描述，可以包含主题、风格、情节、情感等任何相关需求
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