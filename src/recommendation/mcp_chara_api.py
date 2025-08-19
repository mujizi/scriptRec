import os
import sys
from dotenv import load_dotenv
import json
load_dotenv()

# 从环境变量获取并设置 PYTHONPATH
pythonpath = os.getenv("PYTHONPATH")
if pythonpath and pythonpath not in sys.path:
    sys.path.insert(0, pythonpath)

# print(os.getenv("MILVUS_URI"))
# print(os.getenv("PYTHONPATH"))


from pymilvus import MilvusClient
from src.utils.llm import embedding_func
import numpy as np
import uuid
import os




# 连接到Milvus数据库（假设人物集合名为character_analysis）
db_name = "kb"
client = MilvusClient(uri="http://117.36.50.198:40056", db_name=db_name)
collection_name = "character_collection"  # 人物集合名
SIMILARITY_THRESHOLD = 0.4  # 相似度阈值

from character_recommendation_app import vector_query

from mcp.server.fastmcp import FastMCP


mcp = FastMCP(host="0.0.0.0", port=7012)


@mcp.tool()
def script_character_recommendation(text):
    """
    剧本人物推荐工具：根据用户的需求描述，智能推荐最符合要求的剧本角色。
    
    当用户想要寻找特定类型、性格、职业或背景的剧本人物时，请调用此工具。
    支持的推荐场景包括但不限于：
    - 按职业推荐：如"律师"、"医生"、"老师"、"警察"等
    - 按性格推荐：如"勇敢的主角"、"智慧的长者"、"叛逆的青年"等
    - 按关系推荐：如"慈爱的母亲"、"严厉的父亲"、"忠诚的朋友"等
    - 按特征推荐：如"有领导力的人"、"善良的人"、"复杂的反派"等
    - 按背景推荐：如"贫困出身"、"富家子弟"、"异乡人"等
    - 按年龄推荐：如"年轻女性"、"中年男性"、"老年人"等

    Args:   
        text: 用户对想要的剧本人物的描述，可以包含职业、性格、关系、特征、背景等任何相关需求
    """
    
    results = vector_query(client, collection_name, text, top_k=5)
    return json.dumps(results, ensure_ascii=False)
    # formatted_results = []
    # for result in results:
    #     # 排除id和similarity_score，其他字段用key加换行拼接
    #     character_info = []
    #     for key, value in result.items():
    #         if key not in ["id", "similarity_score"]:
    #             character_info.append(f"{key}:\n{value}")
        
    #     # 将单个角色的信息拼接
    #     formatted_results.append("\n".join(character_info))
    # return formatted_results
    # 用"——————"分割不同元素
    # return "\n——————\n".join(formatted_results)


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
    # print(script_character_recommendation("律师"))
