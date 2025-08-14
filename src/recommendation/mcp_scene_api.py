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




# 连接到Milvus数据库（假设场景集合名为scene）
db_name = "kb"
client = MilvusClient(uri="http://10.1.15.222:19530", db_name=db_name)
collection_name = "scene_bm25"  # 场景集合名
SIMILARITY_THRESHOLD = 0.3  # 相似度阈值

from scene_recommendation_app import vector_query

from mcp.server.fastmcp import FastMCP


mcp = FastMCP(host="0.0.0.0", port=7013)


@mcp.tool()
def script_scene_recommendation(text):
    """
    剧本场景推荐工具：根据用户的场景需求描述，智能推荐最符合要求的剧本场景。
    
    当用户想要寻找特定类型、环境、氛围或情节的剧本场景时，请调用此工具。
    支持的推荐场景包括但不限于：
    - 按场景类型推荐：如"办公室场景"、"家庭聚餐"、"户外追逐"等
    - 按环境氛围推荐：如"紧张的审讯室"、"温馨的咖啡厅"、"神秘的地下室"等
    - 按情节功能推荐：如"告白场景"、"冲突爆发"、"和解场面"等
    - 按时间地点推荐：如"深夜街头"、"雨天室内"、"阳光海滩"等
    - 按人物状态推荐：如"独自沉思"、"激烈争吵"、"温情对话"等

    Args:
        text: 用户对想要的剧本场景的描述，可以包含环境、氛围、情节、人物状态等任何相关需求，例如：1.在办公室里，主角正在紧张地准备一份重要的报告。2.在咖啡厅里，两位主角正在讨论一个重要的商业计划。3.在海滩上，主角正在享受阳光和海风。
    """

    
    results = vector_query(client, collection_name, text, top_k=5)
    return json.dumps(results, ensure_ascii=False)
    # formatted_results = []
    # for result in results:
    #     # 排除id和similarity_score，其他字段用key加换行拼接
    #     scene_info = []
    #     for key, value in result.items():
    #         if key not in ["id", "similarity_score"]:
    #             scene_info.append(f"{key}:\n{value}")
        
    #     # 将单个角色的信息拼接
    #     formatted_results.append("\n".join(scene_info))
    # return formatted_results
    # 用"——————"分割不同元素
    # return "\n——————\n".join(formatted_results)


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
    print(script_scene_recommendation("晚上，辛西娅在家中焦急打电话，担心兰登的情绪和家中的困境。兰登独自坐在车里，显得疲惫沮丧。"))