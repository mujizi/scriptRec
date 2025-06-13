from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
import uvicorn
from pymilvus import AsyncMilvusClient
import sys
import os

# 添加项目根目录到 Python 路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# from utils.llm import embedding_func
from src.utils.llm import async_embedding_func
from src.utils.llm import async_model_infer

# --- Milvus 连接和查询逻辑 ---

# 从环境变量或配置文件中获取配置
MILVUS_URI = os.getenv("MILVUS_URI")
DB_NAME = "kb"
scene_collection_name = "scene"
script_collection_name = "script4"
character_collection_name = "character2"

scene_similarity_threshold=0.2
script_similarity_threshold=0.2
character_similarity_threshold=0.2

async def get_client():
    async_client = AsyncMilvusClient(
        uri=MILVUS_URI,
        # token="root:Milvus",
        db_name=DB_NAME
    )
    # try:
    #     client = MilvusClient(uri=MILVUS_URI, db_name=DB_NAME)
    # except Exception as e:
    #     print(f"Error connecting to Milvus: {e}")
    #     client = None
    return async_client


async def search_scene(marking_text: Optional[str],query_text: Optional[str], top_k: int = 5):
    client = await get_client()
    prompt = f"""
    你帮我分析{marking_text}和{query_text}，分三种情况考虑：
    1. 如果{marking_text}不为空，{query_text}为空，你帮我实现：如果{marking_text}字数较长，则提取{marking_text}的核心语义，去掉一些无关信息，返回凝练后的结果，如果{marking_text}字数很短，则直接返回{marking_text}。
    2. 如果{marking_text}不为空，{query_text}不为空，你帮我实现：一起凝练{marking_text}和{query_text}，去掉一些无关信息，返回凝练后的结果。
    3. 如果{marking_text}为空，{query_text}不为空，则将{query_text}凝练，去掉一些无关信息，返回凝练后的结果。
    """
    result_text = await async_model_infer(prompt)
    emb = await async_embedding_func([result_text])
    search_params = {"metric_type": "COSINE", "nprobe": 128}
    try:
        search_res = await client.search(
            collection_name=scene_collection_name,
            data=emb,
            search_params=search_params,
            limit=top_k,
            anns_field="dense_vector",
            output_fields=["id", "scene_name", "scene_specifics", "scene_summary", "script_name"],
        )

        results = []
        if search_res:
            for hit in search_res[0]:
                if hit["distance"] >= scene_similarity_threshold:
                    entity = hit["entity"]
                    results.append({
                        "id": str(entity["id"]),
                        "scene_name": entity["scene_name"],
                        "scene_specifics": entity["scene_specifics"],
                        "scene_summary": entity["scene_summary"],
                        "script_name": entity["script_name"],
                        "similarity_score": hit["distance"]
                    })
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during Milvus search: {e}")
    
async def search_script(query_text: Optional[str], top_k: int = 5):
    client = await get_client()
    prompt = f"""
    你帮我分析{query_text}，分三种情况考虑：
    1. 如果{query_text}不为空，你帮我实现：如果{query_text}字数较长，则提取{query_text}的核心语义，去掉一些无关信息，返回凝练后的结果，如果{query_text}字数很短，则直接返回{query_text}。
    2. 如果{query_text}为空，则返回空。
    """
    result_text = await async_model_infer(prompt)
    emb = await async_embedding_func([result_text])
    search_params = {"metric_type": "COSINE", "nprobe": 128}
    try:
        search_res = await client.search(
            collection_name=script_collection_name,
            data=emb,
            search_params=search_params,
            limit=top_k,
            anns_field="dense_vector",
            output_fields=["id", "script_name", "script_theme", "script_genre", "script_type", "script_subtypes", "script_background", "script_synopsis", "script_structure", "script_summary"],
        )

        results = []
        if search_res: 
            for hit in search_res[0]:
                if hit["distance"] >= script_similarity_threshold:
                    entity = hit["entity"]
                    results.append({
                        "id": str(entity["id"]),
                        "script_name": entity["script_name"],
                        "script_theme": entity["script_theme"],
                        "script_genre": entity["script_genre"],
                        "script_type": entity["script_type"],
                        "script_subtypes": entity["script_subtypes"],
                        "script_background": entity["script_background"],
                        "script_synopsis": entity["script_synopsis"],
                        "script_structure": entity["script_structure"],
                        "script_summary": entity["script_summary"],
                        "similarity_score": hit["distance"]
                    })
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during Milvus search: {e}")
    
async def search_character(query_text: Optional[str], top_k: int = 5):
    client = await get_client()
    prompt = f"""
    你帮我分析{query_text}，分三种情况考虑：
    1. 如果{query_text}不为空，你帮我实现：如果{query_text}字数较长，则提取{query_text}的核心语义，去掉一些无关信息，返回凝练后的结果，如果{query_text}字数很短，则直接返回{query_text}。
    2. 如果{query_text}为空，则返回空。
    """
    result_text = await async_model_infer(prompt)
    emb = await async_embedding_func([result_text])
    search_params = {"metric_type": "COSINE", "nprobe": 128}
    try:
        search_res = await client.search(
            collection_name=character_collection_name,
            data=emb,
            search_params=search_params,
            limit=top_k,
            anns_field="dense_vector",
            output_fields=["id", "character_name", "basic_information", "characteristics",  "biography", "character_summary", "script_name"],
        )

        results = []
        if search_res:
            for hit in search_res[0]:
                if hit["distance"] >= character_similarity_threshold:
                    entity = hit["entity"]
                    results.append({
                        "id": str(entity["id"]),
                        "character_name": entity["character_name"],
                        "basic_information": entity["basic_information"],
                        "characteristics": entity["characteristics"],
                        "biography": entity["biography"],
                        "character_summary": entity["character_summary"],
                        "script_name": entity["script_name"],
                        "similarity_score": hit["distance"]
                    })
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during Milvus search: {e}")
    

# --- FastAPI 应用 ---

app = FastAPI(
    title="场景推荐API",
    description="基于Milvus向量数据库的影视场景推荐服务"
)

# --- Pydantic 模型 ---

class SceneRequest(BaseModel):
    marking_text: Optional[str] = Field(..., description="划词文本", example="一个男人和一个女人在亲吻")
    query_text: Optional[str] = Field(..., description="查询文本", example="雨夜的街头追逐")
    top_k: Optional[int] = Field(5, ge=1, le=20, description="返回最相似场景的数量")

class SceneResult(BaseModel):
    id: str
    scene_name: str
    script_name: str
    scene_specifics: str
    scene_summary: str
    similarity_score: float

class SceneResponse(BaseModel):
    results: List[SceneResult]

class ScriptRequest(BaseModel):
    query_text: Optional[str] = Field(..., description="查询文本", example="一个关于爱情的故事")
    top_k: Optional[int] = Field(5, ge=1, le=20, description="返回最相似剧本的数量")

class ScriptResult(BaseModel):
    id: str
    script_name: str
    script_theme: str
    script_genre: str
    script_type: str
    script_subtypes: str
    script_background: str
    script_synopsis: str
    script_structure: str
    script_summary: str
    similarity_score: float

class ScriptResponse(BaseModel):
    results: List[ScriptResult]

class CharacterRequest(BaseModel):
    query_text: Optional[str] = Field(..., description="查询文本", example="一个日本的冷酷刺客")
    top_k: Optional[int] = Field(5, ge=1, le=20, description="返回最相似人物的数量")

class CharacterResult(BaseModel):
    id: str
    character_name: str
    basic_information: str
    characteristics: str
    biography: str
    character_summary: str
    script_name: str
    similarity_score: float
class CharacterResponse(BaseModel):
    results: List[CharacterResult]


# --- API 端点 ---

@app.post(
    "/recommend/scene",
    response_model=SceneResponse,
    tags=["Scene Recommendation"],
    summary="获取场景推荐",
    description="输入一段场景描述文本，API将返回最相关的场景列表，包含相似度、所属剧本、场景详情和总结。",
)
async def recommend_scene(request: SceneRequest):
    """
    接收场景描述并从Milvus数据库返回推荐场景列表。
    
    - **query**: 场景的文本描述.
    - **top_k**: 返回结果的数量 (默认为5).
    """
    try:
        results = await search_scene(marking_text=request.marking_text, query_text=request.query_text, top_k=request.top_k)
        return {"results": results}
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

@app.post(
    "/recommend/script",
    response_model=ScriptResponse,
    tags=["Script Recommendation"],
    summary="获取剧本推荐",
    description="输入一段剧本描述文本，API将返回最相关的剧本列表，包含相似度、script_theme、script_genre、script_type、script_subtypes、script_background、script_synopsis、script_structure、script_summary。",
)
async def recommend_script(request: ScriptRequest):
    """
    接收剧本描述并从Milvus数据库返回推荐剧本列表。
    
    - **query**: 剧本的文本描述.
    - **top_k**: 返回结果的数量 (默认为5).
    """
    try:
        results = await search_script(query_text=request.query_text, top_k=request.top_k)
        return {"results": results}
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

@app.post(
    "/recommend/character",
    response_model=CharacterResponse,
    tags=["Character Recommendation"],
    summary="获取人物推荐",
    description="输入一段人物描述文本，API将返回最相关的人物列表，包含相似度、characteristics、biography、character_summary、script_name。",
)
async def recommend_character(request: CharacterRequest):
    """
    接收人物描述并从Milvus数据库返回推荐人物列表。
    
    - **query**: 人物的文本描述.
    - **top_k**: 返回结果的数量 (默认为5).
    """
    try:
        results = await search_character(query_text=request.query_text, top_k=request.top_k)
        return {"results": results}
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

# @app.get("/health", summary="健康检查", tags=["Health Check"])
# async def health_check():
#     """
#     检查API服务和Milvus连接状态
#     """
#     if not client:
#         raise HTTPException(status_code=503, detail="Milvus service is unavailable.")
    
#     try:
#         # 简单检查与Milvus的连接
#         client.has_collection(COLLECTION_NAME)
#         return {"status": "ok", "milvus_connection": "ok"}
#     except Exception as e:
#         raise HTTPException(status_code=503, detail=f"Milvus connection failed: {e}")

# --- 启动服务 ---

if __name__ == "__main__":
    uvicorn.run(app, host="10.1.80.18", port=7003) 



