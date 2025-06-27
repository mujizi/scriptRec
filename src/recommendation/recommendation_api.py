from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
import uvicorn
from pymilvus import AsyncMilvusClient
import sys
import os
from openai import AzureOpenAI, AsyncAzureOpenAI
from dotenv import load_dotenv
import json
# 添加项目根目录到 Python 路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
# 加载 .env 文件
load_dotenv('/opt/rag_milvus_kb_project/.env')

# 从环境变量获取配置
AZURE_OPENAI_API_KEY = os.getenv('AZURE_OPENAI_API_KEY')
AZURE_OPENAI_ENDPOINT = os.getenv('AZURE_OPENAI_ENDPOINT')
AZURE_EMBEDDING_DEPLOYMENT = os.getenv('AZURE_EMBEDDING_DEPLOYMENT')
API_VERSION = os.getenv('API_VERSION')
AZURE_EMBEDDING_DEPLOYMENT=os.getenv('AZURE_EMBEDDING_DEPLOYMENT')
AZURE_EMBEDDING_MODEL_NAME=os.getenv('AZURE_EMBEDDING_MODEL_NAME')
AZURE_MODEL_NAME=os.getenv('AZURE_MODEL_NAME')
# from utils.llm import embedding_func
from src.utils.llm import async_embedding_func

async def async_model_infer(text):
    client = AsyncAzureOpenAI(
        azure_endpoint = AZURE_OPENAI_ENDPOINT, 
        api_key=AZURE_OPENAI_API_KEY,  
        api_version=API_VERSION
    )
    response = await client.chat.completions.create(
    model=AZURE_MODEL_NAME, 
    messages=[
        {"role": "user", "content": text},
    ],
    temperature=1,
    top_p=0.7,
    response_format={"type": "json_object"}
    )
    return response.choices[0].message.content

# --- Milvus 连接和查询逻辑 ---

# 从环境变量或配置文件中获取配置
MILVUS_URI = os.getenv("MILVUS_URI")
DB_NAME = "kb"
scene_collection_name = "scene_bm25"
script_collection_name = "script4"
character_collection_name = "character_collection"

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


async def search_scene(marking_text: Optional[str],query_text: Optional[str], top_k: int = 6):
    """
    搜索场景推荐
    
    Args:
        marking_text: 划词文本
        query_text: 查询文本
        top_k: 返回结果数量
    """
    client = await get_client()
    prompt = f"""
你是一个搜索句生成专家，你的任务是根据以下两个变量生成一个简短的场景搜索句和关键词：
## 背景
- 划词文本是自于文章或者剧本中的一段文本划词，通常较长，如果没有请忽略
- 场景命令是用户希望搜索的场景描述，可能是短句或一个或几个关键词，如果没有请忽略

## 划词文本：
{marking_text}

## 场景命令：
{query_text}

要求：
1. 生成格式为"在[场景]里，[角色]正在[动作描述]"
2. 结合划词文本的内容特点和场景命令
3. 语言简洁生动，突出场景氛围
4. 长度控制在15-25字之间
5. 同时提取3-5个关键词，用于BM25搜索

你必须严格按照以下JSON格式返回，不能包含任何其他内容：
{{
    "search_sentence": "在雨夜的街头，两个人在追逐",
    "keywords": ["雨夜", "街头", "追逐", "紧张", "动作"]
}}

注意：
- 必须返回有效的JSON格式
- search_sentence字段包含搜索句
- keywords字段包含关键词数组
- 不要添加任何解释或额外文本
"""
    result_text = await async_model_infer(prompt)
    print('result_text', result_text)
    
    try:
        # 解析JSON结果
        result_json = json.loads(result_text)
        search_sentence = result_json.get("search_sentence", "")
        keywords = result_json.get("keywords", [])
    except json.JSONDecodeError:
        # 如果JSON解析失败，使用原始文本作为搜索句
        search_sentence = result_text
        keywords = []
    
    # 稠密向量搜索
    emb = await async_embedding_func([search_sentence])
    search_params = {"metric_type": "COSINE", "nprobe": 128}
    
    dense_results = []
    try:
        search_res = await client.search(
            collection_name=scene_collection_name,
            data=emb,
            search_params=search_params,
            limit=top_k,
            anns_field="dense_vector",
            output_fields=["id", "scene_name", "scene_specifics", "scene_summary", "script_name"],
        )

        if search_res:
            for hit in search_res[0]:
                if hit["distance"] >= scene_similarity_threshold:
                    entity = hit["entity"]
                    dense_results.append({
                        "id": str(entity["id"]),
                        "scene_name": entity["scene_name"],
                        "scene_specifics": entity["scene_specifics"],
                        "scene_summary": entity["scene_summary"],
                        "script_name": entity["script_name"],
                        "similarity_score": hit["distance"],
                        "search_type": "dense_vector"
                    })
    except Exception as e:
        print(f"Error during dense vector search: {e}")
    
    # BM25搜索
    bm25_results = []
    if keywords:
        try:
            # 构建BM25查询
            keyword_query = " ".join(keywords)
            search_res = await client.search(
                collection_name=scene_collection_name,
                data=[keyword_query],
                search_params={"metric_type": "BM25", "nprobe": 128},
                limit=top_k,
                anns_field="sparse_vector",
                output_fields=["id", "scene_name", "scene_specifics", "scene_summary", "script_name"],
            )

            if search_res:
                for hit in search_res[0]:
                    if hit["distance"] >= scene_similarity_threshold:
                        entity = hit["entity"]
                        bm25_results.append({
                            "id": str(entity["id"]),
                            "scene_name": entity["scene_name"],
                            "scene_specifics": entity["scene_specifics"],
                            "scene_summary": entity["scene_summary"],
                            "script_name": entity["script_name"],
                            "similarity_score": hit["distance"],
                            "search_type": "bm25"
                        })
        except Exception as e:
            print(f"Error during BM25 search: {e}")
    
    # 高效去重和穿插算法
    # 记录每个结果在各自搜索结果中的排名（位置索引）
    dense_ranks = {result["id"]: idx for idx, result in enumerate(dense_results)}
    bm25_ranks = {result["id"]: idx for idx, result in enumerate(bm25_results)}
    
    print("=== 排名信息 ===")
    print("稠密向量排名:", dense_ranks)
    print("BM25排名:", bm25_ranks)
    
    # 为每个结果添加排名信息（在各自搜索结果中的位置）
    for result in dense_results:
        result["rank"] = dense_ranks.get(result["id"], float('inf'))
    for result in bm25_results:
        result["rank"] = bm25_ranks.get(result["id"], float('inf'))
    
    print("=== 添加排名后的结果 ===")
    print("稠密向量结果:")
    for result in dense_results:
        print(f"  ID: {result['id']}, Rank: {result['rank']}, Scene: {result['scene_name']}")
    print("BM25结果:")
    for result in bm25_results:
        print(f"  ID: {result['id']}, Rank: {result['rank']}, Scene: {result['scene_name']}")
    
    # 在各自类型内按排名排序（排名越小越好，即位置越靠前越好）
    dense_results.sort(key=lambda x: x["rank"])
    bm25_results.sort(key=lambda x: x["rank"])
    
    # 边去重边按排名添加
    final_results = []
    seen_ids = set()  # 用于去重
    dense_idx = 0
    bm25_idx = 0
    
    print("=== 开始去重和穿插 ===")
    
    # 按全局排名顺序添加结果
    while len(final_results) < top_k and (dense_idx < len(dense_results) or bm25_idx < len(bm25_results)):
        # 获取当前两个候选结果的排名
        current_dense_rank = dense_results[dense_idx]["rank"] if dense_idx < len(dense_results) else float('inf')
        current_bm25_rank = bm25_results[bm25_idx]["rank"] if bm25_idx < len(bm25_results) else float('inf')
        
        print(f"当前比较: 稠密向量rank={current_dense_rank}, BM25 rank={current_bm25_rank}")
        
        # 选择排名更好的结果（排名越小越好）
        if current_dense_rank <= current_bm25_rank and dense_idx < len(dense_results):
            # 添加稠密向量结果
            current_dense = dense_results[dense_idx]
            if current_dense["id"] not in seen_ids:
                final_results.append(current_dense)
                seen_ids.add(current_dense["id"])
                print(f"添加稠密向量: ID={current_dense['id']}, Rank={current_dense['rank']}, Scene={current_dense['scene_name']}")
            dense_idx += 1
        elif bm25_idx < len(bm25_results):
            # 添加BM25结果
            current_bm25 = bm25_results[bm25_idx]
            if current_bm25["id"] not in seen_ids:
                final_results.append(current_bm25)
                seen_ids.add(current_bm25["id"])
                print(f"添加BM25: ID={current_bm25['id']}, Rank={current_bm25['rank']}, Scene={current_bm25['scene_name']}")
            elif current_bm25["id"] in seen_ids:
                # 如果已存在，比较排名决定是否替换
                existing_idx = next(i for i, r in enumerate(final_results) if r["id"] == current_bm25["id"])
                existing_result = final_results[existing_idx]
                
                # 如果BM25排名更好（排名更小），替换
                if current_bm25["rank"] < existing_result["rank"]:
                    final_results[existing_idx] = current_bm25
                    print(f"替换为BM25: ID={current_bm25['id']}, Rank={current_bm25['rank']}, Scene={current_bm25['scene_name']}")
            bm25_idx += 1
        
        # 检查是否已达到top_k
        if len(final_results) >= top_k:
            break
    
    print("=== 最终结果 ===")
    print(f"请求top_k: {top_k}, 实际返回: {len(final_results)}")
    for i, result in enumerate(final_results):
        print(f"{i+1}. ID: {result['id']}, Rank: {result['rank']}, Type: {result['search_type']}, Scene: {result['scene_name']}")
    
    # 确保返回结果不超过top_k
    final_results = final_results[:top_k]
    
    # 移除rank字段，只保留需要的字段
    for result in final_results:
        result.pop("rank", None)
    
    return final_results

async def search_script(query_text: Optional[str], top_k: int = 6):
    client = await get_client()
    result_text = await async_model_infer(query_text)
    print('result_text', result_text)
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
    
async def search_character(query_text: Optional[str], top_k: int = 6):
    client = await get_client()
    result_text = await async_model_infer(query_text)
    print('result_text', result_text)
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
    search_type: str = Field(..., description="搜索类型：dense_vector 或 bm25")

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
    
    - **marking_text**: 划词文本.
    - **query_text**: 场景的文本描述.
    - **top_k**: 返回结果的数量 (默认为5).
    """
    try:
        results = await search_scene(
            marking_text=request.marking_text, 
            query_text=request.query_text, 
            top_k=request.top_k
        )
        print(results)
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
        print(results)
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
        print(results)
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



