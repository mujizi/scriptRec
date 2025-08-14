# 推理模型及向量化模型
# 目前使用 openai gpt-4.1, text-embedding small


from openai import AzureOpenAI, AsyncAzureOpenAI
import numpy as np
import asyncio
from dotenv import load_dotenv
import os

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
Embedding_API_VERSION=os.getenv('Embedding_API_VERSION')

client = AzureOpenAI(
  azure_endpoint = AZURE_OPENAI_ENDPOINT, 
  api_key=AZURE_OPENAI_API_KEY,  
  api_version=API_VERSION
)


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
    )
    return response.choices[0].message.content


def model_infer(text):
    client = AzureOpenAI( 
        azure_endpoint = AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_API_KEY,  
        api_version=API_VERSION
    )
    response = client.chat.completions.create(
    model=AZURE_MODEL_NAME, 
    messages=[
        {"role": "user", "content": text},
    ],
    temperature=1,
    top_p=0.7,
    )
    return response.choices[0].message.content


def prompt_merge(text, prompt):
    return prompt.format(text)


def embedding_func(texts: list[str]) -> np.ndarray:
    try:
        client = AzureOpenAI(
            api_key=AZURE_OPENAI_API_KEY,
            api_version=API_VERSION,
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
        )
        embedding = client.embeddings.create(model=AZURE_EMBEDDING_DEPLOYMENT, input=texts)
        embeddings = [item.embedding for item in embedding.data]
        return np.array(embeddings)
    except Exception as e:
        print(f"Error occurred while generating embeddings: {e}")
        return np.array([])


async def async_embedding_func(texts: list[str]) -> np.ndarray:
    try:
        client = AsyncAzureOpenAI(
            api_key=AZURE_OPENAI_API_KEY,
            api_version=Embedding_API_VERSION,
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
        )
        embedding = await client.embeddings.create(model=AZURE_EMBEDDING_DEPLOYMENT, input=texts)
        embeddings = [item.embedding for item in embedding.data]
        return np.array(embeddings)
    except Exception as e:
        print(f"Error occurred while generating embeddings: {e}")
        return np.array([])


if  __name__ == '__main__':
    text = "Hello world"
    print(embedding_func([text]))
    # print(model_infer(text))
