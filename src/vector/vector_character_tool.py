from pymilvus import MilvusClient, CollectionSchema, FieldSchema, DataType, connections, Collection, Function, FunctionType
from llm import embedding_func
import pandas as pd
import numpy as np
import openai
import os
import uuid  # 用于生成唯一ID

'''
Create collection
'''
def create_milvus_collection(client, collection_name, summary_embeddings):
    # Define the schema
    id_field = FieldSchema(name="id", dtype=DataType.INT64, is_primary=True)

    # Get the vector dimension from the sparse vector embeddings
    vector_dim = summary_embeddings.shape[1]

    # Define the dense vector filed with dimension
    dense_vector_field = FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=vector_dim)
    #  - "character_name":"XXX",
    # - "basic_information":"XXX",
    # - "characteristics":"XXX",
    # - "biography":"XXX",
    # - "character_summary":"XXX",
    # - "script_name":"XXX"					
    # Define text and book fields
    character_name_field = FieldSchema(name="character_name", dtype=DataType.VARCHAR, max_length=60000)
    basic_information_field = FieldSchema(name="basic_information", dtype=DataType.VARCHAR, max_length=60000)
    characteristics_field = FieldSchema(name="characteristics", dtype=DataType.VARCHAR, max_length=60000)
    biography_field = FieldSchema(name="biography", dtype=DataType.VARCHAR, max_length=60000)
    character_summary_field = FieldSchema(name="character_summary", dtype=DataType.VARCHAR, max_length=60000)
    script_name_field = FieldSchema(name="script_name", dtype=DataType.VARCHAR, max_length=60000)
    # Create the collection schema with all fields
    schema = CollectionSchema(fields=[id_field, dense_vector_field, script_name_field,character_name_field, basic_information_field , characteristics_field ,biography_field,character_summary_field])

    # Check if the collection exists, if not, create it
    try:
        if not client.has_collection(collection_name):
            client.create_collection(collection_name=collection_name, schema=schema)
            print(f"Collection '{collection_name}' created with vector dimension: {vector_dim}")
            
            # 验证集合是否成功创建
            if client.has_collection(collection_name):
                print(f"Collection '{collection_name}' verification successful.")
            else:
                raise Exception(f"Failed to verify collection '{collection_name}' after creation.")
        else:
            print(f"Collection '{collection_name}' already exists. Skipping creation.")

    except Exception as e:
        print(f"Error creating collection: {e}")
        raise  # 重新抛出异常，终止程序

    return vector_dim


'''
Save Dense vector: Insert the QA questions to the vector collection
'''
def insert_dense_data_to_collection(client, collection_name, docs_embeddings, content_list, batch_size=100):
    num_docs = docs_embeddings.shape[0]

    entities = []

    for i in range(len(content_list)):
        row = docs_embeddings[i].tolist()  # 将 numpy 数组转为列表

        # 使用UUID生成唯一ID（转换为整数）
        unique_id = int(uuid.uuid4().int % 1e18)  # 取模确保在int64范围内
        
        entity = {
            "id": unique_id,  # 使用生成的唯一ID
            "dense_vector": row,
            "script_name": content_list[i][0],
            "character_name": content_list[i][1],
            "basic_information": content_list[i][2],
            "characteristics": content_list[i][3],
            "biography": content_list[i][4],
            "character_summary": content_list[i][5],

        }
        # print('entity:', entity)
        entities.append(entity)

    try:
        for start in range(0, len(entities), batch_size):
            end = min(start + 100, len(entities))
            batch = entities[start:end]
            result = client.insert(collection_name=collection_name, data=batch)
            # inserted_ids = result.ids 
            # print(f"Inserted batch {start//batch_size + 1}/{(len(entities)+batch_size-1)//batch_size}, IDs: {inserted_ids[:3]}...")

        # 强制刷新，确保数据可见
        # client.flush([collection_name])
        # print(f"Data flushed to collection '{collection_name}'.")

        index_params = MilvusClient.prepare_index_params()
        index_params.add_index(
            field_name="dense_vector",
            index_type="IVF_FLAT",
            metric_type="COSINE",
            params={"drop_ratio_build": 0.1, "nlist": 50},
        )
        # sparse_index = {
        #     "index_type": "SPARSE_INVERTED_INDEX",
        #     "metric_type": "IP"  # Inner Product
        # }
        # client.create_index(collection_name, "sparse_vector", sparse_index)
        # client.create_index("sparse_vector", sparse_index)
        print(f"Creating index on collection '{collection_name}'...")
        client.create_index(collection_name=collection_name, index_params=index_params)
        # client.create_index(collection_name=collection_name, index_params=index_params)
        
        # 加载集合用于搜索
        print(f"Loading collection '{collection_name}' for search...")
        client.load_collection(collection_name=collection_name)
        print("Load Collection Done !!!")

    except Exception as e:
        print(f"Error inserting data or creating index: {e}")
        raise  # 重新抛出异常，终止程序



def vector_query(client, collection_name, text, top_k=1):
    try:
        # 验证集合是否存在
        if not client.has_collection(collection_name):
            print(f"Error: Collection '{collection_name}' does not exist.")
            return None
            
        emb = embedding_func([text])
    except openai.NotFoundError as e:
        print(f"嵌入模型部署未找到，请检查部署名称或等待部署完成: {e}")
        return None
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return None

    try:
        search_params = {
            "metric_type": "COSINE",
            "nprobe": 128,  # 可以增加 nprobe 的值来提高召回率
        }

        print(f"Searching in collection '{collection_name}'...")
        search_res = client.search(
            collection_name=collection_name,
            data=emb,
            search_params=search_params,
            limit=top_k,
            anns_field="dense_vector",
            output_fields=["script_name", "character_name", "basic_information", "characteristics", "biography", "character_summary"],
        )

        if search_res:
            search_res = search_res[0][0]["entity"]
            print(f"Search successful. Result: {search_res.get('character_name', 'No result')}")
        else:
            print("No results found.")
            
        return search_res

    except Exception as e:
        print(f"Error during search: {e}")
        return None


if __name__ == "__main__":
    # insert
    try:
        # 尝试连接默认数据库
        db_name = "kb"
        client = MilvusClient(uri="http://10.1.15.222:19530",db_name=db_name)

        
        if db_name not in client.list_databases():
            # 若需要，可创建新数据库
            client.create_database(db_name=db_name)
            print(f"Database '{db_name}' created.")
        else:
            print(f"Database '{db_name}' already exists.")

        print(f"Using database '{db_name}'. Current collections: {client.list_collections()}")

    except Exception as e:
        print(f"连接数据库时出错: {e}")
        exit(1)

    collection_name = "character2"  # 集合名称
    processed_files = []
    
    try:
        # 遍历指定目录下的所有 xlsx 文件
        directory = '/opt/Filmdataset/demo/character2'
        if not os.path.exists(directory):
            raise FileNotFoundError(f"Directory '{directory}' does not exist.")
        # 获取目录下所有xlsx文件
        all_files = [f for f in os.listdir(directory) if f.endswith('.xlsx')]
        
        for filename in all_files:
            if filename not in processed_files:
                file_path = os.path.join(directory, filename)
                try:
                    print(f"\nProcessing file: {filename}")
                    df = pd.read_excel(file_path)
                    # df = df.head(10)
                    # 读取摘要列数据
                    summary_list = df['character_summary'].tolist()
                    
                    # 对摘要列数据进行embedding
                    all_embeddings = []
                    for i in range(0, len(summary_list), 5):
                        batch = summary_list[i:i + 5]
                        batch_embeddings = embedding_func(batch)
                        all_embeddings.extend(batch_embeddings)
                    summary_embeddings = np.array(all_embeddings)

                    # 从原始数据中提取其他字段（不包含id列）
                    data_list = df[['script_name', 'character_name','basic_information','characteristics','biography','character_summary']].values.tolist()
                    # print(data_list)
                    vector_dim = create_milvus_collection(client, collection_name, summary_embeddings)
                    print(f"向集合 {collection_name} 中插入数据...")
                    insert_dense_data_to_collection(client, collection_name, summary_embeddings, data_list)
                    processed_files.append(filename)
                    print(f"File '{filename}' processed successfully.")

                except Exception as e:
                    print(f"处理文件 {filename} 时出错: {e}")
                    continue

    except FileNotFoundError as e:
        print(f"错误：未找到目录 {directory}。")
        exit(1)
    except Exception as e:
        print(f"错误：读取文件时发生未知错误：{e}")
        exit(1)

    # 查询前再次验证集合
    print(f"\nCollections before query: {client.list_collections()}")
    
    # query
    text = "成熟稳重的男人"
    print(f"\nQuerying: '{text}'")
    res = vector_query(client, collection_name, text, 3)
    if res:
        print("\n查询结果:")
        print(f"场景名: {res['script_name']}")
        print(f"人物名: {res['character_name']}")
        print(f"人物基本信息: {res['basic_information']}")
        print(f"人物特点: {res['characteristics']}")
        print(f"人物传记: {res['biography']}")
        print(f"人物摘要: {res['character_summary']}")
    else:
        print("未找到匹配结果。")
        # Hybrid search (dense + sparse)
    
    