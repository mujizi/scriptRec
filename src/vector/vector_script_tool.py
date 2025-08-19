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

    # Get the vector dimension from the summary embeddings
    vector_dim = summary_embeddings.shape[1]

    # Define the dense vector field with dimension
    dense_vector_field = FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=vector_dim)
                                                                
    # Define text and book fields
    script_name_field = FieldSchema(name="script_name", dtype=DataType.VARCHAR, max_length=65535)
    script_theme_field = FieldSchema(name="script_theme", dtype=DataType.VARCHAR, max_length=65535)
    script_type_field = FieldSchema(name="script_type", dtype=DataType.VARCHAR, max_length=65535)
    script_genre_field = FieldSchema(name="script_genre", dtype=DataType.VARCHAR, max_length=65535)
    script_subtypes_field = FieldSchema(name="script_subtypes", dtype=DataType.VARCHAR, max_length=65535)
    script_background_field = FieldSchema(name="script_background", dtype=DataType.VARCHAR, max_length=65535)
    
    # Define script_synopsis_field for text content, enabling analyzer for BM25 function input
    script_synopsis_field = FieldSchema(name="script_synopsis", dtype=DataType.VARCHAR, max_length=65535, enable_analyzer=True, analyzer_params={"type": "chinese"})
    
    script_structure_field = FieldSchema(name="script_structure", dtype=DataType.VARCHAR, max_length=65535)
    script_summary_field = FieldSchema(name="script_summary", dtype=DataType.VARCHAR, max_length=65535, enable_analyzer=True)

    # Define the sparse vector field
    sparse_vector_field = FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR)

    # Define the BM25 function that processes script_synopsis and outputs to sparse_vector
    bm25_function = Function(
        name="text_bm25_emb",
        input_field_names=["script_synopsis"], # Input text field
        output_field_names=["sparse_vector"], # Output sparse vector field
        function_type=FunctionType.BM25, # Model for processing mapping relationship
    )

    # Create the collection schema with all fields
    # Ensure script_synopsis_field and sparse_vector_field are included
    schema = CollectionSchema(
        fields=[
            id_field, 
            dense_vector_field, 
            script_name_field, 
            script_theme_field, 
            script_genre_field,
            script_type_field, 
            script_subtypes_field, 
            script_background_field, 
            script_synopsis_field, # Text field for synopsis
            script_structure_field, 
            script_summary_field,
            sparse_vector_field # Sparse vector field
        ]
    )
    
    # Add the BM25 function to the schema
    schema.add_function(bm25_function)

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
def insert_dense_data_to_collection(client, collection_name, docs_embeddings, content_list, batch_size=300):
    num_docs = docs_embeddings.shape[0]
    total_inserted = 0

    entities = []

    for i in range(len(content_list)):
        row = docs_embeddings[i].tolist()  # 将 numpy 数组转为列表

        # 使用UUID生成唯一ID（转换为整数）
        unique_id = int(uuid.uuid4().int % 1e18)  # 取模确保在int64范围内
        
        entity = {
            "id": unique_id,  # 使用生成的唯一ID
            "dense_vector": row,
            "script_name": content_list[i][0],
            "script_theme": content_list[i][1],
            "script_genre": content_list[i][2],
            "script_type": content_list[i][3],
            "script_subtypes": content_list[i][4],
            "script_background": content_list[i][5],
            "script_synopsis": content_list[i][6],
            "script_structure": content_list[i][7],
            "script_summary": content_list[i][8]
            # sparse_vector will be generated by the BM25 function automatically on insertion
        }
        entities.append(entity)

    try:
        # 使用一致的batch_size参数
        for start in range(0, len(entities), batch_size):
            end = min(start + batch_size, len(entities))
            batch = entities[start:end]
            result = client.insert(collection_name=collection_name, data=batch)
            batch_count = len(batch)
            total_inserted += batch_count
            print(f"Inserted batch {start//batch_size + 1}/{(len(entities)+batch_size-1)//batch_size}, Count: {batch_count}, Total: {total_inserted}")

        # 强制刷新，确保数据可见
        client.flush([collection_name])
        print(f"Data flushed to collection '{collection_name}'. Total inserted: {total_inserted}")

    except Exception as e:
        print(f"Error inserting data: {e}")
        raise  # 重新抛出异常，终止程序


def create_index_and_load(client, collection_name):
    try:
        # 检查是否已存在索引
        index_info_dense = client.describe_index(collection_name=collection_name, field_name="dense_vector")
        index_info_sparse = client.describe_index(collection_name=collection_name, field_name="sparse_vector")

        if index_info_dense and index_info_sparse:
            print(f"Indexes already exist on collection '{collection_name}'. Skipping index creation.")
        else:
            index_params = MilvusClient.prepare_index_params()
            index_params.add_index(
                field_name="dense_vector",
                index_type="IVF_FLAT",
                metric_type="COSINE",
                params={"drop_ratio_build": 0.1, "nlist": 50},
            )
            # Add index for sparse_vector field
            index_params.add_index(
                field_name="sparse_vector",
                index_type="AUTOINDEX",  # Default WAND index
                metric_type="BM25", # Configure relevance scoring through metric_type
                params=  {'drop_ratio_search': 0.6},
            )
            print(f"Creating index on collection '{collection_name}'...")
            client.create_index(collection_name=collection_name, index_params=index_params)
        
        # 加载集合用于搜索
        print(f"Loading collection '{collection_name}' for search...")
        client.load_collection(collection_name=collection_name)
        print("Load Collection Done !!!")

    except Exception as e:
        print(f"Error creating index or loading collection: {e}")
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
            output_fields=["script_name", "script_theme", "script_genre", "script_type", "script_subtypes", "script_background", "script_synopsis", "script_structure", "script_summary"],
        )

        if search_res:
            results = []
            for hit in search_res[0]:
                entity = hit["entity"]
                results.append(entity)
                print(f"Search successful. Result: {entity.get('script_name', 'No result')}")
            return results
        else:
            print("No results found.")
            return None

    except Exception as e:
        print(f"Error during search: {e}")
        return None


def get_all_xlsx_files(directory):
    """递归获取目录下所有xlsx文件"""
    xlsx_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.xlsx'):
                xlsx_files.append(os.path.join(root, file))
    return xlsx_files


if __name__ == "__main__":
    # insert
    try:
        # 尝试连接默认数据库
        db_name = "kb"
        client = MilvusClient(uri="http://117.36.50.198:40056", db_name=db_name)

        
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

    collection_name = "script4"  # 集合名称
    processed_files = []
    total_files = 0
    total_records = 0
    total_inserted = 0
    
    try:
        # 目录路径 替换
        directory = '/opt/Filmdataset/demo/script_3000'
        if not os.path.exists(directory):
            raise FileNotFoundError(f"Directory '{directory}' does not exist.")
        
        # 获取目录下包括子目录的所有xlsx文件
        all_files = get_all_xlsx_files(directory)
        
        print(f"在目录 '{directory}' 及其子目录下共找到 {len(all_files)} 个xlsx文件")
        
        # 收集所有文件的数据并一次性创建集合和索引
        all_embeddings_list = []
        all_data_list = []
        
        for file_path in all_files:
            # 提取文件名（不包含路径）
            filename = os.path.basename(file_path)
            
            if filename not in processed_files:
                try:
                    print(f"\nProcessing file: {filename} (完整路径: {file_path})")
                    df = pd.read_excel(file_path)
                    
                    # 记录文件行数
                    file_rows = len(df)
                    total_records += file_rows
                    print(f"File contains {file_rows} rows.")
                    
                    # 读取摘要列数据
                    summary_list = df['script_summary'].tolist()
                    
                    # 对摘要列数据进行embedding
                    file_embeddings = []
                    for i in range(0, len(summary_list), 5):
                        batch = summary_list[i:i + 5]
                        batch_embeddings = embedding_func(batch)
                        file_embeddings.extend(batch_embeddings)
                    
                    file_embeddings_array = np.array(file_embeddings)
                    all_embeddings_list.append(file_embeddings_array)
                    
                    # 从原始数据中提取其他字段（不包含id列）
                    data_list = df[['script_name', 'script_theme', 'script_genre', 'script_type', 'script_subtypes', 'script_background', 'script_synopsis', 'script_structure', 'script_summary']].values.tolist()
                    all_data_list.extend(data_list)
                    
                    processed_files.append(filename)
                    total_files += 1
                    print(f"File '{filename}' processed successfully.")

                except Exception as e:
                    print(f"处理文件 {filename} 时出错: {e}")
                    continue
        
        print(f"\n所有文件处理完成。共处理 {total_files} 个文件，总记录数: {total_records}")
        
        if not all_embeddings_list:
            print("没有有效的嵌入数据可插入。")
            exit(1)
        
        # 合并所有嵌入数据
        all_embeddings = np.vstack(all_embeddings_list)
        
        # 创建集合（如果不存在）
        vector_dim = create_milvus_collection(client, collection_name, all_embeddings)
        
        # 插入所有数据
        print(f"向集合 {collection_name} 中插入所有数据...")
        insert_dense_data_to_collection(client, collection_name, all_embeddings, all_data_list)
        
        # 创建索引并加载集合
        create_index_and_load(client, collection_name)
        
        # 获取最终集合中的记录数
        collection = Collection(name=collection_name, using=client.alias)
        num_entities = collection.num_entities
        print(f"集合 '{collection_name}' 中的最终记录数: {num_entities}")

    except FileNotFoundError as e:
        print(f"错误：未找到目录 {directory}。")
        exit(1)
    except Exception as e:
        print(f"错误：读取文件时发生未知错误：{e}")
        exit(1)

    # 查询前再次验证集合
    print(f"\nCollections before query: {client.list_collections()}")
    
    # query
    text = "西方军官在日本明治维新中自我救赎与文化认同的史诗剧。"
    print(f"\nQuerying: '{text}'")
    res = vector_query(client, collection_name, text, 3)
    if res:
        print("\n查询结果:")
        for i, result in enumerate(res, 1):
            print(f"\n结果 {i}:")
            print(f"场景名: {result['script_name']}")
            print(f"场景摘要: {result['script_summary']}")
            print(f"场景主题: {result['script_theme']}")
            print(f"场景题材: {result['script_genre']}")
            print(f"场景类型: {result['script_type']}")
            print(f"场景亚类型: {result['script_subtypes']}")
            print(f"场景背景: {result['script_background']}")
            print(f"场景梗概: {result['script_synopsis']}")
            print(f"场景结构: {result['script_structure']}")
    else:
        print("未找到匹配结果。")
