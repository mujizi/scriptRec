

from pymilvus import MilvusClient, CollectionSchema, FieldSchema, DataType, connections, Collection
import re
from pymilvus.model.sparse.bm25.tokenizers import build_default_analyzer
from pymilvus.model.sparse import BM25EmbeddingFunction
import pickle
import time
import os
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
# from config import MILVUS_HOST, MILVUS_PORT, VECTOR_DIM
import json
import numpy as np
import jieba




all_codecs = [
    'utf-8', 'gb2312', 'gbk', 'utf_16', 'ascii', 'big5', 'big5hkscs',
    'cp037', 'cp273', 'cp424', 'cp437',
    'cp500', 'cp720', 'cp737', 'cp775', 'cp850', 'cp852', 'cp855', 'cp856', 'cp857',
    'cp858', 'cp860', 'cp861', 'cp862', 'cp863', 'cp864', 'cp865', 'cp866', 'cp869',
    'cp874', 'cp875', 'cp932', 'cp949', 'cp950', 'cp1006', 'cp1026', 'cp1125',
    'cp1140', 'cp1250', 'cp1251', 'cp1252', 'cp1253', 'cp1254', 'cp1255', 'cp1256',
    'cp1257', 'cp1258', 'euc_jp', 'euc_jis_2004', 'euc_jisx0213', 'euc_kr',
    'gb2312', 'gb18030', 'hz', 'iso2022_jp', 'iso2022_jp_1', 'iso2022_jp_2',
    'iso2022_jp_2004', 'iso2022_jp_3', 'iso2022_jp_ext', 'iso2022_kr', 'latin_1',
    'iso8859_2', 'iso8859_3', 'iso8859_4', 'iso8859_5', 'iso8859_6', 'iso8859_7',
    'iso8859_8', 'iso8859_9', 'iso8859_10', 'iso8859_11', 'iso8859_13',
    'iso8859_14', 'iso8859_15', 'iso8859_16', 'johab', 'koi8_r', 'koi8_t', 'koi8_u',
    'kz1048', 'mac_cyrillic', 'mac_greek', 'mac_iceland', 'mac_latin2', 'mac_roman',
    'mac_turkish', 'ptcp154', 'shift_jis', 'shift_jis_2004', 'shift_jisx0213',
    'utf_32', 'utf_32_be', 'utf_32_le''utf_16_be', 'utf_16_le', 'utf_7'
]


def find_codec(blob):
    global all_codecs
    for c in all_codecs:
        try:
            blob[:1024].decode(c)
            return c
        except Exception as e:
            pass
        try:
            blob.decode(c)
            return c
        except Exception as e:
            pass

    return "utf-8"


class TxtParser:
    def __call__(self, fnm, binary=None, chunk_token_num=128, delimiter="\n!?;。；！？"):
        txt = ""
        if binary:
            encoding = find_codec(binary)
            txt = binary.decode(encoding, errors="ignore")
        else:
            with open(fnm, "r") as f:
                while True:
                    l = f.readline()
                    if not l:
                        break
                    txt += l
        # return self.parser_txt(txt, chunk_token_num, delimiter)
        return self.split_chunk(txt, 512)

    @classmethod
    def parser_txt(cls, txt, chunk_token_num=128, delimiter="\n!?;。；！？"):
        if type(txt) != str:
            raise TypeError("txt type should be str!")
        sections = []
        for sec in re.split(r"[%s]+"%delimiter, txt):
            # if sections and sec in delimiter:
            #     sections[-1][0] += sec
            #     continue
            if len(jieba.lcut(sec)) > 10 * int(chunk_token_num):
                sections.append(sec[: int(len(sec) / 2)])
                sections.append(sec[int(len(sec) / 2) :])
            else:
                sections.append(sec)
        return sections
    

    def split_chunk(self, text, chunk_size):
        return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]



'''
Preprocessing the text
'''


# Delete the blank lines
def remove_blank_lines(text):
    return "\n".join([line for line in text.split("\n") if line.strip() != ""])


# Make sure the text.length is smaller than 1024
def truncate_text(text, max_length=1024):
    return text if len(text) <= max_length else text[:max_length]



def read_and_process_files(books_dir):
    all_texts = []
    parser = TxtParser()
    for filename in os.listdir(books_dir):
        # Preprocess each txt file
        txt_path = books_dir + filename
        with open(txt_path, 'r', encoding='utf-8') as file:
            text = file.read()
            text = remove_blank_lines(text)
            chunks = parser.split_chunk(text, 512)
            chunks = [[i, filename.split(".")[0]]for i in chunks]
            all_texts.extend(chunks)
    return all_texts


    
'''
Calculating the BM25
'''
    
def create_bm25_embeddings(chunks):
    analyzer = build_default_analyzer(language="zh")
    bm25_ef = BM25EmbeddingFunction(analyzer)

    start_time = time.time()
    bm25_ef.fit(chunks)
    end_time = time.time()
    print("bm25运行时间: {:.2f} 秒".format(end_time - start_time))
    docs_embeddings = bm25_ef.encode_documents(chunks)
    return bm25_ef, docs_embeddings

'''
Create collection
'''
def create_milvus_collection(client, collection_name, docs_embeddings):
    # Define the schema
    id_field = FieldSchema(name="id", dtype=DataType.INT64, is_primary=True)
    vector_dim = docs_embeddings.shape[1]
    # vector_field = FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=docs_embeddings.shape[1])
    vector_field = FieldSchema(name="vector", dtype=DataType.SPARSE_FLOAT_VECTOR)
    
    text_field = FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=2000)  

    book_field = FieldSchema(name="book", dtype=DataType.VARCHAR, max_length=50)

    schema = CollectionSchema(fields=[id_field, vector_field, text_field, book_field], description="BM25 text collection")

    # Check if the collection already exists
    if client.has_collection(collection_name=collection_name):
        client.drop_collection(collection_name=collection_name)

    # Create a new collection
    client.create_collection(collection_name=collection_name,schema=schema)
    print(f"Collection created with vector dimension: {vector_dim}")

    return docs_embeddings, vector_dim


def insert_data_to_collection(client, collection_name, docs_embeddings, chunks, batch_size=100):
    num_docs = len(docs_embeddings)
    
    # Print the dimensionality of the vectors
    if num_docs > 0:
        vector_dim = len(docs_embeddings[0])
        # 查看插入数据的维度
        print(f"Dimensionality of inserted vectors: {vector_dim}")
    
    
    for start in range(0, num_docs, batch_size):
        end = min(start + batch_size, num_docs)  # Make sure end does not exceed the array size
        data_batch = [
            {"id": i, "vector": docs_embeddings[i].tolist(), "text": truncate_text(chunks[i])} for i in range(start, end)]

        for data in data_batch:
            # print(f"ID: {data['id']}, Vector: {data['vector'][:5]}...")
            client.insert(collection_name=collection_name, data=[data])

    index_params = MilvusClient.prepare_index_params()
    index_params.add_index(
        field_name="vector",
        index_type="FLAT",
        metric_type="IP",
        params={"nlist": 128}
    )
   
    # Create index
    
    client.create_index(collection_name=collection_name, index_params=index_params)

    # Load collection
    client.load_collection(collection_name=collection_name)


def insert_sparse_data_to_collection(client, collection_name, docs_embeddings, chunks, batch_size=100):

    num_docs = docs_embeddings.shape[0]

    entities = []
    i = 0

    for i in range(num_docs):
        row = docs_embeddings._getrow(i)
        if row.indices.size == 0 or row.data.size == 0:
            continue

        sparse_vector_dict = dict(zip(row.indices, row.data))
        entity = {
            "id": i,
            "vector": sparse_vector_dict,
            "text": truncate_text(chunks[i][0]),
            "book": chunks[i][1]
        }
        entities.append(entity)


    for start in range(0, len(entities), 100):
        end = min(start + 100, num_docs)
        batch = entities[start:end]
        client.insert(collection_name=collection_name, data=batch)


    index_params = MilvusClient.prepare_index_params()
    index_params.add_index(
        field_name="vector",
        index_type="SPARSE_INVERTED_INDEX",
        metric_type="IP",
        params={"drop_ratio_build": 0.1},
    )
   
    # # Create index
    client.create_index(collection_name=collection_name, index_params=index_params)

    # # Load collection
    client.load_collection(collection_name=collection_name)
    
    
    
'''
Save vector and text data as JSON files
'''

def save_vectors_and_texts_as_json(collection_name, json_file_path):
    # Load the collection
    collection = Collection(name=collection_name)
    
    # Retrieve all data
    # Assumes that the collection has a vector field named "vector" and a text field named "text"
    data = collection.query(expr="*", output_fields=["vector", "text"])

    # Extract vectors and text
    vectors = []
    texts = []
    for item in data:
        vectors.append(item["vector"])
        texts.append(item["text"])
    
    # Save to JSON
    with open(json_file_path, 'w') as f:
        json.dump({"vectors": vectors, "texts": texts}, f)

    
'''
Save bm25_ef object
'''
def save_bm25_model(bm25_ef, path):
    with open(path, "wb") as f:
        pickle.dump(bm25_ef, f)

def load_bm25_ef(filepath):
    with open(filepath, "rb") as f:
        bm25_ef = pickle.load(f)
    return bm25_ef



if __name__ == "__main__":
    books_dir = '/home/vto/WorkSpace/ry_project/milvus_test/bq_book/'
    query_list = ["剧本的三个结构"]
    collection_name = "bq_1"
    pkl_path = "bq_bm25_save.pkl"

    client = MilvusClient(uri="http://10.1.15.222:19530", db_name="test_database")

    chunks = read_and_process_files(books_dir)
    # for i in chunks:
    #     print(i, "\n*******")

    bm25_ef, docs_embeddings = create_bm25_embeddings([i[0] for i in chunks])
    save_bm25_model(bm25_ef, pkl_path)
    docs_embeddings, vector_dim = create_milvus_collection(client, collection_name, docs_embeddings)
    insert_sparse_data_to_collection(client, collection_name, docs_embeddings, chunks)
    
    

